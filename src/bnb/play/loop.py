"""The match loop: the realtime sim at its fixed tick, driven by the display's clock.

One thread and a fixed timestep: each frame adds the elapsed time (times the time scale)
to an accumulator, the sim steps while the accumulator holds a whole tick, and the frame
draws at alpha = accumulator / tick between the last two snapshots. The match draws its
randomness from its own generator (seeded by the setup), and the loop draws none, so the
setup and the per-tick actions decide every state.

Each side (team) has a controller (`bnb.play.sides`): scripted, or a trained agent. One
ship at a time may be flown by the human, which replaces that ship's action only. Each
tick's kills, captures and the match's end are kept as `MatchEvent`s, for the kill feed
and the replay's markers.
"""

import dataclasses
import time

import numpy as np
import torch

from bnb.contracts import GameConfig, Obs
from bnb.contracts.obs import GLOBAL
from bnb.controllers.human import INPUTS, human_action
from bnb.game.api import TEAMS
from bnb.runtime.realtime import Realtime

from .interpolate import interpolate
from .snapshot import Snapshot, snapshot
from .tuning import Tuning

TIME_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)  # playback speeds, times real time
TIME_SCALE_LABELS = {0.25: "×¼", 0.5: "×½"}  # the others are written "×<n>"
CATCH_UP_TICKS = 4  # at most this many ticks of backlog per unit of time scale


@dataclasses.dataclass(frozen=True)
class SideSetup:
    """One side's controller (`bnb.play.sides.CONTROLLERS`): "scripted" (the chaser),
    "scripted_agent" (the scripted agent, with its memory), or "agent" with the policy
    file's directory and the sampling temperature."""

    controller: str = "scripted"  # "scripted", "scripted_agent" or "agent"
    policy: str | None = None
    temperature: float = 1.0


@dataclasses.dataclass(frozen=True)
class MatchSetup:
    """Everything a match starts from: the game, the lineup (each slot's ship type, the
    same for both teams; None for the game's default), fog, the seed of the match's
    generator, each side's controller, and the tuning it starts from (`Tuning`: the rules'
    settings and the ship card where they differ from the game's)."""

    config: GameConfig
    lineup: tuple[int, ...] | None = None
    fogged: bool = True
    seed: int = 0
    sides: tuple[SideSetup, SideSetup] = (SideSetup(), SideSetup())
    tuning: Tuning = Tuning()


@dataclasses.dataclass(frozen=True)
class MatchEvent:
    """Something the feed and the markers show: at `tick` (ticks into the episode, the
    state after that many ticks), a "kill" (`ship` killed by `other`, -1 for the
    boundary), a "capture" (`team` captured a zone) or the match's "end" (`team` won, -1
    for a draw)."""

    tick: int
    kind: str  # "kill", "capture" or "end"
    ship: int = -1
    other: int = -1
    team: int = -1


def team_view(observation: Obs, team: int) -> Obs:
    """One team's view (1, ...) out of the realtime observation (1, 2, ...)."""
    return Obs(
        ships=observation.ships[:, team],
        fields=observation.fields[:, team],
        globals=observation.globals[:, team],
        zones=observation.zones[:, team],
    )


def tick_events(realtime: Realtime, before: Snapshot, after: Snapshot, tick: int) -> list:
    """The kills, the capture and the end in the tick that led from `before` to `after`
    (`tick` is its number in the episode)."""
    events = realtime.events
    died = events.died[0].numpy()
    killer = events.killer[0].numpy()
    found = [
        MatchEvent(tick, "kill", ship=int(ship), other=int(killer[ship]))
        for ship in np.flatnonzero(died)
    ]
    done = bool(events.done[0])
    if not done and after.front != before.front:
        found.append(MatchEvent(tick, "capture", team=0 if after.front > before.front else 1))
    if done:
        result = int(events.result[0])
        found.append(MatchEvent(tick, "end", team={1: 0, -1: 1}.get(result, -1)))
    return found


class MatchLoop:
    """A live match: the realtime sim, a controller per side, and at most one ship flown
    by the human.

    `pilot` is the flown ship's sim index (None for none); it follows `inputs`, a bool
    tensor in `bnb.controllers.human.INPUTS` order that the shell fills each frame.
    `timings` holds the seconds the last tick spent per stage. `ticks` counts every tick
    run; `restarted` (N,) collects the ships that spawned (every ship does at a reset)
    since the shell last cleared it, so their trails restart even when a frame runs several
    ticks. `events` holds this episode's `MatchEvent`s (`kills` and `deaths` count them per
    ship), and `on_tick`, when set, is called after every tick with the actions, the
    realtime match and the tick's events (the replay recorder); `on_retune`, when set, is
    called after every live change (`retune`) with the realtime match.
    """

    replaying = False

    def __init__(
        self,
        setup: MatchSetup,
        sides: list | None = None,
        compiled: bool = True,
        background: bool = False,
        pilot: int | None = None,
        realtime: Realtime | None = None,
    ) -> None:
        from .sides import build_sides

        self.setup = setup
        if realtime is None:
            realtime = Realtime(
                setup.config,
                compiled=compiled,
                fogged=setup.fogged,
                generator=torch.Generator().manual_seed(setup.seed),
                lineup=setup.lineup,
                background=background,
                settings=setup.tuning.rules,
                types=setup.tuning.types(setup.config.ship_set) if setup.tuning.card else None,
            )
        self.realtime = realtime
        self.rules, self.types = realtime.rules, realtime.types
        self.sides = build_sides(setup, self.rules, self.types) if sides is None else sides
        self.pilot = pilot
        self.inputs = torch.zeros(len(INPUTS), dtype=torch.bool)
        self.time_scale = 1.0
        self.paused = False
        self.accumulator = 0.0
        self.timings = {"controllers": 0.0, "sim": 0.0, "snapshot": 0.0}
        self.current = self._snapshot()
        self.previous = self.current
        self.ticks = 0
        self.restarted = np.ones(self.rules.num_ships, dtype=bool)
        self.events: list[MatchEvent] = []
        self.kills = np.zeros(self.rules.num_ships, np.int64)
        self.deaths = np.zeros(self.rules.num_ships, np.int64)
        self.last_actions = torch.zeros(self.rules.num_ships, 3, dtype=torch.int32)
        self.on_tick = None
        self.on_retune = None

    def retune(self, rules=None, types=None) -> None:
        """Play on with new rules and/or a new ship card of the same sizes from the next
        tick (`Realtime.retune`); the shown state is re-read, so a change shows even while
        paused."""
        self.realtime.retune(rules=rules, types=types)
        self.rules, self.types = self.realtime.rules, self.realtime.types
        self.current = self._snapshot()
        self.previous = dataclasses.replace(
            self.previous, zones=self.current.zones, arena_radius=self.current.arena_radius
        )
        if self.on_retune is not None:
            self.on_retune(self.realtime)

    @property
    def tick_length(self) -> float:
        """Seconds of game time per tick."""
        return self.rules.tick_length

    @property
    def alpha(self) -> float:
        """How far the drawn frame is from the previous tick to the current one, in [0, 1]."""
        return min(self.accumulator / self.tick_length, 1.0)

    @property
    def game_time(self) -> float:
        """Seconds of play the drawn frame is at, counted across episodes."""
        return (self.ticks - 1 + self.alpha) * self.tick_length

    def _snapshot(self) -> Snapshot:
        return snapshot(self.rules, self.types, self.realtime.state, self.realtime.obs)

    @torch.no_grad()
    def actions(self) -> torch.Tensor:
        """(N, 3) int32 actions in sim order: each side's controller's, and the pilot's
        from the inputs."""
        observation = self.realtime.obs
        reset = observation.globals[:, 0, GLOBAL["reset"]] > 0.5
        actions = torch.cat(
            [self.sides[team].act(team_view(observation, team), reset) for team in range(TEAMS)]
        )
        if self.pilot is not None:
            actions[self.pilot] = human_action(self.inputs).packed()
        return actions

    def _eager(self) -> bool:
        """Whether the next tick must run uncompiled (a replay repeats ticks as run)."""
        return False

    def _count(self, events: list[MatchEvent]) -> None:
        for event in events:
            if event.kind == "kill":
                self.deaths[event.ship] += 1
                if event.other >= 0:
                    self.kills[event.other] += 1
        if any(event.kind == "end" for event in events):
            self.events = []
            self.kills[:] = 0
            self.deaths[:] = 0
        else:
            self.events += events

    def tick(self, actions: torch.Tensor) -> None:
        """Step the sim once with these (N, 3) actions."""
        started = time.perf_counter()
        episode_tick = self.current.tick + 1
        self.realtime.step(actions, eager=self._eager())
        stepped = time.perf_counter()
        self.previous, self.current = self.current, self._snapshot()
        self.ticks += 1
        self.restarted |= self.current.ships.spawned
        self.last_actions = actions
        events = tick_events(self.realtime, self.previous, self.current, episode_tick)
        if self.on_tick is not None:
            self.on_tick(actions, self.realtime, events)
        self._count(events)
        self.timings["sim"] = stepped - started
        self.timings["snapshot"] = time.perf_counter() - stepped

    def step(self) -> None:
        """One tick with the controllers' actions."""
        started = time.perf_counter()
        actions = self.actions()
        self.timings["controllers"] = time.perf_counter() - started
        self.tick(actions)

    def advance(self, elapsed: float) -> int:
        """Run the ticks `elapsed` seconds of display time hold; returns how many ran.
        Paused, nothing runs. A backlog beyond CATCH_UP_TICKS per unit of time scale is
        dropped, so a stall does not snowball."""
        if self.paused:
            return 0
        limit = CATCH_UP_TICKS * self.time_scale * self.tick_length
        self.accumulator = min(self.accumulator + elapsed * self.time_scale, limit)
        ticks = 0
        while self.accumulator >= self.tick_length:
            self.step()
            self.accumulator -= self.tick_length
            ticks += 1
        return ticks

    def single_step(self) -> None:
        """While paused, advance exactly one tick (and draw it without interpolation)."""
        if self.paused:
            self.step()
            self.accumulator = self.tick_length

    def frame(self) -> Snapshot:
        """The state to draw now."""
        return interpolate(self.previous, self.current, self.alpha)

    def next_pilot(self) -> None:
        """Hand the keyboard to the next ship in sim order (the released ship returns to
        its side's controller)."""
        current = -1 if self.pilot is None else self.pilot
        self.pilot = (current + 1) % self.rules.num_ships

    def faster(self) -> None:
        """The next larger time scale (the largest stays)."""
        index = TIME_SCALES.index(self.time_scale)
        self.time_scale = TIME_SCALES[min(index + 1, len(TIME_SCALES) - 1)]

    def slower(self) -> None:
        """The next smaller time scale (the smallest stays)."""
        index = TIME_SCALES.index(self.time_scale)
        self.time_scale = TIME_SCALES[max(index - 1, 0)]
