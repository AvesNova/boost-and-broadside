"""The match loop: the realtime sim at its fixed tick, driven by the display's clock.

One thread and a fixed timestep: each frame adds the elapsed time (times the time scale)
to an accumulator, the sim steps while the accumulator holds a whole tick, and the frame
draws at alpha = accumulator / tick between the last two snapshots. The loop draws no
random numbers, so with the same seed and the same per-tick actions it gives the same
states as `bnb.runtime.realtime.Realtime` alone.
"""

import time

import numpy as np
import torch

from bnb.contracts import GameConfig, Obs
from bnb.contracts.obs import GLOBAL
from bnb.controllers.human import INPUTS, human_action
from bnb.controllers.scripted import Chaser
from bnb.game.api import TEAMS
from bnb.runtime.realtime import Realtime

from .interpolate import interpolate
from .snapshot import Snapshot, snapshot

TIME_SCALES = (1.0, 2.0, 4.0, 8.0)
CATCH_UP_TICKS = 4  # at most this many ticks of backlog per unit of time scale


def team_view(observation: Obs, team: int) -> Obs:
    """One team's view (1, ...) out of the realtime observation (1, 2, ...)."""
    return Obs(
        ships=observation.ships[:, team],
        fields=observation.fields[:, team],
        globals=observation.globals[:, team],
        zones=observation.zones[:, team],
    )


class MatchLoop:
    """A scripted match with at most one ship flown from the keyboard.

    Every ship plays the chaser except the pilot's (a sim index, or None for none), which
    follows `inputs`, a bool tensor in `bnb.controllers.human.INPUTS` order that the shell
    fills each frame. `timings` holds the seconds the last tick spent per stage. `ticks`
    counts every tick run; `restarted` (N,) collects the ships that spawned (every ship
    does at a reset) since the shell last cleared it, so their trails restart even when a
    frame runs several ticks.
    """

    def __init__(
        self, config: GameConfig, fogged: bool = False, compile: bool = True, pilot: int | None = 0
    ) -> None:
        self.realtime = Realtime(config, compiled=compile, fogged=fogged)
        self.rules, self.types = self.realtime.rules, self.realtime.types
        self.chaser = Chaser(self.rules.ships_per_team)
        self.pilot = pilot
        self.inputs = torch.zeros(len(INPUTS), dtype=torch.bool)
        self.time_scale = TIME_SCALES[0]
        self.paused = False
        self.accumulator = 0.0
        self.timings = {"controllers": 0.0, "sim": 0.0, "snapshot": 0.0}
        self.current = self._snapshot()
        self.previous = self.current
        self.ticks = 0
        self.restarted = np.ones(self.rules.num_ships, dtype=bool)

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
        """(N, 3) int32 actions in sim order: the chaser's, and the pilot's from the inputs."""
        observation = self.realtime.obs
        done = observation.globals[:, 0, GLOBAL["reset"]] > 0.5
        teams = [
            self.chaser.step(None, team_view(observation, team), done)[0].packed()[0]
            for team in range(TEAMS)
        ]
        actions = torch.cat(teams)
        if self.pilot is not None:
            actions[self.pilot] = human_action(self.inputs).packed()
        return actions

    def tick(self, actions: torch.Tensor) -> None:
        """Step the sim once with these (N, 3) actions."""
        started = time.perf_counter()
        self.realtime.step(actions)
        stepped = time.perf_counter()
        self.previous, self.current = self.current, self._snapshot()
        self.ticks += 1
        self.restarted |= self.current.ships.spawned
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
        the chaser)."""
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
