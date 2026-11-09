"""Replays: every live match is recorded, saved on its end or on quit, and played back by
re-simulation.

A recording holds the setup (`MatchSetup`: game, lineup, fog, seed, controllers), a
keyframe every KEYFRAME_TICKS ticks (`Realtime`'s state, views and generator, cloned),
every tick's actions, whether each tick ran compiled, and the match's events (kills,
captures, the end). Playback restores the nearest keyframe and repeats the recorded
actions, each tick compiled or not as it was run, so it reproduces the recorded states and
a seek costs at most KEYFRAME_TICKS ticks.

On disk a replay is a directory `<root>/<date>-<mode>-<sides>/`: `replay.json` (the record),
`actions.safetensors` and `keyframes.safetensors`. The root keeps the newest KEEP replays.
"""

import dataclasses
import datetime
import json
import pathlib
import re
import shutil

import safetensors.torch
import torch
from torch.utils import _pytree

from bnb.runs.serialise import from_json, named_tensors, to_json
from bnb.runtime.realtime import Keyframe, Realtime

from .loop import MatchEvent, MatchLoop, MatchSetup

FORMAT = 1
KEYFRAME_TICKS = 90  # 3 s at 30 Hz
KEEP = 50
RECORD, ACTIONS, KEYFRAMES = "replay.json", "actions.safetensors", "keyframes.safetensors"


class ReplayError(ValueError):
    """A replay that cannot be read, with the reason."""


class Recorder:
    """Records a live match from its current state: a keyframe at the start and every
    KEYFRAME_TICKS ticks, and each tick's actions, compile flag and events (their ticks
    counted from the recording's start). `ended`: the match ended (the last tick's state
    is the next match's start, which playback never shows)."""

    def __init__(self, setup: MatchSetup, realtime: Realtime, name: str = "") -> None:
        self.setup, self.name = setup, name
        self.started = datetime.datetime.now()
        self.keyframes: dict[int, Keyframe] = {0: realtime.keyframe()}
        self.actions: list[torch.Tensor] = []
        self.compiled: list[bool] = []
        self.events: list[MatchEvent] = []
        self.ended = False

    @property
    def ticks(self) -> int:
        """Ticks recorded."""
        return len(self.actions)

    def record(self, actions: torch.Tensor, realtime: Realtime, events: list) -> None:
        """After a tick: its actions, whether it ran compiled, its events; a keyframe when
        one is due."""
        self.actions.append(actions.clone())
        self.compiled.append(realtime.uses_compiled)
        tick = self.ticks
        self.events += [dataclasses.replace(event, tick=tick) for event in events]
        if any(event.kind == "end" for event in events):
            self.ended = True
        elif tick % KEYFRAME_TICKS == 0:
            self.keyframes[tick] = realtime.keyframe()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "match"


def _keyframe_tree(keyframe: Keyframe) -> dict:
    return {"state": keyframe.state, "obs": keyframe.obs, "generator": keyframe.generator}


def _tree_like(template, stored: dict[str, torch.Tensor], prefix: str):
    """A tree shaped like `template` with the stored tensors under `prefix` as its leaves,
    each as stored (a state's integer dtypes can differ between a reset and a tick)."""
    names = list(named_tensors(template, prefix))
    leaves, spec = _pytree.tree_flatten(template)
    missing = [name for name in names if name not in stored]
    if missing or len(names) != len(leaves):
        raise ReplayError(f"the keyframe lacks {missing[:5]}")
    return _pytree.tree_unflatten([stored[name] for name in names], spec)


def save(recorder: Recorder, root: str | pathlib.Path, keep: int = KEEP) -> pathlib.Path:
    """Write the recording under `root` (created if missing), then delete all but the
    newest `keep` replays there; returns the replay's directory."""
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = recorder.started.strftime("%Y%m%d-%H%M%S%f")[
        :-3
    ]  # to the millisecond: newest-first order
    base = f"{stamp}-{_slug(recorder.setup.config.mode)}-{_slug(recorder.name)}"
    directory, copy = root / base, 1
    while directory.exists():
        copy += 1
        directory = root / f"{base}-{copy}"
    partial = directory.with_name(directory.name + ".tmp")
    partial.mkdir()
    actions = torch.stack(recorder.actions) if recorder.actions else torch.zeros(0, 0, 3)
    safetensors.torch.save_file({"actions": actions.contiguous()}, partial / ACTIONS)
    tensors = {}
    for tick, keyframe in recorder.keyframes.items():
        tensors |= {
            name: tensor.contiguous()
            for name, tensor in named_tensors(_keyframe_tree(keyframe), f"{tick}.").items()
        }
    safetensors.torch.save_file(tensors, partial / KEYFRAMES)
    record = dict(
        format=FORMAT,
        setup=to_json(recorder.setup),
        name=recorder.name,
        started=recorder.started.isoformat(timespec="seconds"),
        ticks=recorder.ticks,
        ended=recorder.ended,
        keyframes=sorted(recorder.keyframes),
        compiled=recorder.compiled,
        events=[to_json(event) for event in recorder.events],
    )
    (partial / RECORD).write_text(json.dumps(record))
    partial.rename(directory)
    prune(root, keep)
    return directory


def replays(root: str | pathlib.Path) -> list[pathlib.Path]:
    """The saved replays under `root`, newest first."""
    root = pathlib.Path(root)
    if not root.is_dir():
        return []
    found = [path for path in root.iterdir() if (path / RECORD).is_file()]
    return sorted(found, key=lambda path: path.name, reverse=True)


def prune(root: str | pathlib.Path, keep: int = KEEP) -> None:
    """Delete all but the newest `keep` replays under `root`."""
    for path in replays(root)[keep:]:
        shutil.rmtree(path)


@dataclasses.dataclass
class Replay:
    """A loaded replay. `last`: the last tick playback shows (a match that ended stops one
    tick before its end, whose state is the next match's start)."""

    directory: pathlib.Path
    setup: MatchSetup
    name: str
    actions: torch.Tensor
    keyframes: dict[int, Keyframe]
    compiled: list[bool]
    events: list[MatchEvent]
    ended: bool

    @property
    def last(self) -> int:
        """The last tick playback shows."""
        return max(len(self.actions) - (1 if self.ended else 0), 0)

    @classmethod
    def load(cls, directory: str | pathlib.Path) -> Replay:
        """Read a saved replay (raises ReplayError naming what is wrong)."""
        directory = pathlib.Path(directory)
        try:
            record = json.loads((directory / RECORD).read_text())
        except (OSError, ValueError) as error:
            raise ReplayError(f"{directory} is not a replay: {error}") from error
        if record.get("format") != FORMAT:
            raise ReplayError(
                f"{directory} is replay format {record.get('format')}; "
                f"this code reads format {FORMAT}"
            )
        setup, _ = from_json(MatchSetup, record["setup"])
        actions = safetensors.torch.load_file(directory / ACTIONS)["actions"]
        stored = safetensors.torch.load_file(directory / KEYFRAMES)
        template = _keyframe_tree(
            Realtime(
                setup.config,
                compiled=False,
                fogged=setup.fogged,
                lineup=setup.lineup,
                **tuned(setup),
            ).keyframe()
        )
        keyframes = {}
        for tick in record["keyframes"]:
            tree = _tree_like(template, stored, f"{tick}.")
            keyframes[tick] = Keyframe(tree["state"], tree["obs"], tree["generator"])
        events = [from_json(MatchEvent, event)[0] for event in record["events"]]
        return cls(
            directory,
            setup,
            record["name"],
            actions,
            keyframes,
            record["compiled"],
            events,
            record["ended"],
        )


def tuned(setup: MatchSetup) -> dict:
    """The rule settings and the card a recorded match played (its ship set's default card
    unless it was tuned), as `Realtime` takes them."""
    tuning = setup.tuning
    types = tuning.types(setup.config.ship_set) if tuning.card else None
    return dict(settings=tuning.rules, types=types)


class ReplayLoop(MatchLoop):
    """A replay played back: the loop's clock and controls, with the recorded actions in
    place of the controllers. `position` is the tick shown; playback pauses at the end."""

    replaying = True

    def __init__(self, replay: Replay, compiled: bool = True) -> None:
        setup = replay.setup
        realtime = Realtime(
            setup.config,
            compiled=compiled,
            fogged=setup.fogged,
            lineup=setup.lineup,
            background=compiled,
            **tuned(setup),
        )
        realtime.restore(replay.keyframes[0])
        self.replay = replay
        self.position = 0
        super().__init__(setup, sides=[], realtime=realtime)

    def actions(self) -> torch.Tensor:
        """The recorded actions of the next tick."""
        return self.replay.actions[self.position]

    def _eager(self) -> bool:
        return not self.replay.compiled[self.position]

    def tick(self, actions: torch.Tensor) -> None:
        """Repeat the next recorded tick, compiled or not as it was run."""
        super().tick(actions)
        self.position += 1

    def step(self) -> None:
        """One recorded tick; at the end, pause."""
        if self.position >= self.replay.last:
            self.paused = True
            return
        super().step()
        if self.position >= self.replay.last:
            self.paused = True

    def seek(self, tick: int) -> None:
        """Show tick `tick` (clamped to the replay): restore the nearest keyframe at or
        before it and repeat the recorded ticks from there. It draws at tick `tick`; the
        tick before it (when repeated) gives the speed and the last actions."""
        tick = min(max(int(tick), 0), self.replay.last)
        start = max(t for t in self.replay.keyframes if t <= tick)
        self.realtime.restore(self.replay.keyframes[start])
        self.previous = None
        for position in range(start, tick):
            if position == tick - 1:  # the tick before: for the speed and the last action
                self.previous = self._snapshot()
                self.last_actions = self.replay.actions[position]
            self.realtime.step(
                self.replay.actions[position], eager=not self.replay.compiled[position]
            )
        self.position = tick
        self.current = self._snapshot()
        self.previous = self.current if self.previous is None else self.previous
        self.accumulator = self.tick_length
        self.restarted[:] = True
        self.kills[:] = 0
        self.deaths[:] = 0
        self.events = []
        self._count([event for event in self.replay.events if event.tick <= tick])

    def single_step_back(self) -> None:
        """While paused, show the previous tick."""
        if self.paused:
            self.seek(self.position - 1)

    def markers(self) -> list[MatchEvent]:
        """The replay's events, for the scrubber's markers."""
        return self.replay.events
