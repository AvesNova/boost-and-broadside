"""The live window on a lean match, through v2's play shell (the copy in lean.view.play, NEEDS
REVIEW): v2's window, session, interface and renderer, with the match played by lean.

Adaption layer: the lean game runs in a child process (`produce`: scripted against scripted,
the whole match ahead of the screen, in batches over a queue), so a frame never waits on the
game. The window plays the ticks back as a replay does (`LeanLoop`, v2's replay loop's
interface: pause, step, speed, the scrubber with kill markers, seek), and `LeanSession` is v2's
session with the parts that drive v2's game turned off (listed in lean/view/NOTES.md).

    PYTHONPATH=.:src uv run --no-sync python -m lean.view.live --ships-per-team 4 --seed 0
        [--fields 0] [--vision-eccentricity E] [--draw-vision] [--theme NAME]
"""

from __future__ import annotations

import dataclasses
import multiprocessing
import queue
import time

import numpy as np
import torch
import tyro

from bnb.game import api
from lean.game import card

from . import marks as lean_marks
from .match import Match, MatchSettings
from .play import app
from .play import window as window_module
from .play.interface import fleets, top_bar
from .play.interface import text as interface_text
from .play.loop import MatchEvent, MatchLoop
from .play.tuning import base_rules
from .play.window import Window
from .snapshot import TYPE_LABELS, Snapshot, to_v2

# Adaption: what v2's interface writes from v2's game, as lean has it.
interface_text.type_name = lambda ship_type: TYPE_LABELS[card.TYPE_NAMES[ship_type]]
top_bar.boundary_label = lambda frame: "edge fixed"  # lean's boundary never shrinks
fleets.controller_label_of_setup = lambda side: "lean scripted (yardstick)"

BATCH = 30  # ticks per message from the game process
DISABLED = "not in the lean view yet (lean/view/NOTES.md)"


def produce(settings: MatchSettings, out) -> None:
    """The game process: play the match and send its snapshots in batches, then None."""
    torch.set_num_threads(2)
    batch = []
    for frame in Match(settings).snapshots():
        batch.append(frame)
        if len(batch) == BATCH:
            out.put(batch)
            batch = []
    out.put(batch)
    out.put(None)


def match_events(frame: Snapshot) -> list[MatchEvent]:
    """v2's events (the kill feed, the scrubber's markers) of a lean frame."""
    events = frame.events
    found = [
        MatchEvent(frame.tick, "kill", ship=int(ship), other=int(events.killer[ship]))
        for ship in np.flatnonzero(events.died)
    ]
    if frame.result >= 0:
        found.append(MatchEvent(frame.tick, "end", team=frame.result if frame.result < 2 else -1))
    return found


@dataclasses.dataclass
class LeanReplay:
    """What v2's interface reads of a replay: the last tick shown and the events so far."""

    name: str
    last: int = 0
    events: list = dataclasses.field(default_factory=list)


class LeanLoop(MatchLoop):
    """A lean match played back as it arrives from the game process: v2's replay loop's
    interface (`position`, `seek`, `markers`), its ticks lean frames (`frames`) shown through
    `to_v2`."""

    replaying = True

    def __init__(self, setup, settings: MatchSettings) -> None:
        self.setup, self.settings = setup, settings
        self.realtime = None
        self.rules = base_rules(setup.config, setup.lineup)
        self.types = api.ship_types(ship_set=setup.config.ship_set)
        context = multiprocessing.get_context("spawn")
        self.queue = context.Queue()
        self.process = context.Process(target=produce, args=(settings, self.queue), daemon=True)
        self.process.start()
        self.frames: list[Snapshot] = []
        self.shown = []  # to_v2 of each frame
        self.finished = False
        self.replay = LeanReplay(f"lean {settings.ships_per_team[0]}v{settings.ships_per_team[1]}")
        while not self.frames:  # the deal's frame: the session needs one to start
            self.receive(block=True)
        self.sides, self.pilot = [], None
        self.inputs = torch.zeros(1, dtype=torch.bool)
        self.time_scale, self.paused, self.accumulator = 1.0, False, 0.0
        self.timings = {"controllers": 0.0, "sim": 0.0, "snapshot": 0.0}
        self.position = 0
        self.current = self.previous = self.shown[0]
        self.ticks = 0
        self.restarted = np.ones(self.rules.num_ships, dtype=bool)
        self.events: list[MatchEvent] = []
        self.kills = np.zeros(self.rules.num_ships, np.int64)
        self.deaths = np.zeros(self.rules.num_ships, np.int64)
        self.last_actions = torch.zeros(self.rules.num_ships, 3, dtype=torch.int32)
        self.on_tick = self.on_retune = None

    def receive(self, block: bool = False) -> None:
        """Take what the game process has sent (waiting only when `block`)."""
        while not self.finished:
            try:
                batch = self.queue.get(block=block, timeout=60.0 if block else None)
            except queue.Empty:
                return
            if batch is None:
                self.finished = True
                self.process.join(timeout=1.0)
                return
            for frame in batch:
                self.frames.append(frame)
                self.shown.append(to_v2(frame))
                self.replay.events += match_events(frame)
            self.replay.last = len(self.frames) - 1
            block = False

    def step(self) -> None:
        """The next tick, once it has arrived; at the end, pause."""
        self.receive()
        if self.position >= len(self.frames) - 1:
            if self.finished:
                self.paused = True
            return
        self.position += 1
        self.previous, self.current = self.current, self.shown[self.position]
        self.ticks += 1
        events = match_events(self.frames[self.position])
        if self.on_tick is not None:
            self.on_tick(self.last_actions, None, events)
        self._count(events)

    def seek(self, tick: int) -> None:
        """Show tick `tick` (clamped to what has arrived)."""
        self.receive()
        tick = min(max(int(tick), 0), len(self.frames) - 1)
        self.position = tick
        self.current = self.shown[tick]
        self.previous = self.shown[max(tick - 1, 0)]
        self.accumulator = self.tick_length
        self.restarted[:] = True
        self.kills[:] = 0
        self.deaths[:] = 0
        self.events = []
        self._count([event for event in self.replay.events if event.tick <= tick])

    def single_step_back(self) -> None:
        if self.paused:
            self.seek(self.position - 1)

    def markers(self) -> list[MatchEvent]:
        return self.replay.events

    def retune(self, rules=None, types=None) -> None:
        raise NotImplementedError(DISABLED)


@dataclasses.dataclass
class LeanOptions(app.Options):
    """v2's options, with lean's match settings: the sight ellipse's eccentricity and range
    (None: the card's), the tracking circle, and drawing each ship's sight outline. --fields 0
    plays without fields; --ships-per-team sets the team sizes."""

    ships_per_team: int | None = 4
    record: bool = False  # v2's replays re-simulate v2's game
    compile: bool = False
    vision_eccentricity: float | None = None
    vision_range: float | None = None
    tracking: bool = True
    draw_vision: bool = False


class LeanSession(app.Session):
    """v2's session on a lean match (`LeanLoop`); the parts that drive v2's game are off."""

    def __init__(self, options, *arguments, **keywords) -> None:
        super().__init__(options, *arguments, **keywords)
        draw = self.renderer.draw

        def draw_with_lean(frame, view, camera, theme, trails, now, target, *rest, **named):
            draw(frame, view, camera, theme, trails, now, target, *rest, **named)
            loop = self.loop
            lean_frame = loop.frames[loop.position]
            lean_marks.draw(
                self.renderer,
                camera,
                theme,
                lean_frame,
                lean_marks.marks_at(loop.frames, loop.position),
                view,
                self.options.draw_vision,
                target,
            )

        self.renderer.draw = draw_with_lean

    def start(self, setup, pilot=None, whole=False, realtime=None, sides=None) -> bool:
        options = self.options
        settings = MatchSettings(
            ships_per_team=(setup.config.ships_per_team,) * 2,
            seed=setup.seed,
            fields=setup.config.num_fields != 0,
            tracking=options.tracking,
            vision_range=options.vision_range,
            vision_eccentricity=options.vision_eccentricity,
        )
        if self.loop is not None:
            self.loop.process.terminate()
        loop = LeanLoop(setup, settings)
        self.following = None
        self._adopt(loop, whole)
        eccentricity = float(loop.frames[0].ships.vision_eccentricity[0])
        self.renderer.fog.eccentricity = self.renderer.minimap.fog.eccentricity = eccentricity
        self.recorder = None
        return True

    def open_replay(self, directory, whole: bool = False) -> bool:
        self.say("v2 replays: " + DISABLED)
        return False

    def toggle_tuning(self) -> None:
        self.say("The tuning panel: " + DISABLED)

    def toggle_map_panel(self) -> None:
        self.say("The map editor: " + DISABLED)

    def set_side(self, team: int, side) -> bool:
        self.say("Changing sides: " + DISABLED)
        return False

    def take_control(self, ship: int) -> None:
        self.say("Flying a ship: " + DISABLED)

    def offered_overlays(self) -> tuple[str, ...]:
        # v2's overlays read v2's realtime game and Yemong's heads (lean/view/NOTES.md).
        return ()

    def close(self) -> None:
        self.loop.process.terminate()
        super().close()


class LeanWindow(Window):
    """v2's window on a `LeanSession`."""

    def __init__(self, options: LeanOptions) -> None:
        window_module.Session = LeanSession  # v2's Window builds its session by this name
        super().__init__(options)


def run_offscreen(options: LeanOptions) -> list:
    """v2's offscreen runner (app.run_offscreen) on a LeanSession: options.frames frames at
    60 per second of display time, as PNG files in options.offscreen (a check without a
    display)."""
    context = app.offscreen_context()
    size = (options.width, options.height)
    target = context.framebuffer(color_attachments=[context.texture(size, 4)])
    session = LeanSession(options, context, options.width, options.height)
    session.show_fps = False
    options.offscreen.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(options.frames):
        session.update(app.FRAME_SECONDS)
        session.draw(target)
        path = options.offscreen / f"frame_{index:04d}.png"
        app.write_png(path, app.read_pixels(target))
        paths.append(path)
    session.close()
    context.release()
    return paths


def main() -> None:
    options = tyro.cli(LeanOptions)
    if options.offscreen is not None:
        print(f"wrote {len(run_offscreen(options))} frames to {options.offscreen}")
        return
    started = time.perf_counter()
    window = LeanWindow(options)
    print(f"window open in {time.perf_counter() - started:.1f} s", flush=True)
    window.run()


if __name__ == "__main__":
    main()
