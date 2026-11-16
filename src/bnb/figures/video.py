"""A video of one game: a policy against Imugi, drawn by the play shell offscreen
(`bnb.play.app.Session` on an EGL context, no window) and encoded to MP4 by ffmpeg, which reads
the raw frames on a pipe.

The policy plays team 0 and Imugi team 1 (`VideoConfig.opponent`: Imugi with its memory and the
Baekho doctrine, the evaluations' opponent; or the scripted chaser), in the mode of the preset
asked for (`bnb.play.sides.load_agent` with `any_mode`: a policy trained on a mixed batch plays
every mode of the mix). The teams may be uneven (`VideoConfig.ships`: the policy's ships and
Imugi's, the rest of a team's slots benched as the crossover benches them,
`bnb.league.crossover`), and the video may stop where the game ends (`VideoConfig.until_end`:
the last frame held, with the final kill's mark). The match runs uncompiled and its agent on
whatever device torch sees (`bnb.play.sides.agent_device`; `bnb figures` hides the GPU from
torch unless `--device` names one). Each video frame advances the match by `time_scale` frames
of display time, so a video of 25 s at the default 4x shows the game's first 100 s. The frame is
the shell's own drawing, with its top bar and kill feed (the fleets card and the minimap
collapsed, the camera fitted to the arena), and the prediction overlays chosen
(`VideoConfig.overlays`, `bnb.play.overlays`: the policy's predictions, and the true bullet
paths), drawn by `draw_frame`.
"""

import dataclasses
import math
import pathlib
import shutil
import subprocess
import time
from typing import TYPE_CHECKING

import numpy as np

from bnb.game.api import PRESETS
from bnb.play.loop import TIME_SCALES, MatchSetup, SideSetup
from bnb.play.sides import Imugi, ImugiAgent, load_agent

FFMPEG = "ffmpeg"  # looked up on the PATH (`/usr/bin/ffmpeg` on the dev machine)
POLICY_TEAM = 0  # the policy plays team 0, the scripted chaser team 1
COLLAPSED_CARDS = ("fleets", "selection", "minimap")  # the cards a video leaves closed

if TYPE_CHECKING:
    from bnb.play.app import Session


@dataclasses.dataclass(frozen=True)
class VideoConfig:
    """seconds and frame_rate: the video's length and frames per second; time_scale: game
    seconds per video second (one of `bnb.play.loop.TIME_SCALES`); width and height:
    pixels; seed: the match's generator and the agent's; temperature: the agent's sampling
    temperature; interface: draw the shell's top bar and kill feed; quality: ffmpeg's
    constant rate factor for H.264 (lower is larger and sharper); death_marks: leave a
    team-coloured X where each ship died, for the rest of the video
    (`bnb.play.render.marks`); ships: the policy's ships and the opponent's (each team's
    other slots benched; the game's ships per team grown to the larger when it has fewer;
    None: the preset's teams); opponent: "imugi" (`bnb.play.sides.ImugiAgent`, as the
    evaluations play it) or "chaser" (the scripted chaser, `bnb.play.sides.Imugi`);
    until_end: stop at the first game's end and hold its last frame `hold` seconds
    (`seconds` stays the bound); follow: the camera frames the live ships (`Framing`)
    rather than the whole arena."""

    seconds: float = 25.0
    frame_rate: int = 30
    time_scale: float = 4.0
    width: int = 1280
    height: int = 720
    seed: int = 0
    temperature: float = 1.0
    interface: bool = True
    quality: int = 28
    overlays: tuple[str, ...] = ()
    truth: bool = False
    death_marks: bool = True
    ships: tuple[int, int] | None = None
    opponent: str = "imugi"
    until_end: bool = False
    hold: float = 2.0
    follow: bool = False


def encoder(path: pathlib.Path, config: VideoConfig) -> subprocess.Popen:
    """An ffmpeg process that encodes raw RGB frames from its stdin to an H.264 MP4 at
    `path` (yuv420p, the moov atom first, so a browser starts it at once)."""
    found = shutil.which(FFMPEG)
    if found is None:
        raise SystemExit(f"bnb figures video needs ffmpeg on the PATH ({FFMPEG!r} not found)")
    command = [
        found,
        "-y",
        "-loglevel",
        "error",
        *("-f", "rawvideo", "-pix_fmt", "rgb24"),
        *("-s", f"{config.width}x{config.height}", "-r", str(config.frame_rate)),
        *("-i", "-"),
        *("-c:v", "libx264", "-preset", "veryfast", "-crf", str(config.quality)),
        *("-pix_fmt", "yuv420p", "-movflags", "+faststart"),
        str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


OPPONENTS = ("imugi", "chaser")


def setup_of(policy: pathlib.Path, preset: str, config: VideoConfig) -> MatchSetup:
    """The match: the preset's game (fog as the game has it: none in team deathmatch) at
    the lineup's size (on the preset's arena, as the crossover's preset map rule), the
    policy's side and the opponent's, and each team's ships in play."""
    if config.opponent not in OPPONENTS:
        raise SystemExit(f"--video.opponent {config.opponent!r}: one of {OPPONENTS}")
    game, active = PRESETS[preset], None
    if config.ships is not None:
        counts = [0, 0]
        counts[POLICY_TEAM], counts[1 - POLICY_TEAM] = config.ships
        active = (counts[0], counts[1])
        size = max(game.ships_per_team, *active)
        if size != game.ships_per_team:  # the preset's arena (crossover's preset map rule)
            game = dataclasses.replace(game, ships_per_team=size, spawn_groups=())
    opponent = "imugi" if config.opponent == "imugi" else "scripted"
    sides = [SideSetup(opponent), SideSetup(opponent)]
    sides[POLICY_TEAM] = SideSetup("agent", str(policy), config.temperature)
    return MatchSetup(game, seed=config.seed, sides=tuple(sides), active=active)


class Tally:
    """What happened in the video's game time, by team: kills (the killer's team; a ship
    lost to the boundary counts as a kill for nobody), losses, zone captures and the ends
    of games (the winner, or -1 for a draw); at the first game's end (`finish`), its game
    seconds, each team's ships left and their mean shield (a fraction of full)."""

    def __init__(self, teams) -> None:
        self.teams = teams  # (N,) each sim index's team
        self.kills, self.losses, self.captures = [0, 0], [0, 0], [0, 0]
        self.results: list[int] = []
        self.ended_at: float | None = None
        self.left: list[int] | None = None
        self.shield_left: list[float] | None = None

    def finish(self, alive, shield, seconds: float) -> None:
        """The first game's end: (N,) each ship alive after its last tick and its shield."""
        if self.ended_at is not None:
            return
        self.ended_at = round(seconds, 2)
        self.left = [int(alive[self.teams == team].sum()) for team in (0, 1)]
        self.shield_left = [
            round(float(shield[(self.teams == team) & alive].mean()), 3) if self.left[team] else 0.0
            for team in (0, 1)
        ]

    def count(self, events) -> None:
        """Add one tick's `bnb.play.loop.MatchEvent`s."""
        for event in events:
            if event.kind == "kill":
                self.losses[int(self.teams[event.ship])] += 1
                if event.other >= 0:
                    self.kills[int(self.teams[event.other])] += 1
            elif event.kind == "capture":
                self.captures[event.team] += 1
            elif event.kind == "end":
                self.results.append(event.team)

    def summary(self) -> dict:
        """The tally with the policy's team first: `policy_*` and `scripted_*`, and the
        games ended in the video as wins, losses and draws of the policy."""
        policy, scripted = POLICY_TEAM, 1 - POLICY_TEAM
        return dict(
            policy_kills=self.kills[policy],
            scripted_kills=self.kills[scripted],
            policy_losses=self.losses[policy],
            scripted_losses=self.losses[scripted],
            policy_captures=self.captures[policy],
            scripted_captures=self.captures[scripted],
            policy_wins=self.results.count(policy),
            policy_defeats=self.results.count(scripted),
            draws=self.results.count(-1),
            first_result=(
                None
                if not self.results
                else {policy: "win", scripted: "loss"}.get(self.results[0], "draw")
            ),
            ended_at=self.ended_at,
            policy_left=None if self.left is None else self.left[policy],
            scripted_left=None if self.left is None else self.left[scripted],
            policy_shield_left=None if self.shield_left is None else self.shield_left[policy],
            scripted_shield_left=(None if self.shield_left is None else self.shield_left[scripted]),
        )


def draw_frame(session: Session, target, overlays: tuple[str, ...] | None = None) -> None:
    """One video frame onto `target`: the shell's battlefield, its prediction overlays and
    its interface. overlays: the overlays to turn on from this frame (names of
    `bnb.play.overlays.NAMES`), or None to keep the session's."""
    if overlays is not None and set(overlays) != session.overlays.enabled:
        session.overlays.enabled = set(overlays)
        session.predicted_at = None
        session.predict()  # the overlays' arrays for this frame
    session.draw(target)


def render(
    policy: str | pathlib.Path, preset: str, config: VideoConfig, path: str | pathlib.Path
) -> dict:
    """Write the video of one game of `policy` against Imugi in the game
    `preset` (`bnb.game.api.PRESETS`) to `path` (MP4); returns its numbers: the tally
    (`Tally.summary`), the frames, the game seconds shown, the file's bytes and the
    seconds it took to make."""
    from bnb.play import app  # OpenGL and ImGui: only for a video
    from bnb.play.render.marks import DeathMarks

    if config.time_scale not in TIME_SCALES:
        raise SystemExit(f"--time-scale {config.time_scale}: one of {TIME_SCALES}")
    started = time.perf_counter()
    policy, path = pathlib.Path(policy), pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    setup = setup_of(policy, preset, config)
    options = app.Options(
        preset=preset,
        record=False,
        compile=False,
        width=config.width,
        height=config.height,
        seed=config.seed,
        overlays=config.overlays,
        truth=config.truth,
    )
    context = app.offscreen_context()
    size = (config.width, config.height)
    target = context.framebuffer(color_attachments=[context.texture(size, 4)])
    # The session opens on a scripted match (its options name no agent); the video's match,
    # with the policy loaded for any mode, replaces it.
    session = app.Session(options, context, *size)
    # The match starts with no controllers, which are made for its rules (its size) next.
    session.start(setup, whole=True, sides=[])
    loop = session.loop
    agent = load_agent(
        policy,
        setup.config,
        loop.rules,
        loop.types,
        config.temperature,
        config.seed + POLICY_TEAM,
        any_mode=True,
        benching=setup.active is not None,
    )
    opponent = 1 - POLICY_TEAM
    if config.opponent == "imugi":
        scripted = ImugiAgent(loop.rules, loop.types, seed=config.seed + opponent)
    else:
        scripted = Imugi(loop.rules, loop.types)
    sides = [scripted, scripted]
    sides[POLICY_TEAM] = agent
    loop.sides[:] = sides
    session.show_interface = config.interface
    session.interface.open -= set(COLLAPSED_CARDS)
    session.set_time_scale(config.time_scale)
    if config.death_marks:
        session.death_marks = DeathMarks()
    tally = Tally(session.loop.current.ships.team)
    shell_on_tick = session.loop.on_tick

    def on_tick(actions, realtime, events) -> None:
        shell_on_tick(actions, realtime, events)  # the feed and the death marks first
        tally.count(events)
        if any(event.kind == "end" for event in events):
            alive = loop.previous.ships.alive & ~realtime.events.died[0].numpy()
            tally.finish(alive, loop.previous.ships.shield, loop.ticks * loop.tick_length)
            if config.until_end:
                hold_last_tick(loop, alive)

    session.loop.on_tick = on_tick
    most = round(config.seconds * config.frame_rate)
    frames = held = 0
    framing = Framing() if config.follow else None
    process = encoder(path, config)
    try:
        while frames < most and held < config.hold * config.frame_rate:
            session.update(1.0 / config.frame_rate)
            if framing is not None:
                framing.update(session.camera, session.frame, 1.0 / config.frame_rate)
            draw_frame(session, target)
            process.stdin.write(app.read_pixels(target).tobytes())
            frames += 1
            if tally.ended_at is not None and config.until_end:
                held += 1  # the last tick's frame, held
    finally:
        process.stdin.close()
        process.wait()
        session.close()
        context.release()
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {process.returncode}) writing {path}")
    return dict(
        **tally.summary(),
        frames=frames,
        game_seconds=round(session.loop.ticks * session.loop.tick_length, 2),
        bytes=path.stat().st_size,
        seconds_taken=round(time.perf_counter() - started, 1),
    )


FOLLOW_MARGIN = 2.0  # a new framing shows the live ships' extent this many times over
FOLLOW_SMALLEST = 1.0 / 6.0  # the closest view's half height, as a share of the arena's radius
FOLLOW_EDGE = 0.9  # reframe when a live ship is past this share of the view's half size
FOLLOW_ZOOM_IN = 0.4  # or when the ships' framing needs less than this share of the view
FOLLOW_SECONDS = 0.3  # video seconds over which a reframe closes about 63% of its gap


class Framing:
    """A camera that frames the live ships, in steps: it holds still while they stay well
    inside the view and its zoom suits them, and otherwise moves to a new framing (their
    extent `FOLLOW_MARGIN` times over, never closer than `FOLLOW_SMALLEST` of the arena,
    never wider than the fitted arena) by an exponential approach. A still camera keeps
    the frames alike, which a GIF needs (a moving one changes every pixel of the
    background). The first frame jumps."""

    def __init__(self) -> None:
        self.target: tuple[float, float, float] | None = None

    def _framing(self, camera, frame) -> tuple[float, float, float] | None:
        ships = frame.ships
        if not ships.alive.any():
            return None
        x, y = ships.x[ships.alive], ships.y[ships.alive]
        centre_x, centre_y = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
        scale = max(
            0.5 * (x.max() - x.min()) * FOLLOW_MARGIN / (0.5 * camera.width),
            0.5 * (y.max() - y.min()) * FOLLOW_MARGIN / (0.5 * camera.height),
            FOLLOW_SMALLEST * frame.arena_radius / (0.5 * camera.height),
        )
        return float(centre_x), float(centre_y), float(min(scale, camera.fitted))

    def _holds(self, camera, frame, wanted) -> bool:
        """The current target still frames the live ships."""
        centre_x, centre_y, scale = self.target
        ships = frame.ships
        x, y = ships.x[ships.alive], ships.y[ships.alive]
        inside_x = np.abs(x - centre_x) <= FOLLOW_EDGE * 0.5 * camera.width * scale
        inside_y = np.abs(y - centre_y) <= FOLLOW_EDGE * 0.5 * camera.height * scale
        return bool((inside_x & inside_y).all()) and wanted[2] >= FOLLOW_ZOOM_IN * scale

    def update(self, camera, frame, step: float) -> None:
        """Move `camera` toward the framing over `step` video seconds."""
        wanted = self._framing(camera, frame)
        if wanted is None:
            return
        first = self.target is None
        if first or not self._holds(camera, frame, wanted):
            self.target = wanted
        share = 1.0 if first else 1.0 - math.exp(-step / FOLLOW_SECONDS)
        centre_x, centre_y, scale = self.target
        camera.centre_x += share * (centre_x - camera.centre_x)
        camera.centre_y += share * (centre_y - camera.centre_y)
        camera.metres_per_pixel += share * (scale - camera.metres_per_pixel)
        if abs(camera.metres_per_pixel - scale) < 1e-3 * scale:  # settled: exactly still
            camera.centre_x, camera.centre_y, camera.metres_per_pixel = self.target


def hold_last_tick(loop, alive) -> None:
    """Stop the loop on a game's last tick: the episode's reset (every ship respawned, a
    new map) is never drawn; the shown state is the last one before it with the ships that
    died on the last tick gone (their marks stay)."""
    last = dataclasses.replace(
        loop.previous, ships=dataclasses.replace(loop.previous.ships, alive=alive)
    )
    loop.previous = loop.current = last
    loop.paused = True
    loop.accumulator = 0.0  # the loop's while ends: no tick of the next game runs
    loop.restarted[:] = False


def video_name(preset: str, config: VideoConfig) -> str:
    """The video's file name: the preset and the seed."""
    return f"{preset}_seed{config.seed}.mp4"
