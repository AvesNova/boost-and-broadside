"""Replay videos of a scripted lean match in v2's look: simulate on the CPU (lean.view.match),
draw each tick offscreen with v2's renderer (the copy in lean.view.play, through the adaption
layer `snapshot.to_v2`) in a standalone OpenGL context (EGL), and encode with ffmpeg (H.264,
yuv420p, faststart). One frame per tick (30 Hz), the last held with the result.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:src uv run --no-sync python -m lean.view.video \
        --ships-per-team 4 --seeds 0 --out DIR [--no-fields] [--view spectator|0|1]
        [--vision-eccentricity E] [--vision-range M] [--tracking on|off] [--draw-vision]
        [--theme NAME] [--stills T ...]

See lean/view/NOTES.md.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import time
from pathlib import Path

import moderngl
import numpy as np
from imgui_bundle import imgui

from . import marks as lean_marks
from .interface import Interface, Shown, start
from .match import MatchSettings, record
from .play import themes
from .play.camera import Camera
from .play.index_map import IndexGrid
from .play.render.arcs import arcs
from .play.render.renderer import Renderer
from .play.trails import Trails
from .snapshot import SPECTATOR, Snapshot, to_v2

FPS = 30
HOLD_SECONDS = 3.0
WIDTH, HEIGHT = 1920, 1080
SCALE = 1.5  # the interface scale at 1080 lines (v2's screenshots: 1 at 720)
MARGIN = 450.0  # m around the living ships
SMALLEST_HALF = 1100.0  # m: the camera's smallest half height


def context() -> moderngl.Context:
    """A standalone OpenGL 3.3 context: EGL (no display needed), else the default backend."""
    try:
        return moderngl.create_standalone_context(backend="egl")
    except Exception:  # noqa: BLE001 (any EGL failure falls back)
        return moderngl.create_standalone_context()


def camera_path(frames: list[Snapshot], arena: float, aspect: float) -> np.ndarray:
    """(T, 3) centre x, y and half height (m) per frame: the living ships' bounding box with a
    margin, smoothed forward and backward (1 s), never cutting a living ship off
    (docs/rewrite/sections/replay/replay.py's camera)."""
    whole = arena * 1.08
    raw = np.zeros((len(frames), 3))
    need = np.zeros(len(frames))
    for t, frame in enumerate(frames):
        ships = frame.ships
        live = ships.alive if ships.alive.any() else np.ones_like(ships.alive)
        x, y = ships.x[live], ships.y[live]
        half = max((x.max() - x.min()) / aspect, y.max() - y.min()) / 2 + MARGIN
        raw[t] = (
            (x.min() + x.max()) / 2,
            (y.min() + y.max()) / 2,
            min(max(half, SMALLEST_HALF), whole),
        )

    def smooth(values: np.ndarray, seconds: float) -> np.ndarray:
        keep = math.exp(-1.0 / (seconds * FPS))
        out = values.copy()
        for t in range(1, len(out)):
            out[t] = keep * out[t - 1] + (1 - keep) * out[t]
        for t in range(len(out) - 2, -1, -1):
            out[t] = keep * out[t + 1] + (1 - keep) * out[t]
        return out

    path = smooth(raw, 1.0)
    for t, frame in enumerate(frames):
        ships = frame.ships
        live = ships.alive if ships.alive.any() else np.ones_like(ships.alive)
        reach = np.maximum(
            np.abs(ships.x[live] - path[t, 0]) / aspect, np.abs(ships.y[live] - path[t, 1])
        )
        need[t] = min(reach.max() + 350.0, whole)
    window = FPS // 2
    held = np.array([need[max(0, t - window) : t + window + 1].max() for t in range(len(need))])
    path[:, 2] = np.maximum(smooth(np.maximum(path[:, 2], held), 0.3), need)
    return path


class VideoRenderer:
    """v2's renderer, the interface and lean's extras, offscreen at WIDTH x HEIGHT."""

    def __init__(
        self,
        frames: list[Snapshot],
        grid: IndexGrid,
        values: np.ndarray,
        view: int,
        theme: str,
        title: str,
        draw_vision: bool,
    ) -> None:
        self.frames, self.view, self.title, self.draw_vision = frames, view, title, draw_vision
        self.v2_frames = [to_v2(frame) for frame in frames]
        self.context = context()
        self.fonts = start(WIDTH, HEIGHT)
        self.interface = Interface(self.fonts, SCALE)
        self.renderer = Renderer(self.context, WIDTH, HEIGHT)
        self.renderer.set_map(self.v2_frames[0], grid, values)
        eccentricity = float(frames[0].ships.vision_eccentricity[0])  # one for every type
        self.renderer.fog.eccentricity = self.renderer.minimap.fog.eccentricity = eccentricity
        self.colour = self.context.texture((WIDTH, HEIGHT), 4)
        self.target = self.context.framebuffer(color_attachments=[self.colour])
        self.theme = themes.load(theme)
        self.camera = Camera(WIDTH, HEIGHT)
        self.camera.fit(frames[0].arena_radius)
        side = int(round(210 * SCALE))
        self.minimap_camera = Camera(side, side)
        self.minimap_camera.fit(frames[0].arena_radius)
        self.path = camera_path(frames, frames[0].arena_radius, WIDTH / HEIGHT)
        self.trails = Trails(len(frames[0].ships.x), frames[0].stall_angle)
        self.recorded = -1

    def frame(self, t: int, final: bool = False) -> bytes:
        """The RGB bytes of tick t (top row first)."""
        frame, v2_frame = self.frames[t], self.v2_frames[t]
        if t < self.recorded:  # a still out of order: rebuild the trails
            self.trails = Trails(len(frame.ships.x), frame.stall_angle)
            self.recorded = -1
        for k in range(max(self.recorded + 1, t - 60), t + 1):
            self.trails.record(
                self.v2_frames[k], self.frames[k].clock, np.zeros(len(frame.ships.x), bool)
            )
        self.recorded = t
        centre_x, centre_y, half = self.path[t]
        self.camera.follow(centre_x, centre_y)
        self.camera.metres_per_pixel = 2.0 * half / HEIGHT
        theme = self.theme
        marks = lean_marks.marks_at(self.frames, t)
        self.renderer.draw(
            v2_frame,
            self.view,
            self.camera,
            theme,
            self.trails,
            frame.clock,
            self.target,
            grid=theme.grid_pattern,
            contours=theme.contour_shapes,
        )
        lean_marks.draw(
            self.renderer,
            self.camera,
            theme,
            frame,
            marks,
            self.view,
            self.draw_vision,
            self.target,
        )
        shown = v2_frame.ships.alive & (
            v2_frame.ships.visible[self.view] if self.view != SPECTATOR else True
        )
        texture = self.renderer.draw_minimap(
            v2_frame, self.view, self.minimap_camera, theme, shown, arcs(0)
        )
        self.target.use()
        self.context.viewport = (0, 0, WIDTH, HEIGHT)
        io = imgui.get_io()
        io.display_size = imgui.ImVec2(WIDTH, HEIGHT)
        io.delta_time = 1.0 / FPS
        imgui.new_frame()
        self.interface.draw(
            Shown(
                frames=self.frames,
                index=t,
                view=self.view,
                theme=theme,
                camera=self.camera,
                minimap_camera=self.minimap_camera,
                minimap_texture=texture,
                marks=marks,
                title=self.title,
                draw_vision=self.draw_vision,
                show_result=final,
            )
        )
        imgui.render()
        self.target.use()
        self.renderer.draw_imgui(imgui.get_draw_data())
        pixels = np.frombuffer(self.target.read(components=3), np.uint8)
        return pixels.reshape(HEIGHT, WIDTH, 3)[::-1].tobytes()


def encode(video: VideoRenderer, path: Path) -> None:
    command = [
        "/usr/bin/ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    count = len(video.frames)
    for t in range(count):
        process.stdin.write(video.frame(t))
    final = video.frame(count - 1, final=True)
    for _ in range(int(HOLD_SECONDS * FPS)):
        process.stdin.write(final)
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")


def save_png(rgb: bytes, path: Path) -> None:
    from PIL import Image

    Image.frombytes("RGB", (WIDTH, HEIGHT), rgb).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ships-per-team", type=int, default=4)
    parser.add_argument("--no-fields", action="store_true")
    parser.add_argument("--view", default="spectator", choices=("spectator", "0", "1"))
    parser.add_argument("--vision-range", type=float, default=None)
    parser.add_argument("--vision-eccentricity", type=float, default=None)
    parser.add_argument("--tracking", choices=("on", "off"), default="on")
    parser.add_argument(
        "--draw-vision",
        action="store_true",
        help="each ship's sight ellipse and tracking circle (on with an eccentricity)",
    )
    parser.add_argument("--theme", default=themes.DEFAULT, choices=themes.shipped_names())
    parser.add_argument(
        "--stills",
        type=float,
        nargs="*",
        default=[0.25, 0.5, 1.0],
        help="also save PNG stills at these fractions of the game",
    )
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    view = SPECTATOR if arguments.view == "spectator" else int(arguments.view)
    size = arguments.ships_per_team
    draw_vision = arguments.draw_vision or arguments.vision_eccentricity is not None
    for seed in arguments.seeds:
        started = time.perf_counter()
        settings = MatchSettings(
            ships_per_team=(size, size),
            seed=seed,
            fields=not arguments.no_fields,
            tracking=arguments.tracking == "on",
            vision_range=arguments.vision_range,
            vision_eccentricity=arguments.vision_eccentricity,
        )
        frames, index = record(settings)
        grid = IndexGrid(origin=index.origin, texel=index.texel, size=index.size)
        stem = f"scripted{size}v{size}" + ("_nofields" if arguments.no_fields else "_fields")
        stem += f"_{arguments.view}_seed{seed}"
        if arguments.vision_eccentricity is not None:
            stem += f"_ellipse{arguments.vision_eccentricity:g}"
        title = f"lean scripted {size}v{size}, seed {seed}" + (
            ", no fields" if arguments.no_fields else ""
        )
        video = VideoRenderer(frames, grid, index.values, view, arguments.theme, title, draw_vision)
        print(
            f"seed {seed}: {len(frames)} frames, simulated in "
            f"{time.perf_counter() - started:.0f} s",
            flush=True,
        )
        for fraction in arguments.stills:
            t = min(int(fraction * (len(frames) - 1)), len(frames) - 1)
            save_png(video.frame(t, final=fraction >= 1.0), arguments.out / f"{stem}_t{t:04d}.png")
        path = arguments.out / f"{stem}.mp4"
        encode(video, path)
        print(f"  wrote {path} in {time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
