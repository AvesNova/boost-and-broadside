"""`python -m bnb.play.benchmark`: the renderer's frame time, offscreen, holding the
benchmark lock (run from the repository root, where `bench` imports).

For each game and view it plays a match on the display clock with no frame cap (each
frame waits for the GPU to finish, as a swap without vsync would), and reports the frame
time's percentiles and whether the sim kept its real-time tick rate. A second, shorter
run times each pass with GPU timer queries (which stall the pipeline, so they are not
part of the frame times) and the CPU stages (sim tick, controllers, snapshot, index
tiles, interpolation, and each pass's own Python, which includes its uploads). Last, it
rebuilds the map's index texture as a map change does, a few tiles per frame, and reports
those frames.
"""

import dataclasses
import json
import pathlib
import time

import numpy as np
import torch
import tyro

from .app import Options, Session, offscreen_context
from .render.renderer import PassTimer

GAMES = {
    "deathmatch": dict(preset="deathmatch"),
    "deathmatch_wide": dict(preset="deathmatch_wide"),
    "frontline": dict(preset="frontline"),
    "50v50": dict(preset="deathmatch", ships_per_team=50, fields=40, arena_radius=8222.0),
}


@dataclasses.dataclass
class Benchmark:
    games: tuple[str, ...] = tuple(GAMES)
    views: tuple[str, ...] = ("team0", "spectator")
    seconds: float = 10.0  # timed play per game and view
    profile_frames: int = 240
    warmup_frames: int = 120
    width: int = 1920
    height: int = 1080
    threads: int = 1
    output: pathlib.Path | None = None  # also append the results here, one JSON line each


def _percentiles(values: list[float]) -> dict:
    data = np.array(values) * 1e3
    return {f"p{percentile}": float(np.percentile(data, percentile))
            for percentile in (50, 90, 99)} | {"max": float(data.max())}


def measure(session: Session, target, seconds: float, profile_frames: int,
            warmup_frames: int) -> dict:
    context = session.context
    for _ in range(warmup_frames):  # compiles the tick and fills the caches
        session.update(1.0 / 60.0)
        session.draw(target)
    context.finish()
    loop = session.loop
    frame_times, started_ticks = [], loop.ticks
    started = last = time.perf_counter()
    while last - started < seconds:
        now = time.perf_counter()
        session.update(now - last)
        session.draw(target)
        context.finish()
        frame_end = time.perf_counter()
        frame_times.append(frame_end - now)
        last = now
    wall = time.perf_counter() - started
    tick_rate = (loop.ticks - started_ticks) / wall

    timer = PassTimer(context)
    session.renderer.timer = timer
    gpu, cpu, stages = {}, {}, {}
    for _ in range(profile_frames):
        session.update(1.0 / 60.0)
        session.draw(target)
        for name, value in timer.gpu().items():
            gpu.setdefault(name, []).append(value)
        for name, value in timer.cpu.items():
            cpu.setdefault(name, []).append(value * 1e3)
        for name, value in (session.timings | loop.timings).items():
            stages.setdefault(name, []).append(value * 1e3)
    session.renderer.timer = None

    # A map change: the index texture is rebuilt a few tiles per frame.
    session._new_map(whole=False)
    rebuild, last = [], time.perf_counter()
    while session.builder is not None:
        now = time.perf_counter()
        session.update(now - last)
        session.draw(target)
        context.finish()
        rebuild.append(time.perf_counter() - now)
        last = now
    return dict(frames=len(frame_times), frame_ms=_percentiles(frame_times),
                tick_rate=tick_rate, real_time_rate=1.0 / loop.tick_length,
                gpu_ms={name: float(np.mean(values)) for name, values in gpu.items()},
                pass_cpu_ms={name: float(np.mean(values)) for name, values in cpu.items()},
                stage_ms={name: float(np.mean(values)) for name, values in stages.items()},
                map_change=dict(frames=len(rebuild), frame_ms=_percentiles(rebuild)),
                fog_mask=list(session.renderer.fog.size) if session.renderer.fog.size else None,
                index_grid=dataclasses.asdict(session.renderer.fields.grid))


def main() -> None:
    from bench.lock import bench_lock

    benchmark = tyro.cli(Benchmark)
    torch.set_num_threads(benchmark.threads)
    with bench_lock():
        context = offscreen_context()
        size = (benchmark.width, benchmark.height)
        target = context.framebuffer(color_attachments=[context.texture(size, 4)])
        for game in benchmark.games:
            for view in benchmark.views:
                torch.manual_seed(0)
                options = Options(**GAMES[game], fog=True, view=view, fly=False,
                                  width=benchmark.width, height=benchmark.height)
                started = time.perf_counter()
                session = Session(options, context, *size)
                setup = time.perf_counter() - started
                result = dict(game=game, view=view, setup_s=setup,
                              renderer=context.info["GL_RENDERER"]) | measure(
                    session, target, benchmark.seconds, benchmark.profile_frames,
                    benchmark.warmup_frames)
                session.close()
                print(json.dumps(result), flush=True)
                if benchmark.output is not None:
                    with open(benchmark.output, "a") as file:
                        file.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
