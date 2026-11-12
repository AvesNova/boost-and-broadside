"""Where the compile time goes: one case per process, on cold caches.

Every `torch.compile` site is wrapped so each call that compiles (Dynamo's frame count
rises) is timed with the device synchronised before and after, and labelled by the
compiled function's name. The CUDA-graph capture is timed separately (`Rollout._capture`),
then the case's phases (build, first batch or update, a steady one).

Cases:
  ratings   the match engine: `players` (default: two policies of one build and the
            scripted agent) on `preset`, the default ArenaConfig, two batches
  train     a run's TrainConfig (`run`: a run directory with config.json, or `config`: a
            config file, `bnb train --print-config`'s), `steps` updates (default two);
            heads=0 turns the learner's heads' sums off; `set` overrides fields
            (`path=value;path=value`, JSON values, e.g. `compile_regions=true`); reports the
            peak memory allocated and reserved, the first update's series, the steps per
            update and, with `process_start` (the shell's epoch seconds), each phase's end
            in seconds since the process started
  realtime  the play shell's tick on the CPU (compiled, B = 1), its first call

Usage: `python bench/compile_baseline.py CASE [key=value ...] > result.json`
Run it with fresh TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR for a cold measurement
(`bench/compile_baseline.sh` does). Prints progress lines to stderr and one JSON object
to stdout.
"""

import functools
import json
import os
import sys
import time

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # as `bnb`'s

import torch  # noqa: E402
import torch._dynamo.utils as dynamo_utils  # noqa: E402

START = time.perf_counter()
PROCESS_START = next(
    (float(a.split("=", 1)[1]) for a in sys.argv if a.startswith("process_start=")), None
)
EVENTS: list[dict] = []


def log(message: str) -> None:
    """Print `message` to stderr with the seconds since the start."""
    print(f"[{time.perf_counter() - START:7.1f} s] {message}", file=sys.stderr, flush=True)


def _synchronize() -> None:
    if torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing():
        torch.cuda.synchronize()


def _frames() -> int:
    return dynamo_utils.counters["stats"].get("unique_graphs", 0)


def _ops() -> int:
    return dynamo_utils.counters["stats"].get("calls_captured", 0)


_compile = torch.compile


def _timed_compile(function=None, **options):
    if function is None:
        return lambda f: _timed_compile(f, **options)
    compiled = _compile(function, **options)
    label = getattr(function, "__qualname__", repr(function))

    @functools.wraps(function)
    def call(*arguments, **keywords):
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return compiled(*arguments, **keywords)
        _synchronize()
        frames, ops, started = _frames(), _ops(), time.perf_counter()
        result = compiled(*arguments, **keywords)
        _synchronize()
        if _frames() > frames:
            seconds = time.perf_counter() - started
            EVENTS.append(dict(kind="compile", label=label, seconds=seconds, ops=_ops() - ops))
            log(f"compiled {label}: {seconds:.1f} s, {_ops() - ops} ops")
        return result

    return call


torch.compile = _timed_compile


def _time_captures() -> None:
    from bnb.runtime import rollout

    original = rollout.Rollout._capture

    def capture(self):
        _synchronize()
        started = time.perf_counter()
        original(self)
        _synchronize()
        seconds = time.perf_counter() - started
        EVENTS.append(dict(kind="capture", label="Rollout._capture", seconds=seconds))
        log(f"captured a {self.block}-tick graph: {seconds:.1f} s")

    rollout.Rollout._capture = capture


def phase(name: str, phases: dict):
    """A context that times its block into `phases[name]`, the device synchronised."""

    class Phase:
        def __enter__(self):
            _synchronize()
            self.started = time.perf_counter()
            log(f"{name} ...")

        def __exit__(self, *exc):
            _synchronize()
            phases[name] = time.perf_counter() - self.started
            log(f"{name}: {phases[name]:.1f} s")
            if PROCESS_START is not None:  # the phase's end, since the process started
                phases[f"{name}_end_since_start"] = time.time() - PROCESS_START

    return Phase()


def ratings_case(options: dict, phases: dict) -> dict:
    """The match engine: build, a first batch (compiles, captures) and a second."""
    from bnb.game.api import PRESETS
    from bnb.league.matches import Arena, ArenaConfig, player

    names = (
        options.get("players", "").split(",")
        if options.get("players")
        else [
            "scripted",
            "/home/vizia/avesnova/runs-v2/2026-11-08-bc-arm2d-persistence/policy-exponential",
            "/home/vizia/avesnova/runs-v2/2026-11-08-bc-arm2d-persistence/policy-live",
        ]
    )
    preset = options.get("preset", "frontline")
    game = PRESETS[preset]
    if "ships_per_team" in options:
        import dataclasses

        game = dataclasses.replace(game, ships_per_team=int(options["ships_per_team"]))
    device = torch.device("cuda")
    config = ArenaConfig()
    with phase("build", phases):
        arena = Arena([player(name) for name in names], game, device, config)
    pairs = [(1, 0), (2, 0), (1, 2), (0, 1), (0, 2), (2, 1), (1, 0), (2, 0)][: config.blocks]
    with phase("first_batch", phases):
        arena.play(pairs)
    with phase("second_batch", phases):
        arena.play(pairs)
    return dict(
        lanes=list(arena.lanes), players=names, preset=preset, ships_per_team=game.ships_per_team
    )


def train_case(options: dict, phases: dict) -> dict:
    """A run's TrainConfig: build, then `steps` collects and updates (default two)."""
    from bnb.runs.serialise import from_json
    from bnb.runtime import metrics
    from bnb.runtime.train import TrainConfig, Trainer

    run = options.get("run")
    path = options.get("config") or os.path.join(run, "config.json")
    with open(path) as file:
        data = json.load(file)
    for setting in filter(None, options.get("set", "").split(";")):
        name, value = setting.split("=", 1)
        *parents, leaf = name.split(".")
        node = data
        for parent in parents:
            node = node[parent]
        if leaf not in node:
            raise SystemExit(f"set {name}: the config has no such field")
        node[leaf] = json.loads(value)
    for name in options.get("drop", "").split(","):  # fields the current code reads differently
        if name:
            data.pop(name)
    config, filled = from_json(TrainConfig, data)
    if "updates" in options:
        import dataclasses

        config = dataclasses.replace(config, updates=int(options["updates"]))
    torch.manual_seed(0)
    device = torch.device("cuda")
    with phase("build", phases):
        trainer = Trainer(config, device)
    if options.get("heads") == "0":  # the heads' sums off, to time their share
        trainer.ppo.measures_heads = False
    first_update = {}
    for index in range(1, int(options.get("steps", 2)) + 1):
        with phase(f"collect_{index}", phases):
            store, bootstrap = trainer.collect()
        with phase(f"learn_{index}", phases):
            row = metrics.read(trainer.ppo.update(store, bootstrap, trainer.steps_per_update))
        if index == 1:
            first_update = {name: value for name, value in row.items() if "time" not in name}
    return dict(
        run=run or path,
        objective=config.objective,
        steps_per_update=trainer.steps_per_update,
        filled=filled,
        peak_vram_gb=torch.cuda.max_memory_allocated(device) / 2**30,
        peak_reserved_gb=torch.cuda.max_memory_reserved(device) / 2**30,
        first_update=first_update,  # its series, to compare runs that should agree
    )


def realtime_case(options: dict, phases: dict) -> dict:
    """The play shell's tick on the CPU: build, the first tick (compiles), 100 more."""
    from bnb.game.api import PRESETS
    from bnb.runtime.realtime import Realtime

    game = PRESETS[options.get("preset", "frontline")]
    actions = torch.zeros((game.num_ships, 3), dtype=torch.int32)
    with phase("build", phases):
        sim = Realtime(game, compiled=True)
    with phase("first_tick", phases):
        sim.step(actions)
    with phase("steady_100_ticks", phases):
        for _ in range(100):
            sim.step(actions)
    return dict(preset=options.get("preset", "frontline"), uses_compiled=sim.uses_compiled)


def main() -> None:
    """Run the case named on the command line and print its JSON result."""
    case, options = sys.argv[1], dict(a.split("=", 1) for a in sys.argv[2:])
    _time_captures()
    phases: dict = {}
    details = dict(ratings=ratings_case, train=train_case, realtime=realtime_case)[case](
        options, phases
    )
    compiles = [e for e in EVENTS if e["kind"] == "compile"]
    result = dict(
        case=case,
        options=options,
        details=details,
        phases=phases,
        events=EVENTS,
        compile_seconds=sum(e["seconds"] for e in compiles),
        capture_seconds=sum(e["seconds"] for e in EVENTS if e["kind"] == "capture"),
        total_seconds=time.perf_counter() - START,
        dynamo_counters={
            k: dict(v)
            for k, v in dynamo_utils.counters.items()
            if k in ("stats", "frames", "recompiles", "graph_break")
        },
        compile_times=dynamo_utils.compile_times(repr="csv", aggregate=True),
        cache_dirs={k: os.environ.get(k) for k in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")},
        torch=torch.__version__,
    )
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
