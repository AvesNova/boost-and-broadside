"""Compile tripwire: a cold compile of every training graph, against a recorded baseline.

    PYTHONPATH=. python docs/rewrite/sections/scoreboard/compile_check.py [--update-baseline]
        [--baseline compile_baseline.json] [--operations-tolerance 0.10] [--time-tolerance 0.25]

Run it at the end of every step that touches lean/ (on the GPU; about 4 minutes on the
laptop's 4070). It runs the run's two stages in miniature with lean.learn.compiled's graphs
and a fresh, empty inductor cache (a temporary TORCHINDUCTOR_CACHE_DIR, so the compile is
cold): 64 environments, one 16-tick segment and one BC update with the scripted agent the
only league member (no network), then the league opened (RL's start) with two snapshots, one
more segment (now with the routing graph) and one RL update. With `dynamic=True` the graphs do not depend on these
sizes, so the miniature compiles what a run compiles.

Per compiled function it records each Dynamo graph's traced operations (FX call nodes) and
its AOT forward/backward graphs' operations, the backend's seconds, and in total the number
of graphs, recompiles, graph breaks, and the cold seconds from process start to the end of
the RL update. Against the baseline (compile_baseline.json beside this file) it fails (exit
1) when a function compiles more graphs than recorded (a recompile or a new graph), any
graph breaks, or a function's traced operations grow by more than --operations-tolerance;
it warns when the cold seconds grow by more than --time-tolerance (time is compared only on
the baseline's GPU and power profile), and fails when they exceed the baseline's budget
(`budget_s`, the author's cap on a cold start; on another GPU or profile time is not
compared). --update-baseline writes the run as the new baseline, keeping the old budget
unless --budget sets one (after a deliberate change; say why in the commit). Measure on a
quiet machine (load average below about 4): a busy CPU slows the compile.
"""

import os
import tempfile
import time

PROCESS_START = time.perf_counter()
if "TORCHINDUCTOR_CACHE_DIR" not in os.environ:  # cold by default: an empty cache
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="bnb-compile-check-")
    os.environ["TRITON_CACHE_DIR"] = os.path.join(os.environ["TORCHINDUCTOR_CACHE_DIR"], "triton")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from torch._dynamo.utils import counters  # noqa: E402
from torch._inductor.compile_fx import compile_fx, compile_fx_inner  # noqa: E402

BASELINE = Path(__file__).with_name("compile_baseline.json")
GRAPHS = []  # per compiled graph: function, operations, aot operations, backend seconds


def operations(graph_module) -> int:
    return sum(
        node.op in ("call_function", "call_method", "call_module")
        for node in graph_module.graph.nodes
    )


def counting_backend(name: str):
    """Inductor (as torch.compile's default backend) with each graph's sizes recorded."""

    def backend(graph_module, example_inputs):
        record = {"function": name, "operations": operations(graph_module), "aot": []}

        def inner(graph, inputs, **keywords):
            record["aot"].append(
                ("backward" if keywords.get("is_backward") else "forward", operations(graph))
            )
            return compile_fx_inner(graph, inputs, **keywords)

        began = time.perf_counter()
        compiled = compile_fx(graph_module, example_inputs, inner_compile=inner)
        record["backend_s"] = time.perf_counter() - began
        GRAPHS.append(record)
        return compiled

    return backend


def instrument() -> None:
    """torch.compile with the counting backend (lean.learn.compiled calls torch.compile)."""
    original = torch.compile

    def compile_counted(model=None, **keywords):
        if model is None:
            return lambda function: compile_counted(function, **keywords)
        keywords["backend"] = counting_backend(getattr(model, "__name__", type(model).__name__))
        return original(model, **keywords)

    torch.compile = compile_counted


def run(game_settings: dict) -> float:
    """The miniature run (module docstring); returns seconds from process start to its end."""
    from lean import game, model, scripted
    from lean.learn import (
        League,
        LearnConfig,
        Opponent,
        RolloutStore,
        WeightAverages,
        collect,
        make_optimiser,
        start,
    )
    from lean.learn import update as learn_update

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    # league_capacity_factor 12 (= league_slots): every slot's capacity is the whole league (C = L),
    # so the members placed by hand below fit the table (the graphs are the same at any C).
    config = dataclasses.replace(
        LearnConfig(),
        segments_per_update=1,
        segment_ticks=16,
        minibatches=2,
        epochs=1,
        league_capacity_factor=12.0,
    )
    game_config = game.GameConfig(training_spawn=True, **game_settings)
    policy = model.Policy(model.ModelConfig.for_arena(game_config.arena_radius)).to(device)
    league = League(config, device)
    opponent = Opponent(scripted.ScriptedConfig(), league)
    generator = torch.Generator(device=device).manual_seed(0)
    state, carry = start(config, game_config, policy, opponent, 64, generator)
    store = RolloutStore(1, config.stream_rollout and device.type == "cuda")
    optimiser = make_optimiser(config, policy)
    averages = WeightAverages(policy, config)

    def segment(state, carry):
        rollout = collect(
            config,
            game_config,
            policy,
            opponent,
            state,
            carry,
            config.segment_ticks,
            generator,
            compiled=True,
            store=store,
            segment=0,
        )
        return rollout.state, rollout.carry

    state, carry = segment(state, carry)  # BC: the league has no network yet
    learn_update(config, policy, optimiser, store, cloning=True, generator=generator, compiled=True)
    averages.update(policy, 10**9)
    league.open(policy, averages, slot=carry.slot)  # RL's start: the personas and the EMA join
    for index in range(2):
        step = league.schedule.points(10**12)[index]
        league.after_update(policy, averages, step - 1, step, carry.member, carry.slot)
    # Every loaded slot plays some league games (the yardstick, personas, averages, snapshots).
    loaded = [slot for slot, state in enumerate(league.slot_state) if state == "loaded"]
    slot = torch.tensor(loaded, device=device)[
        torch.arange(carry.member.shape[0], device=device) % len(loaded)
    ]
    carry = dataclasses.replace(carry, member=league.slot_member[slot], slot=slot)
    state, carry = segment(state, carry)  # RL: the routing graph plays the networks
    learn_update(
        config, policy, optimiser, store, teacher_weight=0.5, generator=generator, compiled=True
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - PROCESS_START


def power_profile() -> str:
    try:
        return subprocess.run(
            ["powerprofilesctl", "get"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except OSError, subprocess.SubprocessError:
        return "unknown"


def measure(game_settings: dict | None = None) -> dict:
    instrument()
    cold_s = run(game_settings or {})
    functions = {}
    for record in GRAPHS:
        entry = functions.setdefault(
            record["function"], {"graphs": 0, "operations": 0, "aot": [], "backend_s": 0.0}
        )
        entry["graphs"] += 1
        entry["operations"] = max(entry["operations"], record["operations"])
        entry["aot"] = [list(item) for item in record["aot"]]
        entry["backend_s"] = round(entry["backend_s"] + record["backend_s"], 1)
    breaks = sum(counters["graph_break"].values())
    return {
        "cold_s": round(cold_s, 1),
        "graphs": len(GRAPHS),
        "graph_breaks": breaks,
        "operations": sum(entry["operations"] for entry in functions.values()),
        "functions": functions,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "power_profile": power_profile(),
        "torch": torch.__version__,
    }


def compare(
    now: dict, baseline: dict, operations_tolerance: float, time_tolerance: float
) -> tuple[list, list]:
    failures, warnings = [], []
    if now["graph_breaks"]:
        failures.append(f"{now['graph_breaks']} graph break(s)")
    for name, entry in now["functions"].items():
        before = baseline["functions"].get(name)
        if before is None:
            failures.append(
                f"{name}: a new compiled function ({entry['graphs']} graph(s), {entry['operations']} operations)"
            )
            continue
        if entry["graphs"] > before["graphs"]:
            failures.append(
                f"{name}: {entry['graphs']} graphs, baseline {before['graphs']} (a recompile)"
            )
        growth = entry["operations"] / before["operations"] - 1.0
        if growth > operations_tolerance:
            failures.append(
                f"{name}: {entry['operations']} traced operations, baseline {before['operations']} (+{growth:.0%})"
            )
    if (now["gpu"], now["power_profile"]) == (baseline["gpu"], baseline["power_profile"]):
        growth = now["cold_s"] / baseline["cold_s"] - 1.0
        if now["cold_s"] > baseline["budget_s"]:
            failures.append(
                f"cold compile {now['cold_s']} s, over the budget of {baseline['budget_s']} s"
            )
        elif growth > time_tolerance:
            warnings.append(
                f"cold compile {now['cold_s']} s, baseline {baseline['cold_s']} s (+{growth:.0%})"
            )
    else:
        warnings.append(
            f"time not compared: {now['gpu']} / {now['power_profile']}, baseline "
            f"{baseline['gpu']} / {baseline['power_profile']}"
        )
    return failures, warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--operations-tolerance", type=float, default=0.10)
    parser.add_argument("--time-tolerance", type=float, default=0.25)
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="with --update-baseline: the cold budget in seconds (default: the old baseline's)",
    )
    parser.add_argument(
        "--deflector",
        default=None,
        help="the game's deflector preset (GameConfig.deflector; default: the game's)",
    )
    arguments = parser.parse_args()
    now = measure({} if arguments.deflector is None else {"deflector": arguments.deflector})
    shutil.rmtree(os.environ["TORCHINDUCTOR_CACHE_DIR"], ignore_errors=True)
    for name, entry in now["functions"].items():
        print(
            f"{name}: {entry['graphs']} graph(s), {entry['operations']} traced operations, aot {entry['aot']}, "
            f"backend {entry['backend_s']} s"
        )
    print(
        f"total: {now['graphs']} graphs, {now['operations']} traced operations, {now['graph_breaks']} graph breaks, "
        f"cold {now['cold_s']} s ({now['gpu']}, {now['power_profile']})"
    )
    if arguments.update_baseline:
        budget = arguments.budget
        if budget is None and arguments.baseline.exists():
            budget = json.loads(arguments.baseline.read_text()).get("budget_s")
        now["budget_s"] = budget if budget is not None else round(1.25 * now["cold_s"])
        arguments.baseline.write_text(json.dumps(now, indent=1) + "\n")
        print(f"baseline written: {arguments.baseline}")
        return
    failures, warnings = compare(
        now,
        json.loads(arguments.baseline.read_text()),
        arguments.operations_tolerance,
        arguments.time_tolerance,
    )
    for warning in warnings:
        print(f"WARNING {warning}")
    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        sys.exit(1)
    print("compile check: within the baseline")


if __name__ == "__main__":
    main()
