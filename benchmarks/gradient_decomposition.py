"""What each loss term contributes to the trunk gradient, measured off a checkpoint.

``max_grad_norm`` renormalizes every term together, so the total gradient norm
cannot say which term is asking for the step. ``TermGradientAccumulator``
measures that, but any diagnostic level costs the compiled update for the whole
run (run 739 paid 2.05x), which is why ``DEFAULT_DIAGNOSTIC_LEVEL`` is ``off``
and why the decomposition is meant to be taken *after* the fact instead:

    Both scalers ride in the checkpoint, so resuming with
    ``--gradient-diagnostics reward_full`` for a handful of updates normalizes
    exactly as training did.  -- config/diagnostics.py

This script is that procedure. It resumes a run's latest checkpoint, runs a few
ordinary PPO updates with diagnostics on, and reports the per-term norms, shares
and cosines. It is a read-only instrument: no checkpoint is written, no run
status is recorded, W&B is never contacted, and the run directory is untouched,
so it is safe to point at a run that is merely paused and will be resumed.

The shared-trunk scope is the one to read. Task-specific heads have disjoint
parameters, so a whole-model cosine between two heads' terms is dragged toward
zero by that disjointness; the trunk is where the terms actually compete.

    uv run --no-sync python benchmarks/gradient_decomposition.py --run pious-butterfly-748
    uv run --no-sync python benchmarks/gradient_decomposition.py \
        --run pious-butterfly-748 --updates 3 --level reward_full \
        --output docs/internal/grad-decomposition-748.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

# The scope whose shares and cosines the summary reports. Heads are disjoint, so
# only here do two terms contend for the same parameters.
SUMMARY_SCOPE = "trunk_top_level"
# The term a BC run is actually trying to minimize; every other term is an
# auxiliary spending part of the same clipped step. Reported as the reference
# column of the summary, and skipped when a profile does not train it.
REFERENCE_TERM = "bc"


def _run_profile(run: str, checkpoint_dir: Path) -> str | None:
    """The profile name recorded in a run's manifest, if the run has one."""

    manifest = checkpoint_dir / run / "run.json"
    if not manifest.is_file():
        return None
    return json.loads(manifest.read_text()).get("profile")


def _resume_path(subject: str, checkpoint_dir: Path) -> Path:
    """The checkpoint ``subject`` names -- a run name, or an explicit ``.pt``."""

    from boost_and_broadside.evaluation.run_catalog import (
        resolve_exact_run,
        resolve_explicit_checkpoint,
        select_latest_resumable_checkpoint,
    )

    if subject.endswith(".pt"):
        return resolve_explicit_checkpoint(subject).path
    return select_latest_resumable_checkpoint(resolve_exact_run(subject, checkpoint_dir)).path


def _summarize(records: dict[str, float]) -> dict[str, dict[str, float]]:
    """Per-term trunk statistics, keyed by term, from one update's flat records.

    ``scope_metric_records`` publishes ``grad_norm/<scope>/<term>``,
    ``grad_share/<scope>/<term>`` and ``grad_cos/<scope>/<a>__<b>``. This inverts
    that flattening for the trunk scope alone and attaches each term's cosine
    with :data:`REFERENCE_TERM`, which is the number that says whether an
    auxiliary is pulling the trunk with the objective or against it.
    """

    prefix = f"grad_norm/{SUMMARY_SCOPE}/"
    terms = sorted(key.removeprefix(prefix) for key in records if key.startswith(prefix))
    summary: dict[str, dict[str, float]] = {}
    for term in terms:
        entry = {
            "trunk_norm": records[f"grad_norm/{SUMMARY_SCOPE}/{term}"],
            "trunk_share": records[f"grad_share/{SUMMARY_SCOPE}/{term}"],
        }
        whole = records.get(f"grad_norm/top_level/{term}")
        if whole is not None:
            entry["whole_model_norm"] = whole
        if term != REFERENCE_TERM:
            # Emitted under whichever ordering the accumulator saw the pair in.
            for left, right in ((term, REFERENCE_TERM), (REFERENCE_TERM, term)):
                cosine = records.get(f"grad_cos/{SUMMARY_SCOPE}/{left}__{right}")
                if cosine is not None:
                    entry[f"trunk_cos_with_{REFERENCE_TERM}"] = cosine
                    break
        summary[term] = entry
    return summary


def _mean_records(per_update: list[dict[str, float]]) -> dict[str, float]:
    """Mean of every key present in every diagnosed update.

    A key missing from one update is dropped rather than averaged over a
    different denominator than its neighbours.
    """

    if not per_update:
        return {}
    shared = set(per_update[0]).intersection(*(set(record) for record in per_update))
    return {key: sum(r[key] for r in per_update) / len(per_update) for key in sorted(shared)}


def _format_table(summary: dict[str, dict[str, float]]) -> str:
    """The summary as a fixed-width table, most demanding term first."""

    order = sorted(summary, key=lambda term: -summary[term]["trunk_share"])
    header = f"{'term':<14}{'trunk norm':>14}{'share':>9}{f'cos({REFERENCE_TERM})':>12}"
    lines = [header, "-" * len(header)]
    for term in order:
        entry = summary[term]
        cosine = entry.get(f"trunk_cos_with_{REFERENCE_TERM}")
        lines.append(
            f"{term:<14}{entry['trunk_norm']:>14.4g}{entry['trunk_share']:>9.3f}"
            f"{'--' if cosine is None else f'{cosine:+.3f}':>12}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="Run name, or a path to a .pt checkpoint.")
    parser.add_argument(
        "--profile",
        default=None,
        help="Profile to resolve. Defaults to the one the run's manifest records.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--vram",
        default="auto",
        help=(
            "Device memory sizing, as `bnb train` takes it. The default reuses "
            "the cached measurement, so the rollout is the width the run used."
        ),
    )
    parser.add_argument("--compile", dest="compile_mode", default="default")
    parser.add_argument(
        "--level",
        default="reward_full",
        help="Diagnostic depth: off, top_level, reward_policy, reward_full.",
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=3,
        help="PPO updates to run. Each is diagnosed; the report averages them.",
    )
    parser.add_argument(
        "--minibatches",
        type=int,
        default=1,
        help="Complete optimizer minibatches measured per update.",
    )
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--microbatch-tokens", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    import torch

    from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
    from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
    from boost_and_broadside.config.diagnostics import GradientDiagnosticsConfig
    from boost_and_broadside.launch import resolve_training_launch
    from boost_and_broadside.profiles import resolve_named_profile
    from boost_and_broadside.train.rl.ppo import PPOTrainer

    profile = args.profile or _run_profile(args.run, args.checkpoint_dir)
    if profile is None:
        parser.error(f"no profile recorded for {args.run!r}; pass --profile")
    checkpoint = _resume_path(args.run, args.checkpoint_dir)
    compile_mode = None if args.compile_mode == "none" else args.compile_mode

    # interval=1: every update this script runs is a measurement. A training
    # launch spaces them out because it pays for the eager update either way;
    # here the whole point of the process is the measurement.
    diagnostics = GradientDiagnosticsConfig(
        level=args.level, interval=1, minibatches=args.minibatches
    )
    launch = resolve_training_launch(
        profile=profile,
        vram=args.vram,
        device=args.device,
        seed=None,
        compile_mode=compile_mode,
        wandb=False,
        allow_config_drift=False,
        gradient_diagnostics=diagnostics,
        num_envs=args.num_envs,
        microbatch_tokens=args.microbatch_tokens,
        report=print,
        resolve=resolve_named_profile,
        overrides={},
    )
    resolved = launch.resolved
    trainer = PPOTrainer(
        train_config=resolved.train_config,
        model_config=resolved.model_config,
        ship_config=resolved.ship_config,
        device=args.device,
        use_wandb=False,
        scripted_agent=StochasticScriptedAgent(resolved.ship_config, StochasticAgentConfig()),
        compile_mode=launch.execution.compile_mode,
        gradient_diagnostics=diagnostics,
    )
    trainer.load_checkpoint(str(checkpoint))
    resumed_step = trainer._global_step

    # The ordinary update path, minus everything that would mutate the run:
    # no _maybe_save_checkpoint, no _refresh_training_schedule, no ladder
    # advance, no logging. The schedule is whatever the checkpoint restored, so
    # each coefficient is the one training was applying at `resumed_step`.
    runtime = trainer._initialize_rollout_runtime()
    trainer._train_start_time = time.time()
    trainer._perf_mark_time = trainer._train_start_time
    trainer._perf_mark_step = trainer._global_step
    trainer._perf_mark_ship_steps = trainer._ship_steps

    per_update: list[dict[str, float]] = []
    for index in range(args.updates):
        start = time.perf_counter()
        terminated = trainer._collect_rollout(runtime, False)
        trainer._compute_rollout_gae(runtime, terminated)
        metrics = trainer._update_epochs(
            all_buffers=[trainer.buffer, *trainer.aux_buffers],
            record_histograms=False,
            precomputed=False,
            update=trainer._start_update + index,
        )
        records = {
            key: float(value)
            for key, value in metrics.items()
            if key.startswith(("grad_norm/", "grad_share/", "grad_cos/", "grad_diag/"))
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        }
        if not records:
            raise RuntimeError(
                f"no gradient-diagnostic records at level {args.level!r}; "
                "is the level 'off'?"
            )
        records["loss/total"] = float(metrics.get("loss/total", float("nan")))
        records["train/gradient_norm"] = float(metrics.get("train/gradient_norm", float("nan")))
        per_update.append(records)
        print(f"update {index + 1}/{args.updates}  {time.perf_counter() - start:.1f}s")

    mean = _mean_records(per_update)
    summary = _summarize(mean)
    result = {
        "run": args.run,
        "profile": profile,
        "checkpoint": str(checkpoint),
        "resumed_global_step": resumed_step,
        "device": torch.cuda.get_device_name(args.device) if args.device == "cuda" else "cpu",
        "torch": torch.__version__,
        "compile_mode": args.compile_mode,
        "level": args.level,
        "updates": args.updates,
        "minibatches_per_update": args.minibatches,
        "scope": SUMMARY_SCOPE,
        "reference_term": REFERENCE_TERM,
        "summary": summary,
        "pre_clip_gradient_norm": mean.get("train/gradient_norm"),
        "max_grad_norm": resolved.train_config.max_grad_norm,
        "mean_records": mean,
        "per_update_records": per_update,
    }
    print()
    print(f"{args.run} @ {resumed_step:,} steps -- {SUMMARY_SCOPE}, mean of {args.updates} updates")
    print(_format_table(summary))
    print()
    print(
        f"pre-clip total gradient norm {result['pre_clip_gradient_norm']:.4g}"
        f"  (max_grad_norm {result['max_grad_norm']})"
    )
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.output}")
    trainer.shutdown()


if __name__ == "__main__":
    main()
