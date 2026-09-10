"""Phase-resolved wall-clock profile of the real RL training pipeline.

Builds the trainer through the same launch path ``bnb train`` uses, then runs a
bounded number of updates with monkeypatched timers around every phase. Nothing
in ``src/`` is modified: the instrumentation lives here and is attached at run
time, so the pipeline being measured is the shipped one.

Two timing modes:

``--mode wall``  Sync only at update boundaries. Phase totals are attributed to
                 whichever call the CPU was inside, so they under-report GPU work
                 that was still queued, but the per-update total is the truth.
``--mode sync``  ``torch.cuda.synchronize()`` on entry and exit of every timed
                 region. Attribution is correct; the per-update total inflates by
                 whatever CPU/GPU overlap the syncs destroy. Run both and compare.

Usage:
    uv run --no-sync python benchmarks/rl_pipeline_profile.py \
        --updates 3 --warmup 1 --mode sync --out /tmp/profile.json
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import time
from collections import defaultdict

import torch

# ----------------------------------------------------------------------
# Timer registry
# ----------------------------------------------------------------------

TOTALS: dict[str, float] = defaultdict(float)
COUNTS: dict[str, int] = defaultdict(int)
_SYNC = False
_ENABLED = False
_DEPTH: list[str] = []


def reset_timers() -> None:
    TOTALS.clear()
    COUNTS.clear()


def _sync() -> None:
    if _SYNC and torch.cuda.is_available():
        torch.cuda.synchronize()


@contextlib.contextmanager
def timed(name: str):
    if not _ENABLED:
        yield
        return
    _sync()
    start = time.perf_counter()
    try:
        yield
    finally:
        _sync()
        TOTALS[name] += time.perf_counter() - start
        COUNTS[name] += 1


def wrap(obj, attr: str, name: str) -> None:
    """Wrap a bound method / function attribute in a timer."""
    original = getattr(obj, attr)

    def wrapper(*args, **kwargs):
        with timed(name):
            return original(*args, **kwargs)

    wrapper._perf_original = original  # type: ignore[attr-defined]
    setattr(obj, attr, wrapper)


def wrap_gen(obj, attr: str, name: str) -> None:
    """Wrap a generator-producing method so only value production is timed."""
    original = getattr(obj, attr)

    def wrapper(*args, **kwargs):
        gen = original(*args, **kwargs)
        while True:
            with timed(name):
                try:
                    value = next(gen)
                except StopIteration:
                    return
            yield value

    setattr(obj, attr, wrapper)


# ----------------------------------------------------------------------
# Instrumentation
# ----------------------------------------------------------------------


def instrument(trainer, detail: bool) -> None:
    from boost_and_broadside.train.rl import buffer as buffer_mod
    from boost_and_broadside.train.rl import elo_eval as elo_mod

    # --- top-level update phases -------------------------------------
    for attr, name in [
        ("_collect_rollout", "01_rollout/total"),
        ("_compute_rollout_gae", "02_gae"),
        ("_precompute_ns_labels", "03_ns_labels"),
        ("_prepare_host_rollouts", "05_lambda_aggregates"),
        ("_update_epochs", "06_update/total"),
        ("_refresh_training_schedule", "07_schedule_refresh"),
        ("_assemble_metrics", "08_metrics_assemble"),
        ("_log_training_update", "09_log"),
        ("_maybe_save_checkpoint", "10_checkpoint"),
        ("_maybe_advance_ladder", "11_ladder"),
        # rollout internals
        ("_collect_primary_step", "01a_rollout/primary_step"),
        ("_collect_aux_steps", "01b_rollout/aux_steps"),
        ("_scripted_step_outputs", "01c_rollout/scripted"),
        ("_step_environment_and_network", "01d_rollout/env+net"),
        ("_select_primary_actions", "01e_rollout/select_actions"),
        ("_reset_primary_hidden", "01f_rollout/reset_hidden"),
        ("_rollout_network_forwards", "01g_rollout/net_forward"),
        ("_prepare_league_slots", "01h_rollout/league_slots"),
        ("_allocate_ladder_games", "01i_rollout/ladder_alloc"),
        # update internals
        ("_minibatch_denominators", "06a_update/denominators"),
        ("_compute_minibatch_loss", "06b_update/forward_loss"),
        ("_stage_microbatch", "06c_update/stage_h2d"),
    ]:
        if hasattr(trainer, attr):
            wrap(trainer, attr, name)

    wrap(trainer.optim, "step", "06e_update/optim_step")
    wrap(trainer.buffer, "add", "01j_rollout/buffer_add")
    wrap(trainer.buffer, "compute_gae", "02a_gae/compute")

    # StoredRollout construction is the device->host copy of a whole shard.
    original_stored_init = buffer_mod.StoredRollout.__init__

    def timed_stored_init(self, source):
        with timed("04_rollout_to_host"):
            original_stored_init(self, source)

    buffer_mod.StoredRollout.__init__ = timed_stored_init

    # Host-side minibatch gather (fancy-index on CPU).
    wrap_gen(buffer_mod.StoredRollout, "get_minibatch_iterator", "06f_update/host_gather")

    # backward pass
    original_backward = torch.Tensor.backward

    def timed_backward(self, *args, **kwargs):
        with timed("06d_update/backward"):
            return original_backward(self, *args, **kwargs)

    torch.Tensor.backward = timed_backward

    # gradient clipping
    original_clip = torch.nn.utils.clip_grad_norm_

    def timed_clip(*args, **kwargs):
        with timed("06g_update/clip_grad"):
            return original_clip(*args, **kwargs)

    torch.nn.utils.clip_grad_norm_ = timed_clip
    import torch.nn.utils as nn_utils

    nn_utils.clip_grad_norm_ = timed_clip
    import boost_and_broadside.train.rl.ppo as ppo_mod

    ppo_mod.nn.utils.clip_grad_norm_ = timed_clip

    # Elo evaluator: separate its step from the rollout it rides inside.
    wrap(elo_mod.EloEvaluator, "step", "01k_rollout/elo_step")
    wrap(elo_mod.EloEvaluator, "flush", "01l_rollout/elo_flush")

    # Belief tracking rides inside the primary step and is otherwise invisible.
    from boost_and_broadside.train.rl import belief as belief_mod

    wrap(belief_mod.BeliefTracker, "compose", "01m_rollout/belief_compose")
    wrap(belief_mod.BeliefTracker, "advance", "01n_rollout/belief_advance")
    wrap(trainer.wrapper, "privileged_observation", "01o_rollout/privileged_obs")
    wrap(trainer.coordinator, "get_target_vector", "01p_rollout/target_vector")

    # Elo evaluator internals.
    wrap(elo_mod.EloEvaluator, "_compute_team_actions", "01k1_elo/team_actions")
    wrap(elo_mod.EloEvaluator, "_apply_rating_updates", "01k2_elo/rating_update")
    wrap(elo_mod.EloEvaluator, "_accumulate_match_counts", "01k3_elo/match_counts")
    wrap(elo_mod.EloEvaluator, "_resample_anchor_assignments", "01k4_elo/resample")
    wrap(elo_mod.EloEvaluator, "_reset_agent_hiddens", "01k5_elo/reset_hidden")
    wrap(elo_mod.EloEvaluator, "_policy_actions", "01k6_elo/policy_actions")
    wrap(elo_mod.EloEvaluator, "_anchor_actions", "01k7_elo/anchor_actions")

    if detail:
        # Split env physics from the policy forward. This forces the two CUDA
        # streams to be timed separately; the caller disables the overlap so
        # the numbers are isolation costs, not concurrent ones.
        wrap(trainer.wrapper, "step", "01d1_rollout/env_step")
        wrap(trainer.wrapper, "_get_obs", "01d3_rollout/env_get_obs")
        from boost_and_broadside.env.env import TensorEnv

        wrap(TensorEnv, "step", "01k8_elo/env_step")
        wrap(TensorEnv, "tick", "01d2b_rollout/env_tick_cls")
        from boost_and_broadside.env import wrapper as wrapper_mod

        wrap(wrapper_mod, "perceived_observation_from_state", "01q_rollout/perceive")
        wrap(elo_mod, "perceived_observation_from_state", "01k9_elo/perceive")


def dump(label: str, updates: int, env_steps_per_update: int, wall: float, out: str | None):
    total_steps = updates * env_steps_per_update
    rows = sorted(TOTALS.items())
    payload = {
        "label": label,
        "updates": updates,
        "env_steps_per_update": env_steps_per_update,
        "wall_seconds": wall,
        "sps": total_steps / wall if wall else 0.0,
        "phases": {
            key: {"seconds": value, "calls": COUNTS[key], "per_update": value / updates}
            for key, value in rows
        },
    }
    print(f"\n=== {label} ===")
    print(
        f"updates={updates}  wall={wall:.2f}s  per_update={wall / updates:.2f}s  "
        f"env_steps/update={env_steps_per_update:,}  SPS={payload['sps']:,.0f}"
    )
    print(f"{'phase':<34} {'s/update':>10} {'calls/upd':>10} {'% wall':>8}")
    for key, value in rows:
        print(
            f"{key:<34} {value / updates:>10.3f} {COUNTS[key] / updates:>10.1f} "
            f"{100 * value / wall:>7.1f}%"
        )
    if out:
        with open(out, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {out}")
    return payload


def main() -> None:
    global _SYNC, _ENABLED

    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--mode", choices=("wall", "sync"), default="wall")
    parser.add_argument("--detail", action="store_true", help="split env/net (disables overlap)")
    parser.add_argument("--no-overlap", action="store_true", help="serialize env and net streams")
    parser.add_argument("--compile", dest="compile_mode", default="reduce-overhead")
    parser.add_argument("--out", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--torch-profile", default=None, help="write a chrome trace here")
    parser.add_argument("--no-checkpoint", action="store_true", help="skip periodic saves")
    parser.add_argument("--microbatch-tokens", type=int, default=None)
    parser.add_argument(
        "--no-microbatch",
        action="store_true",
        help="set microbatch_tokens=None: one backward pass per shard minibatch",
    )
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument(
        "--no-elo", action="store_true", help="ablation: skip the Elo evaluator's per-step work"
    )
    parser.add_argument(
        "--checkpoint-dir", default=None, help="redirect checkpoint writes off the repo"
    )
    args = parser.parse_args()

    _SYNC = args.mode == "sync"

    from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
    from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
    from boost_and_broadside.launch import resolve_training_launch
    from boost_and_broadside.train.rl.ppo import PPOTrainer

    overrides = dict(o.split("=", 1) for o in args.override)
    launch = resolve_training_launch(
        profile="rl",
        vram="auto",
        device="cuda",
        seed=1234,
        compile_mode=None if args.compile_mode == "none" else args.compile_mode,
        wandb=False,
        allow_probe=False,
        report=print,
        overrides=overrides,
        num_envs=args.num_envs,
        microbatch_tokens=args.microbatch_tokens,
    )
    resolved = launch.resolved
    train_config = resolved.train_config
    if args.checkpoint_dir:
        train_config = dataclasses.replace(train_config, checkpoint_dir=args.checkpoint_dir)
    if args.no_microbatch:
        train_config = dataclasses.replace(train_config, microbatch_tokens=None)
    scale = train_config.scales[0]
    env_steps_per_update = (
        scale.num_envs * train_config.num_steps * train_config.rollouts_per_update
    )
    print(
        f"num_envs={scale.num_envs} num_steps={train_config.num_steps} "
        f"rollouts_per_update={train_config.rollouts_per_update} "
        f"minibatches={train_config.num_minibatches} "
        f"microbatch_tokens={train_config.microbatch_tokens} "
        f"num_ships={scale.env_config.num_ships} num_fields={scale.env_config.num_fields} "
        f"env_steps/update={env_steps_per_update:,}"
    )

    torch.manual_seed(1234)
    trainer = PPOTrainer(
        train_config=train_config,
        model_config=resolved.model_config,
        ship_config=resolved.ship_config,
        device="cuda",
        use_wandb=False,
        scripted_agent=StochasticScriptedAgent(resolved.ship_config, StochasticAgentConfig()),
        compile_mode=launch.execution.compile_mode,
        gradient_diagnostics=launch.execution.gradient_diagnostics,
    )
    trainer.beliefs_present = True
    instrument(trainer, args.detail)

    if args.no_elo:
        from boost_and_broadside.train.rl import elo_eval as _elo

        _elo.EloEvaluator.step = lambda self, rollout_step, avg_active: None

    runtime = trainer._initialize_rollout_runtime()
    if args.no_overlap or args.detail:
        runtime.env_stream = None
        runtime.net_stream = None
    trainer._train_start_time = time.time()

    def one_update(update: int):
        avg_eval_active = trainer._avg_update_count > 0
        if trainer.cfg.rollouts_per_update == 1:
            terminated = trainer._collect_rollout(runtime, avg_eval_active)
            trainer._compute_rollout_gae(runtime, terminated)
            update_buffers = [trainer.buffer, *trainer.aux_buffers]
            precomputed = False
        else:
            update_buffers = trainer._collect_host_rollouts(runtime, avg_eval_active)
            precomputed = True
        record_hist = update % trainer.cfg.histogram_interval == 0
        metrics = trainer._update_epochs(
            all_buffers=update_buffers,
            record_histograms=record_hist,
            precomputed=precomputed,
            update=update,
        )
        trainer._refresh_training_schedule(metrics, runtime.elo_eval)
        sps, ship_tps = trainer._assemble_metrics(metrics, update, runtime.ship_tokens_per_update)
        trainer._log_training_update(metrics, update, sps, ship_tps)
        if not args.no_checkpoint:
            trainer._maybe_save_checkpoint(update)
            trainer._maybe_advance_ladder(update, runtime.elo_eval)
        trainer._completed_update = update
        return metrics

    _ENABLED = False
    for update in range(1, args.warmup + 1):
        start = time.perf_counter()
        one_update(update)
        torch.cuda.synchronize()
        print(f"[warmup {update}] {time.perf_counter() - start:.2f}s")

    reset_timers()
    _ENABLED = True
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    per_update = []
    epochs: list[float] = []
    profiler_ctx = contextlib.nullcontext()
    if args.torch_profile:
        profiler_ctx = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            with_stack=False,
        )

    wall_start = time.perf_counter()
    with profiler_ctx as prof:
        for i in range(args.updates):
            update = args.warmup + 1 + i
            start = time.perf_counter()
            metrics = one_update(update)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            per_update.append(elapsed)
            epochs.append(metrics.get("train/epochs_completed", float("nan")))
            print(f"[measure {update}] {elapsed:.2f}s  epochs={epochs[-1]:.0f}")
    wall = time.perf_counter() - wall_start

    if args.torch_profile and prof is not None:
        prof.export_chrome_trace(args.torch_profile)
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=45))

    label = args.label or f"rl/{args.mode}"
    payload = dump(label, args.updates, env_steps_per_update, wall, args.out)
    payload["per_update_seconds"] = per_update
    payload["epochs_completed"] = epochs
    total_epochs = sum(epochs)
    rollout = TOTALS.get("01_rollout/total", 0.0)
    update_total = TOTALS.get("06_update/total", 0.0)
    payload["rollout_s_per_update"] = rollout / args.updates
    payload["update_s_per_epoch"] = update_total / total_epochs if total_epochs else 0.0
    print(f"epochs per update: {epochs}")
    print(
        f"rollout {rollout / args.updates:.2f} s/update   "
        f"update {update_total / max(total_epochs, 1):.2f} s/epoch"
    )
    payload["peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
    payload["peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
    print("per-update: " + ", ".join(f"{value:.2f}" for value in per_update))
    print(
        f"peak allocated {payload['peak_allocated_mb']:.0f} MiB  "
        f"reserved {payload['peak_reserved_mb']:.0f} MiB"
    )
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=2)

    trainer.shutdown()


if __name__ == "__main__":
    main()
