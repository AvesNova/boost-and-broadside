"""Single-config VRAM/SPS benchmark for the PPO trainer.

Runs ONE configuration in a fresh process (so allocator state / compile caches
never leak between configs) and prints a single JSON line of results. Loop it
over configs from a shell to build a sweep; running each config in its own
process keeps the VRAM numbers clean.

Methodology: decompose peak VRAM into
  - persistent   = params + optimizer state + rollout buffer (resident between updates)
  - activation   = update_peak - persistent  (governed by micro-batch shape + depth,
                   independent of num_envs -> representative at reduced num_envs)
  - buffer_mb    = summed analytically from the buffer tensors (linear in num_envs)

The trainer's own logged `sps` is a cumulative average that folds in compile
warmup, so this times steady-state windows itself after skipping W warmups.
Rollout is measured separately from the update phase (gradient checkpointing only
affects the update-time backward).

Usage (from the repo root):
    uv run python scripts/bench_mem.py \
        --blocks 4 --num-envs 36 --minibatches 1 --microbatch-tokens 37500 \
        --compile default --grad-checkpoint --warmup 2 --measure 3
"""

import argparse
import json
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import constant
from boost_and_broadside.config.defaults import MODEL_CONFIG, SHIP_CONFIG
from boost_and_broadside.profiles import RL_TRAIN_CONFIG
from boost_and_broadside.train.rl.ppo import PPOTrainer

MB = 1024.0 * 1024.0


def _buffer_bytes(trainer: PPOTrainer) -> int:
    """Sum bytes of every tensor the primary rollout buffer holds."""
    buf = trainer.buffer
    total = 0
    for v in buf.obs.values():
        total += v.numel() * v.element_size()
    for name in (
        "actions",
        "logprobs",
        "rewards",
        "values",
        "dones",
        "alive_mask",
        "advantages",
        "returns",
        "adv_agg",
        "ret_agg",
        "actor_masks",
        "expert_probs",
        "terminated",
    ):
        t = getattr(buf, name, None)
        if isinstance(t, torch.Tensor):
            total += t.numel() * t.element_size()
    if isinstance(buf.initial_hidden, torch.Tensor):
        total += buf.initial_hidden.numel() * buf.initial_hidden.element_size()
    return total


def _host_buffer_bytes(buffers) -> int:
    """Sum pageable tensors retained by host-backed logical buffers."""
    total = 0
    for buffer in buffers:
        for shard in getattr(buffer, "shards", ()):
            tensors = [
                *shard.obs.values(),
                shard.actions,
                shard.logprobs,
                shard.advantages,
                shard.returns,
                shard.alive_mask,
                shard.actor_masks,
                shard.expert_probs,
                shard.terminated,
                shard.initial_hidden,
                shard.adv_agg,
                shard.ret_agg,
                shard.ns_labels,
            ]
            total += sum(
                tensor.numel() * tensor.element_size()
                for tensor in tensors
                if isinstance(tensor, torch.Tensor)
            )
    return total


def _lean_config(
    base,
    num_envs,
    minibatches,
    microbatch_tokens,
    rollouts_per_update,
    num_ships=None,
    epochs=None,
):
    """Reduce to a policy-only benchmark: no scripted/avg/league opponents, tiny eval."""
    schedule = replace(
        base.schedule,
        scripted_fraction=constant(0.0),
        avg_model_fraction=constant(0.0),
        league_fraction=constant(0.0),
        behavior_cloning_coef=constant(0.0),
    )
    if epochs is not None:
        schedule = replace(schedule, num_epochs=constant(epochs))
    scale0 = base.scales[0]
    if num_ships is not None:
        scale0 = replace(scale0, env_config=replace(scale0.env_config, num_ships=num_ships))
    scales = (replace(scale0, num_envs=num_envs),) + tuple(base.scales[1:])
    return replace(
        base,
        scales=scales,
        schedule=schedule,
        num_minibatches=minibatches,
        microbatch_tokens=microbatch_tokens,
        rollouts_per_update=rollouts_per_update,
        elo_eval=replace(base.elo_eval, envs_per_matchup=4),
        obstacle_cache=None,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, required=True)
    ap.add_argument("--num-envs", type=int, default=36)
    ap.add_argument("--num-ships", type=int, default=None)
    ap.add_argument("--minibatches", type=int, default=1)
    ap.add_argument("--microbatch-tokens", type=int, default=37500)
    ap.add_argument("--rollouts-per-update", type=int, default=1)
    ap.add_argument(
        "--compile",
        default="none",
        choices=["none", "reduce-overhead", "default", "max-autotune"],
    )
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override PPO epochs/update (default: schedule's 4)",
    )
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--measure", type=int, default=3)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = "cuda"
    compile_mode = None if args.compile == "none" else args.compile
    model_kwargs = {"n_yemong_blocks": args.blocks}
    if args.grad_checkpoint:
        model_kwargs["grad_checkpoint"] = True  # field only exists where the feature is present
    model_cfg = replace(MODEL_CONFIG, **model_kwargs)
    cfg = _lean_config(
        RL_TRAIN_CONFIG,
        args.num_envs,
        args.minibatches,
        args.microbatch_tokens,
        args.rollouts_per_update,
        num_ships=args.num_ships,
        epochs=args.epochs,
    )
    agent = StochasticScriptedAgent(SHIP_CONFIG, StochasticAgentConfig())

    result = {
        "tag": args.tag,
        "blocks": args.blocks,
        "num_envs": args.num_envs,
        "num_ships": cfg.scales[0].env_config.num_ships,
        "minibatches": args.minibatches,
        "microbatch_tokens": args.microbatch_tokens,
        "rollouts_per_update": args.rollouts_per_update,
        "compile": args.compile,
        "grad_checkpoint": args.grad_checkpoint,
        "oom": False,
    }

    try:
        torch.manual_seed(0)
        trainer = PPOTrainer(
            train_config=cfg,
            model_config=model_cfg,
            ship_config=SHIP_CONFIG,
            device=device,
            use_wandb=False,
            scripted_agent=agent,
            compile_mode=compile_mode,
        )
        torch.cuda.synchronize()
        result["alloc_after_init_mb"] = torch.cuda.memory_allocated() / MB
        result["buffer_mb"] = _buffer_bytes(trainer) / MB
        result["params_m"] = sum(p.numel() for p in trainer._policy_module.parameters()) / 1e6

        runtime = trainer._initialize_rollout_runtime()

        def collect_update():
            if cfg.rollouts_per_update > 1:
                return trainer._collect_host_rollouts(runtime, avg_eval_active=False)
            dones = trainer._collect_rollout(runtime, avg_eval_active=False)
            trainer._compute_rollout_gae(runtime, dones)
            return [trainer.buffer] + trainer.aux_buffers

        for _ in range(args.warmup):
            all_buffers = collect_update()
            trainer._update_epochs(
                all_buffers=all_buffers,
                precomputed=cfg.rollouts_per_update > 1,
            )
        torch.cuda.synchronize()
        result["persistent_mb"] = torch.cuda.memory_allocated() / MB
        result["host_buffer_mb"] = _host_buffer_bytes(all_buffers) / MB

        rollout_ms, rollout_peak = [], 0
        for _ in range(max(1, args.measure // 2)):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            all_buffers = collect_update()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            rollout_peak = max(rollout_peak, torch.cuda.max_memory_allocated())
            rollout_ms.append((t1 - t0) * 1e3)

        update_ms, update_peak = [], 0
        for _ in range(args.measure):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            trainer._update_epochs(
                all_buffers=all_buffers,
                precomputed=cfg.rollouts_per_update > 1,
            )
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            update_peak = max(update_peak, torch.cuda.max_memory_allocated())
            update_ms.append((t1 - t0) * 1e3)

        roll = sum(rollout_ms) / len(rollout_ms)
        upd = sum(update_ms) / len(update_ms)
        result.update(
            {
                "rollout_ms": roll,
                "update_ms": upd,
                "rollout_peak_mb": rollout_peak / MB,
                "update_peak_mb": update_peak / MB,
                "activation_mb": (update_peak / MB) - result["persistent_mb"],
                "reserved_peak_mb": torch.cuda.max_memory_reserved() / MB,
                "sps": int(
                    args.num_envs * cfg.num_steps * cfg.rollouts_per_update / ((roll + upd) / 1e3)
                ),
                "ship_tokens_per_sec": int(
                    args.num_envs
                    * cfg.num_steps
                    * cfg.scales[0].env_config.num_ships
                    * cfg.rollouts_per_update
                    / ((roll + upd) / 1e3)
                ),
                "epochs": trainer._schedule_state.num_epochs,
            }
        )
        trainer.shutdown()
    except torch.cuda.OutOfMemoryError as e:
        result["oom"] = True
        result["error"] = str(e)[:200]
    except Exception as e:  # noqa: BLE001 — surface any failure in the JSON
        result["error"] = f"{type(e).__name__}: {e}"[:300]

    print("BENCH_JSON " + json.dumps(result))


if __name__ == "__main__":
    main()
