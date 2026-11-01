"""Command-line entry point: `bnb train ...`. The one place that sets process-wide options
(the seed and the matmul precision)."""

import dataclasses
import os

import tyro


@dataclasses.dataclass(frozen=True)
class Train:
    """Train a model with PPO (learner spec): Yemong, or the toy relational model.

    preset: a training preset (deathmatch, deathmatch_wide, frontline), which sets the mode,
    the sizes and the spawn groups, or mixed, the default training mix of all three in one
    batch; without one, the mode and sizes below.
    precision: float32, tf32 or bfloat16 (the model's matmuls); old_log_probabilities:
    rollout or recomputed (by unroll before the first epoch).
    """

    preset: str | None = None
    mode: str = "team_deathmatch"
    ships_per_team: int = 5
    num_fields: int = 10
    arena_radius: float = 2600.0
    num_envs: int = 1024
    updates: int = 100
    shards: int = 1
    model: str = "yemong"
    precision: str = "float32"
    old_log_probabilities: str = "rollout"
    micro_batch_rows: int = 160
    seed: int = 0
    device: str = "cuda"


def main() -> None:
    os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")  # Triton 3.9's default, made explicit
    args = tyro.cli(Train)
    import torch

    from bnb.contracts import GameConfig
    from bnb.contracts.pools import environment_multiple
    from bnb.game.modes.rules import PRESETS
    from bnb.learn.ppo import LearnerConfig
    from bnb.runtime.train import LINEUP, TrainConfig, mixed, set_precision, train

    pools = LINEUP
    if args.preset == "mixed":
        game, pools = mixed()
    elif args.preset is not None:
        if args.preset not in PRESETS:
            raise SystemExit(f"unknown preset {args.preset!r}; one of "
                             f"{sorted(PRESETS) + ['mixed']}")
        game = PRESETS[args.preset]
    else:
        game = GameConfig(args.ships_per_team, args.num_fields, args.arena_radius,
                          mode=args.mode)
    multiple = environment_multiple(pools)
    if args.num_envs % multiple:
        raise SystemExit(f"--num-envs {args.num_envs} does not split into the pools' shares: "
                         f"use a multiple of {multiple}")
    learner = LearnerConfig(old_log_probabilities=args.old_log_probabilities,
                            micro_batch_rows=args.micro_batch_rows, seed=args.seed)
    cfg = TrainConfig(game=game, pools=pools, num_envs=args.num_envs, updates=args.updates,
                      shards=args.shards, model=args.model, precision=args.precision,
                      learner=learner, seed=args.seed)
    torch.manual_seed(cfg.seed)
    set_precision(cfg.precision)

    def log(row: dict) -> None:
        rewards = " ".join(f"{name} {sum(pool['reward_per_s'].values()):+.3f}"
                           for name, pool in row["pools"].items())
        print(f"update {row['update']:4d}  rollout {row['rollout_s'] * 1e3:6.0f} ms  "
              f"learn {row['learn_s'] * 1e3:6.0f} ms  entropy {row['loss/entropy']:.2f}  "
              f"kl {row['approximate_kl']:+.4f}  clip {row['clip_fraction']:.3f}  "
              f"reward/s: {rewards}", flush=True)

    train(cfg, torch.device(args.device), log)
