"""Command-line entry point: `bnb train ...`. The one place that sets process-wide options."""

import dataclasses
import os

import tyro


@dataclasses.dataclass(frozen=True)
class Train:
    """Train the skeleton MLP with PPO."""

    ships_per_team: int = 5
    num_fields: int = 10
    arena_radius: float = 2600.0
    num_envs: int = 1024
    updates: int = 100
    seed: int = 0
    device: str = "cuda"


def main() -> None:
    os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")  # Triton 3.9's default, made explicit
    args = tyro.cli(Train)
    import torch

    from bnb.contracts import GameConfig
    from bnb.runtime.train import TrainConfig, train

    game = GameConfig(args.ships_per_team, args.num_fields, args.arena_radius)
    cfg = TrainConfig(game=game, num_envs=args.num_envs, updates=args.updates, seed=args.seed)

    def log(row: dict) -> None:
        rewards = " ".join(f"{k} {v:+.3f}" for k, v in row["reward_per_s"].items())
        print(f"update {row['update']:4d}  rollout {row['rollout_s'] * 1e3:6.0f} ms  "
              f"learn {row['learn_s'] * 1e3:6.0f} ms  entropy {row['entropy']:.2f}  "
              f"kl {row['approx_kl']:+.4f}  reward/s: {rewards}", flush=True)

    train(cfg, torch.device(args.device), log)
