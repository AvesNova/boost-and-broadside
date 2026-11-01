"""Command-line entry point: `bnb train ...` and `bnb ship-card ...`. The one place that sets
process-wide options."""

import dataclasses
import os

import tyro


@dataclasses.dataclass(frozen=True)
class Train:
    """Train the skeleton MLP with PPO.

    preset: a training preset (deathmatch, deathmatch_wide, frontline), which sets the mode,
    the sizes and the spawn groups, or mixed, the default training mix of all three in one
    batch; without one, the mode and sizes below.
    """

    preset: str | None = None
    mode: str = "team_deathmatch"
    ships_per_team: int = 5
    num_fields: int = 10
    arena_radius: float = 2600.0
    num_envs: int = 1024
    updates: int = 100
    seed: int = 0
    device: str = "cuda"


@dataclasses.dataclass(frozen=True)
class ShipCard:
    """Print every handle, parameter and shown quantity of a ship card per type, and write its
    charts (the Ps curve, the E-M doghouse, the similarity ratios against the P-51D). The
    charts need the optional tools extra (uv sync --extra tools).

    card: a JSON file of ship-card fields (per-type lists; missing fields keep the default);
    the default card without one.
    out: the directory the charts are written to.
    """

    card: str | None = None
    out: str = "ship-card"


def main() -> None:
    os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")  # Triton 3.9's default, made explicit
    args = tyro.extras.subcommand_cli_from_dict({"train": Train, "ship-card": ShipCard})
    if isinstance(args, ShipCard):
        ship_card(args)
    else:
        train(args)


def ship_card(args: ShipCard) -> None:
    from bnb.contracts import GameConfig
    from bnb.game.api import rules_of
    from bnb.game.handling import chart

    rules = rules_of(GameConfig(1, 1, 1000.0))  # flight and weapons need no arena
    types = chart.load_card(args.card)
    card = chart.report(rules, types)
    print("\n".join(chart.lines(rules, types, card)))
    for path in chart.write_charts(rules, types, card, args.out):
        print(f"wrote {path}")


def train(args: Train) -> None:
    import torch

    from bnb.contracts import GameConfig
    from bnb.contracts.pools import environment_multiple
    from bnb.game.modes.rules import PRESETS
    from bnb.runtime.train import LINEUP, TrainConfig, mixed
    from bnb.runtime.train import train as run

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
    cfg = TrainConfig(game=game, pools=pools, num_envs=args.num_envs, updates=args.updates,
                      seed=args.seed)

    def log(row: dict) -> None:
        rewards = " ".join(f"{k} {v:+.3f}" for k, v in row["reward_per_s"].items())
        print(f"update {row['update']:4d}  rollout {row['rollout_s'] * 1e3:6.0f} ms  "
              f"learn {row['learn_s'] * 1e3:6.0f} ms  entropy {row['entropy']:.2f}  "
              f"kl {row['approx_kl']:+.4f}  reward/s: {rewards}", flush=True)

    run(cfg, torch.device(args.device), log)
