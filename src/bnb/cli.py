"""Command-line entry point: `bnb ...` trains or clones, `bnb ship-card ...` charts a ship
card. The one place that sets process-wide options (the seed, the matmul precision, the
charts' backend)."""

import dataclasses
import os
import sys

import tyro


@dataclasses.dataclass(frozen=True)
class Train:
    """Train a model with PPO (learner spec), or clone the scripted agent (BC): Yemong, or
    the toy relational model.

    preset: a training preset (deathmatch, deathmatch_wide, frontline), which sets the mode,
    the sizes and the spawn groups, or mixed, the default training mix of all three in one
    batch; without one, the mode and sizes below.
    precision: float32, tf32 or bfloat16 (the model's matmuls); old_log_probabilities:
    rollout or recomputed (by unroll before the first epoch).
    objective: ppo, or bc (behaviour cloning against self-play and the scripted agent,
    until a stop rule fires; updates is then unused); teacher_temperature and teacher_floor
    set the labels' entropy; bc_budget_steps: BC's budget in environment steps.
    calibrate_entropy: print the scripted agent's natural entropy over its own games of the
    preset and the temperature whose labels reach this mean entropy (nats per living ship
    step) at teacher_floor, then exit.
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
    objective: str = "ppo"
    teacher_temperature: float = 1.0
    teacher_floor: float = 0.0
    bc_budget_steps: int = 500_000_000
    calibrate_entropy: float | None = None


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
    """The `bnb` command: `bnb ship-card ...` charts a card, anything else trains."""
    os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")  # Triton 3.9's default, made explicit
    if sys.argv[1:2] == ["ship-card"]:
        os.environ.setdefault("MPLBACKEND", "Agg")  # the charts go to files: no display
        ship_card(tyro.cli(ShipCard, args=sys.argv[2:], prog="bnb ship-card"))
    else:
        train(tyro.cli(Train))


def ship_card(args: ShipCard) -> None:
    """`bnb ship-card`: print the card's chart at 30 Hz and write its charts to args.out;
    exits with the reason when matplotlib is missing."""
    from bnb.contracts import GameConfig
    from bnb.game.api import rules_of
    from bnb.game.handling import chart

    rules = rules_of(GameConfig(1, 1, 1000.0))  # flight and weapons need no arena
    types = chart.load_card(args.card)
    card = chart.report(rules, types)
    print("\n".join(chart.lines(rules, types, card)))
    try:
        paths = chart.write_charts(rules, types, card, args.out)
    except ImportError as error:
        raise SystemExit(str(error)) from error
    for path in paths:
        print(f"wrote {path}")


def train(args: Train) -> None:
    """`bnb`: train with PPO, or clone the scripted agent, on the preset (or the mode and
    sizes) given, logging a line per update; or calibrate the teacher's temperature. Exits
    with the reason for an unknown preset or an environment count that the pools cannot
    share."""
    import torch

    from bnb.contracts import GameConfig
    from bnb.contracts.pools import environment_multiple
    from bnb.game.modes.rules import PRESETS
    from bnb.learn import bc
    from bnb.learn.ppo import LearnerConfig
    from bnb.runtime import cloning
    from bnb.runtime.train import LINEUP, TrainConfig, mixed, set_precision
    from bnb.runtime.train import train as run

    cloned = args.objective == "bc"
    pools = cloning.LINEUP if cloned else LINEUP
    if args.preset == "mixed":
        game, pools = mixed(pools)
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
    settings = dict(old_log_probabilities=args.old_log_probabilities,
                    micro_batch_rows=args.micro_batch_rows, seed=args.seed)
    learner = bc.config(**settings) if cloned else LearnerConfig(**settings)
    cfg = TrainConfig(game=game, pools=pools, num_envs=args.num_envs, updates=args.updates,
                      shards=args.shards, model=args.model, precision=args.precision,
                      learner=learner, seed=args.seed, objective=args.objective,
                      teacher=bc.TeacherConfig(args.teacher_temperature, args.teacher_floor))
    torch.manual_seed(cfg.seed)
    set_precision(cfg.precision)
    device = torch.device(args.device)

    if args.calibrate_entropy is not None:
        tokens = cloning.teacher_distributions(game, pools, args.num_envs, 4, device)
        natural = bc.mean_entropy(tokens, bc.TeacherConfig())
        print(f"natural entropy {natural:.4f} nats per living ship-step over "
              f"{tokens.shape[0]} ship-steps", flush=True)
        try:
            chosen = bc.calibrate(tokens, args.calibrate_entropy, args.teacher_floor)
        except ValueError as error:
            raise SystemExit(str(error)) from None
        print(f"temperature {chosen.temperature:.4f} at floor {chosen.floor} gives "
              f"{bc.mean_entropy(tokens, chosen):.4f}")
        return

    def log(row: dict) -> None:
        rewards = " ".join(f"{name} {sum(pool['reward_per_s'].values()):+.3f}"
                           for name, pool in row["pools"].items())
        print(f"update {row['update']:4d}  rollout {row['rollout_s'] * 1e3:6.0f} ms  "
              f"learn {row['learn_s'] * 1e3:6.0f} ms  entropy {row['loss/entropy']:.2f}  "
              f"kl {row['approximate_kl']:+.4f}  clip {row['clip_fraction']:.3f}  "
              f"reward/s: {rewards}", flush=True)

    if cloned:
        _, reason = cloning.clone(cfg, device, bc.StopConfig(budget_steps=args.bc_budget_steps),
                                  log_cloning)
        print(f"stopped by {reason}", flush=True)
        return
    run(cfg, device, log)


def log_cloning(row: dict) -> None:
    scores = " ".join(f"{kind} {value['score']:.3f}" for kind, value in row["scores"].items()
                      if value["score"] is not None)
    print(f"update {row['update']:4d}  steps {row['env_steps']:>11,}  "
          f"rollout {row['rollout_s'] * 1e3:6.0f} ms  learn {row['learn_s'] * 1e3:6.0f} ms  "
          f"bc kl {row['bc_kl']:.4f}  teacher {row['teacher_entropy']:.3f}  "
          f"student {row['student_entropy']:.3f}  scores: {scores}", flush=True)
