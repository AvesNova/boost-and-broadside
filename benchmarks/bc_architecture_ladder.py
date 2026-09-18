"""Launch the four controlled behaviour-cloning arms of the spatial ladder.

Each arm adds exactly one mechanism to the one before it, so a difference in the
learning curves attributes to that mechanism and nothing else:

    BC-1  two 64-wide spatial heads                      (control)
    BC-2  + 3D rotary Q/K over world x, y and attitude
    BC-3  + ally/enemy presence scalars
    BC-4  + shared pairwise relational attention bias

Everything else is held fixed *by construction* rather than by restatement: the
arms are CLI overrides on one registered profile, so the environment, the
schedules, the batch geometry, the optimizer and the evaluation procedure are
the same objects in all four runs. The only differences are the four
``model_config`` flags below.

Seed
----
One shared seed across all four arms. These runs differ by a handful of
parameters against a ~1.9M-parameter network and are compared on a supervised
loss curve, so the dominant nuisance is rollout sampling, not initialisation --
and a shared seed makes the scripted teacher present the same scenes in the same
order to every arm. It does *not* make the runs paired after the first update
(the arms' own actions diverge and they then see different states), so a
difference smaller than the within-arm noise is still not a result. The shared
seed is recorded in each run's config.

Budget
------
200M environment steps per arm. On an RTX 4070 Laptop the shipped pipeline runs
at roughly 2,700 steps/second, so each arm is about 20 hours and the ladder is
about 3.5 days of wall clock. ``--dry-run`` prints the commands without running
them; ``--steps`` shortens the budget for a health check.

Usage:
    uv run --no-sync python benchmarks/bc_architecture_ladder.py --dry-run
    uv run --no-sync python benchmarks/bc_architecture_ladder.py --arm BC-1
    uv run --no-sync python benchmarks/bc_architecture_ladder.py           # all four
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field

TOTAL_TIMESTEPS = 200_000_000
SHARED_SEED = 20260918


@dataclass(frozen=True)
class Arm:
    """One rung of the architecture ladder."""

    name: str
    summary: str
    model_overrides: dict[str, str] = field(default_factory=dict)

    def overrides(self) -> list[str]:
        base = {
            # Every arm uses the new head layout; BC-1 is the control for it.
            "model_config.n_spatial_heads": "2",
            "model_config.spatial_rope": "false",
            "model_config.local_presence": "false",
            "model_config.relational_bias": "false",
            "total_timesteps": str(TOTAL_TIMESTEPS),
        }
        base.update(self.model_overrides)
        return [f"{key}={value}" for key, value in base.items()]


ARMS: tuple[Arm, ...] = (
    Arm("BC-1", "two 64-wide spatial heads"),
    Arm("BC-2", "+ 3D rotary x/y/attitude", {"model_config.spatial_rope": "true"}),
    Arm(
        "BC-3",
        "+ ally/enemy presence",
        {"model_config.spatial_rope": "true", "model_config.local_presence": "true"},
    ),
    Arm(
        "BC-4",
        "+ relational attention bias",
        {
            "model_config.spatial_rope": "true",
            "model_config.local_presence": "true",
            "model_config.relational_bias": "true",
        },
    ),
)


def command(arm: Arm, steps: int, seed: int, compile_mode: str) -> list[str]:
    """The exact ``bnb train`` invocation for one arm.

    W&B logging is on: it is the default, and the ladder is compared on learning
    curves that have to survive the session that launched them.
    """

    overrides = [o for o in arm.overrides() if not o.startswith("total_timesteps=")]
    return [
        "uv",
        "run",
        "--no-sync",
        "bnb",
        "train",
        "--profile",
        "bc",
        "--compile",
        compile_mode,
        "--seed",
        str(seed),
        "--vram",
        "8",
        *overrides,
        f"total_timesteps={steps}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", choices=[a.name for a in ARMS])
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=SHARED_SEED)
    parser.add_argument("--compile", dest="compile_mode", default="default")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    wanted = set(args.arm) if args.arm else {a.name for a in ARMS}
    selected = [arm for arm in ARMS if arm.name in wanted]

    for arm in selected:
        argv = command(arm, args.steps, args.seed, args.compile_mode)
        print(f"\n=== {arm.name}: {arm.summary} ===")
        print(" ".join(argv))
        if args.dry_run:
            continue
        result = subprocess.run(argv)
        if result.returncode != 0:
            print(f"{arm.name} exited {result.returncode}; stopping the ladder", file=sys.stderr)
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
