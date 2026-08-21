"""Installed ``bnb`` command-line interface.

Parser construction is deliberately lightweight. Runtime engines are imported
only after a complete subcommand has parsed, so ``bnb`` and ``bnb --help`` do
not allocate an environment, construct a trainer, or probe an accelerator.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from boost_and_broadside.config.diagnostics import GRADIENT_DIAGNOSTICS_LEVELS
from boost_and_broadside.errors import UserFacingError
from boost_and_broadside.profiles import PROFILES


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _exact_run(value: str) -> str:
    if not value or Path(value).name != value or value in {".", "..", "latest", "none"}:
        raise argparse.ArgumentTypeError(
            f"expected an exact run name, got {value!r}; magic or path-like run names are invalid"
        )
    return value


def _exact_agent(value: str) -> str:
    if not value or value != value.strip() or value in {"latest", "none"}:
        raise argparse.ArgumentTypeError(
            f"expected an exact agent name or checkpoint path, got {value!r}"
        )
    return value


def _checkpoint_path(value: str) -> str:
    path = Path(value)
    if not value or value != value.strip() or path.suffix != ".pt" or path.name == ".pt":
        raise argparse.ArgumentTypeError(
            f"expected an explicit .pt checkpoint path, got {value!r}"
        )
    return value


def _resume_subject(value: str) -> str:
    """Accept either an explicit checkpoint path or one exact run name."""

    return _checkpoint_path(value) if Path(value).suffix == ".pt" else _exact_run(value)


def _vram_policy(value: str) -> str:
    from boost_and_broadside.config.vram import VramError, parse_vram_policy

    try:
        parse_vram_policy(value)
    except VramError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return value


def _matchup(value: str) -> str:
    from boost_and_broadside.evaluation.sizes import MatchupParseError, parse_matchup

    try:
        parse_matchup(value)
    except MatchupParseError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return value


class _AgentListAction(argparse.Action):
    """Require a real Bradley--Terry field at the parsing boundary."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: list[str],
        option_string: str | None = None,
    ) -> None:
        if len(values) < 2:
            raise argparse.ArgumentError(self, "--agents requires at least two exact agents")
        if len(set(values)) != len(values):
            raise argparse.ArgumentError(self, "--agents must not repeat an agent")
        setattr(namespace, self.dest, values)


@dataclass(frozen=True)
class OptionSpec:
    """One declarative option owned by one command."""

    flags: tuple[str, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)
    exclusive_group: str | None = None
    group_required: bool = False


@dataclass(frozen=True)
class CommandSpec:
    """Parser and dispatch metadata for one installed subcommand."""

    name: str
    help: str
    options: tuple[OptionSpec, ...] = ()


def _option(*flags: str, exclusive_group: str | None = None, **kwargs: Any) -> OptionSpec:
    return OptionSpec(flags, kwargs, exclusive_group)


_DEVICE = _option(
    "--device",
    default="auto",
    metavar="DEVICE",
    help="Execution device (default: auto-select CUDA when available).",
)
_SEED = _option(
    "--seed",
    type=int,
    default=None,
    help="Execution RNG seed (default: record the process-generated PyTorch seed).",
)
_ALLOW_DRIFT = _option(
    "--allow-config-drift",
    action="store_true",
    help="Allow checkpoint physics/config drift, with warnings.",
)
_TEAM0 = _option(
    "--team0", type=_exact_agent, required=True, metavar="AGENT", help="Exact team-0 agent."
)
_TEAM1 = _option(
    "--team1", type=_exact_agent, required=True, metavar="AGENT", help="Exact team-1 agent."
)
_RUN = _option(
    "--run", type=_exact_run, required=True, metavar="RUN", help="Exact run directory name."
)
_CALIBRATION = (
    _option(
        "--games-per-batch",
        type=_positive_int,
        default=None,
        metavar="GAMES",
        help="Parallel games played by each adaptive tournament batch.",
    ),
    _option(
        "--target-stderr",
        type=float,
        default=None,
        metavar="ELO",
        help="Stop target for every conditional Elo standard error.",
    ),
    _option(
        "--max-batches",
        type=_positive_int,
        default=None,
        metavar="BATCHES",
        help="Maximum adaptive tournament batches.",
    ),
)


COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec(
        "train",
        "Train one exact registered profile.",
        (
            _option(
                "--profile",
                choices=tuple(sorted(PROFILES)),
                required=True,
                help="Independent training profile.",
            ),
            _option(
                "--resume",
                type=_resume_subject,
                exclusive_group="training-source",
                metavar="PATH_OR_RUN",
                help="Resume from an explicit .pt file or the latest numeric step in an exact run.",
            ),
            _option(
                "--resume-last",
                action="store_true",
                exclusive_group="training-source",
                help=(
                    "Resume the most recently written run of this profile. "
                    "Runs recorded before run.json existed are not eligible."
                ),
            ),
            _option(
                "--pretrain-from",
                type=_checkpoint_path,
                exclusive_group="training-source",
                metavar="PATH",
                help="Warm-start policy/scaler weights from an explicit .pt checkpoint.",
            ),
            _option(
                "--compile",
                dest="compile_mode",
                choices=("none", "reduce-overhead", "default", "max-autotune"),
                default="reduce-overhead",
                metavar="MODE",
                help="torch.compile profile.",
            ),
            _option("--no-wandb", action="store_true", help="Disable W&B logging."),
            _option(
                "--print-config",
                action="store_true",
                help="Resolve, validate, fingerprint, and print the launch without allocation.",
            ),
            _option(
                "--vram",
                type=_vram_policy,
                default="auto",
                metavar="POLICY",
                help=(
                    "Device memory sizing: auto (use a matching cached measurement), "
                    "probe, reprobe, off, or a provisional preset 8|16|24|32 (GB)."
                ),
            ),
            _option(
                "--num-envs",
                type=_positive_int,
                default=None,
                metavar="ENVS",
                help="Explicit aligned environment width; preserves the fixed logical token batch.",
            ),
            _option(
                "--microbatch-tokens",
                type=_positive_int,
                default=None,
                metavar="TOKENS",
                help="Explicit entity tokens per gradient microbatch.",
            ),
            _option(
                "--gradient-diagnostics",
                choices=GRADIENT_DIAGNOSTICS_LEVELS,
                default="off",
                metavar="LEVEL",
                help=(
                    "Decompose the update's gradient by loss term (top_level), also by "
                    "reward component for the policy (reward_policy), or for the policy "
                    "and the critic (reward_full). Default: off."
                ),
            ),
            _option(
                "--gradient-diagnostics-interval",
                type=_positive_int,
                default=1,
                metavar="UPDATES",
                help="Measure gradient diagnostics every N PPO updates (default: 1).",
            ),
            _option(
                "--gradient-diagnostics-minibatches",
                type=_positive_int,
                default=1,
                metavar="MINIBATCHES",
                help="Complete optimizer minibatches measured per diagnostic update.",
            ),
            _DEVICE,
            _SEED,
            _ALLOW_DRIFT,
        ),
    ),
    CommandSpec("play", "Play the fixed human 1v1 duel.", (_DEVICE, _SEED)),
    CommandSpec(
        "watch",
        "Watch two exact agents in the fixed interactive arena.",
        (_TEAM0, _TEAM1, _DEVICE, _SEED, _ALLOW_DRIFT),
    ),
    CommandSpec(
        "capture",
        "Capture seeded scratch gameplay clips from an exact run.",
        (
            _RUN,
            _option("--scenarios", nargs="+", default=["self", "vs_scripted"], metavar="SCENARIO"),
            _option(
                "--seeds",
                default="0-7",
                metavar="SEEDS",
                help="Seed range (0-7) or comma list (0,3,9).",
            ),
            _option("--sizes", nargs="+", type=_matchup, default=None, metavar="MATCHUP"),
            _option("--fps", type=_positive_int, default=60, help="Playback frames per second."),
            _option(
                "--max-steps",
                type=_positive_int,
                default=1024,
                metavar="PHYSICS_TICKS",
                help="Maximum physics ticks before a tie.",
            ),
            _option("--hold-ms", type=_positive_int, default=1500),
            _option("--gif", action="store_true", help="Also write a downscaled GIF."),
            _option(
                "--out",
                type=Path,
                default=Path("out"),
                metavar="DIR",
                help="Scratch output directory (default: out).",
            ),
            _DEVICE,
            _SEED,
            _ALLOW_DRIFT,
        ),
    ),
    CommandSpec(
        "collect-stats",
        "Collect outcome statistics for two exact agents.",
        (
            _TEAM0,
            _TEAM1,
            _option(
                "--sizes",
                nargs="+",
                type=_matchup,
                default=["4v4"],
                metavar="MATCHUP",
            ),
            _option(
                "--games",
                type=_positive_int,
                default=1024,
                help="Parallel games per matchup.",
            ),
            _DEVICE,
            _SEED,
            _ALLOW_DRIFT,
        ),
    ),
    CommandSpec(
        "crossover",
        "Measure the scripted fleet count needed to beat a trained fleet.",
        (
            _RUN,
            _option("--sizes", default="1,2,4,8,16,32,64", metavar="TRAINED_COUNTS"),
            _option(
                "--games-per-matchup",
                type=_positive_int,
                default=256,
                help="Parallel games per tested matchup.",
            ),
            _DEVICE,
            _SEED,
            _ALLOW_DRIFT,
        ),
    ),
    CommandSpec(
        "elo-calibrate",
        "Post-hoc Bradley-Terry calibration for one exact run or agent field.",
        (
            OptionSpec(
                ("--run",),
                {"type": _exact_run, "metavar": "RUN", "help": "Exact finished run."},
                "calibration-subject",
                True,
            ),
            OptionSpec(
                ("--agents",),
                {
                    "nargs": "+",
                    "type": _exact_agent,
                    "action": _AgentListAction,
                    "metavar": "AGENT",
                    "help": "At least two exact agents for an arbitrary calibration field.",
                },
                "calibration-subject",
                True,
            ),
            OptionSpec(
                ("--from-artifact",),
                {
                    "type": Path,
                    "metavar": "ARTIFACT",
                    "help": (
                        "Refit a stored elo-calibration artifact from its match "
                        "matrices, without playing a game."
                    ),
                },
                "calibration-subject",
                True,
            ),
            *_CALIBRATION,
            _DEVICE,
            _SEED,
            _ALLOW_DRIFT,
        ),
    ),
    CommandSpec(
        "elo-scale",
        "Evaluate one exact run across symmetric fleet sizes.",
        (
            _RUN,
            _option("--sizes", default="1,2,4,8,16,32,64", metavar="TEAM_COUNTS"),
            *_CALIBRATION,
            _DEVICE,
            _SEED,
            _ALLOW_DRIFT,
        ),
    ),
    CommandSpec(
        "semi-random",
        "Measure a random-to-scripted reference ladder for one exact profile or run.",
        (
            OptionSpec(
                ("--profile",),
                {
                    "choices": tuple(sorted(PROFILES)),
                    "help": "Exact profile whose environment defines the ladder.",
                },
                "ladder-subject",
                True,
            ),
            OptionSpec(
                ("--run",),
                {"type": _exact_run, "metavar": "RUN", "help": "Exact finished run."},
                "ladder-subject",
                True,
            ),
            _option("--sizes", default="1,2,4,8,16,32,64", metavar="TEAM_COUNTS"),
            _option(
                "--scripted-probabilities",
                default="0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1",
                metavar="PROBABILITIES",
            ),
            _option(
                "--games-per-pair",
                type=_positive_int,
                default=128,
                help="Side-balanced games per unordered agent pair and fleet size.",
            ),
            _option(
                "--max-parallel-games",
                type=_positive_int,
                default=None,
                help="Maximum simultaneous games.",
            ),
            _DEVICE,
            _SEED,
            _ALLOW_DRIFT,
        ),
    ),
    CommandSpec(
        "ar-report",
        "Generate autoregressive diagnostics for two exact agents.",
        (
            _TEAM0,
            _TEAM1,
            _option(
                "--decision-steps",
                type=_positive_int,
                default=512,
                help="Policy decision steps sampled per scenario.",
            ),
            _DEVICE,
            _SEED,
            _ALLOW_DRIFT,
        ),
    ),
    CommandSpec(
        "noise-calibration",
        "Measure next-state prediction error for two exact agents.",
        (_TEAM0, _TEAM1, _DEVICE, _SEED, _ALLOW_DRIFT),
    ),
    CommandSpec(
        "feature-stats",
        "Collect label-scale calibration statistics for two exact agents.",
        (
            _TEAM0,
            _TEAM1,
            _option("--games", type=_positive_int, default=128, help="Parallel sample games."),
            _option(
                "--decision-steps",
                type=_positive_int,
                default=1024,
                help="Policy decision steps per game stream.",
            ),
            _DEVICE,
            _SEED,
            _ALLOW_DRIFT,
        ),
    ),
    CommandSpec(
        "runs",
        "List the most recently written training runs.",
        (
            _option(
                "--limit",
                type=_positive_int,
                default=10,
                metavar="COUNT",
                help="How many runs to list, newest first.",
            ),
            _option("--all", action="store_true", help="List every run, ignoring --limit."),
            _option(
                "--profile",
                choices=tuple(sorted(PROFILES)),
                help="List only runs recorded as this profile.",
            ),
            _option(
                "--resumable",
                action="store_true",
                help="List only runs holding a resumable step_*.pt checkpoint.",
            ),
        ),
    ),
    CommandSpec(
        "figures",
        "Render one run's figure set into that run's own artifacts.",
        (
            _RUN,
            _option(
                "--only",
                nargs="+",
                default=(),
                metavar="NAME",
                help="Render only these figures (default: the whole set).",
            ),
        ),
    ),
    CommandSpec(
        "publish",
        "Render manifest-selected canonical publications offline.",
        (
            _option("--target", metavar="NAME", help="Render one manifest entry."),
            _option(
                "--check",
                action="store_true",
                help="Check without modifying canonical output.",
            ),
        ),
    ),
    CommandSpec(
        "smoke",
        "Run the isolated smoke matrix.",
        (
            _option(
                "--case",
                metavar="NAME",
                help="Run one registered case for focused diagnosis (default: full matrix).",
            ),
        ),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bnb", description="Boost and Broadside")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for command in COMMANDS:
        subparser = subparsers.add_parser(command.name, help=command.help, description=command.help)
        groups: dict[str, argparse._MutuallyExclusiveGroup] = {}
        for option in command.options:
            target: argparse._ActionsContainer = subparser
            if option.exclusive_group is not None:
                if option.exclusive_group not in groups:
                    groups[option.exclusive_group] = subparser.add_mutually_exclusive_group(
                        required=option.group_required
                    )
                target = groups[option.exclusive_group]
            target.add_argument(*option.flags, **option.kwargs)
    return parser


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = build_parser()
    return parser, parser.parse_args(argv)


def _dispatch_command(command: str, args: argparse.Namespace, argv: Sequence[str]) -> None:
    from boost_and_broadside.cli_commands import execute

    execute(command, args, argv)


def _invocation(argv: Sequence[str] | None) -> tuple[str, ...]:
    """The argument vector to record, whether launched by shell or by test."""

    return tuple(sys.argv) if argv is None else ("bnb", *argv)


def main(argv: Sequence[str] | None = None) -> int:
    parser, args = parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "train" and args.print_config:
            from boost_and_broadside.cli_commands import gradient_diagnostics_from_args
            from boost_and_broadside.config.service import print_resolved_config
            from boost_and_broadside.launch import resolve_training_launch

            launch = resolve_training_launch(
                profile=args.profile,
                vram=args.vram,
                device=args.device,
                seed=args.seed,
                compile_mode=None if args.compile_mode == "none" else args.compile_mode,
                wandb=not args.no_wandb,
                allow_config_drift=args.allow_config_drift,
                gradient_diagnostics=gradient_diagnostics_from_args(args),
                num_envs=args.num_envs,
                microbatch_tokens=args.microbatch_tokens,
                allow_probe=False,
            )
            print_resolved_config(
                launch.resolved, file=sys.stdout, launch=launch.document()
            )
            return 0

        if args.command == "smoke":
            from boost_and_broadside.smoke import SmokeError, run_smoke_matrix

            try:
                run_smoke_matrix(args.case)
            except SmokeError as error:
                parser.error(str(error))
            return 0

        _dispatch_command(args.command, args, _invocation(argv))
    except (UserFacingError, FileNotFoundError, KeyError, ValueError) as error:
        parser.error(str(error))
    return 0


__all__ = ["COMMANDS", "CommandSpec", "OptionSpec", "build_parser", "main", "parse_args"]
