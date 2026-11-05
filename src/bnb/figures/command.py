"""What `bnb figures` does: resolve the policy and the output directory, capture (or read a
capture), and draw the figure sets."""

import pathlib

import torch

from bnb.runs.records import RECORD, Run

from . import capture as capturing
from . import scenarios
from .sets import FIGURE_SETS

CAPTURES = "captures"  # the output directory's subdirectory for capture artifacts


def policy_path(name: str, runs_directory: pathlib.Path) -> pathlib.Path:
    """A policy directory (one with a checkpoint.json), or the newest checkpoint of the run
    `name` under runs_directory."""
    path = pathlib.Path(name)
    if (path / "checkpoint.json").is_file():
        return path
    if (runs_directory / name / RECORD).is_file():
        return Run.open(runs_directory, name).checkpoint_at()
    raise SystemExit(f"{name!r} is neither a policy directory nor a run in {runs_directory}")


def run_of(policy: pathlib.Path) -> pathlib.Path | None:
    """The run directory a policy directory lies in (its parent's, or its grandparent's,
    run.json), else None."""
    for parent in (policy.parent, policy.parent.parent):
        if (parent / RECORD).is_file():
            return parent
    return None


def output_directory(policy: pathlib.Path | None, capture_path, given: str | None) -> pathlib.Path:
    """`given`; else a run's artifacts/figures/<policy directory's name>; else beside the
    capture read; else figures/ in the current directory."""
    if given is not None:
        return pathlib.Path(given)
    if policy is not None and (run := run_of(policy)) is not None:
        return run / "artifacts" / "figures" / policy.name
    if capture_path is not None:
        capture_path = pathlib.Path(capture_path)
        parent = capture_path.parent
        return parent.parent if parent.name == CAPTURES else parent
    return pathlib.Path("figures")


def figures(options, runs_directory: pathlib.Path) -> None:
    """`bnb figures`: see `bnb.cli.Figures`."""
    names = list(options.sets)
    capture_only = names == ["capture"]
    if "all" in names:
        names = list(FIGURE_SETS)
    unknown = [name for name in names if name not in FIGURE_SETS and not capture_only]
    if unknown:
        raise SystemExit(f"unknown figure sets {unknown}; choose from {sorted(FIGURE_SETS)}")
    if options.capture is None and options.policy is None:
        raise SystemExit("bnb figures needs --policy (to capture) or --capture (to read one)")
    policy = None if options.policy is None else policy_path(options.policy, runs_directory)
    directory = output_directory(policy, options.capture, options.out)
    if options.capture is not None:
        artifact = capturing.load(options.capture)
    else:
        config = capturing.CaptureConfig(
            preset=options.preset,
            environments=options.environments,
            ticks=options.ticks,
            seed=options.seed,
            spread_samples=options.spread_samples,
            game_segments=options.game_segments,
        )
        artifact = capturing.capture(
            policy, config, torch.device(options.device), scenarios=scenarios.measure
        )
        path = directory / CAPTURES / f"{options.preset}_seed{options.seed}.pt"
        print(f"wrote {capturing.save(artifact, path)}")
    if capture_only:
        return
    for name in names:
        written, numbers = FIGURE_SETS[name](artifact, directory)
        for path in written:
            print(f"wrote {path}")
        print(f"{name}: {numbers}")
