"""`bnb.game.VERSION` rises with what the game computes: a short seeded match in each game
preset (the final state and observation, on the CPU, uncompiled) is kept in
`game_reference.pt` beside the version it was made at, so a change to the game's numbers
fails here until the version is raised and the reference regenerated (a refactor that
computes the same numbers passes unchanged).

Regenerate after a meant change, with `bnb.game.VERSION` raised:
`uv run --no-sync python -m tests.sections.test_game_version --write`.
"""

import sys
from pathlib import Path

import torch
from torch.utils import _pytree

from bnb import game
from bnb.game.api import PRESETS
from bnb.runtime.realtime import Realtime

TICKS = 48
# the game presets the reference plays: the training and rating games (the fleet-size
# presets are these games at other sizes, and a 40v40 match would make the test slow)
REFERENCED = (
    "deathmatch",
    "deathmatch_wide",
    "frontline",
    "real_scale_deathmatch",
    "real_scale_frontline",
    "real_scale_elimination",
    "clover",
    "real_scale_elimination_hemisphere",
)
REFERENCE_PATH = Path(__file__).with_name("game_reference.pt")
# "Within rounding", not bitwise: machines differ in the last bit or two of a float32 (a
# vector instruction set, a fused multiply-add), and 48 ticks of a smooth sim grow that to
# a few parts in 1e6 at most. rtol 1e-4 is two orders above that and far below any rule or
# constant change; atol 1e-6 covers values that are zero up to rounding.
RELATIVE_TOLERANCE = 1e-4
ABSOLUTE_TOLERANCE = 1e-6


def play(name: str) -> dict[str, torch.Tensor]:
    """The leaves of (state, observation) after the preset's seeded match, by tree path."""
    config = PRESETS[name]
    sim = Realtime(config, compiled=False, generator=torch.Generator().manual_seed(1))
    draws = torch.Generator().manual_seed(2)
    for _ in range(TICKS):
        actions = torch.stack(
            [
                torch.randint(0, 3, (config.num_ships,), generator=draws),
                torch.randint(0, 7, (config.num_ships,), generator=draws),
                torch.randint(0, 2, (config.num_ships,), generator=draws),
            ],
            -1,
        ).to(torch.int32)
        obs = sim.step(actions)
    leaves, _ = _pytree.tree_flatten_with_path((sim.state, obs))
    return {
        _pytree.keystr(path): leaf.detach().clone()
        for path, leaf in leaves
        if isinstance(leaf, torch.Tensor)
    }


def write_reference() -> None:
    reference = {"version": game.VERSION, "presets": {name: play(name) for name in REFERENCED}}
    torch.save(reference, REFERENCE_PATH)
    print(f"wrote {REFERENCE_PATH} (game version {game.VERSION})")


def _mismatch(name: str, path: str, expected: torch.Tensor, actual: torch.Tensor) -> str | None:
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return (
            f"{name} {path}: shape/dtype {tuple(actual.shape)} {actual.dtype} "
            f"against the reference's {tuple(expected.shape)} {expected.dtype}"
        )
    if expected.is_floating_point():
        close = torch.isclose(
            actual, expected, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE, equal_nan=True
        )
        difference = (actual.double() - expected.double()).abs()
    else:
        close = actual == expected
        difference = (actual.double() - expected.double()).abs()
    if close.all():
        return None
    difference = torch.where(close, torch.zeros_like(difference), difference)
    return f"{name} {path}: largest difference {difference.max().item():.6g}"


def test_the_game_version_names_what_the_game_computes():
    reference = torch.load(REFERENCE_PATH)
    assert reference["version"] == game.VERSION, (
        f"game_reference.pt was made at game version {reference['version']} but "
        f"bnb.game.VERSION is {game.VERSION}: regenerate it with "
        "`uv run --no-sync python -m tests.sections.test_game_version --write`"
    )
    assert tuple(reference["presets"]) == REFERENCED
    failures = []
    for name in REFERENCED:
        expected_leaves = reference["presets"][name]
        actual_leaves = play(name)
        if actual_leaves.keys() != expected_leaves.keys():
            failures.append(
                f"{name}: leaves differ, {sorted(actual_leaves.keys() ^ expected_leaves.keys())}"
            )
            continue
        for path, expected in expected_leaves.items():
            message = _mismatch(name, path, expected, actual_leaves[path])
            if message:
                failures.append(message)
    assert not failures, (
        "the game computes other numbers than the reference records (floats beyond "
        f"rtol {RELATIVE_TOLERANCE} / atol {ABSOLUTE_TOLERANCE}, integers and flags "
        "exactly):\n"
        + "\n".join(failures)
        + "\nif the change is meant, raise bnb.game.VERSION and regenerate the reference "
        "with `uv run --no-sync python -m tests.sections.test_game_version --write`; "
        "if it is a refactor, it should not change them"
    )


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        raise SystemExit("usage: python -m tests.sections.test_game_version --write")
    write_reference()
