"""Git-ignore rules for artifacts, scratch output, and raw sample payloads.

The landmark run is whitelisted back into the index, so every rule that has to
survive that whitelist is asserted here rather than read off the file.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_LANDMARK = "checkpoints/resilient-resonance-682"


def _ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", path],
        cwd=_ROOT,
        capture_output=True,
    )
    if result.returncode not in (0, 1):
        raise AssertionError(result.stderr.decode())
    return result.returncode == 0


@pytest.mark.parametrize(
    "path",
    [
        "artifacts/elo-calibration/20260809T142500Z-a81bc39e/result.json",
        "out/replays/self_4v4_seed00.mp4",
        ".vram.json",
        f"{_LANDMARK}/artifacts/noise-calibration/20260809T142500Z-a81bc39e/samples/errors.npz",
        f"{_LANDMARK}/artifacts/ar-report/20260809T142500Z-a81bc39e/samples/rollout.npz",
        "artifacts/ar-report/20260809T142500Z-a81bc39e/samples/rollout.npz",
    ],
)
def test_local_only_outputs_are_ignored(path):
    assert _ignored(path)


@pytest.mark.parametrize(
    "path",
    [
        f"{_LANDMARK}/artifacts/elo-scale/20260809T142500Z-a81bc39e/artifact.json",
        f"{_LANDMARK}/artifacts/elo-scale/20260809T142500Z-a81bc39e/result.json",
        "docs/publications.toml",
        "docs/results/elo_curve.png",
    ],
)
def test_promotable_landmark_aggregates_remain_trackable(path):
    assert not _ignored(path)


@pytest.mark.parametrize(
    "path",
    [
        f"{_LANDMARK}/artifacts/wandb-export/20260809T142500Z-a81bc39e/files/output.log",
        f"{_LANDMARK}/artifacts/wandb-export/20260809T142500Z-a81bc39e/files/archive.zip",
        f"{_LANDMARK}/artifacts/crossover/20260809T142500Z-a81bc39e/data/matrix.npz",
        # The raw export an artifact was promoted from is evidence on the same
        # terms: the recorded ingestion command has to rebuild the same payload.
        f"{_LANDMARK}/wandb_export/files/output.log",
    ],
)
def test_a_broad_local_rule_does_not_reach_inside_an_artifact_payload(path):
    """A recorded payload file must be trackable whatever its name happens to be.

    ``verify_artifact`` requires every file ``artifact.json`` records, so one the
    repository refuses to track makes a clean checkout unable to verify — or
    publish — anything citing that artifact. That is not hypothetical: the
    landmark W&B export records ``files/output.log``, and the repository-wide
    ``*.log`` rule aimed at stray training logs quietly swallowed it, leaving a
    fresh clone unable to render a single canonical output.
    """

    assert not _ignored(path)
