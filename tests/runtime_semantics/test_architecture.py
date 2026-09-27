"""Source-level guards for the shared decision-runtime boundary."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_CALL = re.compile(r"\.(?:get_actions|get_action_and_value)\(")
ENV_ADVANCE = re.compile(
    r"\b(?:\w*env|\w*wrapper|standard|interactive|aux_w)"
    r"\.(?:step|tick|step_interactive)\("
)
RUNTIME_MARKERS = (
    "PendingActionState",
    "advance_autonomous_decision",
    "MatchRunner",
)
RAW_PENDING_ACCESS = re.compile(r"(?:self\.)?(?:team1_)?data\[ObsKey\.PREVIOUS_ACTION\]")


def _python_files(root: Path):
    return sorted(root.rglob("*.py"))


def test_controller_bearing_advance_loops_declare_a_runtime_mechanism() -> None:
    """A new controller loop must visibly opt into authoritative action timing."""

    files = [
        *_python_files(ROOT / "src" / "boost_and_broadside"),
        *_python_files(ROOT / "benchmarks"),
    ]
    offenders = []
    for path in files:
        source = path.read_text()
        if CONTROLLER_CALL.search(source) and ENV_ADVANCE.search(source):
            if not any(marker in source for marker in RUNTIME_MARKERS):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_raw_pending_observation_access_has_explicit_owners() -> None:
    """Only raw-view, imagination, queue, and belief composers own this channel."""

    owners = {
        str(path.relative_to(ROOT))
        for path in _python_files(ROOT / "src" / "boost_and_broadside")
        if RAW_PENDING_ACCESS.search(path.read_text())
    }
    assert owners == {
        "src/boost_and_broadside/env/observation.py",
        "src/boost_and_broadside/evaluation/next_state.py",
        "src/boost_and_broadside/runtime/actions.py",
        "src/boost_and_broadside/train/rl/belief.py",
    }
