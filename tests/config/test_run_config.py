"""``checkpoints/<run>/config.json``: a run's configuration as a function of step."""

from __future__ import annotations

import json

import pytest

from boost_and_broadside.config.run_config import (
    ConfigSegment,
    RunConfigError,
    append_segment,
    config_at,
    latest_config,
    read_segments,
)


def _segment(step: int, **overrides) -> ConfigSegment:
    return ConfigSegment(
        from_step=step,
        profile="rl",
        config={"clip_coef": 0.15},
        overrides=overrides,
    )


def test_a_run_with_no_recorded_history_reads_as_empty(tmp_path) -> None:
    """Runs 682, 716 and 719 all predate this file and must stay readable."""

    assert read_segments(tmp_path) == ()
    assert config_at(tmp_path, 0) is None
    assert latest_config(tmp_path) is None


def test_continuing_a_run_appends_rather_than_replacing(tmp_path) -> None:
    append_segment(tmp_path, _segment(0))
    append_segment(tmp_path, _segment(250_000_000, clip_coef="0.2"))

    segments = read_segments(tmp_path)
    assert [segment.from_step for segment in segments] == [0, 250_000_000]
    assert segments[0].overrides == {}
    assert segments[1].overrides == {"clip_coef": "0.2"}


def test_the_config_in_force_depends_on_the_step(tmp_path) -> None:
    """The question "what did this run train with" has no answer without a step."""

    append_segment(tmp_path, _segment(0))
    append_segment(tmp_path, _segment(250_000_000, clip_coef="0.2"))

    assert config_at(tmp_path, 0).from_step == 0
    assert config_at(tmp_path, 249_999_999).from_step == 0
    assert config_at(tmp_path, 250_000_000).from_step == 250_000_000
    assert config_at(tmp_path, 10**12).from_step == 250_000_000
    # Rating a final checkpoint wants the last segment, and says so.
    assert latest_config(tmp_path).from_step == 250_000_000


def test_relaunching_at_the_newest_step_corrects_it_instead_of_duplicating(tmp_path) -> None:
    """A run that has not trained past its newest segment is being corrected."""

    append_segment(tmp_path, _segment(0, clip_coef="0.2"))
    append_segment(tmp_path, _segment(0, clip_coef="0.3"))

    segments = read_segments(tmp_path)
    assert len(segments) == 1
    assert segments[0].overrides == {"clip_coef": "0.3"}


def test_history_cannot_be_rewritten_behind_a_step_already_recorded(tmp_path) -> None:
    append_segment(tmp_path, _segment(250_000_000))

    with pytest.raises(RunConfigError, match="already has one at 250000000"):
        append_segment(tmp_path, _segment(100_000_000))


def test_an_unreadable_or_future_history_is_an_error_not_an_empty_one(tmp_path) -> None:
    """Silently reading a damaged history as "no history" is the failure to avoid."""

    path = tmp_path / "config.json"
    path.write_text("{not json")
    with pytest.raises(RunConfigError, match="not readable JSON"):
        read_segments(tmp_path)

    path.write_text(json.dumps({"schema_version": 99, "segments": []}))
    with pytest.raises(RunConfigError, match="schema_version"):
        read_segments(tmp_path)
