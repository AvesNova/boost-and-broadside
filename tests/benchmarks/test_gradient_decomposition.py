"""CPU tests for the offline gradient-decomposition report's pure helpers.

The measurement itself needs a checkpoint and a GPU; what is testable here is
the inversion of ``scope_metric_records``' flattening and the averaging, which
is where a wrong key silently reports the wrong term.
"""

import json

from benchmarks.gradient_decomposition import (
    REFERENCE_TERM,
    SUMMARY_SCOPE,
    _format_table,
    _mean_records,
    _run_profile,
    _summarize,
)
from boost_and_broadside.train.rl.grad_diagnostics import ScopeStatistics, scope_metric_records


def _records(norms, cosines):
    """One update's flat records, built the way the trainer publishes them."""
    total = sum(norms.values())
    stats = ScopeStatistics(
        norms=norms,
        cosines=cosines,
        shares={name: value / total for name, value in norms.items()},
        total_norm=total,
        agreement=1.0,
    )
    return scope_metric_records(SUMMARY_SCOPE, stats)


def test_summary_inverts_the_metric_flattening_the_trainer_publishes():
    records = _records(
        {"bc": 20.0, "next_state": 60.0, "density": 20.0},
        {("bc", "next_state"): -0.5, ("bc", "density"): 0.25},
    )
    summary = _summarize(records)

    assert set(summary) == {"bc", "next_state", "density"}
    assert summary["next_state"]["trunk_norm"] == 60.0
    assert summary["next_state"]["trunk_share"] == 0.6
    # The reference term has no cosine with itself, and every other term does
    # regardless of which way round the accumulator emitted the pair.
    assert f"trunk_cos_with_{REFERENCE_TERM}" not in summary["bc"]
    assert summary["next_state"][f"trunk_cos_with_{REFERENCE_TERM}"] == -0.5
    assert summary["density"][f"trunk_cos_with_{REFERENCE_TERM}"] == 0.25


def test_summary_finds_the_cosine_when_the_pair_is_emitted_reference_first():
    records = _records({"bc": 1.0, "value": 1.0}, {("value", "bc"): -0.125})
    assert _summarize(records)["value"][f"trunk_cos_with_{REFERENCE_TERM}"] == -0.125


def test_summary_is_empty_when_the_scope_published_nothing():
    assert _summarize({"loss/total": 1.0}) == {}


def test_mean_drops_keys_absent_from_any_update_rather_than_averaging_unevenly():
    mean = _mean_records([{"a": 1.0, "b": 2.0}, {"a": 3.0}])
    assert mean == {"a": 2.0}


def test_mean_of_no_updates_is_empty():
    assert _mean_records([]) == {}


def test_table_orders_terms_by_the_share_of_the_step_they_ask_for():
    table = _format_table(_summarize(_records({"bc": 1.0, "next_state": 9.0}, {})))
    rows = table.splitlines()[2:]
    assert rows[0].startswith("next_state")
    assert rows[1].startswith("bc")


def test_profile_comes_from_the_run_manifest(tmp_path):
    run = tmp_path / "some-run-1"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"profile": "bc", "global_step": 1}))
    assert _run_profile("some-run-1", tmp_path) == "bc"
    assert _run_profile("no-such-run", tmp_path) is None
