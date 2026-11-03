"""The league inside the captured rollout (league spec 2; acceptance 5): a rollout with the
league's seat is CUDA-graph captured once, swaps and drains change only buffers in place so
the graph is never captured again, and nothing synchronises with the host per step."""

import warnings

import league_cases as cases
import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.usefixtures("global_generator", "fresh_compiler"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]


def _collect_and_count_syncs(trainer) -> int:
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            trainer.collect()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    return sum("called a synchronizing" in str(warning.message) for warning in caught)


def test_swaps_and_drains_never_capture_the_rollout_again(tmp_path):
    trainer, league = cases.league(tmp_path, "cuda")
    trainer.collect()
    graph = trainer.rollout.graph
    assert graph is not None
    drains = 0
    for _ in range(40):  # 640 ticks: about ten episodes
        before = trainer.opponents.draining_slot
        trainer.collect()
        drains += before is not None and trainer.opponents.draining_slot != before
        assert trainer.rollout.graph is graph
    assert drains >= 3, drains


def test_the_league_adds_host_synchronisations_per_segment_not_per_step(tmp_path):
    """Capturing already forbids a synchronisation inside the ticks; this checks the hooks:
    doubling the segment adds the same synchronisations with the league as without it."""
    added = {}
    for league in (True, False):
        counts = []
        for length in (32, 64):
            directory = tmp_path / f"{league}-{length}"
            if league:
                trainer, _ = cases.league(directory, "cuda", length=length)
            else:
                trainer = cases.Trainer(
                    cases.config("cuda", length=length, pools=cases.SELF_PLAY), torch.device("cuda")
                )
            trainer.collect()  # compiles and captures
            counts.append(_collect_and_count_syncs(trainer))
        added[league] = counts[1] - counts[0]
    assert added[True] == added[False], added
