"""The league inside the captured rollout (league spec 2; acceptance 5): a rollout with the
league's seat is CUDA-graph captured once, cuts change only buffers in place so the graph is
never captured again, and nothing synchronises with the host per step."""

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


def _count_syncs(work) -> int:
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            work()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    return sum("called a synchronizing" in str(warning.message) for warning in caught)


def test_cuts_never_capture_the_rollout_again(tmp_path):
    trainer, league = cases.league(
        tmp_path, "cuda", league=cases.LeagueConfig(cheese=(), snapshot_every=256, cut_every=1)
    )
    trainer.collect()
    graph = trainer.rollout.graph
    assert graph is not None
    cuts = 0
    for _ in range(40):  # 640 ticks: about ten episodes
        trainer.collect()
        league.collected(trainer)
        trainer.ppo.environment_steps += trainer.steps_per_update
        cuts += league.updated(trainer)["pool/cut/cuts"]
        assert trainer.rollout.graph is graph
    assert cuts >= 3, cuts


def test_a_league_rollout_synchronises_no_more_than_self_play(tmp_path):
    """Capturing already forbids a synchronisation inside the ticks; a segment's rollout with
    the league synchronises as often as one without it, and the league's own reads happen
    once per segment, in its hook."""
    league_trainer, league = cases.league(tmp_path / "league", "cuda", length=64)
    alone = cases.Trainer(
        cases.config("cuda", length=64, pools=cases.SELF_PLAY), torch.device("cuda")
    )
    counts = {}
    for name, trainer in (("league", league_trainer), ("self_play", alone)):
        trainer.collect()  # compiles and captures
        counts[name] = _count_syncs(lambda: trainer.rollout.run(0))
    assert counts["league"] == counts["self_play"], counts
    hook = _count_syncs(lambda: league.after_shard(league_trainer, 0))
    assert hook <= 8, hook  # the tally, the drain check and the slot loads, per segment
