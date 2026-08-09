"""Collect-stats typed-input failure contracts."""

import pytest

from boost_and_broadside.config import EnvConfig, ShipConfig
from boost_and_broadside.config.defaults import MODEL_CONFIG
from boost_and_broadside.evaluation.sizes import MatchupParseError
from boost_and_broadside.modes.collect import run_collect_stats_mode


def test_collect_stats_rejects_an_invalid_matchup_instead_of_skipping(monkeypatch) -> None:
    monkeypatch.setattr(
        "boost_and_broadside.modes.collect.resolve_agent_spec",
        lambda *args, **kwargs: pytest.fail("resolved an agent for an invalid matchup"),
    )

    with pytest.raises(MatchupParseError):
        run_collect_stats_mode(
            "scripted",
            "random",
            1,
            ShipConfig(),
            EnvConfig(num_ships=8, max_bullets=4, max_episode_steps=8),
            MODEL_CONFIG,
            "cpu",
            matchups=["0v4"],
        )
