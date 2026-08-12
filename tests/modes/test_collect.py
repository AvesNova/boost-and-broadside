"""Collect-stats typed-input failure contracts."""

import pytest

from boost_and_broadside.config import EnvConfig, FieldMapConfig, ShipConfig
from boost_and_broadside.config.defaults import MODEL_CONFIG
from boost_and_broadside.evaluation.agents import ResolvedAgent
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


def test_collect_stats_uses_the_field_distribution_declared_by_a_checkpoint(monkeypatch) -> None:
    field_env = EnvConfig(num_ships=8, max_bullets=4, max_episode_steps=8, num_fields=2)
    field_map = FieldMapConfig(cache_size=3)
    bundle = type("Bundle", (), {"env_config": field_env, "field_map_config": field_map})()
    policy = ResolvedAgent("policy", object(), bundle=bundle)
    captured = {}
    monkeypatch.setattr(
        "boost_and_broadside.modes.collect.resolve_agent_spec",
        lambda *args, **kwargs: policy,
    )
    monkeypatch.setattr(
        "boost_and_broadside.modes.collect.evaluate_matchup",
        lambda *args, **kwargs: captured.update(env=args[6], field_map=kwargs["field_map_config"])
        or (1, 0, 0, 1.0),
    )

    run_collect_stats_mode(
        "field-policy.pt",
        "scripted",
        1,
        ShipConfig(),
        EnvConfig(num_ships=8, max_bullets=4, max_episode_steps=8),
        MODEL_CONFIG,
        "cpu",
        matchups=["4v4"],
    )
    assert captured["env"].num_fields == 2
    assert captured["field_map"] == field_map
