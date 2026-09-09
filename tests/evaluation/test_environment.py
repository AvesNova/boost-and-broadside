"""Task-aware evaluation environment construction."""

from types import SimpleNamespace

import pytest

from boost_and_broadside.config import EnvConfig, FieldMapConfig, FrontlineConfig, ShipConfig
from boost_and_broadside.evaluation import environment as evaluation_environment
from boost_and_broadside.evaluation.agents import ResolvedAgent
from boost_and_broadside.evaluation.subjects import describe_environment


class _FieldMap:
    num_fields = 1


def _env_config(num_fields: int) -> EnvConfig:
    return EnvConfig(
        num_ships=8,
        max_bullets=4,
        max_episode_steps=32,
        num_fields=num_fields,
    )


def test_zero_field_factory_builds_without_a_map():
    env = evaluation_environment.create_evaluation_env(
        2, ShipConfig(), _env_config(num_fields=0), "cpu"
    )
    assert env.field_map is None


def test_field_factory_generates_and_attaches_the_declared_map(monkeypatch):
    captured = {}

    def generate(ship_config, env_config, map_config, device, seed=None):
        captured.update(
            ship=ship_config,
            env=env_config,
            map=map_config,
            device=device,
            seed=seed,
        )
        return _FieldMap()

    monkeypatch.setattr(evaluation_environment.FieldMapCache, "generate", generate)
    ship = ShipConfig()
    env_config = _env_config(num_fields=1)
    map_config = FieldMapConfig(cache_size=3)
    env = evaluation_environment.create_evaluation_env(
        2,
        ship,
        env_config,
        "cpu",
        field_map_config=map_config,
        field_map_seed=17,
    )
    assert env.field_map is not None
    assert captured == {
        "ship": ship,
        "env": env_config,
        "map": map_config,
        "device": env.device,
        "seed": 17,
    }


def test_field_environment_requires_map_generation_intent():
    try:
        evaluation_environment.create_evaluation_env(
            1, ShipConfig(), _env_config(num_fields=1), "cpu"
        )
    except ValueError as error:
        assert "field_map_config" in str(error)
    else:
        raise AssertionError("field evaluation unexpectedly constructed without map intent")


def _frontline_env() -> EnvConfig:
    return EnvConfig(
        num_ships=8,
        max_bullets=4,
        max_episode_steps=1800,
        action_repeat=2,
        frontline=FrontlineConfig(
            zone_radius=220.0,
            zone_ring_radius=1200.0,
            playable_radius=2600.0,
            capture_seconds=6.0,
            defense_damage_per_second=2.0,
            respawn_health=25.0,
            spawn_heal_per_second=12.0,
            enemy_spawn_damage_per_second=8.0,
            boundary_damage_per_second=5.0,
            boundary_damage_per_pixel_second=0.05,
            front_win_threshold=5,
        ),
    )


def _policy_with_task(env_config: EnvConfig, ship_config: ShipConfig) -> ResolvedAgent:
    bundle = SimpleNamespace(
        env_config=env_config,
        field_map_config=None,
        ship_config=ship_config,
    )
    return ResolvedAgent("policy", object(), bundle=bundle)


def test_explicit_match_rejects_combat_frontline_mode_mismatch():
    ship = ShipConfig(world_size=(16384.0, 16384.0))
    with pytest.raises(ValueError, match="game mode"):
        evaluation_environment.resolve_evaluation_environment(
            _env_config(0),
            (_policy_with_task(_frontline_env(), ship),),
            ship_config=ship,
        )


def test_explicit_match_rejects_world_size_mismatch():
    with pytest.raises(ValueError, match="world size"):
        evaluation_environment.resolve_evaluation_environment(
            _frontline_env(),
            (_policy_with_task(_frontline_env(), ShipConfig()),),
            ship_config=ShipConfig(world_size=(16384.0, 16384.0)),
        )


def test_frontline_artifact_identity_records_world_and_map_translation():
    described = describe_environment(
        _frontline_env(),
        ship_config=ShipConfig(world_size=(16384.0, 16384.0)),
    )

    assert described["game_mode"] == "frontline"
    assert described["world_size"] == [16384.0, 16384.0]
    assert described["action_repeat"] == 2
    assert described["frontline"]["front_win_threshold"] == 5
    assert described["map_translation"]["distribution"] == "uniform_toroidal_per_episode"
