"""Field-aware evaluation environment construction."""

from boost_and_broadside.config import EnvConfig, FieldMapConfig, ShipConfig
from boost_and_broadside.evaluation import environment as evaluation_environment


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
