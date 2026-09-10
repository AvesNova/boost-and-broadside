"""``key=value`` overrides: what a sweep arm changes, and what it may not."""

from __future__ import annotations

import pytest

from boost_and_broadside.config.overrides import OverrideError, apply_overrides, parse_override
from boost_and_broadside.config.resolve import resolve_profile
from boost_and_broadside.profiles import PROFILES


def _apply(*arguments: str):
    return apply_overrides(PROFILES["rl"], dict(parse_override(text) for text in arguments))


def test_a_value_is_coerced_to_the_type_of_the_field_it_lands_on() -> None:
    changed = _apply(
        "clip_coef=0.2",
        "total_timesteps=5e8",
        "league_uniform_sampling=true",
        "elo_eval.window_size=64",
        "max_episode_steps=none",
    )

    assert changed.clip_coef == 0.2
    assert changed.total_timesteps == 500_000_000  # scientific notation, as an int
    assert changed.league_uniform_sampling is True
    assert changed.elo_eval.window_size == 64  # dotted path into a sub-config
    assert changed.max_episode_steps is None  # optional fields can be cleared


def test_an_override_lands_before_anything_is_derived_from_it() -> None:
    """The whole reason overrides are applied to the profile and not the result.

    Dropping to zero fields makes each environment cheaper, which increases the
    shard width that fits the fixed rollout-token target. An override applied
    after resolution would retain the smaller width of the ten-field environment.
    """

    resolved = resolve_profile(_apply("num_fields=0"))
    unchanged = resolve_profile(PROFILES["rl"])

    assert resolved.env_config.num_fields == 0
    assert resolved.train_config.scales[0].num_envs > unchanged.train_config.scales[0].num_envs


def test_a_misspelled_key_is_refused_with_the_nearest_real_one() -> None:
    """A silently ignored override produces an arm that did not test what it claims."""

    with pytest.raises(OverrideError, match="did you mean clip_coef"):
        _apply("clip_coeff=0.2")


def test_a_value_that_does_not_fit_its_field_is_refused() -> None:
    with pytest.raises(OverrideError, match="could not convert"):
        _apply("clip_coef=high")
    with pytest.raises(OverrideError, match="cannot be set from the command line"):
        _apply("rewards=1")
    with pytest.raises(OverrideError, match="expected key=value"):
        parse_override("clip_coef")


def test_a_literal_model_mode_can_be_selected_from_the_command_line() -> None:
    changed = _apply("model_config.map_read_mode=kv_memory")

    assert changed.model_config.map_read_mode == "kv_memory"


def test_an_unknown_literal_value_lists_the_supported_modes() -> None:
    with pytest.raises(OverrideError, match="expected one of.*full_attention.*kv_memory"):
        _apply("model_config.map_read_mode=compressed")


def test_zero_fields_needs_no_separate_map_configuration() -> None:
    assert resolve_profile(_apply("num_fields=0")).env_config.num_fields == 0
