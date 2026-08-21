"""Every run-subject evaluation mode, driven against a run that has fields.

This file exists because of a specific failure. Fields were added to training and
the evaluation stack did not follow, and nothing noticed: the synthetic run every
other test and the whole smoke matrix use is field-free, so six modes passed
their checks while being unable to measure the only kind of run being trained.
Three failed outright when finally pointed at one; two produced a plausible
report of a policy fighting in an empty arena it had never trained in, and one
silently dropped the field tokens from its own output layout.

So these tests assert the *environment that was built*, not the exit status. An
exit-code test would have caught the three that raised and none of the three
that did not.
"""

from __future__ import annotations

import pytest

from boost_and_broadside.config import EloCalibrateConfig, EnvConfig, ShipConfig
from boost_and_broadside.config.defaults import MODEL_CONFIG, REWARDS
from boost_and_broadside.smoke import build_synthetic_run

_RUN = "fields-run"


@pytest.fixture
def fields_run(tmp_path):
    """A minimal complete run trained with four refractive fields."""

    return build_synthetic_run(tmp_path, profile="rl-fields", run_name=_RUN)


@pytest.fixture
def built_envs(monkeypatch):
    """Record the configuration of every environment any mode constructs.

    Patched at ``TensorEnv`` itself rather than at one of the several helpers
    that build one, because the modes reach the simulator by different routes --
    ``create_evaluation_env``, ``evaluate_matchup``, and ``YemongEnvWrapper`` --
    and a spy on any single route would leave the others unwatched.
    """

    import boost_and_broadside.env.env as env_module

    built: list[tuple[EnvConfig, bool]] = []
    original = env_module.TensorEnv.__init__

    def recording_init(self, num_envs, ship_config, env_config, device, field_map=None, *a, **kw):
        built.append((env_config, field_map is not None))
        return original(self, num_envs, ship_config, env_config, device, field_map, *a, **kw)

    monkeypatch.setattr(env_module.TensorEnv, "__init__", recording_init)
    return built


def _assert_played_with_fields(built) -> None:
    assert built, "no environment was built at all"
    fielded = [(config, has_map) for config, has_map in built if config.num_fields > 0]
    assert fielded, (
        "every environment was built field-free for a run trained with fields; "
        f"saw num_fields={[c.num_fields for c, _ in built]}"
    )
    missing_map = [config for config, has_map in fielded if not has_map]
    assert not missing_map, "fields were configured but no map distribution was generated"


def _calibration(**overrides) -> EloCalibrateConfig:
    return EloCalibrateConfig(num_envs=2, target_stderr=1_000_000.0, max_batches=1, **overrides)


def test_elo_calibrate_plays_a_fields_run_with_its_fields(fields_run, built_envs, tmp_path):
    from boost_and_broadside.modes.elo_calibrate import run_elo_calibrate_mode

    run_elo_calibrate_mode(
        run_spec=_RUN,
        ship_config=ShipConfig(),
        device="cpu",
        config=_calibration(),
        checkpoint_dir=str(tmp_path),
    )
    _assert_played_with_fields(built_envs)


def test_elo_scale_plays_a_fields_run_with_its_fields(fields_run, built_envs, tmp_path):
    from boost_and_broadside.modes.elo_scale import run_elo_scale_mode

    run_elo_scale_mode(
        run_spec=_RUN,
        team_sizes=[1],
        ship_config=ShipConfig(),
        device="cpu",
        checkpoint_dir=str(tmp_path),
        config=_calibration(),
    )
    _assert_played_with_fields(built_envs)


def test_semi_random_plays_a_fields_run_with_its_fields(fields_run, built_envs, tmp_path):
    from boost_and_broadside.modes.semi_random_tournament import run_semi_random_tournament

    run_semi_random_tournament(
        run_spec=_RUN,
        train_config=None,
        team_sizes=[1],
        probabilities=(0.0, 1.0),
        games_per_pair=2,
        max_parallel_envs=2,
        ship_config=ShipConfig(),
        device="cpu",
        checkpoint_dir=str(tmp_path),
    )
    _assert_played_with_fields(built_envs)


def test_crossover_plays_a_fields_run_with_its_fields(fields_run, built_envs, tmp_path):
    from boost_and_broadside.modes.crossover import run_crossover_mode

    run_crossover_mode(
        run_spec=_RUN,
        trained_counts=[1],
        ship_config=ShipConfig(),
        model_config=MODEL_CONFIG,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        num_envs=1,
        max_total_ships=2,
    )
    _assert_played_with_fields(built_envs)


def test_ar_report_diagnoses_a_fields_policy_in_a_field_arena(fields_run, built_envs, tmp_path):
    """The silent one: this used to pass field_map=None unconditionally."""

    from boost_and_broadside.modes.ar_report import run_ar_report_mode

    # The policy plays both sides, as the canonical report does: the AR phase
    # replays an *imagined* rollout, and a scripted controller has no real state
    # to read there.
    run_ar_report_mode(
        team0_spec=str(fields_run.checkpoint),
        team1_spec=str(fields_run.checkpoint),
        num_steps=2,
        ship_config=ShipConfig(),
        env_config=EnvConfig(num_ships=2, max_bullets=2, max_episode_steps=16),
        rewards=REWARDS,
        model_config=MODEL_CONFIG,
        device="cpu",
        checkpoint_dir=str(tmp_path),
    )
    _assert_played_with_fields(built_envs)


def test_noise_calibration_measures_a_fields_policy_in_a_field_arena(
    fields_run, built_envs, tmp_path
):
    from boost_and_broadside.modes.noise_calibration import run_noise_calibration_mode

    run_noise_calibration_mode(
        team0_spec=str(fields_run.checkpoint),
        team1_spec="scripted",
        num_envs=2,
        num_steps=2,
        num_ar_envs=2,
        num_ar_windows=1,
        ship_config=ShipConfig(),
        env_config=EnvConfig(num_ships=2, max_bullets=2, max_episode_steps=2),
        model_config=MODEL_CONFIG,
        device="cpu",
        checkpoint_dir=str(tmp_path),
    )
    _assert_played_with_fields(built_envs)


def test_noise_calibration_sizes_its_hidden_state_for_the_field_tokens(
    fields_run, tmp_path, monkeypatch
):
    """``num_tokens`` was derived before the environment was resolved, so a fields
    policy was given recurrent state for its ships only, and the field tokens it
    also predicts were dropped from its own error report."""

    import boost_and_broadside.modes.noise_calibration as mode

    widths: list[int] = []
    original = mode.init_hidden

    def recording_init_hidden(agent, batch, num_tokens, device):
        widths.append(num_tokens)
        return original(agent, batch, num_tokens, device)

    monkeypatch.setattr(mode, "init_hidden", recording_init_hidden)
    mode.run_noise_calibration_mode(
        team0_spec=str(fields_run.checkpoint),
        team1_spec="scripted",
        num_envs=2,
        num_steps=2,
        num_ar_envs=2,
        num_ar_windows=1,
        ship_config=ShipConfig(),
        env_config=EnvConfig(num_ships=2, max_bullets=2, max_episode_steps=2),
        model_config=MODEL_CONFIG,
        device="cpu",
        checkpoint_dir=str(tmp_path),
    )

    # Two ships plus four fields. Before the fix this read 2.
    assert widths and set(widths) == {6}
