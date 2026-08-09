"""The single policy construction and loading path.

A policy and the feature pipelines feeding it are two halves of one contract, and
before this module they were assembled separately at five call sites. These tests
pin the properties that make them impossible to separate again.
"""

import re
from pathlib import Path

import pytest
import torch

from boost_and_broadside.config import ModelConfig, ShipConfig
from boost_and_broadside.train.rl.policy_io import (
    FEATURE_SHIP_CONFIG_FIELDS,
    CheckpointProvenanceWarning,
    build_policy,
    feature_signature,
    load_policy_bundle,
)

_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "boost_and_broadside"

# The two modules that turn ShipConfig into what the encoders see.
_FEATURE_SOURCES = (
    _SOURCE_ROOT / "train" / "rl" / "features.py",
    _SOURCE_ROOT / "env" / "observation.py",
)


class TestFeatureSignature:
    def test_covers_every_ship_config_field_the_encoders_read(self):
        """Pins the drift check to its sources.

        The signature decides whether a checkpoint's weights can be trusted
        against the current physics. A new ``ship_config.<field>`` reference in
        either feature module silently widens what the weights depend on, so this
        fails until the field is listed. Over-listing is harmless; under-listing
        is a mismatch nobody is told about.
        """
        referenced = set()
        for source in _FEATURE_SOURCES:
            referenced |= set(re.findall(r"ship_config\.([a-z_]+)", source.read_text()))

        missing = sorted(referenced - set(FEATURE_SHIP_CONFIG_FIELDS))
        assert not missing, f"feature-relevant ShipConfig fields not in the signature: {missing}"

    def test_distinguishes_configs_that_change_what_weights_mean(self):
        import dataclasses

        base = ShipConfig()
        assert feature_signature(base) == feature_signature(dataclasses.replace(base))
        wider = dataclasses.replace(base, world_size=(2048.0, 2048.0))
        assert feature_signature(base) != feature_signature(wider)


class TestBuildPolicy:
    @staticmethod
    def _config(**overrides) -> ModelConfig:
        return ModelConfig(d_model=32, n_heads=4, n_yemong_blocks=1, **overrides)

    def test_pipelines_follow_the_config_without_being_passed(self):
        reads = build_policy(
            self._config(n_bullet_cross_per_block=1),
            ShipConfig(),
            num_value_components=3,
            num_ships=4,
            team_pma_k=(),
        )
        silent = build_policy(
            self._config(),
            ShipConfig(),
            num_value_components=3,
            num_ships=4,
            team_pma_k=(),
        )
        assert reads.bullet_encoder is not None
        assert silent.bullet_encoder is None

    def test_encoder_width_follows_ship_config(self):
        """The feature list is fixed, so a physics change moves constants, not dims —
        which is exactly why a drifted ShipConfig loads cleanly and plays wrong."""
        import dataclasses

        narrow = build_policy(
            self._config(),
            ShipConfig(),
            num_value_components=3,
            num_ships=4,
            team_pma_k=(),
        )
        wide = build_policy(
            self._config(),
            dataclasses.replace(ShipConfig(), world_size=(2048.0, 2048.0)),
            num_value_components=3,
            num_ships=4,
            team_pma_k=(),
        )
        assert set(narrow.state_dict()) == set(wide.state_dict())


class TestCheckpointProvenance:
    """A checkpoint is rebuilt as what it was, not as what the reader is running."""

    def test_the_checkpoints_config_wins_over_the_callers(self, tmp_path):
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(
            checkpoint_dir=str(tmp_path),
            model_config=ModelConfig(d_model=32, n_heads=4, n_yemong_blocks=1),
        )
        path = trainer._save_ladder_snapshot()

        # The caller is running a wider, deeper model — as a later run would be.
        bundle = load_policy_bundle(
            str(path),
            device="cpu",
            num_ships=trainer.wrapper.num_ships,
            ship_config=trainer.ship_config,
            model_config=ModelConfig(d_model=64, n_heads=4, n_yemong_blocks=3),
        )

        assert bundle.model_config == trainer.model_config
        assert bundle.policy.state_dict()["encoder.feature_extractor.0.weight"].shape[0] == 64

    def test_physics_drift_is_refused_and_named(self, tmp_path):
        import dataclasses

        from boost_and_broadside.train.rl.policy_io import ConfigDriftError
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        path = trainer._save_ladder_snapshot()
        moved = dataclasses.replace(trainer.ship_config, max_health=1234.0)

        with pytest.raises(ConfigDriftError, match="max_health"):
            load_policy_bundle(
                str(path), device="cpu", num_ships=4, ship_config=moved, model_config=None
            )

    def test_drift_can_be_allowed_explicitly(self, tmp_path):
        import dataclasses

        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        path = trainer._save_ladder_snapshot()
        moved = dataclasses.replace(trainer.ship_config, max_health=1234.0)

        with pytest.warns(CheckpointProvenanceWarning, match="max_health"):
            bundle = load_policy_bundle(
                str(path),
                device="cpu",
                num_ships=4,
                ship_config=moved,
                allow_config_drift=True,
            )

        # It reads the world through the constants it trained on, not the new ones.
        assert bundle.ship_config.max_health == trainer.ship_config.max_health


class TestLegacyCheckpoints:
    def test_missing_provenance_falls_back_and_says_so(self, tmp_path):
        """Payloads written before provenance stay loadable, but not silently."""
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        path = trainer._save_ladder_snapshot()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload.pop("model_config", None)
        payload.pop("ship_config", None)
        torch.save(payload, path)

        with pytest.warns(CheckpointProvenanceWarning, match="model_config"):
            bundle = load_policy_bundle(
                str(path),
                device="cpu",
                num_ships=trainer.wrapper.num_ships,
                ship_config=trainer.ship_config,
                model_config=trainer.model_config,
                team_pma_k=trainer._win_k,
            )

        assert bundle.model_config == trainer.model_config
        assert bundle.ship_config == trainer.ship_config

    def test_a_checkpoint_with_no_provenance_and_no_fallback_is_refused(self, tmp_path):
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        path = trainer._save_ladder_snapshot()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload.pop("model_config", None)
        torch.save(payload, path)

        with pytest.raises(ValueError, match="records no model_config"):
            load_policy_bundle(
                str(path),
                device="cpu",
                num_ships=trainer.wrapper.num_ships,
                ship_config=trainer.ship_config,
            )
