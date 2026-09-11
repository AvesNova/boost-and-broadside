"""The single policy construction and loading path.

A policy and the feature pipelines feeding it are two halves of one contract, and
before this module they were assembled separately at five call sites. These tests
pin the properties that make them impossible to separate again.
"""

import gc
import re
import weakref
from pathlib import Path

import pytest
import torch

from boost_and_broadside.config import ModelConfig, ShipConfig
from boost_and_broadside.train.rl.checkpoint_schema import (
    observation_contract,
    position_fourier_frequencies,
)
from boost_and_broadside.train.rl.policy_io import (
    FEATURE_SHIP_CONFIG_FIELDS,
    CheckpointProvenanceWarning,
    build_policy,
    compile_policy,
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
        """The large world widens position features to preserve 128 px detail."""
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
            dataclasses.replace(ShipConfig(), world_size=(16384.0, 16384.0)),
            num_value_components=3,
            num_ships=4,
            team_pma_k=(),
        )
        assert set(narrow.state_dict()) == set(wide.state_dict())
        narrow_input = narrow.state_dict()["encoder.feature_extractor.0.weight"].shape[1]
        wide_input = wide.state_dict()["encoder.feature_extractor.0.weight"].shape[1]
        assert wide_input - narrow_input == 16

    def test_position_frequency_contract_is_explicit_and_scale_preserving(self):
        assert position_fourier_frequencies(1024.0) == 4
        assert position_fourier_frequencies(16384.0) == 8
        assert observation_contract(ShipConfig(world_size=(16384.0, 16384.0)))[
            "position_frequencies"
        ] == (8, 8)


def _cuda_obs(envs: int, ships: int):
    """A minimal well-formed observation on CUDA, for shape-specialization tests."""
    from boost_and_broadside.env.observation import ObsKey, YemongObservation

    def f(*shape):
        return torch.randn(*shape, device="cuda")

    return YemongObservation(
        data={
            ObsKey.POS: f(envs, ships, 2) * 100,
            ObsKey.VEL: f(envs, ships, 2),
            ObsKey.ATT: torch.nn.functional.normalize(f(envs, ships, 2), dim=-1),
            ObsKey.ANG_VEL: f(envs, ships, 1),
            ObsKey.HEALTH: f(envs, ships, 1).abs(),
            ObsKey.POWER: f(envs, ships, 1).abs(),
            ObsKey.COOLDOWN: f(envs, ships, 1).abs(),
            ObsKey.TEAM_ID: torch.randint(0, 2, (envs, ships), device="cuda"),
            ObsKey.ALIVE: torch.ones(envs, ships, dtype=torch.bool, device="cuda"),
            ObsKey.VISIBLE: torch.ones(envs, ships, dtype=torch.bool, device="cuda"),
            ObsKey.BELIEF_VALID: torch.ones(envs, ships, dtype=torch.bool, device="cuda"),
            ObsKey.TIME_SINCE_OBSERVATION: f(envs, ships, 1).abs(),
            ObsKey.OBJECT_TYPE: torch.zeros(envs, ships, dtype=torch.long, device="cuda"),
            ObsKey.RADIUS: f(envs, ships, 1).abs(),
            ObsKey.PREVIOUS_ACTION: torch.zeros(envs, ships, 3, dtype=torch.long, device="cuda"),
            ObsKey.LOCAL_LOG_INDEX: f(envs, ships, 1),
            ObsKey.LOCAL_INDEX_GRADIENT: f(envs, ships, 2),
            ObsKey.FIELD_TRANSITION_WIDTH: f(envs, ships, 1).abs(),
            ObsKey.FIELD_TARGET_LOG_INDEX: f(envs, ships, 1),
            ObsKey.FIELD_DAMAGE: f(envs, ships, 1).abs(),
            ObsKey.ZONE_ROLE: torch.zeros(envs, ships, dtype=torch.long, device="cuda"),
            ObsKey.CAPTURE_PROGRESS: f(envs, ships, 1),
            ObsKey.CAPTURE_DIRECTION: f(envs, ships, 1),
            ObsKey.FRONT_POSITION: f(envs, ships, 1),
            ObsKey.FRONT_WIN_THRESHOLD: f(envs, ships, 1),
            ObsKey.TIME_REMAINING: f(envs, ships, 1),
            ObsKey.GAME_MODE: f(envs, ships, 1),
        }
    )


class TestCompilePolicy:
    """The entry points callers actually use must be the compiled ones.

    ``torch.compile(module)`` wraps ``forward`` and nothing else, and
    ``OptimizedModule.__getattr__`` hands every other attribute back from the
    original module. Nothing here calls a policy's ``forward``, so wrapping the
    module alone left both hot paths running eager and dynamo compiling zero
    frames -- silently, for as long as the flag existed.
    """

    @staticmethod
    def _policy():
        return build_policy(
            ModelConfig(d_model=32, n_heads=4, n_yemong_blocks=1),
            ShipConfig(),
            num_value_components=3,
            num_ships=4,
            team_pma_k=(),
        )

    def test_no_mode_leaves_the_policy_untouched(self):
        policy = self._policy()
        assert compile_policy(policy, None) is policy

    def test_the_rollout_entry_point_does_not_resolve_to_the_eager_bound_method(self):
        policy = self._policy()
        compiled = compile_policy(policy, "default")
        entry = compiled.get_action_and_value
        assert getattr(entry, "__self__", None) is not policy, (
            "get_action_and_value resolves to the policy's own bound method, so it "
            "bypasses torch.compile and runs eager"
        )

    def test_the_update_entry_point_is_compiled_too(self):
        """Which of the two the update *uses* is PPOTrainer's decision."""
        policy = self._policy()
        compiled = compile_policy(policy, "default")
        assert getattr(compiled.evaluate_actions, "__self__", None) is not policy

    def test_the_policy_itself_comes_back_unchanged_otherwise(self):
        policy = self._policy()
        compiled = compile_policy(policy, "default")
        assert compiled is policy
        assert compiled.coordinator is policy.coordinator
        assert compiled.num_recurrent_tokens == policy.num_recurrent_tokens
        assert compiled.n_hidden_layers == policy.n_hidden_layers
        assert set(compiled.state_dict()) == set(policy.state_dict())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_a_second_batch_shape_does_not_break_the_compile(self):
        """Two shapes must give two static graphs, not one dynamic one.

        Letting dynamo generalize makes T symbolic, and ``_parallel_scan`` pads T
        to the next power of two with ``1 << (T_real - 1).bit_length()`` --
        inductor cannot express that and fails the whole compile with
        "ValueError: Exponent must be non-negative". The rollout, the evaluator
        and the update all call at different widths, and any
        ``--microbatch-tokens`` whose split is uneven adds another; the shipped
        divisor survived only because it happened to divide evenly.
        """
        policy = compile_policy(self._policy().to("cuda"), "default")
        hidden_width = policy.n_hidden_layers

        for envs in (7, 5):  # deliberately not equal, and not the same as before
            obs = _cuda_obs(envs, ships=4)
            hidden = torch.zeros(hidden_width, envs * 4, 4 * 32, device="cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                action, _, _, _, _ = policy.get_action_and_value(obs, hidden)
            assert action.shape == (envs, 4, 3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_a_cuda_graph_mode_returns_outputs_that_survive_the_next_call(self):
        """The property every caller here depends on, and the one graphs break.

        ``reduce-overhead`` writes its outputs into static buffers that the next
        replay overwrites. This pipeline holds them: the recurrent hidden state
        is carried into the following rollout step, and the Elo evaluator keeps
        five policies' sampled actions alive at once while it builds the team
        action. Without a copy the second call either raises "accessing tensor
        output of CUDAGraphs that has been overwritten by a subsequent run" or,
        worse, hands back the next step's numbers.
        """
        policy = compile_policy(self._policy().to("cuda"), "reduce-overhead")
        hidden = torch.zeros(policy.n_hidden_layers, 6 * 4, 4 * 32, device="cuda")

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            first = policy.get_action_and_value(_cuda_obs(6, ships=4), hidden)
        kept = [tensor.clone() for tensor in first]

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            policy.get_action_and_value(_cuda_obs(6, ships=4), first[4])

        for index, (held, expected) in enumerate(zip(first, kept)):
            torch.testing.assert_close(
                held, expected, msg=f"output {index} changed under a later call"
            )

    def test_an_evicted_compiled_policy_is_reclaimed_by_the_roster(self):
        """Dropping a compiled policy needs a collection pass, and gets one.

        Dynamo keeps the traced instance alive from its own caches, so reference
        counting alone does not return an evicted league entry's weights to the
        card. ``EloRoster._unload`` collects for exactly this reason; without
        that, ``max_size`` would stop bounding device memory.
        """
        policy = self._policy()
        compiled = compile_policy(policy, "default")
        alive = weakref.ref(policy)
        del policy, compiled
        assert alive() is not None, (
            "reference counting now reclaims a compiled policy; the collect in "
            "EloRoster._unload is no longer needed and should be removed"
        )
        gc.collect()
        assert alive() is None


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

    def test_incompatible_policy_tensors_are_reported_as_checkpoint_input(self, tmp_path):
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        path = trainer._save_ladder_snapshot()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        key = next(iter(payload["policy_state_dict"]))
        payload["policy_state_dict"][key] = torch.zeros(1)
        torch.save(payload, path)

        with pytest.raises(ValueError, match="incompatible policy weights"):
            load_policy_bundle(
                str(path),
                device="cpu",
                num_ships=trainer.wrapper.num_ships,
                ship_config=trainer.ship_config,
            )

    def test_non_mapping_policy_state_is_reported_as_checkpoint_input(self, tmp_path):
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        path = trainer._save_ladder_snapshot()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["policy_state_dict"] = None
        torch.save(payload, path)

        with pytest.raises(ValueError, match="expected a mapping, got NoneType"):
            load_policy_bundle(
                str(path),
                device="cpu",
                num_ships=trainer.wrapper.num_ships,
                ship_config=trainer.ship_config,
            )


class TestLegacyCheckpoints:
    def test_v4_checkpoint_without_ship_feature_contract_is_refused(self, tmp_path):
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        path = trainer._save_ladder_snapshot()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload.pop("ship_config")
        torch.save(payload, path)

        with pytest.raises(ValueError, match="ship_config.world_size"):
            load_policy_bundle(
                str(path),
                device="cpu",
                num_ships=trainer.wrapper.num_ships,
                ship_config=trainer.ship_config,
            )

    def test_missing_model_provenance_falls_back_and_says_so(self, tmp_path):
        """Model provenance can fall back; v4 ship semantics cannot."""
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        path = trainer._save_ladder_snapshot()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload.pop("model_config", None)
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
