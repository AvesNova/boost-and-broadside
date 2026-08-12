"""Checkpoint serialization integrity.

These tests exist because a corrupt checkpoint is silent: the file loads, every
tensor has the right shape and dtype, and the values look plausible. The damage
only surfaces later as a diverged resume or a NaN with no traceable origin.
"""

import dataclasses
import threading
from pathlib import Path

import pytest
import torch

from boost_and_broadside.train.rl.checkpoint import (
    OPTIONAL_CHECKPOINT_FIELDS,
    POLICY_CHECKPOINT_FIELDS,
    RESUMABLE_CHECKPOINT_FIELDS,
    _check_resolved_config_provenance,
    build_policy_checkpoint_payload,
    build_training_checkpoint_payload,
    clone_to_cpu,
    require_resumable_checkpoint,
)
from boost_and_broadside.train.rl.checkpoint_schema import (
    OBSERVATION_SCHEMA,
    require_observation_schema,
)
from boost_and_broadside.train.rl.policy_io import infer_num_value_components

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _queue_gpu_work(rounds: int = 20) -> None:
    """Leave enough work in the stream that an unsynchronized D2H copy loses the race."""
    for _ in range(rounds):
        torch.randn(1024, 1024, device="cuda") @ torch.randn(1024, 1024, device="cuda")


class TestCloneToCpu:
    @requires_cuda
    def test_copies_are_correct_behind_queued_gpu_work(self):
        """Regression: a non-blocking D2H copy read before the DMA lands returns
        recycled buffer contents — usually the *previous* tensor's bytes, so the
        corruption reads as plausible data rather than as garbage."""
        source = {f"t{i}": torch.full((256, 256), float(i), device="cuda") for i in range(16)}
        torch.cuda.synchronize()
        _queue_gpu_work()

        result = clone_to_cpu(source)

        for key, value in result.items():
            expected = float(key[1:])
            assert (value == expected).all(), (
                f"{key} came back holding {torch.unique(value).tolist()[:4]}, expected {expected}"
            )

    @requires_cuda
    def test_result_is_detached_from_device_memory(self):
        """Mutating the source afterwards must not change the snapshot."""
        source = torch.ones(64, device="cuda")
        snapshot = clone_to_cpu(source)
        source.fill_(9.0)
        torch.cuda.synchronize()
        assert (snapshot == 1.0).all()

    def test_walks_nested_containers(self):
        payload = {
            "list": [torch.ones(3)],
            "tuple": (torch.zeros(2),),
            "nested": {"deep": torch.full((2,), 5.0)},
            "scalar": 7,
            "text": "unchanged",
        }
        out = clone_to_cpu(payload)
        assert out["list"][0].tolist() == [1.0, 1.0, 1.0]
        assert out["tuple"][0].tolist() == [0.0, 0.0]
        assert out["nested"]["deep"].tolist() == [5.0, 5.0]
        assert out["scalar"] == 7
        assert out["text"] == "unchanged"

    def test_cpu_tensors_are_copied_not_aliased(self):
        """A CPU tensor still needs a real copy, or the checkpoint would track
        later mutations of live training state."""
        source = torch.arange(4).float()
        snapshot = clone_to_cpu(source)
        source.fill_(9.0)
        assert snapshot.tolist() == [0.0, 1.0, 2.0, 3.0]


class TestSavedCheckpointIntegrity:
    """End-to-end sanity checks on what a save actually writes.

    These assert invariants rather than reproducing the copy race: at test model
    scale the payload tensors are small enough that pageable device-to-host
    copies complete synchronously, so the race cannot be provoked here.
    ``TestCloneToCpu`` covers that directly. These stay valuable as a tripwire —
    a second moment can only go negative if a save captured the wrong bytes.
    """

    def test_optimizer_moments_are_self_consistent(self, tmp_path):
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        trainer.train()
        trainer._save_checkpoint(update=1)
        save_thread = getattr(trainer, "_active_save_thread", None)
        if save_thread is not None:
            save_thread.join(timeout=60)

        saved = list(tmp_path.rglob("step_*.pt"))
        assert saved, "checkpoint file was not written"
        state = torch.load(saved[0], map_location="cpu", weights_only=False)
        moments = state["optimizer_state_dict"]["state"]
        assert moments, "optimizer state was empty"

        for index, entry in moments.items():
            exp_avg_sq = entry["exp_avg_sq"]
            assert (exp_avg_sq >= 0).all(), f"param {index} has a negative second moment"
            assert not torch.equal(entry["exp_avg"], exp_avg_sq), (
                f"param {index} has identical first and second moments"
            )

    def test_avg_param_cumsum_is_fp32(self, tmp_path):
        """The running snapshot sum must stay fp32 — a narrower accumulator lets
        each += round away and the averaged policy drifts without bound."""
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        assert all(c.dtype == torch.float32 for c in trainer._avg_param_cumsum)

        trainer.train()
        trainer._save_checkpoint(update=1)
        save_thread = getattr(trainer, "_active_save_thread", None)
        if save_thread is not None:
            save_thread.join(timeout=60)
        saved = list(tmp_path.rglob("step_*.pt"))
        trainer.load_checkpoint(str(saved[0]))
        assert all(c.dtype == torch.float32 for c in trainer._avg_param_cumsum)

    def test_resume_re_registers_the_scripted_league_entry(self, tmp_path):
        """load_json replaces the entry list wholesale.

        A roster written before the scripted agent was a league entry must not
        resume into a run that can never draw it.
        """
        import json

        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(league_fraction=0.5, checkpoint_dir=str(tmp_path))
        trainer._global_step = 1
        trainer._save_checkpoint(update=1)
        trainer._save_roster_json()
        save_thread = getattr(trainer, "_active_save_thread", None)
        if save_thread is not None:
            save_thread.join(timeout=60)
        saved = list(tmp_path.rglob("step_*.pt"))[0]

        # Rewrite the roster as an older run would have left it: no scripted entry.
        roster_path = saved.parent / "roster.json"
        data = json.loads(roster_path.read_text())
        data["entries"] = [e for e in data["entries"] if e["kind"] != "scripted"]
        roster_path.write_text(json.dumps(data))

        resumed = _make_trainer(league_fraction=0.5, checkpoint_dir=str(tmp_path))
        resumed.load_checkpoint(str(saved))

        assert sum(e.kind == "scripted" for e in resumed.roster.entries) == 1
        slots = resumed._prepare_league_slots(resumed.wrapper.num_ships)
        assert slots and all(slot.entry.kind == "scripted" for slot in slots)

    def test_shutdown_waits_for_inflight_checkpoint(self, tmp_path):
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        trainer._global_step = 1
        trainer._save_checkpoint(update=1)
        trainer.shutdown()

        assert not trainer._active_save_thread.is_alive()
        assert list(tmp_path.rglob("step_*.pt"))
        assert not list(tmp_path.rglob("*.tmp"))


class TestObservationSchema:
    def test_pure_policy_payload_builder_matches_current_provenance_schema(self):
        from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig

        payload = build_policy_checkpoint_payload(
            policy_state_dict={"weight": torch.ones(2)},
            num_value_components=3,
            team_pma_k=(0, 2),
            global_step=17,
            live_elo=42.0,
            model_config=ModelConfig(d_model=32, n_heads=4, n_yemong_blocks=1),
            env_config=EnvConfig(num_ships=2, max_bullets=4, max_episode_steps=8),
            ship_config=ShipConfig(),
            paradigm="ego_pass",
            resolved_config={"resolved_config_fingerprint": "abc"},
            launch={"device": "cpu", "seed": 7},
        )

        assert payload["observation_schema"] == OBSERVATION_SCHEMA
        assert payload["num_value_components"] == 3
        assert payload["team_pma_k"] == (0, 2)
        assert payload["global_step"] == 17
        assert payload["resolved_config"]["resolved_config_fingerprint"] == "abc"
        assert payload["launch"] == {"device": "cpu", "seed": 7}

    def test_full_payload_builder_matches_resumable_checkpoint_schema(self, tmp_path):
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        built = build_training_checkpoint_payload(
            policy_payload=trainer._provenance(),
            optimizer_state_dict=trainer.optim.state_dict(),
            scaler_state_dict=trainer.scaler.state_dict(),
            adv_scaler_state_dict=trainer.adv_scaler.state_dict(),
            avg_policy_state_dict=trainer._avg_policy_module.state_dict(),
            avg_param_cumsum=list(trainer._avg_param_cumsum),
            avg_update_count=0,
            update=0,
            ship_steps=0,
            grad_tokens=0,
            elapsed_train_time=0.0,
            avg_live_elo=0.0,
            floating_games=0,
            eval_window_rand=[],
            eval_window_sc=[],
            eval_window_ladder=[],
            eval_window_floating=[],
            eval_window_live_vs_avg=[],
            elo_milestone=0.0,
            train_config=trainer.cfg,
        )
        production = trainer.checkpoint_payload(update=0)

        assert set(built) == set(production)
        assert built["train_config"] == production["train_config"]
        assert built["optimizer_state_dict"] == production["optimizer_state_dict"]

    def test_all_payload_families_are_versioned(self, tmp_path):
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        assert trainer.checkpoint_payload(0)["observation_schema"] == OBSERVATION_SCHEMA
        assert (
            trainer._checkpoint_payload_lightweight(0)["observation_schema"] == OBSERVATION_SCHEMA
        )
        ladder = torch.load(trainer._save_ladder_snapshot(), map_location="cpu", weights_only=False)
        assert ladder["observation_schema"] == OBSERVATION_SCHEMA

    def test_every_payload_family_records_what_it_was_trained_under(self, tmp_path):
        """Including the ladder snapshots, which are the files most often reloaded
        and were the ones carrying the least about themselves."""
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        trainer.resolved_config_document = {"profile": "rl", "sources": {"x": "profile"}}
        trainer.launch_provenance = {"device": "cpu", "seed": 7}
        ladder = torch.load(trainer._save_ladder_snapshot(), map_location="cpu", weights_only=False)
        families = [
            trainer.checkpoint_payload(0),
            trainer._checkpoint_payload_lightweight(0),
            trainer._avg_checkpoint_payload_lightweight(0),
            ladder,
        ]

        for payload in families:
            for key in ("model_config", "env_config", "ship_config", "team_pma_k"):
                assert key in payload, f"payload family is missing {key}"
            assert payload["ship_config"] == dataclasses.asdict(trainer.ship_config)
            assert payload["resolved_config"] == trainer.resolved_config_document
            assert payload["launch"] == trainer.launch_provenance

    def test_legacy_obstacle_checkpoint_fails_clearly(self):
        with pytest.raises(ValueError, match="Observation feature semantics are incompatible"):
            require_observation_schema({"policy_state_dict": {}}, "legacy.pt")


class TestResolvedConfigProvenance:
    def test_resume_rejects_a_different_complete_resolved_config(self):
        checkpoint = {
            "resolved_config": {"resolved_config_fingerprint": "recorded"},
        }
        current = {"resolved_config_fingerprint": "current"}
        with pytest.raises(ValueError, match="--allow-config-drift"):
            _check_resolved_config_provenance(
                checkpoint,
                current,
                allow_config_drift=False,
            )

    def test_explicit_drift_override_is_loud(self):
        checkpoint = {
            "resolved_config": {"resolved_config_fingerprint": "recorded"},
        }
        current = {"resolved_config_fingerprint": "current"}
        with pytest.warns(UserWarning, match="config drift is allowed"):
            _check_resolved_config_provenance(
                checkpoint,
                current,
                allow_config_drift=True,
            )

    def test_real_resume_loader_enforces_drift_while_pretraining_allows_it(self, tmp_path):
        from tests.train.test_ppo import _make_trainer

        source = _make_trainer(checkpoint_dir=str(tmp_path / "source"))
        payload = source.checkpoint_payload(0)
        payload["resolved_config"] = {"resolved_config_fingerprint": "recorded-bc"}
        checkpoint = tmp_path / "cross-profile.pt"
        torch.save(payload, checkpoint)

        resumed = _make_trainer(checkpoint_dir=str(tmp_path / "resume"))
        resumed.resolved_config_document = {"resolved_config_fingerprint": "current-rl"}
        resumed.launch_provenance = {"allow_config_drift": False}
        with pytest.raises(ValueError, match="--allow-config-drift"):
            resumed.load_checkpoint(str(checkpoint))

        allowed = _make_trainer(checkpoint_dir=str(tmp_path / "allowed"))
        allowed.resolved_config_document = {"resolved_config_fingerprint": "current-rl"}
        allowed.launch_provenance = {"allow_config_drift": True}
        with pytest.warns(UserWarning, match="config drift is allowed"):
            assert allowed.load_checkpoint(str(checkpoint)) == 0

        pretrained = _make_trainer(checkpoint_dir=str(tmp_path / "pretrain"))
        pretrained.resolved_config_document = {"resolved_config_fingerprint": "current-rl"}
        pretrained.launch_provenance = {"allow_config_drift": False}
        pretrained.load_pretrained_weights(str(checkpoint))

        source.shutdown()
        resumed.shutdown()
        allowed.shutdown()
        pretrained.shutdown()


class TestResumableCheckpointContract:
    """A resume restores the complete training state, or it is refused.

    The failure this closes is silent by construction: a payload written under
    the pre-rename live-Elo field names loaded, resumed at the right update, and
    continued with the weights and optimizer intact — while the live rating, its
    running average, and the milestone grid restarted at zero. With
    ``elo_milestone_gap=200`` the run then re-freezes ladder snapshots at heights
    it had already passed. Nothing in the output said so.
    """

    @staticmethod
    def _production_payload(tmp_path):
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        payload = trainer.checkpoint_payload(update=6)
        trainer.shutdown()
        return payload

    def test_required_fields_are_exactly_what_the_builder_always_writes(self, tmp_path):
        """The tripwire: adding a payload field without deciding its resume
        behavior fails here rather than becoming another silent default."""
        payload = self._production_payload(tmp_path)

        assert set(payload) - set(OPTIONAL_CHECKPOINT_FIELDS) == set(RESUMABLE_CHECKPOINT_FIELDS)
        assert len(RESUMABLE_CHECKPOINT_FIELDS) == len(set(RESUMABLE_CHECKPOINT_FIELDS))

    def test_the_policy_family_is_pinned_to_what_its_own_builder_writes(self, tmp_path):
        """The block every family starts with, and the whole of a ladder file.
        S15 migrates against these key sets, so they are frozen here."""
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        ladder = torch.load(trainer._save_ladder_snapshot(), map_location="cpu", weights_only=False)
        trainer.shutdown()

        assert set(ladder) - set(OPTIONAL_CHECKPOINT_FIELDS) == set(POLICY_CHECKPOINT_FIELDS)
        assert RESUMABLE_CHECKPOINT_FIELDS[: len(POLICY_CHECKPOINT_FIELDS)] == (
            POLICY_CHECKPOINT_FIELDS
        )

    @pytest.mark.parametrize("field", RESUMABLE_CHECKPOINT_FIELDS)
    def test_every_required_field_is_named_when_it_is_missing(self, field, tmp_path):
        payload = dict(self._production_payload(tmp_path))
        payload.pop(field)

        with pytest.raises(ValueError, match=f"not a resumable training checkpoint.*{field}"):
            require_resumable_checkpoint(payload, "step_000000000042.pt")

    def test_optional_provenance_fields_do_not_block_a_resume(self, tmp_path):
        """A trainer built without a resolved-config document or launch record —
        every hermetic fixture — still writes a payload that resumes."""
        payload = self._production_payload(tmp_path)

        assert not set(OPTIONAL_CHECKPOINT_FIELDS) & set(payload)
        require_resumable_checkpoint(payload, "step_000000000042.pt")

    @pytest.mark.parametrize("with_resolved_config", [False, True])
    def test_real_resume_refuses_the_pre_rename_live_elo_payload(
        self, with_resolved_config, tmp_path
    ):
        """Both shapes, because the drift check returns early on a payload that
        records no resolved config and so cannot be what stands between a legacy
        file and a reset rating."""
        from tests.train.test_ppo import _make_trainer

        source = _make_trainer(checkpoint_dir=str(tmp_path / "source"))
        source._live_elo = 1547.3
        source._avg_live_elo = 1500.0
        source._elo_milestone = 1400.0
        payload = source.checkpoint_payload(update=6)
        payload["training_elo"] = payload.pop("live_elo")
        payload["avg_training_elo"] = payload.pop("avg_live_elo")
        if with_resolved_config:
            payload["resolved_config"] = {"resolved_config_fingerprint": "recorded"}
        legacy = tmp_path / "legacy.pt"
        torch.save(payload, legacy)

        resumed = _make_trainer(checkpoint_dir=str(tmp_path / "resume"))
        resumed.resolved_config_document = {"resolved_config_fingerprint": "recorded"}
        resumed.launch_provenance = {"allow_config_drift": False}
        before = (
            resumed._live_elo,
            resumed._avg_live_elo,
            resumed._elo_milestone,
            resumed._start_update,
        )

        with pytest.raises(ValueError, match="predates the current live-Elo naming"):
            resumed.load_checkpoint(str(legacy))

        # Refused before any state was restored, so the trainer is untouched.
        assert (
            resumed._live_elo,
            resumed._avg_live_elo,
            resumed._elo_milestone,
            resumed._start_update,
        ) == before
        source.shutdown()
        resumed.shutdown()

    def test_real_resume_refuses_a_policy_only_checkpoint(self, tmp_path):
        """The ladder and best_*.pt families are policy-only by design; resuming
        one would silently start a run with a fresh optimizer."""
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        ladder = trainer._save_ladder_snapshot()

        with pytest.raises(ValueError, match="--pretrain-from"):
            trainer.load_checkpoint(str(ladder))
        trainer.shutdown()

    def test_a_complete_payload_restores_every_rating_field(self, tmp_path):
        """The positive side of the contract, at the real loader."""
        from tests.train.test_ppo import _make_trainer

        source = _make_trainer(checkpoint_dir=str(tmp_path / "source"))
        source._live_elo = 1547.3
        source._avg_live_elo = 1500.0
        source._elo_milestone = 1400.0
        source._floating_games = 11
        source._ship_steps = 4242
        source._grad_tokens = 9999
        source._global_step = 512
        saved = tmp_path / "step.pt"
        torch.save(clone_to_cpu(source.checkpoint_payload(update=6)), saved)

        resumed = _make_trainer(checkpoint_dir=str(tmp_path / "resume"))

        assert resumed.load_checkpoint(str(saved)) == 6
        assert resumed._live_elo == 1547.3
        assert resumed._avg_live_elo == 1500.0
        assert resumed._elo_milestone == 1400.0
        assert resumed._floating_games == 11
        assert resumed._ship_steps == 4242
        assert resumed._grad_tokens == 9999
        assert resumed._global_step == 512
        assert resumed._start_update == 7
        source.shutdown()
        resumed.shutdown()


class TestNumValueComponents:
    """AUDIT-018: critic width K is saved explicitly, not reverse-engineered.

    Every loader used to read K off a hardcoded state-dict key
    (``value_head_local.3.weight``). Saving it as a field decouples the loaders
    from the value head's internal structure; the shape introspection survives
    only as a legacy fallback for checkpoints written before the field existed.
    """

    def test_payloads_record_active_component_count(self, tmp_path):
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        expected = trainer.wrapper.num_active_components

        assert trainer.checkpoint_payload(update=0)["num_value_components"] == expected
        assert trainer._checkpoint_payload_lightweight(update=0)["num_value_components"] == expected

        ladder_path = trainer._save_ladder_snapshot()
        ladder = torch.load(ladder_path, map_location="cpu", weights_only=False)
        assert ladder["num_value_components"] == expected

    def test_infer_reads_explicit_field_over_state_dict_shape(self):
        """The stored field wins even when the state-dict shape would disagree."""
        ckpt = {
            "num_value_components": 7,
            "policy_state_dict": {"value_head_local.3.weight": torch.zeros(3, 4)},
        }
        assert infer_num_value_components(ckpt) == 7

    def test_infer_falls_back_to_state_dict_shape_for_legacy_checkpoints(self):
        """Checkpoints written before the field recover K from the value head."""
        ckpt = {"policy_state_dict": {"value_head_local.3.weight": torch.zeros(5, 4)}}
        assert infer_num_value_components(ckpt) == 5


class TestBulletReadingCheckpoints:
    """A checkpoint's weights and the policy rebuilt to hold them must agree.

    Bullet cross-attention adds a ``bullet_encoder`` submodule. Every checkpoint
    written by a bullet-reading run carries those tensors, and a policy built
    without the matching feature pipeline has nowhere to put them — the load then
    fails mid-run, the first time a league opponent is sampled. ``build_policy``
    derives the pipeline from the config, so the two cannot disagree.
    """

    @staticmethod
    def _bullet_model_config():
        from boost_and_broadside.config import ModelConfig

        return ModelConfig(d_model=32, n_heads=4, n_yemong_blocks=1, n_bullet_cross_per_block=1)

    def test_league_loader_round_trips_a_bullet_reading_snapshot(self, tmp_path):
        from boost_and_broadside.train.rl.policy_io import load_policy_bundle
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(
            checkpoint_dir=str(tmp_path), model_config=self._bullet_model_config()
        )
        path = trainer._save_ladder_snapshot()

        bundle = load_policy_bundle(
            str(path),
            device=trainer.device,
            num_ships=trainer.wrapper.num_ships,
            ship_config=trainer.ship_config,
            model_config=trainer.model_config,
            team_pma_k=trainer._win_k,
        )

        assert bundle.policy.bullet_encoder is not None
        assert bundle.reads_bullets
        saved = torch.load(path, map_location="cpu", weights_only=False)["policy_state_dict"]
        assert set(saved) == set(bundle.policy.state_dict())

    def test_a_bullet_reading_config_always_gets_its_encoder(self):
        """The invariant that replaced the bug: the pipeline is derived, not passed.

        There is no argument a caller can omit to produce a bullet-reading policy
        without a bullet encoder, which is exactly how the league loader ended up
        building one that could not hold the weights it was about to load.
        """
        from boost_and_broadside.config import ShipConfig
        from boost_and_broadside.train.rl.policy_io import build_policy

        reads = build_policy(
            self._bullet_model_config(),
            ShipConfig(),
            num_value_components=3,
            num_ships=4,
            team_pma_k=(),
        )
        silent = build_policy(
            dataclasses.replace(self._bullet_model_config(), n_bullet_cross_per_block=0),
            ShipConfig(),
            num_value_components=3,
            num_ships=4,
            team_pma_k=(),
        )

        assert reads.bullet_encoder is not None
        assert silent.bullet_encoder is None

    def test_sampled_league_opponent_reads_bullets(self, tmp_path):
        """The crash path itself: sample a checkpoint opponent mid-rollout."""
        from tests.train.test_ppo import _make_trainer

        # No scripted agent, so the checkpoint is the only thing a slot can draw.
        trainer = _make_trainer(
            checkpoint_dir=str(tmp_path),
            league_fraction=0.5,
            with_scripted=False,
            model_config=self._bullet_model_config(),
        )
        path = trainer._save_ladder_snapshot()
        trainer.roster.add_checkpoint(
            path=str(path), global_step=1, update=1, initial_elo=trainer._live_elo
        )

        slots = trainer._prepare_league_slots(trainer.wrapper.num_ships)

        assert slots
        assert all(slot.policy.bullet_encoder is not None for slot in slots)
        assert all(slot.hidden is not None for slot in slots)

    def test_elo_evaluator_observes_bullets_when_the_policy_reads_them(self, tmp_path):
        """Rating a bullet-reading policy on a bullet-free observation would
        measure a blindfolded agent and report it as the run's Elo."""
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(
            checkpoint_dir=str(tmp_path), model_config=self._bullet_model_config()
        )
        runtime = trainer._initialize_rollout_runtime()

        assert runtime.elo_eval.include_bullets is True


class TestHeterogeneousLeague:
    """A roster spans a run's history, and a run's architecture can change.

    Nothing in the policy is sized by ship count and every entry is rebuilt from
    its own recorded config, so an opponent need not share the live policy's
    shape. Each carries its own recurrent width, which is what makes this work at
    all — the trainer never allocates hidden state on an opponent's behalf.
    """

    @staticmethod
    def _config(d_model: int, blocks: int = 1):
        from boost_and_broadside.config import ModelConfig

        return ModelConfig(d_model=d_model, n_heads=4, n_yemong_blocks=blocks)

    def test_a_narrower_opponent_plays_a_wider_trainee(self, tmp_path):
        from tests.train.test_ppo import _make_trainer

        older = _make_trainer(
            checkpoint_dir=str(tmp_path / "old"), model_config=self._config(d_model=32)
        )
        snapshot = older._save_ladder_snapshot()

        trainer = _make_trainer(
            checkpoint_dir=str(tmp_path / "new"),
            league_fraction=0.5,
            with_scripted=False,
            model_config=self._config(d_model=64, blocks=2),
        )
        trainer.roster.add_checkpoint(
            path=str(snapshot), global_step=1, update=1, initial_elo=trainer._live_elo
        )

        slots = trainer._prepare_league_slots(trainer.wrapper.num_ships)

        assert slots[0].entry.bundle.model_config.d_model == 32
        # The opponent's hidden state is its own width, not the trainee's.
        assert slots[0].hidden.shape[0] == 1  # one temporal sublayer, from *its* config
        trainer.train()  # a full update with the two architectures interleaved

    def test_a_bullet_reading_opponent_in_a_bullet_free_run_is_retired(self, tmp_path):
        """The rollout observation is shaped once and cannot widen to suit an
        opponent, so the alternative is an opponent silently playing blind.

        Retired rather than raised: with the league drawing every rollout, a
        single incompatible entry would otherwise end training hours in."""
        from tests.train.test_ppo import _make_trainer

        reader = _make_trainer(
            checkpoint_dir=str(tmp_path / "old"),
            model_config=dataclasses.replace(self._config(d_model=32), n_bullet_cross_per_block=1),
        )
        snapshot = reader._save_ladder_snapshot()

        trainer = _make_trainer(
            checkpoint_dir=str(tmp_path / "new"),
            league_fraction=0.5,
            with_scripted=False,
            model_config=self._config(d_model=32),
        )
        entry = trainer.roster.add_checkpoint(
            path=str(snapshot), global_step=1, update=1, initial_elo=trainer._live_elo
        )

        slots = trainer._prepare_league_slots(trainer.wrapper.num_ships)

        assert not entry.usable
        assert entry.elo == trainer._live_elo  # still on the ladder, just not drawn
        # Nothing else on this roster to draw, so the block falls back to self-play.
        assert slots == []


def _save_checkpoint_and_join(trainer, update: int) -> None:
    trainer._save_checkpoint(update=update)
    trainer._active_save_thread.join(timeout=60)


class TestCheckpointRetention:
    """AUDIT-017: the Elo ladder keeps every snapshot; regular saves keep a rolling window.

    Previously ``_save_checkpoint`` kept only the single newest ``step_*.pt``
    file and a single, non-rotated ``recent_avg.pt``. This exercises the
    replacement policy: the newest ``_KEEP_LAST_N_CHECKPOINTS`` live and avg
    checkpoints survive, older ones in each family are pruned, and neither the
    best-model files nor the Elo ladder's own snapshots are ever touched.
    """

    def test_prune_keeps_only_last_n_live_checkpoints(self, tmp_path):
        from boost_and_broadside.train.rl.checkpoint import _KEEP_LAST_N_CHECKPOINTS
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        for step in range(1, _KEEP_LAST_N_CHECKPOINTS + 3):
            trainer._global_step = step
            _save_checkpoint_and_join(trainer, update=step)

        remaining = sorted(p.name for p in tmp_path.rglob("step_*.pt"))
        expected_steps = range(3, _KEEP_LAST_N_CHECKPOINTS + 3)
        assert remaining == [f"step_{s:012d}.pt" for s in expected_steps]

    def test_prune_keeps_only_last_n_avg_checkpoints(self, tmp_path):
        from boost_and_broadside.train.rl.checkpoint import _KEEP_LAST_N_CHECKPOINTS
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        trainer._avg_update_count = 1  # avg policy is ready to be checkpointed
        for step in range(1, _KEEP_LAST_N_CHECKPOINTS + 3):
            trainer._global_step = step
            _save_checkpoint_and_join(trainer, update=step)

        remaining = sorted(p.name for p in tmp_path.rglob("avg_step_*.pt"))
        expected_steps = range(3, _KEEP_LAST_N_CHECKPOINTS + 3)
        assert remaining == [f"avg_step_{s:012d}.pt" for s in expected_steps]

    def test_best_and_ladder_files_survive_rolling_prune(self, tmp_path):
        from boost_and_broadside.train.rl.checkpoint import _KEEP_LAST_N_CHECKPOINTS
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        trainer._save_best_checkpoint("best_training.pt")
        trainer._active_best_thread.join(timeout=60)
        ladder_path = trainer._save_ladder_snapshot()

        for step in range(1, _KEEP_LAST_N_CHECKPOINTS + 3):
            trainer._global_step = step
            _save_checkpoint_and_join(trainer, update=step)

        ckpt_dir = Path(tmp_path) / trainer.run_name
        assert (ckpt_dir / "best_training.pt").exists()
        assert ladder_path.exists()

    def test_roster_referenced_path_is_protected_from_pruning(self, tmp_path):
        """Defense in depth: a roster "checkpoint" entry protects its path even
        though, in practice, roster entries always name ``ladder_step_*`` files
        that this glob never matches in the first place."""
        from boost_and_broadside.train.rl.checkpoint import _KEEP_LAST_N_CHECKPOINTS
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        ckpt_dir = Path(tmp_path) / trainer.run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        protected_path = ckpt_dir / "step_000000000001.pt"
        protected_path.write_bytes(b"placeholder")
        trainer.roster.add_checkpoint(str(protected_path), global_step=1, update=1)

        for step in range(2, _KEEP_LAST_N_CHECKPOINTS + 4):
            trainer._global_step = step
            _save_checkpoint_and_join(trainer, update=step)

        remaining = sorted(p.name for p in ckpt_dir.glob("step_*.pt"))
        expected_steps = [1, *range(4, _KEEP_LAST_N_CHECKPOINTS + 4)]
        assert remaining == [f"step_{s:012d}.pt" for s in expected_steps]

    def test_save_is_skipped_while_the_previous_write_is_in_flight(self, tmp_path):
        """Saving every update means a save can land while the last one is still
        writing. The new one is dropped rather than queued, and the step it would
        have written is simply absent -- the next update writes the next step."""
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        ckpt_dir = Path(tmp_path) / trainer.run_name
        release = threading.Event()
        in_flight = threading.Thread(target=release.wait, daemon=True)
        in_flight.start()
        trainer._active_save_thread = in_flight

        trainer._global_step = 7
        trainer._save_checkpoint(update=7)
        assert not list(ckpt_dir.glob("step_*.pt"))

        release.set()
        in_flight.join(timeout=60)
        trainer._global_step = 8
        _save_checkpoint_and_join(trainer, update=8)
        assert [p.name for p in ckpt_dir.glob("step_*.pt")] == ["step_000000000008.pt"]


class TestFinalCheckpoint:
    """An interrupted run has to leave behind something resumable.

    Before this, ``shutdown`` waited for in-flight writes and exited, so a run
    stopped between scheduled saves discarded every update since the last one.

    The shared test schedule sets a checkpoint interval of zero, which turns
    checkpointing off, so each case that expects a file has to switch it on.
    """

    @staticmethod
    def _interrupted_at(tmp_path, *, global_step: int, completed_update: int):
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        trainer._schedule_state.checkpoint_interval = 1
        trainer._global_step = global_step
        trainer._completed_update = completed_update
        return trainer

    def test_final_save_writes_the_last_completed_update(self, tmp_path):
        trainer = self._interrupted_at(tmp_path, global_step=4096, completed_update=12)

        trainer.save_final_checkpoint()

        ckpt_dir = Path(tmp_path) / trainer.run_name
        written = ckpt_dir / "step_000000004096.pt"
        assert written.exists()
        assert not list(ckpt_dir.glob("*.tmp")), "writer was not joined before returning"
        assert torch.load(written, map_location="cpu", weights_only=False)["update"] == 12
        assert (ckpt_dir / "roster.json").exists()

    def test_final_save_is_a_no_op_when_the_step_is_already_saved(self, tmp_path):
        trainer = self._interrupted_at(tmp_path, global_step=4096, completed_update=12)
        _save_checkpoint_and_join(trainer, update=12)
        written = Path(tmp_path) / trainer.run_name / "step_000000004096.pt"
        before = written.stat().st_mtime_ns

        trainer.save_final_checkpoint()

        assert written.stat().st_mtime_ns == before

    def test_final_save_writes_nothing_before_the_first_update_completes(self, tmp_path):
        """An interrupt during startup or the first rollout has no consistent
        state to record: the run has advanced the step counter but has not
        carried a single update through."""
        trainer = self._interrupted_at(tmp_path, global_step=512, completed_update=0)

        trainer.save_final_checkpoint()

        assert not list(Path(tmp_path).rglob("step_*.pt"))

    def test_final_save_honours_checkpointing_being_switched_off(self, tmp_path):
        """A zero interval means this run does not write checkpoints, and the
        exit path is not an exception to that."""
        trainer = self._interrupted_at(tmp_path, global_step=4096, completed_update=12)
        trainer._schedule_state.checkpoint_interval = 0

        trainer.save_final_checkpoint()

        assert not list(Path(tmp_path).rglob("step_*.pt"))

    def test_final_save_dispatches_after_an_in_flight_write_finishes(self, tmp_path):
        """``_run_async_save`` drops a save that collides with a running one, so
        the final save has to wait the writer out rather than be skipped by it."""
        trainer = self._interrupted_at(tmp_path, global_step=4096, completed_update=12)
        release = threading.Event()
        in_flight = threading.Thread(target=release.wait, daemon=True)
        in_flight.start()
        trainer._active_save_thread = in_flight

        finished = threading.Event()
        threading.Thread(
            target=lambda: (trainer.save_final_checkpoint(), finished.set()), daemon=True
        ).start()
        assert not finished.wait(timeout=0.5), "returned without waiting for the running write"
        release.set()
        assert finished.wait(timeout=60)

        assert (Path(tmp_path) / trainer.run_name / "step_000000004096.pt").exists()

    def test_resume_after_a_final_save_restarts_on_the_next_update(self, tmp_path):
        from tests.train.test_ppo import _make_trainer

        trainer = self._interrupted_at(tmp_path, global_step=4096, completed_update=12)
        trainer.save_final_checkpoint()

        resumed = _make_trainer(checkpoint_dir=str(tmp_path))
        resumed.load_checkpoint(str(Path(tmp_path) / trainer.run_name / "step_000000004096.pt"))

        assert resumed._start_update == 13
        # Interrupting again before finishing update 13 must not claim update 13.
        assert resumed._completed_update == 12


class TestRunManifest:
    """The manifest tracks the run alongside its checkpoints, and only those runs.

    Selecting a run to resume reads this file rather than loading a 27 MB
    payload, so what matters is that it exists exactly where something is
    resumable and that its status tells the truth about how the run ended.
    """

    @staticmethod
    def _trainer(tmp_path):
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        trainer._schedule_state.checkpoint_interval = 1
        trainer.resolved_config_document = {
            "profile": "rl",
            "resolved_config_fingerprint": "0bf3a3b5",
        }
        trainer.launch_provenance = {"device": "cpu", "seed": 7}
        return trainer

    def test_a_scheduled_save_records_the_run(self, tmp_path):
        from boost_and_broadside.run_manifest import RunStatus, read_manifest

        trainer = self._trainer(tmp_path)
        trainer._global_step = 4096
        trainer._maybe_save_checkpoint(update=3)
        trainer._active_save_thread.join(timeout=60)

        manifest = read_manifest(Path(tmp_path) / trainer.run_name)

        assert manifest is not None
        assert manifest.run == trainer.run_name
        assert (manifest.profile, manifest.device, manifest.seed) == ("rl", "cpu", 7)
        assert (manifest.global_step, manifest.update) == (4096, 3)
        assert manifest.status is RunStatus.RUNNING
        assert manifest.resolved_config_fingerprint == "0bf3a3b5"

    def test_the_final_save_records_the_run_it_wrote(self, tmp_path):
        from boost_and_broadside.run_manifest import read_manifest

        trainer = self._trainer(tmp_path)
        trainer._global_step = 4096
        trainer._completed_update = 12
        trainer.save_final_checkpoint()

        manifest = read_manifest(Path(tmp_path) / trainer.run_name)

        assert manifest is not None
        assert (manifest.global_step, manifest.update) == (4096, 12)

    def test_status_moves_to_a_terminal_value_on_an_existing_manifest(self, tmp_path):
        from boost_and_broadside.run_manifest import RunStatus, read_manifest

        trainer = self._trainer(tmp_path)
        trainer._global_step = 4096
        trainer._maybe_save_checkpoint(update=3)
        trainer._active_save_thread.join(timeout=60)

        trainer.record_run_status(RunStatus.INTERRUPTED)

        manifest = read_manifest(Path(tmp_path) / trainer.run_name)
        assert manifest is not None
        assert manifest.status is RunStatus.INTERRUPTED
        # The record it was tracking is untouched by the status change.
        assert (manifest.global_step, manifest.update) == (4096, 3)

    def test_a_run_with_no_checkpoint_gets_no_manifest(self, tmp_path):
        """Checkpointing switched off means nothing to resume, so nothing to list
        -- and no directory written for a trainer that only ever ran."""
        from boost_and_broadside.run_manifest import RunStatus, read_manifest

        trainer = self._trainer(tmp_path)
        trainer._schedule_state.checkpoint_interval = 0
        trainer._global_step = 4096
        trainer._completed_update = 12

        trainer._maybe_save_checkpoint(update=12)
        trainer.save_final_checkpoint()
        trainer.record_run_status(RunStatus.COMPLETE)

        assert read_manifest(Path(tmp_path) / trainer.run_name) is None
        assert not list(Path(tmp_path).rglob("run.json"))


class TestBestCheckpoints:
    """The best-model checkpoints (live and avg) overwrite in place as Elo improves."""

    def test_best_training_is_saved_only_when_live_elo_improves(self, tmp_path):
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        ckpt_dir = Path(tmp_path) / trainer.run_name

        trainer._live_elo = 10.0
        trainer._maybe_save_best_checkpoints()
        trainer._active_best_thread.join(timeout=60)
        assert (ckpt_dir / "best_training.pt").exists()
        first_mtime = (ckpt_dir / "best_training.pt").stat().st_mtime_ns

        # Elo regresses: the file must not be rewritten.
        trainer._live_elo = 5.0
        trainer._maybe_save_best_checkpoints()
        assert (ckpt_dir / "best_training.pt").stat().st_mtime_ns == first_mtime

    def test_best_avg_is_not_saved_before_avg_model_is_ready(self, tmp_path):
        """AUDIT-adjacent: _best_avg_live_elo previously had no writer at all."""
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        ckpt_dir = Path(tmp_path) / trainer.run_name

        assert trainer._avg_update_count == 0
        trainer._avg_live_elo = 1000.0  # would trip the threshold if checked
        trainer._maybe_save_best_checkpoints()
        assert not (ckpt_dir / "best_avg.pt").exists()

    def test_best_avg_checkpoint_holds_avg_policy_weights(self, tmp_path):
        """The previously-dead best-avg trigger now writes the avg policy's
        weights, not the live policy's, into best_avg.pt."""
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        ckpt_dir = Path(tmp_path) / trainer.run_name
        with torch.no_grad():
            for p in trainer._avg_policy_module.parameters():
                p.add_(1.0)

        trainer._avg_update_count = 1
        trainer._avg_live_elo = 50.0
        trainer._maybe_save_best_checkpoints()
        trainer._active_best_avg_thread.join(timeout=60)

        saved = torch.load(ckpt_dir / "best_avg.pt", map_location="cpu", weights_only=False)
        live_state = trainer._policy_module.state_dict()
        for name, avg_param in saved["policy_state_dict"].items():
            assert not torch.equal(avg_param, live_state[name])

    def test_best_avg_saves_when_live_elo_improves_in_the_same_update(self, tmp_path):
        """AUDIT-027: best_avg and best_training must not share a save thread.

        When both ELOs improve in one call, the best_training save is in flight
        by the time best_avg is attempted; a shared thread slot would skip the
        best_avg save while its high-water mark advanced anyway, leaving
        best_avg.pt unwritten. Separate slots must let both persist.
        """
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        ckpt_dir = Path(tmp_path) / trainer.run_name

        trainer._avg_update_count = 1
        trainer._live_elo = 10.0  # live improves
        trainer._avg_live_elo = 10.0  # avg improves in the same call
        trainer._maybe_save_best_checkpoints()
        trainer._active_best_thread.join(timeout=60)
        trainer._active_best_avg_thread.join(timeout=60)

        assert (ckpt_dir / "best_training.pt").exists()
        assert (ckpt_dir / "best_avg.pt").exists()

    def test_skipped_best_save_does_not_advance_high_water_mark(self, tmp_path):
        """AUDIT-027: a save skipped for an in-flight write must not raise the bar.

        If the mark advanced on a skipped save, the peak would be recorded as
        captured and never retried. Simulate an in-flight save with a live dummy
        thread and assert the mark stays put so the next update retries.
        """
        import threading
        import time

        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))

        blocker = threading.Thread(target=lambda: time.sleep(2.0), daemon=True)
        blocker.start()
        trainer._active_best_thread = blocker  # occupy the live-best slot

        mark_before = trainer._best_live_elo
        trainer._live_elo = 10.0
        trainer._maybe_save_best_checkpoints()

        # Save was skipped (slot busy), so the bar must not have moved.
        assert trainer._best_live_elo == mark_before
        blocker.join(timeout=60)
