"""Checkpoint serialization integrity.

These tests exist because a corrupt checkpoint is silent: the file loads, every
tensor has the right shape and dtype, and the values look plausible. The damage
only surfaces later as a diverged resume or a NaN with no traceable origin.
"""

from pathlib import Path

import pytest
import torch

from boost_and_broadside.modes.agent_factory import infer_num_value_components
from boost_and_broadside.train.rl.checkpoint import clone_to_cpu

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

        trainer = _make_trainer(checkpoint_dir=str(tmp_path), avg_model_fraction=0.25)
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

        trainer = _make_trainer(checkpoint_dir=str(tmp_path), avg_model_fraction=0.25)
        assert all(c.dtype == torch.float32 for c in trainer._avg_param_cumsum)

        trainer.train()
        trainer._save_checkpoint(update=1)
        save_thread = getattr(trainer, "_active_save_thread", None)
        if save_thread is not None:
            save_thread.join(timeout=60)
        saved = list(tmp_path.rglob("step_*.pt"))
        trainer.load_checkpoint(str(saved[0]))
        assert all(c.dtype == torch.float32 for c in trainer._avg_param_cumsum)


class TestNumValueComponents:
    """AUDIT-018: critic width K is saved explicitly, not reverse-engineered.

    Every loader used to read K off a hardcoded state-dict key
    (``value_head_local.3.weight``). Saving it as a field decouples the loaders
    from the value head's internal structure; the shape introspection survives
    only as a legacy fallback for checkpoints written before the field existed.
    """

    def test_payloads_record_active_component_count(self, tmp_path):
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path), avg_model_fraction=0.25)
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


def _save_checkpoint_and_join(trainer, update: int) -> None:
    trainer._save_checkpoint(update=update)
    trainer._active_save_thread.join(timeout=60)


class TestCheckpointRetention:
    """AUDIT-017: the ELO ladder keeps every snapshot; regular saves keep a rolling window.

    Previously ``_save_checkpoint`` kept only the single newest ``step_*.pt``
    file and a single, non-rotated ``recent_avg.pt``. This exercises the
    replacement policy: the newest ``_KEEP_LAST_N_CHECKPOINTS`` live and avg
    checkpoints survive, older ones in each family are pruned, and neither the
    best-model files nor the ELO ladder's own snapshots are ever touched.
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

        trainer = _make_trainer(checkpoint_dir=str(tmp_path), avg_model_fraction=0.25)
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


class TestBestCheckpoints:
    """The best-model checkpoints (live and avg) overwrite in place as ELO improves."""

    def test_best_training_is_saved_only_when_live_elo_improves(self, tmp_path):
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        ckpt_dir = Path(tmp_path) / trainer.run_name

        trainer._training_elo = 10.0
        trainer._maybe_save_best_checkpoints(random_elo=0.0)
        trainer._active_best_thread.join(timeout=60)
        assert (ckpt_dir / "best_training.pt").exists()
        first_mtime = (ckpt_dir / "best_training.pt").stat().st_mtime_ns

        # ELO regresses: the file must not be rewritten.
        trainer._training_elo = 5.0
        trainer._maybe_save_best_checkpoints(random_elo=0.0)
        assert (ckpt_dir / "best_training.pt").stat().st_mtime_ns == first_mtime

    def test_best_avg_is_not_saved_before_avg_model_is_ready(self, tmp_path):
        """AUDIT-adjacent: _best_avg_elo_norm previously had no writer at all."""
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path), avg_model_fraction=0.25)
        ckpt_dir = Path(tmp_path) / trainer.run_name

        assert trainer._avg_update_count == 0
        trainer._avg_training_elo = 1000.0  # would trip the threshold if checked
        trainer._maybe_save_best_checkpoints(random_elo=0.0)
        assert not (ckpt_dir / "best_avg.pt").exists()

    def test_best_avg_checkpoint_holds_avg_policy_weights(self, tmp_path):
        """The previously-dead best-avg trigger now writes the avg policy's
        weights, not the live policy's, into best_avg.pt."""
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path), avg_model_fraction=0.25)
        ckpt_dir = Path(tmp_path) / trainer.run_name
        with torch.no_grad():
            for p in trainer._avg_policy_module.parameters():
                p.add_(1.0)

        trainer._avg_update_count = 1
        trainer._avg_training_elo = 50.0
        trainer._maybe_save_best_checkpoints(random_elo=0.0)
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

        trainer = _make_trainer(checkpoint_dir=str(tmp_path), avg_model_fraction=0.25)
        ckpt_dir = Path(tmp_path) / trainer.run_name

        trainer._avg_update_count = 1
        trainer._training_elo = 10.0  # live improves
        trainer._avg_training_elo = 10.0  # avg improves in the same call
        trainer._maybe_save_best_checkpoints(random_elo=0.0)
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

        trainer._training_elo = 10.0
        trainer._maybe_save_best_checkpoints(random_elo=0.0)

        # Save was skipped (slot busy), so the bar must not have moved.
        assert trainer._best_training_elo_norm == 0.0
        blocker.join(timeout=60)
