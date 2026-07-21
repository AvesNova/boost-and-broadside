"""Checkpoint serialization integrity.

These tests exist because a corrupt checkpoint is silent: the file loads, every
tensor has the right shape and dtype, and the values look plausible. The damage
only surfaces later as a diverged resume or a NaN with no traceable origin.
"""

import pytest
import torch

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
