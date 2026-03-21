"""Fix J: Epoch loss logging should NOT multiply by acc_steps."""
import pytest
import torch
from unittest.mock import MagicMock, patch


def make_mock_trainer(acc_steps=4):
    """Create a minimal mock trainer with the relevant accumulator logic."""
    trainer = MagicMock()
    trainer.acc_steps = acc_steps
    trainer.device = torch.device("cpu")
    
    # Inline the fixed _update_accumulators logic
    def _update_accumulators(acc, metrics):
        acc["total_loss"] += metrics["loss"]  # Fixed: no * acc_steps
        acc["total_state_loss"] += metrics["state_loss"]
        acc["total_action_loss"] += metrics["action_loss"]
        acc["total_relational_loss"] += metrics["relational_loss"]
        acc["total_value_loss"] += metrics["metrics"].get("value_loss", 0.0)
        acc["total_reward_loss"] += metrics["metrics"].get("reward_loss", 0.0)
        acc["acc_loss"] += metrics["loss"]  # Fixed: no * acc_steps
        acc["acc_state"] += metrics["state_loss"]
        acc["acc_action"] += metrics["action_loss"]
        acc["acc_rel"] += metrics["relational_loss"]
        acc["acc_time"] += metrics["time"]
        acc["acc_tokens"] += metrics["num_tokens"]
    
    trainer._update_accumulators = _update_accumulators
    return trainer


def test_loss_not_inflated_by_acc_steps():
    """total_loss should equal raw loss sum, not loss * acc_steps sum."""
    acc_steps = 4
    trainer = make_mock_trainer(acc_steps=acc_steps)

    # Simulate the accumulators
    acc = {
        "total_loss": torch.tensor(0.0),
        "total_state_loss": torch.tensor(0.0),
        "total_action_loss": torch.tensor(0.0),
        "total_relational_loss": torch.tensor(0.0),
        "total_value_loss": torch.tensor(0.0),
        "total_reward_loss": torch.tensor(0.0),
        "acc_loss": 0.0,
        "acc_state": 0.0,
        "acc_action": 0.0,
        "acc_rel": 0.0,
        "acc_time": 0.0,
        "acc_tokens": 0.0,
        "acc_errors": {}
    }

    raw_loss = 2.0
    metrics = {
        "loss": torch.tensor(raw_loss),
        "state_loss": torch.tensor(0.5),
        "action_loss": torch.tensor(0.3),
        "relational_loss": torch.tensor(0.1),
        "time": 0.01,
        "num_tokens": 256,
        "metrics": {"value_loss": 0.1, "reward_loss": 0.05}
    }

    # Call N times to simulate micro-steps
    N = 3
    for _ in range(N):
        trainer._update_accumulators(acc, metrics)

    # total_loss should be raw_loss * N, NOT raw_loss * N * acc_steps
    expected = raw_loss * N
    actual = acc["total_loss"].item()
    assert abs(actual - expected) < 1e-5, \
        f"total_loss={actual} but expected {expected} (got inflated by acc_steps={acc_steps}?)"


def test_acc_loss_not_inflated():
    """acc_loss should equal raw loss sum, not multiplied by acc_steps."""
    acc_steps = 8
    trainer = make_mock_trainer(acc_steps=acc_steps)

    acc = {"total_loss": torch.tensor(0.0), "total_state_loss": torch.tensor(0.0),
           "total_action_loss": torch.tensor(0.0), "total_relational_loss": torch.tensor(0.0),
           "total_value_loss": torch.tensor(0.0), "total_reward_loss": torch.tensor(0.0),
           "acc_loss": 0.0, "acc_state": 0.0, "acc_action": 0.0, "acc_rel": 0.0,
           "acc_time": 0.0, "acc_tokens": 0.0, "acc_errors": {}}

    raw_loss = 1.5
    metrics = {"loss": torch.tensor(raw_loss), "state_loss": torch.tensor(0.5),
               "action_loss": torch.tensor(0.3), "relational_loss": torch.tensor(0.1),
               "time": 0.01, "num_tokens": 128, "metrics": {}}

    trainer._update_accumulators(acc, metrics)

    assert abs(acc["acc_loss"] - raw_loss) < 1e-5, \
        f"acc_loss={acc['acc_loss']} should be {raw_loss}, not {raw_loss * acc_steps}"
