import torch
import pytest
from omegaconf import OmegaConf
from boost_and_broadside.models.yemong.scaffolds import YemongDynamicsInterleaved

from boost_and_broadside.core.constants import STATE_DIM


def test_interleaved_ppo_unpacking():
    # Setup dummy inputs
    B = 2
    T = 3
    N = 4
    state_dim = STATE_DIM

    config = OmegaConf.create(
        {
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 2,
            "input_dim": state_dim,
            "action_dim": 12,
            "target_dim": 14,
            "use_soft_bin_targets": True,
            "loss": {
                "_target_": "boost_and_broadside.models.components.losses.CompositeLoss",
                "losses": [],
            },
        }
    )

    model = YemongDynamicsInterleaved(config)
    device = torch.device("cpu")

    x = {
        "state": torch.zeros(B, T, N, state_dim),
        "prev_action": torch.zeros(B, T, N, 3),
        "pos": torch.zeros(B, T, N, 2),
        "vel": torch.zeros(B, T, N, 2),
        "alive": torch.ones(B, T, N, dtype=torch.bool),
    }

    target_actions = torch.zeros(B, T, N, 3)
    seq_idx = torch.zeros(B, T)

    # In PPO update phase, action is provided and step_type is None
    try:
        results = model.get_action_and_value(x, action=target_actions, seq_idx=seq_idx)
        # Should return 8 items: action(None), logprob, entropy, pred_values, mamba_state(None), pred_states, pred_rewards, *extras
        assert len(results) >= 7
        print("PPO Tuple Unpacking Test Passed!")
    except ValueError as e:
        pytest.fail(f"Tuple unpacking failed: {e}")


if __name__ == "__main__":
    test_interleaved_ppo_unpacking()
