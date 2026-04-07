import pytest
import torch
from pathlib import Path
from omegaconf import OmegaConf

from boost_and_broadside.models.yemong.scaffolds import YemongDynamicsInterleaved
from boost_and_broadside.core.constants import STATE_DIM, TARGET_DIM, StateFeature


def test_yemong_dynamics_interleaved_parallel_forward():
    d_model = 128
    config = OmegaConf.create(
        {
            "d_model": d_model,
            "n_layers": 2,
            "n_heads": 4,
            "input_dim": STATE_DIM,
            "target_dim": TARGET_DIM,
            "action_dim": 12,
            "loss_type": "fixed",
            "spatial_layer": {
                "_target_": "boost_and_broadside.models.components.layers.attention.RelationalAttention",
                "d_model": d_model,
                "n_heads": 4,
            },
            "loss": {
                "_target_": "boost_and_broadside.models.components.losses.CompositeLoss",
                "losses": [
                    {
                        "_target_": "boost_and_broadside.models.components.losses.StateLoss",
                        "weight": 1.0,
                    },
                    {
                        "_target_": "boost_and_broadside.models.components.losses.ActionLoss",
                        "weight": 1.0,
                    },
                    {
                        "_target_": "boost_and_broadside.models.components.losses.ValueLoss",
                        "weight": 1.0,
                    },
                    {
                        "_target_": "boost_and_broadside.models.components.losses.RewardLoss",
                        "weight": 1.0,
                    },
                ],
            },
        }
    )
    model = YemongDynamicsInterleaved(config)

    B, T, N = 2, 4, 3
    state = torch.randn(B, T, N, STATE_DIM)
    state[..., StateFeature.HEALTH] = 1.0  # Health > 0

    prev_action = torch.randint(0, 2, (B, T, N, 3))

    pos = torch.randn(B, T, N, 2)
    vel = torch.randn(B, T, N, 2)
    att = torch.randn(B, T, N, 2)
    team_ids = torch.randint(0, 2, (B, N))
    seq_idx = torch.zeros(B, T, dtype=torch.long)

    target_actions = torch.randint(0, 2, (B, T, N, 3))

    action_logits, _, _, value_pred, _, next_state_pred, reward_pred, _, _, _ = model(
        state=state,
        prev_action=prev_action,
        pos=pos,
        vel=vel,
        att=att,
        team_ids=team_ids,
        seq_idx=seq_idx,
        target_actions=target_actions,
    )

    # State token outputs
    assert action_logits.shape == (B, T, N, 12)
    assert value_pred.shape == (B, T, 1)

    # Action token outputs
    assert next_state_pred.shape == (B, T, N, TARGET_DIM)
    assert reward_pred.shape == (B, T, 1)


def test_yemong_dynamics_interleaved_missing_targets():
    config = OmegaConf.create(
        {
            "d_model": 64,
            "n_layers": 1,
            "n_heads": 2,
            "spatial_layer": {
                "_target_": "boost_and_broadside.models.components.layers.attention.RelationalAttention",
                "d_model": 64,
                "n_heads": 2,
            },
            "loss": {
                "_target_": "boost_and_broadside.models.components.losses.CompositeLoss",
                "losses": [],
            },
        }
    )
    model = YemongDynamicsInterleaved(config)

    B, T, N = 1, 2, 2
    state = torch.randn(B, T, N, STATE_DIM)
    prev_action = torch.randint(0, 2, (B, T, N, 3))
    pos = torch.randn(B, T, N, 2)
    vel = torch.randn(B, T, N, 2)

    with pytest.raises(ValueError, match="target_actions must be provided"):
        model(state=state, prev_action=prev_action, pos=pos, vel=vel)
