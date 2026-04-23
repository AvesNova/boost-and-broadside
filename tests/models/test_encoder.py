"""Tests for the unified encoder and policy forward passes."""

import pytest
import torch

from boost_and_broadside.config import ShipConfig, ModelConfig, EnvConfig
from boost_and_broadside.models.mvp.encoder import UnifiedEncoder
from boost_and_broadside.models.mvp.attention import TransformerBlock
from boost_and_broadside.models.mvp.policy import MVPPolicy
from boost_and_broadside.models.mvp.griffin import CONV_KERNEL


@pytest.fixture
def ship_cfg() -> ShipConfig:
    return ShipConfig()


@pytest.fixture
def model_cfg() -> ModelConfig:
    return ModelConfig(d_model=64, n_heads=4, n_fourier_freqs=8, n_transformer_blocks=2)


NUM_VALUE_COMPONENTS = 12  # fixed K for encoder/policy unit tests


def _make_obs(B: int, N: int, M: int = 0) -> dict[str, torch.Tensor]:
    """Build a minimal random obs dict matching MVPEnvWrapper output."""
    return {
        "pos": torch.rand(B, N, 2),
        "vel": torch.randn(B, N, 2) * 100,
        "att": torch.randn(B, N, 2),
        "ang_vel": torch.randn(B, N, 1),
        "scalars": torch.rand(B, N, 3),
        "team_id": torch.randint(0, 2, (B, N)),
        "alive": torch.ones(B, N, dtype=torch.bool),
        "prev_action": torch.zeros(B, N, 3, dtype=torch.long),
        "obstacle_pos": torch.rand(B, M, 2),
        "obstacle_vel": torch.randn(B, M, 2),
        "obstacle_radius": torch.rand(B, M, 1),
        "obstacle_gravity_center": torch.rand(B, M, 2),
        "obstacle_hit": torch.zeros(B, M, 1),
    }


class TestUnifiedEncoder:
    def test_encode_ships_shape(self, ship_cfg, model_cfg):
        """encode_ships must output (B, N, d_model)."""
        B, N = 4, 8
        encoder = UnifiedEncoder(model_cfg, ship_cfg)
        out = encoder.encode_ships(_make_obs(B, N))
        assert out.shape == (B, N, model_cfg.d_model)

    def test_encode_ships_finite(self, ship_cfg, model_cfg):
        B, N = 2, 4
        encoder = UnifiedEncoder(model_cfg, ship_cfg)
        out = encoder.encode_ships(_make_obs(B, N))
        assert torch.isfinite(out).all()

    def test_forward_alias(self, ship_cfg, model_cfg):
        """forward() is an alias for encode_ships()."""
        B, N = 2, 4
        encoder = UnifiedEncoder(model_cfg, ship_cfg)
        obs = _make_obs(B, N)
        assert torch.allclose(encoder(obs), encoder.encode_ships(obs))

    def test_dead_ships_produce_different_token(self, ship_cfg, model_cfg):
        """alive=False should change the ship token."""
        B, N = 1, 2
        encoder = UnifiedEncoder(model_cfg, ship_cfg)
        obs_alive = _make_obs(B, N)
        obs_dead = {k: v.clone() for k, v in obs_alive.items()}
        obs_dead["alive"][0, 0] = False

        out_alive = encoder.encode_ships(obs_alive)
        out_dead = encoder.encode_ships(obs_dead)

        assert not torch.allclose(out_alive[0, 0], out_dead[0, 0])
        assert torch.allclose(out_alive[0, 1], out_dead[0, 1])

    def test_encode_obstacles_shape(self, ship_cfg, model_cfg):
        """encode_obstacles must output (B, M, d_model)."""
        B, N, M = 3, 4, 5
        encoder = UnifiedEncoder(model_cfg, ship_cfg)
        out = encoder.encode_obstacles(_make_obs(B, N, M))
        assert out.shape == (B, M, model_cfg.d_model)

    def test_encode_obstacles_finite(self, ship_cfg, model_cfg):
        B, N, M = 2, 4, 3
        encoder = UnifiedEncoder(model_cfg, ship_cfg)
        out = encoder.encode_obstacles(_make_obs(B, N, M))
        assert torch.isfinite(out).all()

    def test_obstacle_tokens_differ_from_ships(self, ship_cfg, model_cfg):
        """Obstacles and ships with the same position should encode differently."""
        B, N, M = 1, 1, 1
        encoder = UnifiedEncoder(model_cfg, ship_cfg)
        obs = _make_obs(B, N, M)
        ship_out = encoder.encode_ships(obs)
        obs_out = encoder.encode_obstacles(obs)
        # They share the same feature_extractor but different raw vectors —
        # team slot alone ensures distinct outputs.
        assert not torch.allclose(ship_out, obs_out)

    def test_handles_batch_of_one(self, ship_cfg, model_cfg):
        encoder = UnifiedEncoder(model_cfg, ship_cfg)
        out = encoder.encode_ships(_make_obs(1, 1))
        assert out.shape == (1, 1, model_cfg.d_model)

    def test_encode_obstacles_zero_when_no_obstacles(self, ship_cfg, model_cfg):
        """encode_obstacles with M=0 returns empty (B, 0, D) tensor."""
        B, N = 2, 4
        encoder = UnifiedEncoder(model_cfg, ship_cfg)
        out = encoder.encode_obstacles(_make_obs(B, N, M=0))
        assert out.shape == (B, 0, model_cfg.d_model)


class TestTransformerBlock:
    def test_output_shape_unchanged(self, model_cfg):
        """Transformer block output must match input shape (B, N, D)."""
        B, N = 3, 8
        block = TransformerBlock(model_cfg)
        x = torch.randn(B, N, model_cfg.d_model)

        out = block(x)

        assert out.shape == (B, N, model_cfg.d_model)

    def test_alive_mask_does_not_crash(self, model_cfg):
        B, N = 2, 4
        block = TransformerBlock(model_cfg)
        x = torch.randn(B, N, model_cfg.d_model)
        alive = torch.ones(B, N, dtype=torch.bool)
        alive[0, 2] = False

        out = block(x, alive_mask=alive)

        assert out.shape == (B, N, model_cfg.d_model)
        assert torch.isfinite(out).all()

    def test_all_dead_mask_does_not_produce_nan(self, model_cfg):
        B, N = 1, 4
        block = TransformerBlock(model_cfg)
        x = torch.randn(B, N, model_cfg.d_model)
        alive = torch.zeros(B, N, dtype=torch.bool)

        out = block(x, alive_mask=alive)

        assert not torch.isnan(out).any()


class TestMVPPolicy:
    def test_get_action_and_value_shapes_no_obstacles(self, model_cfg, ship_cfg):
        """get_action_and_value with M=0 returns correct shapes."""
        B, N = 2, 8
        policy = MVPPolicy(model_cfg, ship_cfg, num_value_components=NUM_VALUE_COMPONENTS)
        obs = _make_obs(B, N, M=0)
        hidden = policy.initial_hidden(B, N, torch.device("cpu"))

        action, logprob, value, new_hidden = policy.get_action_and_value(obs, hidden)

        K = NUM_VALUE_COMPONENTS
        assert action.shape == (B, N, 3)
        assert logprob.shape == (B, N)
        assert value.shape == (B, N, K)
        assert new_hidden.shape == (
            model_cfg.n_transformer_blocks, B * N, CONV_KERNEL * model_cfg.d_model
        )

    def test_get_action_and_value_shapes_with_obstacles(self, model_cfg, ship_cfg):
        """get_action_and_value with M>0 returns correct shapes (hidden covers N+M tokens)."""
        B, N, M = 2, 4, 3
        policy = MVPPolicy(model_cfg, ship_cfg, num_value_components=NUM_VALUE_COMPONENTS)
        obs = _make_obs(B, N, M)
        hidden = policy.initial_hidden(B, N + M, torch.device("cpu"))

        action, logprob, value, new_hidden = policy.get_action_and_value(obs, hidden)

        K = NUM_VALUE_COMPONENTS
        assert action.shape == (B, N, 3)
        assert logprob.shape == (B, N)
        assert value.shape == (B, N, K)
        assert new_hidden.shape == (
            model_cfg.n_transformer_blocks, B * (N + M), CONV_KERNEL * model_cfg.d_model
        )

    def test_action_indices_in_valid_range(self, model_cfg, ship_cfg):
        from boost_and_broadside.constants import (
            NUM_POWER_ACTIONS,
            NUM_TURN_ACTIONS,
            NUM_SHOOT_ACTIONS,
        )
        B, N = 2, 4
        policy = MVPPolicy(model_cfg, ship_cfg, num_value_components=NUM_VALUE_COMPONENTS)
        obs = _make_obs(B, N)
        hidden = policy.initial_hidden(B, N, torch.device("cpu"))

        action, _, _, _ = policy.get_action_and_value(obs, hidden)

        assert (action[..., 0] >= 0).all() and (action[..., 0] < NUM_POWER_ACTIONS).all()
        assert (action[..., 1] >= 0).all() and (action[..., 1] < NUM_TURN_ACTIONS).all()
        assert (action[..., 2] >= 0).all() and (action[..., 2] < NUM_SHOOT_ACTIONS).all()

    def test_evaluate_actions_shapes(self, model_cfg, ship_cfg):
        T, B, N = 4, 2, 8
        policy = MVPPolicy(model_cfg, ship_cfg, num_value_components=NUM_VALUE_COMPONENTS)
        K = NUM_VALUE_COMPONENTS

        obs = {k: v.unsqueeze(0).expand(T, *v.shape) for k, v in _make_obs(B, N).items()}
        actions = torch.zeros(T, B, N, 3, dtype=torch.long)
        hidden = policy.initial_hidden(B, N, torch.device("cpu"))
        alive_mask = torch.ones(T, B, N, dtype=torch.bool)

        logprob, entropy, new_value, logits = policy.evaluate_actions(
            obs, actions, hidden, alive_mask
        )

        assert logprob.shape == (T, B, N)
        assert entropy.shape == (T, B, N)
        assert new_value.shape == (T, B, N, K)
        assert logits.shape == (T, B, N, 12)

    def test_evaluate_actions_shapes_with_obstacles(self, model_cfg, ship_cfg):
        """evaluate_actions with M>0 still produces ship-sized outputs."""
        T, B, N, M = 4, 2, 4, 3
        policy = MVPPolicy(model_cfg, ship_cfg, num_value_components=NUM_VALUE_COMPONENTS)
        K = NUM_VALUE_COMPONENTS

        obs = {k: v.unsqueeze(0).expand(T, *v.shape) for k, v in _make_obs(B, N, M).items()}
        actions = torch.zeros(T, B, N, 3, dtype=torch.long)
        hidden = policy.initial_hidden(B, N + M, torch.device("cpu"))
        alive_mask = torch.ones(T, B, N, dtype=torch.bool)

        logprob, entropy, new_value, logits = policy.evaluate_actions(
            obs, actions, hidden, alive_mask
        )

        assert logprob.shape == (T, B, N)
        assert entropy.shape == (T, B, N)
        assert new_value.shape == (T, B, N, K)
        assert logits.shape == (T, B, N, 12)

    def test_hidden_reset_zeros_done_envs(self, model_cfg, ship_cfg):
        B, N = 3, 4
        policy = MVPPolicy(model_cfg, ship_cfg, num_value_components=NUM_VALUE_COMPONENTS)
        hidden = torch.ones(
            model_cfg.n_transformer_blocks, B * N, CONV_KERNEL * model_cfg.d_model
        )
        done = torch.tensor([True, False, True])

        new_hidden = policy.reset_hidden_for_envs(hidden, done, N)

        for ship in range(N):
            assert (new_hidden[0, ship, :] == 0).all()          # env 0
            assert (new_hidden[0, N + ship, :] != 0).any()       # env 1 unchanged
            assert (new_hidden[0, 2 * N + ship, :] == 0).all()  # env 2
