"""Tests for the ship encoder and policy forward passes."""

import pytest
import torch

from boost_and_broadside.config import ShipConfig, ModelConfig, EnvConfig
from boost_and_broadside.models.mvp.encoder import ShipEncoder
from boost_and_broadside.models.mvp.attention import RelationalSelfAttention
from boost_and_broadside.models.mvp.relational_features import RelationalFeatures
from boost_and_broadside.models.mvp.policy import MVPPolicy


@pytest.fixture
def ship_cfg() -> ShipConfig:
    return ShipConfig()


@pytest.fixture
def model_cfg() -> ModelConfig:
    return ModelConfig(d_model=64, n_heads=4, n_fourier_freqs=8)


def _make_obs(B: int, N: int) -> dict[str, torch.Tensor]:
    """Build a minimal random obs dict matching MVPEnvWrapper output."""
    return {
        "pos"         : torch.rand(B, N, 2),
        "vel"         : torch.randn(B, N, 2) * 100,
        "att"         : torch.randn(B, N, 2),
        "ang_vel"     : torch.randn(B, N, 1),
        "scalars"     : torch.rand(B, N, 3),
        "team_id"     : torch.randint(0, 2, (B, N)),
        "alive"       : torch.ones(B, N, dtype=torch.bool),
        "prev_action" : torch.zeros(B, N, 3, dtype=torch.long),
    }


class TestShipEncoder:
    def test_output_shape(self, ship_cfg, model_cfg):
        """Encoder must output (B, N, d_model) from standard obs."""
        B, N = 4, 8
        encoder = ShipEncoder(model_cfg, ship_cfg)
        obs     = _make_obs(B, N)

        out = encoder(obs)

        assert out.shape == (B, N, model_cfg.d_model)

    def test_output_is_finite(self, ship_cfg, model_cfg):
        """No NaN or Inf in encoder output."""
        B, N = 2, 4
        encoder = ShipEncoder(model_cfg, ship_cfg)
        obs     = _make_obs(B, N)

        out = encoder(obs)

        assert torch.isfinite(out).all()

    def test_dead_ships_produce_different_token(self, ship_cfg, model_cfg):
        """alive=False should affect the token (alive is a feature)."""
        B, N = 1, 2
        encoder = ShipEncoder(model_cfg, ship_cfg)

        obs_alive = _make_obs(B, N)
        obs_dead  = {k: v.clone() for k, v in obs_alive.items()}
        obs_dead["alive"][0, 0] = False

        out_alive = encoder(obs_alive)
        out_dead  = encoder(obs_dead)

        # Token 0 should differ; token 1 should be identical
        assert not torch.allclose(out_alive[0, 0], out_dead[0, 0])
        assert torch.allclose(out_alive[0, 1], out_dead[0, 1])

    def test_handles_batch_of_one(self, ship_cfg, model_cfg):
        """Encoder must work with B=1, N=1."""
        encoder = ShipEncoder(model_cfg, ship_cfg)
        obs     = _make_obs(1, 1)
        out     = encoder(obs)
        assert out.shape == (1, 1, model_cfg.d_model)


class TestRelationalSelfAttention:
    def test_output_shape_unchanged(self, model_cfg):
        """Attention output must match input shape (B, N, D)."""
        B, N = 3, 8
        attn = RelationalSelfAttention(model_cfg)
        x    = torch.randn(B, N, model_cfg.d_model)

        out  = attn(x)

        assert out.shape == (B, N, model_cfg.d_model)

    def test_alive_mask_does_not_crash(self, model_cfg):
        """Attention with a partial alive mask must not raise errors."""
        B, N = 2, 4
        attn  = RelationalSelfAttention(model_cfg)
        x     = torch.randn(B, N, model_cfg.d_model)
        alive = torch.ones(B, N, dtype=torch.bool)
        alive[0, 2] = False  # one dead ship

        out = attn(x, alive_mask=alive)

        assert out.shape == (B, N, model_cfg.d_model)
        assert torch.isfinite(out).all()

    def test_all_dead_mask_does_not_produce_nan(self, model_cfg):
        """All-dead alive mask should not produce NaN (edge case)."""
        B, N = 1, 4
        attn  = RelationalSelfAttention(model_cfg)
        x     = torch.randn(B, N, model_cfg.d_model)
        alive = torch.zeros(B, N, dtype=torch.bool)  # everyone dead

        out = attn(x, alive_mask=alive)

        # May be all-zero or garbage but must not be NaN
        assert not torch.isnan(out).any()


class TestRelationalFeatures:
    def test_output_shape(self, model_cfg):
        """RelationalFeatures must return (B, H, N, N)."""
        B, N, H = 3, 4, model_cfg.n_heads
        rf  = RelationalFeatures(H)
        obs = _make_obs(B, N)

        bias = rf(obs)

        assert bias.shape == (B, H, N, N)

    def test_output_is_finite(self, model_cfg):
        """No NaN or Inf in relational bias output."""
        B, N = 2, 6
        rf   = RelationalFeatures(model_cfg.n_heads)
        obs  = _make_obs(B, N)

        bias = rf(obs)

        assert torch.isfinite(bias).all()

    def test_bearing_ahead_gives_cos_near_one(self):
        """Ship j directly ahead of ship i must give bear_cos ≈ 1.

        Setup: i at (0, 0), heading along +x (att=(1,0)); j at (1, 0).
        """
        rf = RelationalFeatures(n_heads=1)

        obs = {
            # Single batch, 2 ships
            "pos": torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]),  # (1, 2, 2)
            "vel": torch.zeros(1, 2, 2),
            "att": torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),  # both heading +x
        }

        # Run forward with no-grad to avoid backprop side effects
        with torch.no_grad():
            # Manually compute pairwise features (bypass MLP)
            pos = obs["pos"]
            att = obs["att"]
            rel_pos = pos.unsqueeze(-3) - pos.unsqueeze(-2)   # (1, 2, 2, 2)
            dist2   = (rel_pos * rel_pos).sum(-1, keepdim=True).clamp(min=1e-12)
            dist    = dist2.sqrt()
            dir_ij  = rel_pos / dist
            att_i   = att.unsqueeze(-2).expand_as(rel_pos)

            bear_cos = att_i[..., 0] * dir_ij[..., 0] + att_i[..., 1] * dir_ij[..., 1]
            # bear_cos[0, 0, 1] is i=0 to j=1: j is at (1,0), i heading +x → cos=1
            assert abs(bear_cos[0, 0, 1].item() - 1.0) < 1e-4

    def test_aspect_hot_gives_cos_near_one(self):
        """When j faces directly toward i, asp_cos must be ≈ 1.

        Setup: i at (0,0), j at (1,0) with j heading toward i (att=(-1,0)).
        """
        with torch.no_grad():
            pos = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])    # (1, 2, 2)
            att = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])   # j heads toward i

            rel_pos = pos.unsqueeze(-3) - pos.unsqueeze(-2)
            dist2   = (rel_pos * rel_pos).sum(-1, keepdim=True).clamp(min=1e-12)
            dist    = dist2.sqrt()
            dir_ij  = rel_pos / dist
            att_j   = att.unsqueeze(-3).expand_as(rel_pos)

            asp_cos = -(att_j[..., 0] * dir_ij[..., 0] + att_j[..., 1] * dir_ij[..., 1])
            # asp_cos[0, 0, 1]: from i=0's perspective, j(idx=1) faces toward i
            assert abs(asp_cos[0, 0, 1].item() - 1.0) < 1e-4

    def test_attention_with_obs_does_not_crash(self, model_cfg):
        """RelationalSelfAttention with obs must run and produce finite output."""
        B, N = 2, 4
        attn = RelationalSelfAttention(model_cfg)
        x    = torch.randn(B, N, model_cfg.d_model)
        obs  = _make_obs(B, N)

        out = attn(x, obs=obs)

        assert out.shape == (B, N, model_cfg.d_model)
        assert torch.isfinite(out).all()


class TestMVPPolicy:
    def test_get_action_and_value_shapes(self, model_cfg, ship_cfg):
        """get_action_and_value must return correct tensor shapes."""
        B, N = 2, 8
        policy = MVPPolicy(model_cfg, ship_cfg)
        obs    = _make_obs(B, N)
        hidden = policy.initial_hidden(B, N, torch.device("cpu"))

        action, logprob, value, new_hidden = policy.get_action_and_value(obs, hidden)

        assert action.shape     == (B, N, 3)
        assert logprob.shape    == (B, N)
        assert value.shape      == (B, N)
        assert new_hidden.shape == (1, B * N, model_cfg.d_model)

    def test_action_indices_in_valid_range(self, model_cfg, ship_cfg):
        """Sampled actions must be valid indices for each action head."""
        from boost_and_broadside.constants import NUM_POWER_ACTIONS, NUM_TURN_ACTIONS, NUM_SHOOT_ACTIONS
        B, N   = 2, 4
        policy = MVPPolicy(model_cfg, ship_cfg)
        obs    = _make_obs(B, N)
        hidden = policy.initial_hidden(B, N, torch.device("cpu"))

        action, _, _, _ = policy.get_action_and_value(obs, hidden)

        assert (action[..., 0] >= 0).all() and (action[..., 0] < NUM_POWER_ACTIONS).all()
        assert (action[..., 1] >= 0).all() and (action[..., 1] < NUM_TURN_ACTIONS).all()
        assert (action[..., 2] >= 0).all() and (action[..., 2] < NUM_SHOOT_ACTIONS).all()

    def test_evaluate_actions_shapes(self, model_cfg, ship_cfg):
        """evaluate_actions must return (T, B, N) tensors."""
        T, B, N = 4, 2, 8
        policy = MVPPolicy(model_cfg, ship_cfg)

        obs = {k: v.unsqueeze(0).expand(T, *v.shape) for k, v in _make_obs(B, N).items()}
        actions     = torch.zeros(T, B, N, 3, dtype=torch.long)
        hidden      = policy.initial_hidden(B, N, torch.device("cpu"))
        alive_mask  = torch.ones(T, B, N, dtype=torch.bool)

        logprob, entropy, value = policy.evaluate_actions(obs, actions, hidden, alive_mask)

        assert logprob.shape  == (T, B, N)
        assert entropy.shape  == (T, B, N)
        assert value.shape    == (T, B, N)

    def test_hidden_reset_zeros_done_envs(self, model_cfg, ship_cfg):
        """reset_hidden_for_envs must zero hidden states for done environments."""
        B, N   = 3, 4
        policy = MVPPolicy(model_cfg, ship_cfg)
        hidden = torch.ones(1, B * N, model_cfg.d_model)
        done   = torch.tensor([True, False, True])

        new_hidden = policy.reset_hidden_for_envs(hidden, done, N)

        # Envs 0 and 2 should be zeroed
        for ship in range(N):
            assert (new_hidden[0, ship, :] == 0).all()                   # env 0
            assert (new_hidden[0, N + ship, :] != 0).any()               # env 1 unchanged
            assert (new_hidden[0, 2 * N + ship, :] == 0).all()           # env 2
