"""Tests for the ship encoder and policy forward passes."""

import pytest
import torch

from boost_and_broadside.config import ShipConfig, ModelConfig, EnvConfig
from boost_and_broadside.models.mvp.encoder import ShipEncoder
from boost_and_broadside.models.mvp.attention import TransformerBlock
from boost_and_broadside.models.mvp.policy import MVPPolicy
from boost_and_broadside.env.observation import MVPObservation, ObsKey
from boost_and_broadside.train.rl.features import FeatureCoordinator, build_standard_coordinator


@pytest.fixture
def ship_cfg() -> ShipConfig:
    return ShipConfig()


@pytest.fixture
def model_cfg() -> ModelConfig:
    return ModelConfig(d_model=64, n_heads=4, n_transformer_blocks=2)


@pytest.fixture
def coordinator(ship_cfg) -> FeatureCoordinator:
    return build_standard_coordinator(ship_cfg)


NUM_VALUE_COMPONENTS = 12  # fixed K for encoder/policy unit tests


def _make_obs(B: int, N: int) -> MVPObservation:
    """Build a minimal random obs dict matching MVPEnvWrapper output."""

    return MVPObservation(data={
        ObsKey.POS: torch.rand(B, N, 2),
        ObsKey.VEL: torch.randn(B, N, 2) * 100,
        ObsKey.ATT: torch.randn(B, N, 2),
        ObsKey.ANG_VEL: torch.randn(B, N, 1),
        ObsKey.HEALTH: torch.rand(B, N, 1),
        ObsKey.POWER: torch.rand(B, N, 1),
        ObsKey.COOLDOWN: torch.rand(B, N, 1),
        ObsKey.TEAM_ID: torch.randint(0, 2, (B, N)),
        ObsKey.ALIVE: torch.ones(B, N, dtype=torch.bool),
        ObsKey.PREVIOUS_ACTION: torch.zeros(B, N, 3, dtype=torch.long),
        ObsKey.RADIUS: torch.rand(B, N, 1),
    })


class TestShipEncoder:
    def test_output_shape(self, coordinator, model_cfg):
        """Encoder must output (B, N, d_model) from standard obs."""
        B, N = 4, 8
        encoder = ShipEncoder(model_cfg, coordinator)
        obs = _make_obs(B, N)

        out = encoder(obs)

        assert out.shape == (B, N, model_cfg.d_model)

    def test_output_is_finite(self, coordinator, model_cfg):
        """No NaN or Inf in encoder output."""
        B, N = 2, 4
        encoder = ShipEncoder(model_cfg, coordinator)
        obs = _make_obs(B, N)

        out = encoder(obs)

        assert torch.isfinite(out).all()

    def test_dead_ships_produce_different_token(self, coordinator, model_cfg):
        """alive=False should affect the token (alive is a feature)."""
        B, N = 1, 2
        encoder = ShipEncoder(model_cfg, coordinator)

        obs_alive = _make_obs(B, N)
        obs_dead = {k: v.clone() for k, v in obs_alive.items()}
        obs_dead["alive"][0, 0] = False

        out_alive = encoder(obs_alive)
        out_dead = encoder(obs_dead)

        # Token 0 should differ; token 1 should be identical
        assert not torch.allclose(out_alive[0, 0], out_dead[0, 0])
        assert torch.allclose(out_alive[0, 1], out_dead[0, 1])

    def test_handles_batch_of_one(self, coordinator, model_cfg):
        """Encoder must work with B=1, N=1."""
        encoder = ShipEncoder(model_cfg, coordinator)
        obs = _make_obs(1, 1)
        out = encoder(obs)
        assert out.shape == (1, 1, model_cfg.d_model)


class TestTransformerBlock:
    def test_output_shape_unchanged(self, model_cfg):
        """Transformer block output must match input shape (B, N, D)."""
        B, N = 3, 8
        block = TransformerBlock(model_cfg)
        x = torch.randn(B, N, model_cfg.d_model)

        out = block(x)

        assert out.shape == (B, N, model_cfg.d_model)

    def test_alive_mask_does_not_crash(self, model_cfg):
        """Transformer block with a partial alive mask must not raise errors."""
        B, N = 2, 4
        block = TransformerBlock(model_cfg)
        x = torch.randn(B, N, model_cfg.d_model)
        alive = torch.ones(B, N, dtype=torch.bool)
        alive[0, 2] = False  # one dead ship

        out = block(x, alive_mask=alive)

        assert out.shape == (B, N, model_cfg.d_model)
        assert torch.isfinite(out).all()

    def test_all_dead_mask_does_not_produce_nan(self, model_cfg):
        """All-dead alive mask should not produce NaN (edge case)."""
        B, N = 1, 4
        block = TransformerBlock(model_cfg)
        x = torch.randn(B, N, model_cfg.d_model)
        alive = torch.zeros(B, N, dtype=torch.bool)  # everyone dead

        out = block(x, alive_mask=alive)

        # May be all-zero or garbage but must not be NaN
        assert not torch.isnan(out).any()


class TestRGLRU:
    """Verify forward_sequence (parallel scan) matches the sequential _step reference."""

    def _make_rglru(self, d_model: int = 32):
        from boost_and_broadside.models.mvp.griffin import RGLRU
        torch.manual_seed(0)
        return RGLRU(d_model).eval()

    def _sequential_reference(self, rglru, x_seq, h0):
        T = x_seq.shape[1]
        h = h0.clone()
        outputs = []
        with torch.no_grad():
            for t in range(T):
                h = rglru._step(x_seq[:, t], h)
                outputs.append(h)
        return torch.stack(outputs, dim=1), h

    def test_matches_sequential_T128(self):
        B, T, D = 4, 128, 32
        rglru = self._make_rglru(D)
        torch.manual_seed(1)
        x_seq = torch.randn(B, T, D)
        h0 = torch.randn(B, D)
        ref_out, ref_h = self._sequential_reference(rglru, x_seq, h0)
        with torch.no_grad():
            scan_out, scan_h = rglru.forward_sequence(x_seq, h0)
        assert scan_out.shape == (B, T, D)
        assert torch.allclose(scan_out, ref_out, atol=1e-4), \
            f"max diff: {(scan_out - ref_out).abs().max().item()}"
        assert torch.allclose(scan_h, ref_h, atol=1e-4)

    def test_matches_sequential_T1(self):
        """T=1 path used during rollout via YemongBlock.step."""
        B, T, D = 8, 1, 32
        rglru = self._make_rglru(D)
        torch.manual_seed(2)
        x_seq = torch.randn(B, T, D)
        h0 = torch.randn(B, D)
        ref_out, ref_h = self._sequential_reference(rglru, x_seq, h0)
        with torch.no_grad():
            scan_out, scan_h = rglru.forward_sequence(x_seq, h0)
        assert scan_out.shape == (B, 1, D)
        assert torch.allclose(scan_out, ref_out, atol=1e-5)
        assert torch.allclose(scan_h, ref_h, atol=1e-5)

    def test_done_mask_resets_hidden(self):
        """done_mask=True at step t must zero h[t+1]'s dependence on prior state."""
        B, T, D = 2, 8, 32
        rglru = self._make_rglru(D)
        torch.manual_seed(4)
        x_seq = torch.randn(B, T, D)
        h0 = torch.randn(B, D)

        # Place a done at step 3 for env 0 only.
        done_mask = torch.zeros(B, T, dtype=torch.bool)
        done_mask[0, 3] = True

        with torch.no_grad():
            out_with_done, _ = rglru.forward_sequence(x_seq, h0, done_mask)
            out_no_done, _ = rglru.forward_sequence(x_seq, h0)

        # Steps 0-3 for env 0: done is at step 3, resets affect step 4+.
        # Steps 0-3 should be identical between the two runs.
        assert torch.allclose(out_with_done[0, :4], out_no_done[0, :4], atol=1e-6)

        # Step 4+ for env 0 must differ (h0 influence is cut).
        assert not torch.allclose(out_with_done[0, 4:], out_no_done[0, 4:], atol=1e-6)

        # Env 1 (no done) must be completely unaffected.
        assert torch.allclose(out_with_done[1], out_no_done[1], atol=1e-6)

        # Verify the reset is real: rerun from step 4 with h=0 and it must match.
        h_zero = torch.zeros(B, D)
        out_fresh, _ = rglru.forward_sequence(x_seq[:, 4:], h_zero)
        assert torch.allclose(out_with_done[0, 4:], out_fresh[0], atol=1e-5)

    def test_bfloat16_outputs_finite(self):
        """Outputs must be finite under the training dtype."""
        B, T, D = 4, 128, 32
        rglru = self._make_rglru(D).bfloat16()
        torch.manual_seed(3)
        x_seq = torch.randn(B, T, D, dtype=torch.bfloat16)
        h0 = torch.randn(B, D, dtype=torch.bfloat16)
        with torch.no_grad():
            scan_out, scan_h = rglru.forward_sequence(x_seq, h0)
        assert torch.isfinite(scan_out).all()
        assert torch.isfinite(scan_h).all()


class TestMVPPolicy:
    def test_get_action_and_value_shapes(self, model_cfg, coordinator):
        """get_action_and_value must return correct tensor shapes."""
        B, N = 2, 8
        policy = MVPPolicy(
            model_cfg, coordinator, num_value_components=NUM_VALUE_COMPONENTS, num_ships=N
        )
        obs = _make_obs(B, N)
        hidden = policy.initial_hidden(B, N, torch.device("cpu"))

        action, logprob, value, pred_next, new_hidden = policy.get_action_and_value(obs, hidden)

        K = NUM_VALUE_COMPONENTS
        assert action.shape == (B, N, 3)
        assert logprob.shape == (B, N)
        assert value.shape == (B, N, K)
        from boost_and_broadside.models.mvp.griffin import CONV_KERNEL
        assert new_hidden.shape == (model_cfg.n_transformer_blocks, B * N, CONV_KERNEL * model_cfg.d_model)

    def test_action_indices_in_valid_range(self, model_cfg, coordinator):
        """Sampled actions must be valid indices for each action head."""
        from boost_and_broadside.constants import (
            NUM_POWER_ACTIONS,
            NUM_TURN_ACTIONS,
            NUM_SHOOT_ACTIONS,
        )

        B, N = 2, 4
        policy = MVPPolicy(
            model_cfg, coordinator, num_value_components=NUM_VALUE_COMPONENTS, num_ships=N
        )
        obs = _make_obs(B, N)
        hidden = policy.initial_hidden(B, N, torch.device("cpu"))

        action, _, _, _, _ = policy.get_action_and_value(obs, hidden)

        assert (action[..., 0] >= 0).all() and (
            action[..., 0] < NUM_POWER_ACTIONS
        ).all()
        assert (action[..., 1] >= 0).all() and (action[..., 1] < NUM_TURN_ACTIONS).all()
        assert (action[..., 2] >= 0).all() and (
            action[..., 2] < NUM_SHOOT_ACTIONS
        ).all()

    def test_evaluate_actions_shapes(self, model_cfg, coordinator):
        """evaluate_actions must return (T, B, N) for logprob/entropy and (T, B, N, K) for new_value."""
        T, B, N = 4, 2, 8
        policy = MVPPolicy(
            model_cfg, coordinator, num_value_components=NUM_VALUE_COMPONENTS, num_ships=N
        )
        K = NUM_VALUE_COMPONENTS

        obs_dict = {
            k: v.unsqueeze(0).expand(T, *v.shape) for k, v in _make_obs(B, N).items()
        }
        obs = MVPObservation(data=obs_dict)
        actions = torch.zeros(T, B, N, 3, dtype=torch.long)
        hidden = policy.initial_hidden(B, N, torch.device("cpu"))
        alive_mask = torch.ones(T, B, N, dtype=torch.bool)

        logprob, entropy, new_value, logits, _, _ = policy.evaluate_actions(
            obs, actions, hidden, alive_mask
        )

        assert logprob.shape == (T, B, N)
        assert entropy.shape == (T, B, N)
        assert new_value.shape == (T, B, N, K)
        assert logits.shape == (T, B, N, 12)

    def test_hidden_reset_zeros_done_envs(self, model_cfg, coordinator):
        """reset_hidden_for_envs must zero hidden states for done environments."""
        B, N = 3, 4
        policy = MVPPolicy(
            model_cfg, coordinator, num_value_components=NUM_VALUE_COMPONENTS, num_ships=N
        )
        from boost_and_broadside.models.mvp.griffin import CONV_KERNEL
        hidden = torch.ones(model_cfg.n_transformer_blocks, B * N, CONV_KERNEL * model_cfg.d_model)
        done = torch.tensor([True, False, True])

        new_hidden = policy.reset_hidden_for_envs(hidden, done, N)

        # Envs 0 and 2 should be zeroed
        for ship in range(N):
            assert (new_hidden[0, ship, :] == 0).all()  # env 0
            assert (new_hidden[0, N + ship, :] != 0).any()  # env 1 unchanged
            assert (new_hidden[0, 2 * N + ship, :] == 0).all()  # env 2
