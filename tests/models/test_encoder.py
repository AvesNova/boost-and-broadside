"""Tests for the ship encoder and policy forward passes."""

import math

import pytest
import torch

from boost_and_broadside.config import ModelConfig, ShipConfig
from boost_and_broadside.env.observation import ObsKey, YemongObservation
from boost_and_broadside.models.yemong.attention import TransformerBlock
from boost_and_broadside.models.yemong.encoder import ShipEncoder
from boost_and_broadside.models.yemong.policy import YemongPolicy
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


def _make_obs(B: int, N: int) -> YemongObservation:
    """Build a minimal random obs dict matching YemongEnvWrapper output."""

    return YemongObservation(
        data={
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
            ObsKey.LOCAL_LOG_INDEX: torch.zeros(B, N, 1),
            ObsKey.FIELD_TRANSITION_WIDTH: torch.zeros(B, N, 1),
            ObsKey.FIELD_INSIDE_LOG_INDEX: torch.zeros(B, N, 1),
            ObsKey.FIELD_OUTSIDE_LOG_INDEX: torch.zeros(B, N, 1),
            ObsKey.FIELD_LOG_INDEX_RATIO: torch.zeros(B, N, 1),
            ObsKey.FIELD_DAMAGE: torch.zeros(B, N, 1),
        }
    )


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

    def test_attn_mask_dtype_matches_query_under_autocast(self, model_cfg, monkeypatch):
        """The SDPA attn_mask must match the query dtype so the fused kernels apply.

        Regression: under bf16 autocast the qkv Linear emits bf16 while RMSNorm keeps
        x fp32. Building the mask from x's dtype yields an fp32 mask on bf16 q/k/v,
        which disqualifies the flash/mem-efficient SDPA kernels and silently falls
        back to the math kernel. The mask must follow q's dtype instead.
        """
        from boost_and_broadside.models.yemong import attention as attention_mod

        B, N = 2, 4
        block = TransformerBlock(model_cfg)
        x = torch.randn(B, N, model_cfg.d_model)
        alive = torch.ones(B, N, dtype=torch.bool)
        alive[0, 2] = False  # partial mask so attn_bias is actually built

        captured = {}
        real_sdpa = attention_mod.F.scaled_dot_product_attention

        def spy(q, k, v, attn_mask=None, **kwargs):
            captured["q_dtype"] = q.dtype
            captured["mask_dtype"] = None if attn_mask is None else attn_mask.dtype
            return real_sdpa(q, k, v, attn_mask=attn_mask, **kwargs)

        monkeypatch.setattr(attention_mod.F, "scaled_dot_product_attention", spy)

        # CPU autocast supports bf16 and reproduces the fp32-norm / bf16-Linear split.
        with torch.autocast("cpu", dtype=torch.bfloat16):
            block(x, alive_mask=alive)

        assert captured["mask_dtype"] is not None, "attn_bias was not built"
        assert captured["mask_dtype"] == captured["q_dtype"], (
            f"attn_mask dtype {captured['mask_dtype']} != query dtype "
            f"{captured['q_dtype']} — fused SDPA kernel would be disqualified"
        )


class TestRGLRU:
    """Verify forward_sequence (parallel scan) matches the sequential _step reference."""

    def _make_rglru(self, d_model: int = 32):
        from boost_and_broadside.models.yemong.griffin import RGLRU

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
        assert torch.allclose(scan_out, ref_out, atol=1e-4), (
            f"max diff: {(scan_out - ref_out).abs().max().item()}"
        )
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


class TestYemongPolicy:
    def test_field_tokens_carry_hidden_state_but_heads_emit_ships_only(
        self, model_cfg, coordinator
    ):
        B, N, M = 2, 3, 2
        policy = YemongPolicy(
            model_cfg, coordinator, num_value_components=NUM_VALUE_COMPONENTS, num_ships=N
        )
        obs = _make_obs(B, N + M)
        obs.data[ObsKey.TEAM_ID][:, N:] = 2
        hidden = policy.initial_hidden(B, N + M, torch.device("cpu"))

        action, logprob, value, pred_next, new_hidden = policy.get_action_and_value(obs, hidden)

        from boost_and_broadside.models.yemong.griffin import CONV_KERNEL

        assert action.shape == (B, N, 3)
        assert logprob.shape == (B, N)
        assert value.shape == (B, N, NUM_VALUE_COMPONENTS)
        assert pred_next.shape[:2] == (B, N)
        assert new_hidden.shape == (
            model_cfg.n_transformer_blocks,
            B * (N + M),
            CONV_KERNEL * model_cfg.d_model,
        )

    def test_get_action_and_value_shapes(self, model_cfg, coordinator):
        """get_action_and_value must return correct tensor shapes."""
        B, N = 2, 8
        policy = YemongPolicy(
            model_cfg, coordinator, num_value_components=NUM_VALUE_COMPONENTS, num_ships=N
        )
        obs = _make_obs(B, N)
        hidden = policy.initial_hidden(B, N, torch.device("cpu"))

        action, logprob, value, pred_next, new_hidden = policy.get_action_and_value(obs, hidden)

        K = NUM_VALUE_COMPONENTS
        assert action.shape == (B, N, 3)
        assert logprob.shape == (B, N)
        assert value.shape == (B, N, K)
        from boost_and_broadside.models.yemong.griffin import CONV_KERNEL

        assert new_hidden.shape == (
            model_cfg.n_transformer_blocks,
            B * N,
            CONV_KERNEL * model_cfg.d_model,
        )

    def test_action_indices_in_valid_range(self, model_cfg, coordinator):
        """Sampled actions must be valid indices for each action head."""
        from boost_and_broadside.constants import (
            NUM_POWER_ACTIONS,
            NUM_SHOOT_ACTIONS,
            NUM_TURN_ACTIONS,
        )

        B, N = 2, 4
        policy = YemongPolicy(
            model_cfg, coordinator, num_value_components=NUM_VALUE_COMPONENTS, num_ships=N
        )
        obs = _make_obs(B, N)
        hidden = policy.initial_hidden(B, N, torch.device("cpu"))

        action, _, _, _, _ = policy.get_action_and_value(obs, hidden)

        assert (action[..., 0] >= 0).all() and (action[..., 0] < NUM_POWER_ACTIONS).all()
        assert (action[..., 1] >= 0).all() and (action[..., 1] < NUM_TURN_ACTIONS).all()
        assert (action[..., 2] >= 0).all() and (action[..., 2] < NUM_SHOOT_ACTIONS).all()

    def test_evaluate_actions_shapes(self, model_cfg, coordinator):
        """evaluate_actions returns (T,B,N) logprob/entropy and (T,B,N,K) new_value."""
        T, B, N = 4, 2, 8
        policy = YemongPolicy(
            model_cfg, coordinator, num_value_components=NUM_VALUE_COMPONENTS, num_ships=N
        )
        K = NUM_VALUE_COMPONENTS

        obs_dict = {k: v.unsqueeze(0).expand(T, *v.shape) for k, v in _make_obs(B, N).items()}
        obs = YemongObservation(data=obs_dict)
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
        policy = YemongPolicy(
            model_cfg, coordinator, num_value_components=NUM_VALUE_COMPONENTS, num_ships=N
        )
        from boost_and_broadside.models.yemong.griffin import CONV_KERNEL

        hidden = torch.ones(model_cfg.n_transformer_blocks, B * N, CONV_KERNEL * model_cfg.d_model)
        done = torch.tensor([True, False, True])

        new_hidden = policy.reset_hidden_for_envs(hidden, done, N)

        # Envs 0 and 2 should be zeroed
        for ship in range(N):
            assert (new_hidden[0, ship, :] == 0).all()  # env 0
            assert (new_hidden[0, N + ship, :] != 0).any()  # env 1 unchanged
            assert (new_hidden[0, 2 * N + ship, :] == 0).all()  # env 2


class TestOrthogonalHeadInit:
    """AUDIT-011: head init locates Linear layers by type, not fixed index."""

    def test_finds_linears_regardless_of_position(self):
        """Inserting a non-Linear layer before the first Linear must not shift which
        module gets orthogonal-initialized (previously a hardcoded head[0]/head[3])."""
        import torch.nn as nn

        from boost_and_broadside.models.yemong.policy import _init_head_orthogonal

        torch.manual_seed(0)
        head = nn.Sequential(
            nn.Dropout(0.1),  # pushes the first Linear off index 0
            nn.Linear(4, 8),
            nn.RMSNorm(8),
            nn.GELU(),
            nn.Linear(8, 4),
        )
        _init_head_orthogonal(head)

        first_linear, last_linear = head[1], head[4]
        assert torch.allclose(first_linear.bias, torch.zeros_like(first_linear.bias))
        assert torch.allclose(last_linear.bias, torch.zeros_like(last_linear.bias))
        # orthogonal_ with gain sqrt(2) on an (8, 4) weight: columns are orthonormal
        # up to that gain, i.e. W^T W == gain^2 * I.
        gram = first_linear.weight.T @ first_linear.weight
        assert torch.allclose(gram, 2.0 * torch.eye(4), atol=1e-4)

    def test_policy_heads_are_orthogonal_initialized(self, model_cfg, coordinator):
        """YemongPolicy's actual heads (including the team_pma_k win/loss head) still
        get orthogonal-initialized end to end after the by-type refactor."""
        team_pma_k = (0, 1)
        policy = YemongPolicy(
            model_cfg,
            coordinator,
            num_value_components=NUM_VALUE_COMPONENTS,
            num_ships=4,
            team_pma_k=team_pma_k,
        )

        for head in [policy.action_head, policy.value_head_local, policy.value_head_win]:
            linears = [m for m in head if isinstance(m, torch.nn.Linear)]
            first, last = linears[0], linears[-1]
            assert torch.allclose(first.bias, torch.zeros_like(first.bias))
            assert torch.allclose(last.bias, torch.zeros_like(last.bias))
            out_f, in_f = first.weight.shape
            gram = first.weight.T @ first.weight if out_f >= in_f else first.weight @ first.weight.T
            n = min(out_f, in_f)
            assert torch.allclose(gram, 2.0 * torch.eye(n), atol=1e-3)


class TestFeatureCoordinatorLayout:
    """AUDIT-012: the five prediction-layout views derive from one cached spec."""

    def test_label_scale_vector_is_cached_per_device(self, coordinator):
        """label_scale_vector is constant per coordinator, so it is built once."""
        device = torch.device("cpu")
        first = coordinator.label_scale_vector(device)
        assert coordinator.label_scale_vector(device) is first

    def test_prediction_layout_widths_agree_across_views(self, coordinator):
        """Feature names, label scales, and target slices all size from one source.

        A drift between the previously-independent recomputations would surface as
        these widths disagreeing, so pinning them together guards the refactor.
        """
        names = coordinator.get_feature_names()
        scales = coordinator.label_scale_vector(torch.device("cpu"))
        target_end = max(sl.stop for sl in coordinator.target_slices().values())
        assert len(names) == coordinator.total_prediction_dimension
        assert scales.shape[0] == coordinator.total_prediction_dimension
        assert target_end == coordinator.total_target_dimension


class TestFeatureCoordinatorDecode:
    """AUDIT-022: decode_targets inverts each feature's target Transform."""

    def test_decode_targets_round_trips_raw_observation(self, ship_cfg):
        """Encoding an observation then decoding it must recover the raw channels.

        Exercises every invertible target Transform (Fourier, SymlogVelocity,
        Identity, Symlog, UnitCircle) through the public coordinator boundary that
        _decode_targets_to_obs relies on.
        """
        coordinator = build_standard_coordinator(ship_cfg)
        w, h = ship_cfg.world_size
        att_angle = 0.7
        obs = YemongObservation(
            data={
                ObsKey.POS: torch.tensor([[[0.3 * w, 0.65 * h]]]),
                ObsKey.VEL: torch.tensor([[[18.0, -7.0]]]),
                ObsKey.ATT: torch.tensor([[[math.cos(att_angle), math.sin(att_angle)]]]),
                ObsKey.ANG_VEL: torch.tensor([[[1.4]]]),
                ObsKey.HEALTH: torch.tensor([[[0.4 * ship_cfg.max_health]]]),
                ObsKey.POWER: torch.tensor([[[0.6 * ship_cfg.max_power]]]),
                ObsKey.COOLDOWN: torch.tensor([[[0.5 * ship_cfg.firing_cooldown]]]),
                ObsKey.LOCAL_LOG_INDEX: torch.tensor([[[0.25]]]),
            }
        )

        raw = coordinator.decode_targets(coordinator.get_target_vector(obs))
        pos = torch.cat([raw["position_x"], raw["position_y"]], dim=-1)

        assert torch.allclose(pos, obs[ObsKey.POS], atol=1e-2)
        assert torch.allclose(raw["velocity"], obs[ObsKey.VEL], atol=1e-2)
        assert torch.allclose(raw["angular_velocity"], obs[ObsKey.ANG_VEL], atol=1e-3)
        assert torch.allclose(raw["health"], obs[ObsKey.HEALTH], atol=1e-2)
        assert torch.allclose(raw["cooldown"], obs[ObsKey.COOLDOWN], atol=1e-4)


class TestGradCheckpoint:
    """grad_checkpoint must be a memory-only change: identical outputs and gradients."""

    def test_checkpoint_matches_non_checkpoint(self, model_cfg, coordinator):
        from dataclasses import replace

        T, B, N = 4, 2, 8
        K = NUM_VALUE_COMPONENTS
        torch.manual_seed(0)
        base = YemongPolicy(model_cfg, coordinator, num_value_components=K, num_ships=N)
        ckpt = YemongPolicy(
            replace(model_cfg, grad_checkpoint=True),
            coordinator,
            num_value_components=K,
            num_ships=N,
        )
        ckpt.load_state_dict(base.state_dict())  # identical weights

        obs_dict = {
            k: v.unsqueeze(0).expand(T, *v.shape).clone() for k, v in _make_obs(B, N).items()
        }
        obs = YemongObservation(data=obs_dict)
        actions = torch.zeros(T, B, N, 3, dtype=torch.long)
        hidden = base.initial_hidden(B, N, torch.device("cpu"))
        alive = torch.ones(T, B, N, dtype=torch.bool)

        lp0, _, val0, _, _, pn0 = base.evaluate_actions(obs, actions, hidden, alive)
        lp1, _, val1, _, _, pn1 = ckpt.evaluate_actions(obs, actions, hidden, alive)

        assert torch.allclose(lp0, lp1, atol=1e-6)
        assert torch.allclose(val0, val1, atol=1e-6)
        assert torch.allclose(pn0, pn1, atol=1e-6)

        # Gradients must match exactly — checkpointing recomputes the same forward.
        (lp0.sum() + val0.sum() + pn0.sum()).backward()
        (lp1.sum() + val1.sum() + pn1.sum()).backward()
        g0 = base.yemong_layers[0].temporal.linear1.weight.grad
        g1 = ckpt.yemong_layers[0].temporal.linear1.weight.grad
        assert g0 is not None and g1 is not None
        assert torch.allclose(g0, g1, atol=1e-5)
