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
    return ModelConfig(d_model=64, n_heads=4, n_yemong_blocks=2)


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
            ObsKey.LOCAL_INDEX_GRADIENT: torch.zeros(B, N, 2),
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
    def test_only_ships_carry_hidden_state_and_heads_emit_ships_only(self, model_cfg, coordinator):
        """Fields participate in attention but take the non-recurrent path.

        Their state is static within an episode, so allocating recurrent slots for
        them would cost a third of the hidden tensor for nothing.
        """
        B, N, M = 2, 3, 2
        policy = YemongPolicy(
            model_cfg,
            coordinator,
            num_value_components=NUM_VALUE_COMPONENTS,
            num_ships=N,
            team_pma_k=(),
        )
        obs = _make_obs(B, N + M)
        obs.data[ObsKey.TEAM_ID][:, N:] = 2
        hidden = policy.initial_hidden(B, policy.num_recurrent_tokens, torch.device("cpu"))

        action, logprob, value, pred_next, new_hidden = policy.get_action_and_value(obs, hidden)

        from boost_and_broadside.models.yemong.griffin import CONV_KERNEL

        assert policy.num_recurrent_tokens == N
        assert action.shape == (B, N, 3)
        assert logprob.shape == (B, N)
        assert value.shape == (B, N, NUM_VALUE_COMPONENTS)
        assert pred_next.shape[:2] == (B, N)
        assert new_hidden.shape == (
            model_cfg.n_yemong_blocks,
            B * N,  # ships only — not B*(N+M)
            CONV_KERNEL * model_cfg.d_model,
        )

    def test_get_action_and_value_shapes(self, model_cfg, coordinator):
        """get_action_and_value must return correct tensor shapes."""
        B, N = 2, 8
        policy = YemongPolicy(
            model_cfg,
            coordinator,
            num_value_components=NUM_VALUE_COMPONENTS,
            num_ships=N,
            team_pma_k=(),
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
            model_cfg.n_yemong_blocks,
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
            model_cfg,
            coordinator,
            num_value_components=NUM_VALUE_COMPONENTS,
            num_ships=N,
            team_pma_k=(),
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
            model_cfg,
            coordinator,
            num_value_components=NUM_VALUE_COMPONENTS,
            num_ships=N,
            team_pma_k=(),
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
            model_cfg,
            coordinator,
            num_value_components=NUM_VALUE_COMPONENTS,
            num_ships=N,
            team_pma_k=(),
        )
        from boost_and_broadside.models.yemong.griffin import CONV_KERNEL

        hidden = torch.ones(model_cfg.n_yemong_blocks, B * N, CONV_KERNEL * model_cfg.d_model)
        done = torch.tensor([True, False, True])

        new_hidden = policy.reset_hidden_for_envs(hidden, done, N)

        # Envs 0 and 2 should be zeroed
        for ship in range(N):
            assert (new_hidden[0, ship, :] == 0).all()  # env 0
            assert (new_hidden[0, N + ship, :] != 0).any()  # env 1 unchanged
            assert (new_hidden[0, 2 * N + ship, :] == 0).all()  # env 2


class TestYemongBlockStructure:
    """Configurable spatial/temporal sublayer counts inside each Yemong block."""

    @staticmethod
    def _policy(coordinator, n_blocks: int, n_spatial: int, n_temporal: int, num_ships: int):
        cfg = ModelConfig(
            d_model=64,
            n_heads=4,
            n_yemong_blocks=n_blocks,
            n_spatial_per_block=n_spatial,
            n_temporal_per_block=n_temporal,
        )
        torch.manual_seed(0)
        policy = YemongPolicy(
            cfg,
            coordinator,
            num_value_components=NUM_VALUE_COMPONENTS,
            num_ships=num_ships,
            team_pma_k=(),
        )
        return cfg, policy.eval()

    @pytest.mark.parametrize(
        ("n_blocks", "n_spatial", "n_temporal"),
        [(2, 1, 1), (2, 2, 1), (1, 4, 2), (3, 2, 1), (2, 2, 0)],
    )
    def test_sublayer_counts_match_config(self, coordinator, n_blocks, n_spatial, n_temporal):
        _, policy = self._policy(coordinator, n_blocks, n_spatial, n_temporal, num_ships=3)
        assert len(policy.yemong_layers) == n_blocks
        for block in policy.yemong_layers:
            assert len(block.spatial) == n_spatial
            assert len(block.temporal) == n_temporal

    @pytest.mark.parametrize(
        ("n_blocks", "n_temporal"),
        [(2, 1), (2, 2), (1, 3), (3, 1), (2, 0)],
    )
    def test_hidden_state_has_one_slot_per_temporal_sublayer(
        self, coordinator, n_blocks, n_temporal
    ):
        from boost_and_broadside.models.yemong.griffin import CONV_KERNEL

        B, N, M = 2, 3, 2
        cfg, policy = self._policy(coordinator, n_blocks, 2, n_temporal, num_ships=N)
        hidden = policy.initial_hidden(B, N + M, torch.device("cpu"))

        assert hidden.shape == (
            n_blocks * n_temporal,
            B * (N + M),
            CONV_KERNEL * cfg.d_model,
        )
        assert cfg.n_hidden_layers == n_blocks * n_temporal

        obs = _make_obs(B, N + M)
        obs.data[ObsKey.TEAM_ID][:, N:] = 2
        _, _, _, _, new_hidden = policy.get_action_and_value(obs, hidden)
        assert new_hidden.shape == hidden.shape

    @pytest.mark.parametrize(
        ("n_blocks", "n_spatial", "n_temporal"),
        [(2, 1, 1), (2, 2, 1), (1, 3, 2)],
    )
    def test_step_matches_sequence(self, coordinator, n_blocks, n_spatial, n_temporal):
        """Rollout stepping and full-sequence re-evaluation must agree.

        This is the load-bearing property of the trunk: PPO collects with
        ``get_action_and_value`` one step at a time, then re-evaluates the whole
        rollout with ``evaluate_actions``. If the two paths diverge, the ratio
        ``exp(new_logprob - old_logprob)`` is wrong and the update is silently
        corrupted, so it is pinned across sublayer configurations.
        """
        B, N, M, T = 2, 3, 2, 6
        _, policy = self._policy(coordinator, n_blocks, n_spatial, n_temporal, num_ships=N)

        torch.manual_seed(7)
        obs_seq = []
        for _ in range(T):
            obs = _make_obs(B, N + M)
            obs.data[ObsKey.TEAM_ID][:, N:] = 2
            obs_seq.append(obs)

        initial_hidden = policy.initial_hidden(B, N + M, torch.device("cpu"))

        # Step path: advance one timestep at a time, carrying hidden forward.
        hidden = initial_hidden
        step_values, actions = [], []
        for obs in obs_seq:
            action, _, value, _, hidden = policy.get_action_and_value(obs, hidden)
            step_values.append(value)
            actions.append(action)
        step_value = torch.stack(step_values, dim=0)  # (T, B, N, K)

        # Sequence path: one parallel re-evaluation from the same initial state.
        stacked = YemongObservation(
            data={
                key: torch.stack([o.data[key] for o in obs_seq], dim=0) for key in obs_seq[0].data
            }
        )
        alive_mask = stacked.data[ObsKey.ALIVE]  # (T, B, N+M)
        with torch.no_grad():
            _, _, seq_value, _, _, _ = policy.evaluate_actions(
                stacked, torch.stack(actions, dim=0), initial_hidden, alive_mask
            )

        assert torch.allclose(step_value, seq_value, atol=1e-5), (
            f"max diff: {(step_value - seq_value).abs().max().item()}"
        )

    def test_step_matches_sequence_across_an_episode_boundary(self, coordinator):
        """The same equivalence, with an episode ending mid-rollout.

        Regression: ``reset_hidden_for_envs`` zeroes the whole packed hidden —
        RG-LRU state *and* causal-conv buffer — but the sequence path applied
        ``done_mask`` only inside the RG-LRU. The conv therefore kept reading the
        previous episode's inputs for CONV_KERNEL-1 steps after every boundary,
        so re-evaluation disagreed with rollout exactly at episode starts and the
        PPO ratio was wrong there.
        """
        B, N, M, T = 2, 3, 2, 8
        done_step = 3  # episode ends here; step done_step+1 starts fresh
        _, policy = self._policy(coordinator, n_blocks=2, n_spatial=2, n_temporal=1, num_ships=N)

        torch.manual_seed(11)
        obs_seq = []
        for _ in range(T):
            obs = _make_obs(B, N + M)
            obs.data[ObsKey.TEAM_ID][:, N:] = 2
            obs_seq.append(obs)

        done_mask = torch.zeros(T, B, dtype=torch.bool)
        done_mask[done_step] = True

        initial_hidden = policy.initial_hidden(B, N + M, torch.device("cpu"))

        # Step path: reset the hidden state after the env reports done, exactly
        # as the rollout loop does.
        hidden = initial_hidden
        step_values, actions = [], []
        for t, obs in enumerate(obs_seq):
            action, _, value, _, hidden = policy.get_action_and_value(obs, hidden)
            step_values.append(value)
            actions.append(action)
            hidden = policy.reset_hidden_for_envs(hidden, done_mask[t], N + M)
        step_value = torch.stack(step_values, dim=0)

        stacked = YemongObservation(
            data={
                key: torch.stack([o.data[key] for o in obs_seq], dim=0) for key in obs_seq[0].data
            }
        )
        with torch.no_grad():
            _, _, seq_value, _, _, _ = policy.evaluate_actions(
                stacked,
                torch.stack(actions, dim=0),
                initial_hidden,
                stacked.data[ObsKey.ALIVE],
                done_mask=done_mask,
            )

        assert torch.allclose(step_value, seq_value, atol=1e-5), (
            f"max diff: {(step_value - seq_value).abs().max().item()}"
        )

    def test_defaults_reproduce_single_sublayer_trunk(self, coordinator):
        """Omitting the new knobs must give the pre-refactor 1S+1T structure."""
        cfg = ModelConfig(d_model=64, n_heads=4, n_yemong_blocks=2)
        assert cfg.n_spatial_per_block == 1
        assert cfg.n_temporal_per_block == 1
        assert cfg.n_hidden_layers == 2

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_yemong_blocks": -1},
            {"n_yemong_blocks": 2, "n_spatial_per_block": -1},
            {"n_yemong_blocks": 2, "n_temporal_per_block": -1},
        ],
    )
    def test_rejects_negative_sublayer_counts(self, kwargs):
        kwargs.setdefault("n_yemong_blocks", 2)
        with pytest.raises(ValueError):
            ModelConfig(d_model=64, n_heads=4, **kwargs)


def _make_bullets(B: int, NB: int, active: bool = True) -> dict:
    """Random bullet-axis channels matching bullet_observation_from_state output."""
    from boost_and_broadside.env.observation import BulletObsKey

    return {
        BulletObsKey.POS: torch.rand(B, NB, 2) * 1024.0,
        BulletObsKey.VEL: torch.randn(B, NB, 2) * 400.0,
        BulletObsKey.DAMAGE: torch.rand(B, NB, 1),
        BulletObsKey.LIFETIME: torch.rand(B, NB, 1),
        BulletObsKey.LOCAL_LOG_INDEX: torch.randn(B, NB, 1) * 0.1,
        BulletObsKey.LOCAL_INDEX_GRADIENT: torch.randn(B, NB, 2) * 0.1,
        BulletObsKey.TEAM_ID: torch.randint(0, 2, (B, NB)),
        BulletObsKey.ACTIVE: torch.full((B, NB), active, dtype=torch.bool),
    }


class TestBulletCrossAttention:
    """Bullets are key/value-only tokens: never queried, never recurrent."""

    @staticmethod
    def _cfg(n_cross: int, n_spatial: int = 2) -> ModelConfig:
        return ModelConfig(
            d_model=64,
            n_heads=4,
            n_yemong_blocks=2,
            n_spatial_per_block=n_spatial,
            n_temporal_per_block=1,
            n_bullet_cross_per_block=n_cross,
        )

    def _policy(self, ship_cfg, coordinator, n_cross, N):
        from boost_and_broadside.train.rl.features import build_bullet_coordinator

        torch.manual_seed(0)
        return YemongPolicy(
            self._cfg(n_cross),
            coordinator,
            num_value_components=NUM_VALUE_COMPONENTS,
            num_ships=N,
            team_pma_k=(),
            bullet_coordinator=build_bullet_coordinator(ship_cfg) if n_cross else None,
        ).eval()

    def _obs(self, B, N, M, NB, active=True):
        obs = _make_obs(B, N + M)
        obs.data[ObsKey.TEAM_ID][:, N:] = 2
        return YemongObservation(data=obs.data, bullets=_make_bullets(B, NB, active))

    def test_bullet_observation_flattens_the_ring_buffer(self, ship_cfg):
        """Every slot is emitted; inactive ones are masked, not compacted."""
        from boost_and_broadside.config import EnvConfig
        from boost_and_broadside.config.defaults import REWARDS
        from boost_and_broadside.env.observation import BulletObsKey, observation_from_state
        from boost_and_broadside.env.wrapper import YemongEnvWrapper

        B, N, K = 2, 4, 10
        env_cfg = EnvConfig(num_ships=N, max_bullets=K, max_episode_steps=32)
        wrapper = YemongEnvWrapper(B, ship_cfg, env_cfg, REWARDS, "cpu")
        wrapper.reset()

        obs = observation_from_state(wrapper.env.state, ship_cfg, include_bullets=True)

        assert obs.bullets is not None
        assert obs.bullets[BulletObsKey.POS].shape == (B, N * K, 2)
        assert obs.bullets[BulletObsKey.ACTIVE].shape == (B, N * K)
        # Ring-buffer slots start empty, and log(0) must not leak a -inf.
        assert torch.isfinite(obs.bullets[BulletObsKey.LOCAL_LOG_INDEX]).all()

    def test_bullet_position_shares_the_ship_frequency_basis(self, ship_cfg):
        """Attention computes relative geometry only if both bases match.

        ``q.k`` over Fourier features reduces to a function of the displacement
        only when ship and bullet positions expand on one shared basis. Mismatched
        frequencies leave cross terms that never form relative geometry.
        """
        from boost_and_broadside.train.rl.features import (
            build_bullet_coordinator,
            build_standard_coordinator,
        )

        ship_feats = {f.name: f for f in build_standard_coordinator(ship_cfg).features}
        bullet_feats = {f.name: f for f in build_bullet_coordinator(ship_cfg).features}

        for ship_name, bullet_name in [
            ("position_x", "bullet_position_x"),
            ("position_y", "bullet_position_y"),
        ]:
            ship_enc = ship_feats[ship_name].input_encoder
            bullet_enc = bullet_feats[bullet_name].input_encoder
            assert ship_enc.n_freqs == bullet_enc.n_freqs
            assert ship_enc.periods == bullet_enc.periods

        # Velocity is compared against the ship's own in the FFN, so it matches too.
        assert type(ship_feats["velocity"].input_encoder) is type(
            bullet_feats["bullet_velocity"].input_encoder
        )

    def test_no_ship_index_onehot_in_bullet_features(self, ship_cfg):
        """A per-ship one-hot would fix N in the weights and kill zero-shot transfer."""
        from boost_and_broadside.train.rl.features import OneHot, build_bullet_coordinator

        for feature in build_bullet_coordinator(ship_cfg).features:
            if isinstance(feature.input_encoder, OneHot):
                assert feature.input_encoder.n <= 3, (
                    f"{feature.name} one-hot of width {feature.input_encoder.n} looks "
                    "like a per-ship index — that would fix the fleet size"
                )

    def test_inactive_bullets_do_not_influence_output(self, ship_cfg, coordinator):
        """Masked ring-buffer slots must contribute nothing, whatever they contain."""
        B, N, M, NB = 2, 3, 2, 12
        policy = self._policy(ship_cfg, coordinator, n_cross=1, N=N)

        torch.manual_seed(5)
        obs = self._obs(B, N, M, NB, active=False)
        hidden = policy.initial_hidden(B, N, torch.device("cpu"))
        _, _, base_value, _, _ = policy.get_action_and_value(obs, hidden)

        # Same masked-out slots, wildly different contents.
        garbled = dict(obs.bullets)
        for key, val in garbled.items():
            if val.dtype.is_floating_point:
                garbled[key] = val + 500.0
        moved = YemongObservation(data=obs.data, bullets=garbled)
        _, _, moved_value, _, _ = policy.get_action_and_value(moved, hidden)

        assert torch.allclose(base_value, moved_value, atol=1e-5)

    def test_partially_masked_bullets_ignore_only_the_dead_slots(self, ship_cfg, coordinator):
        """Perturbing masked slots must not move the output when others are live.

        The all-masked case is handled by an explicit gate; this covers the path
        where the softmax genuinely runs and the bias must suppress dead keys.
        """
        from boost_and_broadside.env.observation import BulletObsKey

        B, N, M, NB = 2, 3, 2, 12
        policy = self._policy(ship_cfg, coordinator, n_cross=1, N=N)

        torch.manual_seed(5)
        obs = self._obs(B, N, M, NB, active=True)
        active = obs.bullets[BulletObsKey.ACTIVE].clone()
        active[:, NB // 2 :] = False  # second half dead
        base_bullets = dict(obs.bullets)
        base_bullets[BulletObsKey.ACTIVE] = active
        base = YemongObservation(data=obs.data, bullets=base_bullets)

        garbled = dict(base_bullets)
        for key, val in garbled.items():
            if val.dtype.is_floating_point:
                val = val.clone()
                val[:, NB // 2 :] += 500.0  # only the dead half
                garbled[key] = val
        moved = YemongObservation(data=obs.data, bullets=garbled)

        hidden = policy.initial_hidden(B, N, torch.device("cpu"))
        _, _, base_value, _, _ = policy.get_action_and_value(base, hidden)
        _, _, moved_value, _, _ = policy.get_action_and_value(moved, hidden)

        assert torch.allclose(base_value, moved_value, atol=1e-5)

    def test_active_bullets_do_influence_output(self, ship_cfg, coordinator):
        """The mask test is only meaningful if active bullets actually matter."""
        B, N, M, NB = 2, 3, 2, 12
        policy = self._policy(ship_cfg, coordinator, n_cross=1, N=N)

        torch.manual_seed(5)
        obs = self._obs(B, N, M, NB, active=True)
        hidden = policy.initial_hidden(B, N, torch.device("cpu"))
        _, _, base_value, _, _ = policy.get_action_and_value(obs, hidden)

        moved_bullets = dict(obs.bullets)
        moved_bullets[list(moved_bullets)[0]] = moved_bullets[list(moved_bullets)[0]] + 300.0
        moved = YemongObservation(data=obs.data, bullets=moved_bullets)
        _, _, moved_value, _, _ = policy.get_action_and_value(moved, hidden)

        assert not torch.allclose(base_value, moved_value, atol=1e-5)

    def test_bullets_add_no_recurrent_state(self, ship_cfg, coordinator):
        B, N, M, NB = 2, 3, 2, 12
        policy = self._policy(ship_cfg, coordinator, n_cross=1, N=N)
        hidden = policy.initial_hidden(B, N, torch.device("cpu"))
        obs = self._obs(B, N, M, NB)

        _, _, _, _, new_hidden = policy.get_action_and_value(obs, hidden)

        assert new_hidden.shape == hidden.shape
        assert hidden.shape[1] == B * N  # not B*(N+M), and not B*(N+M+NB)

    @pytest.mark.parametrize("n_cross", [1, 2])
    def test_step_matches_sequence_with_bullets(self, ship_cfg, coordinator, n_cross):
        B, N, M, NB, T = 2, 3, 2, 12, 5
        policy = self._policy(ship_cfg, coordinator, n_cross=n_cross, N=N)

        torch.manual_seed(9)
        obs_seq = [self._obs(B, N, M, NB) for _ in range(T)]
        initial_hidden = policy.initial_hidden(B, N, torch.device("cpu"))

        hidden = initial_hidden
        step_values, actions = [], []
        for obs in obs_seq:
            action, _, value, _, hidden = policy.get_action_and_value(obs, hidden)
            step_values.append(value)
            actions.append(action)

        stacked = YemongObservation(
            data={
                key: torch.stack([o.data[key] for o in obs_seq], dim=0) for key in obs_seq[0].data
            },
            bullets={
                key: torch.stack([o.bullets[key] for o in obs_seq], dim=0)
                for key in obs_seq[0].bullets
            },
        )
        with torch.no_grad():
            _, _, seq_value, _, _, _ = policy.evaluate_actions(
                stacked,
                torch.stack(actions, dim=0),
                initial_hidden,
                stacked.data[ObsKey.ALIVE],
            )

        assert torch.allclose(torch.stack(step_values, dim=0), seq_value, atol=1e-5)

    def test_flip_team_flips_bullet_teams(self):
        from boost_and_broadside.env.observation import BulletObsKey

        B, N, M, NB = 2, 3, 2, 6
        obs = _make_obs(B, N + M)
        obs.data[ObsKey.TEAM_ID][:, N:] = 2
        bullets = _make_bullets(B, NB)
        bullets[BulletObsKey.TEAM_ID] = torch.tensor([[0, 1, 0, 1, 0, 1]] * B)
        obs = YemongObservation(data=obs.data, bullets=bullets)

        flipped = obs.flip_team(N)

        assert torch.equal(
            flipped.bullets[BulletObsKey.TEAM_ID], torch.tensor([[1, 0, 1, 0, 1, 0]] * B)
        )
        # Field tokens keep team 2 through the flip.
        assert (flipped.data[ObsKey.TEAM_ID][:, N:] == 2).all()

    def test_masked_flip_team_flips_ships_and_bullets_in_the_same_envs(self):
        """Per-env perspective: the flip must reach the bullet axis too.

        Post-hoc calibration mirrors team IDs only in the envs where a policy
        plays team 1. Flipping ships there without flipping bullet shooter teams
        shows that policy its own fire as the enemy's — silently, since a bullet
        carries no other identity.
        """
        from boost_and_broadside.env.observation import BulletObsKey

        B, N, M, NB = 2, 3, 2, 4
        obs = _make_obs(B, N + M)
        obs.data[ObsKey.TEAM_ID][:, :N] = torch.tensor([[0, 1, 0]] * B)
        obs.data[ObsKey.TEAM_ID][:, N:] = 2
        bullets = _make_bullets(B, NB)
        bullets[BulletObsKey.TEAM_ID] = torch.tensor([[0, 1, 0, 1]] * B)
        obs = YemongObservation(data=obs.data, bullets=bullets)

        # Env 0 flips, env 1 does not.
        flipped = obs.flip_team(N, mask=torch.tensor([True, False]))

        assert torch.equal(flipped.data[ObsKey.TEAM_ID][0, :N], torch.tensor([1, 0, 1]))
        assert torch.equal(flipped.data[ObsKey.TEAM_ID][1, :N], torch.tensor([0, 1, 0]))
        assert torch.equal(flipped.bullets[BulletObsKey.TEAM_ID][0], torch.tensor([1, 0, 1, 0]))
        assert torch.equal(flipped.bullets[BulletObsKey.TEAM_ID][1], torch.tensor([0, 1, 0, 1]))
        # Field tokens keep team 2 in both.
        assert (flipped.data[ObsKey.TEAM_ID][:, N:] == 2).all()

    def test_structural_ops_carry_bullets(self):
        B, N, M, NB = 4, 3, 2, 6
        obs = YemongObservation(data=_make_obs(B, N + M).data, bullets=_make_bullets(B, NB))

        assert obs.slice_envs(slice(0, 2)).bullets[list(obs.bullets)[0]].shape[0] == 2
        assert obs.concat_batch(obs).bullets[list(obs.bullets)[0]].shape[0] == 2 * B
        assert obs.map(lambda t: t).bullets is not None
        # Bullets are not on the entity-token axis, so token slicing leaves them alone.
        assert obs.slice_tokens(0, N).bullets is obs.bullets

    def test_zero_bullet_cross_builds_no_bullet_encoder(self, ship_cfg, coordinator):
        policy = self._policy(ship_cfg, coordinator, n_cross=0, N=3)
        assert policy.bullet_encoder is None
        for block in policy.yemong_layers:
            assert not any(layer.reads_bullets for layer in block.spatial)

    def test_bullet_reads_are_counted_from_the_first_spatial_layer(self, ship_cfg, coordinator):
        """A read must precede a later spatial layer, or ships can only see fire
        aimed at themselves — never fire aimed at an ally they might support."""
        policy = self._policy(ship_cfg, coordinator, n_cross=1, N=3)
        for block in policy.yemong_layers:
            assert block.spatial[0].reads_bullets
            assert not block.spatial[1].reads_bullets

    def test_runs_at_unseen_fleet_sizes(self, ship_cfg, coordinator):
        """Zero-shot transfer: nothing in the bullet path may fix N or N*K."""
        from boost_and_broadside.train.rl.features import build_bullet_coordinator

        torch.manual_seed(0)
        policy = YemongPolicy(
            self._cfg(1),
            coordinator,
            num_value_components=NUM_VALUE_COMPONENTS,
            num_ships=4,
            team_pma_k=(),
            bullet_coordinator=build_bullet_coordinator(ship_cfg),
        ).eval()

        for N, M, NB in [(4, 2, 40), (16, 2, 160), (1, 0, 10)]:
            policy._num_ships = N
            obs = self._obs(1, N, M, NB)
            hidden = policy.initial_hidden(1, N, torch.device("cpu"))
            action, _, value, _, _ = policy.get_action_and_value(obs, hidden)
            assert action.shape == (1, N, 3)
            assert torch.isfinite(value).all()


class TestEncoderSplit:
    """Per-type first projection with a shared second layer."""

    @staticmethod
    def _cfg(split: bool) -> ModelConfig:
        return ModelConfig(
            d_model=64,
            n_heads=4,
            n_yemong_blocks=2,
            n_spatial_per_block=2,
            n_temporal_per_block=1,
            encoder_split=split,
        )

    def test_default_is_shared_encoder(self):
        assert self._cfg(False).encoder_split is False

    def test_split_encoder_output_shape_matches_shared(self, coordinator):
        B, N, M = 2, 3, 2
        obs = _make_obs(B, N + M)
        obs.data[ObsKey.TEAM_ID][:, N:] = 2

        shared = ShipEncoder(self._cfg(False), coordinator, num_ships=N)
        split = ShipEncoder(self._cfg(True), coordinator, num_ships=N)

        assert shared(obs).shape == split(obs).shape == (B, N + M, 64)

    def test_split_projections_see_only_their_own_channels(self, coordinator):
        """Each per-type projection is narrower than the full shared vector."""
        from boost_and_broadside.train.rl.features import FeatureScope

        ship_dim = coordinator.scoped_input_dimension(FeatureScope.SHIP)
        field_dim = coordinator.scoped_input_dimension(FeatureScope.FIELD)
        total = coordinator.total_input_dimension

        assert ship_dim < total
        assert field_dim < total
        # Shared channels are counted in both, so the two overlap rather than sum.
        assert ship_dim + field_dim > total

        encoder = ShipEncoder(self._cfg(True), coordinator, num_ships=3)
        assert encoder.ship_proj[0].in_features == ship_dim
        assert encoder.field_proj[0].in_features == field_dim

    def test_split_field_tokens_ignore_ship_only_channels(self, coordinator):
        """Ship-only channels are zeros on field tokens and must not reach them."""
        B, N, M = 2, 3, 2
        encoder = ShipEncoder(self._cfg(True), coordinator, num_ships=N).eval()

        obs = _make_obs(B, N + M)
        obs.data[ObsKey.TEAM_ID][:, N:] = 2
        perturbed = YemongObservation(data={k: v.clone() for k, v in obs.data.items()})
        # Write nonsense into a ship-only channel on the field slots only.
        perturbed.data[ObsKey.ANG_VEL][:, N:] = 12.5
        perturbed.data[ObsKey.POWER][:, N:] = 77.0

        with torch.no_grad():
            base, moved = encoder(obs), encoder(perturbed)

        assert torch.allclose(base, moved, atol=1e-6)

    def test_split_requires_num_ships(self, coordinator):
        with pytest.raises(ValueError, match="num_ships"):
            ShipEncoder(self._cfg(True), coordinator)

    def test_policy_runs_end_to_end_with_split_encoder(self, coordinator):
        B, N, M = 2, 3, 2
        torch.manual_seed(0)
        policy = YemongPolicy(
            self._cfg(True),
            coordinator,
            num_value_components=NUM_VALUE_COMPONENTS,
            num_ships=N,
            team_pma_k=(),
        ).eval()
        obs = _make_obs(B, N + M)
        obs.data[ObsKey.TEAM_ID][:, N:] = 2
        hidden = policy.initial_hidden(B, N, torch.device("cpu"))

        action, _, value, _, _ = policy.get_action_and_value(obs, hidden)

        assert action.shape == (B, N, 3)
        assert torch.isfinite(value).all()

    def test_slice_tokens_handles_both_channel_layouts(self):
        """team_id/alive end at the token axis; every other channel has a feature dim."""
        B, N, M = 2, 3, 2
        obs = _make_obs(B, N + M)

        ships = obs.slice_tokens(0, N)
        fields = obs.slice_tokens(N, N + M)

        assert ships.data[ObsKey.TEAM_ID].shape == (B, N)
        assert ships.data[ObsKey.POS].shape == (B, N, 2)
        assert fields.data[ObsKey.ALIVE].shape == (B, M)
        assert fields.data[ObsKey.VEL].shape == (B, M, 2)
        assert torch.equal(fields.data[ObsKey.POS], obs.data[ObsKey.POS][:, N:])


class TestLocalIndexGradientObservation:
    """grad(n) at the ship is a physics term the policy previously could not see."""

    def test_gradient_channel_is_present_and_normalised(self, ship_cfg):
        from boost_and_broadside.env.observation import index_gradient_scale

        scale = index_gradient_scale(ship_cfg)
        assert scale > 0.0
        # Quintic smoothstep peak slope 15/8, widest single-interface index span.
        step = ship_cfg.field_index_step
        expected = 1.875 * (step**2 - step**-2) / ship_cfg.field_transition_width_min
        assert scale == pytest.approx(expected)

    def test_feature_is_registered_as_input_only(self, ship_cfg):
        coordinator = build_standard_coordinator(ship_cfg)
        feature = next(f for f in coordinator.features if f.name == "local_index_gradient")
        assert feature.predictor is None, "gradient is a deterministic function of position"

    def test_gradient_reaches_the_encoder_input(self, ship_cfg):
        """Perturbing the channel must move the encoded token."""
        coordinator = build_standard_coordinator(ship_cfg)
        B, N = 1, 2
        obs = _make_obs(B, N)
        moved = YemongObservation(data={k: v.clone() for k, v in obs.data.items()})
        moved.data[ObsKey.LOCAL_INDEX_GRADIENT][0, 0] = torch.tensor([0.8, -0.4])

        base_vec = coordinator.get_input_vector(obs)
        moved_vec = coordinator.get_input_vector(moved)

        assert not torch.allclose(base_vec[0, 0], moved_vec[0, 0])
        assert torch.allclose(base_vec[0, 1], moved_vec[0, 1])


class TestNonRecurrentFieldPath:
    """Field tokens skip the causal conv and RG-LRU but keep the rest of the block."""

    @staticmethod
    def _policy(coordinator, num_ships, n_spatial=2, n_temporal=1, n_blocks=2):
        cfg = ModelConfig(
            d_model=64,
            n_heads=4,
            n_yemong_blocks=n_blocks,
            n_spatial_per_block=n_spatial,
            n_temporal_per_block=n_temporal,
        )
        torch.manual_seed(0)
        return YemongPolicy(
            cfg,
            coordinator,
            num_value_components=NUM_VALUE_COMPONENTS,
            num_ships=num_ships,
            team_pma_k=(),
        ).eval()

    def test_field_sub_is_identity_initialised(self, coordinator):
        """Zero init would null b1_out and erase the whole gated branch for fields."""
        policy = self._policy(coordinator, num_ships=3, n_temporal=2)
        for block in policy.yemong_layers:
            assert len(block.field_sub) == len(block.temporal)
            for sub in block.field_sub:
                assert torch.equal(sub.weight, torch.eye(policy._d_model))

    def test_forward_nonrecurrent_matches_manual_composition(self):
        """The field path must be the Griffin block with only the scan swapped out."""
        from boost_and_broadside.models.yemong.griffin import GriffinTemporalBlock

        D = 32
        torch.manual_seed(0)
        block = GriffinTemporalBlock(D).eval()
        sub = torch.nn.Linear(D, D, bias=False)
        torch.nn.init.eye_(sub.weight)
        x = torch.randn(2, 5, D)

        with torch.no_grad():
            got = block.forward_nonrecurrent(x, sub)

            normed = block.norm1(x)
            b1 = sub(block.linear1(normed))
            b2 = torch.nn.functional.gelu(block.linear2(normed))
            x1 = x + block.linear_out(b1 * b2)
            want = x1 + block.gated_mlp(block.norm2(x1))

        assert torch.allclose(got, want, atol=1e-6)

    @staticmethod
    def _run_trunk(policy, obs, hidden, B, N):
        """Run the trunk directly — heads emit ship tokens, so fields aren't visible."""
        from boost_and_broadside.models.yemong.griffin import CONV_KERNEL

        D = policy._d_model
        n_layers = hidden.shape[0]
        rg = hidden[:, :, :D]
        cb = hidden[:, :, D:].reshape(n_layers, B * N, CONV_KERNEL - 1, D)
        x = policy.encoder(obs)
        for i, layer in enumerate(policy.yemong_layers):
            sl = slice(i * policy._n_temporal, (i + 1) * policy._n_temporal)
            x, _, _ = layer.step(x, obs.data[ObsKey.ALIVE], rg[sl], cb[sl], N)
        return x

    def test_single_block_field_tokens_ignore_recurrent_state(self, coordinator):
        """The defining property: the field path never reads recurrent state.

        Checked with one block, where no later attention can reintroduce it. Ship
        tokens in the same trunk must still move, or the test proves nothing.
        """
        B, N, M = 2, 3, 2
        policy = self._policy(coordinator, num_ships=N, n_blocks=1)
        obs = _make_obs(B, N + M)
        obs.data[ObsKey.TEAM_ID][:, N:] = 2

        zero_hidden = policy.initial_hidden(B, N, torch.device("cpu"))
        torch.manual_seed(3)
        big_hidden = torch.randn_like(zero_hidden) * 5.0

        with torch.no_grad():
            out_zero = self._run_trunk(policy, obs, zero_hidden, B, N)
            out_big = self._run_trunk(policy, obs, big_hidden, B, N)

        assert torch.allclose(out_zero[:, N:], out_big[:, N:], atol=1e-6), (
            "field tokens moved with the recurrent state"
        )
        assert not torch.allclose(out_zero[:, :N], out_big[:, :N], atol=1e-4), (
            "ship tokens should still depend on the recurrent state"
        )

    def test_later_blocks_relay_recurrent_state_into_fields(self, coordinator):
        """Fields still see ship history indirectly, via attention in a later block.

        This is intentional and is what makes keeping fields in full attention worth
        the cost: a field token aggregates 'which ships are near me' in one block and
        other ships can read that back in the next.
        """
        B, N, M = 2, 3, 2
        policy = self._policy(coordinator, num_ships=N, n_blocks=2)
        obs = _make_obs(B, N + M)
        obs.data[ObsKey.TEAM_ID][:, N:] = 2

        zero_hidden = policy.initial_hidden(B, N, torch.device("cpu"))
        torch.manual_seed(3)
        big_hidden = torch.randn_like(zero_hidden) * 5.0

        with torch.no_grad():
            out_zero = self._run_trunk(policy, obs, zero_hidden, B, N)
            out_big = self._run_trunk(policy, obs, big_hidden, B, N)

        assert not torch.allclose(out_zero[:, N:], out_big[:, N:], atol=1e-4), (
            "second-block attention should carry ship history into field tokens"
        )

    @pytest.mark.parametrize("n_temporal", [1, 2])
    def test_step_matches_sequence_with_fields(self, coordinator, n_temporal):
        """Step/sequence equivalence must survive the ship/field split."""
        B, N, M, T = 2, 3, 2, 5
        policy = self._policy(coordinator, num_ships=N, n_temporal=n_temporal)

        torch.manual_seed(11)
        obs_seq = []
        for _ in range(T):
            obs = _make_obs(B, N + M)
            obs.data[ObsKey.TEAM_ID][:, N:] = 2
            obs_seq.append(obs)

        initial_hidden = policy.initial_hidden(B, N, torch.device("cpu"))
        hidden = initial_hidden
        step_values, actions = [], []
        for obs in obs_seq:
            action, _, value, _, hidden = policy.get_action_and_value(obs, hidden)
            step_values.append(value)
            actions.append(action)

        stacked = YemongObservation(
            data={
                key: torch.stack([o.data[key] for o in obs_seq], dim=0) for key in obs_seq[0].data
            }
        )
        with torch.no_grad():
            _, _, seq_value, _, _, _ = policy.evaluate_actions(
                stacked,
                torch.stack(actions, dim=0),
                initial_hidden,
                stacked.data[ObsKey.ALIVE],
            )

        assert torch.allclose(torch.stack(step_values, dim=0), seq_value, atol=1e-5)

    def test_zero_field_profile_is_unaffected(self, coordinator):
        """With M=0 every token is a ship and the split is a no-op."""
        B, N = 2, 4
        policy = self._policy(coordinator, num_ships=N)
        obs = _make_obs(B, N)
        hidden = policy.initial_hidden(B, N, torch.device("cpu"))

        action, _, _, _, new_hidden = policy.get_action_and_value(obs, hidden)

        assert action.shape == (B, N, 3)
        assert new_hidden.shape == hidden.shape


class TestOrthogonalHeadInit:
    """AUDIT-011: head init locates Linear layers by type, not fixed index."""

    def test_finds_linears_regardless_of_position(self):
        """Inserting a non-Linear layer before the first Linear must not shift which
        module gets orthogonal-initialized (previously a hardcoded head[0]/head[3])."""
        import torch.nn as nn

        from boost_and_broadside.models.yemong.predictive import initialize_head_orthogonal

        torch.manual_seed(0)
        head = nn.Sequential(
            nn.Dropout(0.1),  # pushes the first Linear off index 0
            nn.Linear(4, 8),
            nn.RMSNorm(8),
            nn.GELU(),
            nn.Linear(8, 4),
        )
        initialize_head_orthogonal(head)

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
        the shared next-state decoder relies on.
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
        base = YemongPolicy(
            model_cfg,
            coordinator,
            num_value_components=K,
            num_ships=N,
            team_pma_k=(),
        )
        ckpt = YemongPolicy(
            replace(model_cfg, grad_checkpoint=True),
            coordinator,
            num_value_components=K,
            num_ships=N,
            team_pma_k=(),
        )
        ckpt.load_state_dict(base.state_dict())  # identical weights

        obs_dict = {
            k: v.unsqueeze(0).expand(T, *v.shape).clone() for k, v in _make_obs(B, N).items()
        }
        obs = YemongObservation(data=obs_dict)
        actions = torch.zeros(T, B, N, 3, dtype=torch.long)
        hidden = base.initial_hidden(B, N, torch.device("cpu"))
        alive = torch.ones(T, B, N, dtype=torch.bool)

        lp0, _, val0, _, _, latent0 = base.evaluate_actions(
            obs, actions, hidden, alive, return_predictive_latent=True
        )
        lp1, _, val1, _, _, latent1 = ckpt.evaluate_actions(
            obs, actions, hidden, alive, return_predictive_latent=True
        )

        assert torch.allclose(lp0, lp1, atol=1e-6)
        assert torch.allclose(val0, val1, atol=1e-6)
        assert torch.allclose(latent0, latent1, atol=1e-6)

        # Gradients must match exactly — checkpointing recomputes the same forward.
        (lp0.sum() + val0.sum() + latent0.sum()).backward()
        (lp1.sum() + val1.sum() + latent1.sum()).backward()
        g0 = base.yemong_layers[0].temporal[0].linear1.weight.grad
        g1 = ckpt.yemong_layers[0].temporal[0].linear1.weight.grad
        assert g0 is not None and g1 is not None
        assert torch.allclose(g0, g1, atol=1e-5)
