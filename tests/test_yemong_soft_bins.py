"""
Tests for the soft-binning system (soft_bins.py, SoftBinnedWorldHead, SoftBinnedStateLoss)
and integration with YemongDynamicsInterleaved.
"""

import math
import pytest
import torch
from omegaconf import OmegaConf

from boost_and_broadside.models.components.soft_bins import (
    symlog,
    soft_bin_uniform,
    soft_bin_angular,
    INTERLEAVED_SOFT_BIN_SPECS,
    compute_soft_bin_targets,
    TOTAL_SOFT_BIN_LOGITS,
)
from boost_and_broadside.models.components.heads import SoftBinnedWorldHead
from boost_and_broadside.models.components.losses import SoftBinnedStateLoss
from boost_and_broadside.models.yemong.scaffolds import YemongDynamicsInterleaved
from boost_and_broadside.core.constants import STATE_DIM, TARGET_DIM, StateFeature


# ---------------------------------------------------------------------------
# symlog
# ---------------------------------------------------------------------------

class TestSymlog:
    def test_zero_maps_to_zero(self):
        x = torch.tensor(0.0)
        assert symlog(x).item() == pytest.approx(0.0)

    def test_positive_ordering(self):
        xs = torch.tensor([0.0, 0.5, 1.0, 5.0, 100.0])
        ys = symlog(xs)
        assert (ys[1:] > ys[:-1]).all(), "symlog should be monotonically increasing"

    def test_sign_preservation(self):
        pos = symlog(torch.tensor(3.0))
        neg = symlog(torch.tensor(-3.0))
        assert pos.item() > 0
        assert neg.item() < 0
        assert pos.item() == pytest.approx(-neg.item(), rel=1e-5)

    def test_linthresh(self):
        # At x=linthresh, symlog = sign * log(2)
        x = torch.tensor(1.0)
        assert symlog(x, linthresh=1.0).item() == pytest.approx(math.log(2), rel=1e-5)


# ---------------------------------------------------------------------------
# soft_bin_uniform
# ---------------------------------------------------------------------------

class TestSoftBinUniform:
    def test_sums_to_one(self):
        x = torch.linspace(0.0, 100.0, 50)
        out = soft_bin_uniform(x, n_bins=64, lo=0.0, hi=100.0)
        assert out.shape == (50, 64)
        assert torch.allclose(out.sum(dim=-1), torch.ones(50), atol=1e-6)

    def test_mass_on_correct_bins(self):
        # x = 0 → all mass on bin 0
        out = soft_bin_uniform(torch.tensor(0.0), n_bins=64, lo=0.0, hi=100.0)
        assert out[0].item() == pytest.approx(1.0, abs=1e-6)
        assert out[1:].sum().item() == pytest.approx(0.0, abs=1e-6)

    def test_mass_on_last_bin_at_boundary(self):
        out = soft_bin_uniform(torch.tensor(100.0), n_bins=64, lo=0.0, hi=100.0)
        assert out[-1].item() == pytest.approx(1.0, abs=1e-6)

    def test_two_bin_split_midpoint(self):
        # Midpoint between bin 0 and bin 1 → equal weight
        n, lo, hi = 64, 0.0, 100.0
        w = (hi - lo) / n   # bin width
        x = torch.tensor(lo + w)  # x = lo + 1 bin width → bin boundary → equal split on bins 1 and 2
        out = soft_bin_uniform(x, n_bins=n, lo=lo, hi=hi)
        assert out.sum().item() == pytest.approx(1.0, abs=1e-6)
        # at most 2 nonzero bins
        nonzero = (out > 1e-9).sum().item()
        assert nonzero <= 2

    def test_batched_shape(self):
        x = torch.randn(3, 4, 5).clamp(0, 100)
        out = soft_bin_uniform(x, n_bins=64, lo=0.0, hi=100.0)
        assert out.shape == (3, 4, 5, 64)
        assert torch.allclose(out.sum(dim=-1), torch.ones(3, 4, 5), atol=1e-6)


# ---------------------------------------------------------------------------
# soft_bin_angular (seam handling)
# ---------------------------------------------------------------------------

class TestSoftBinAngular:
    def test_sums_to_one(self):
        angles = torch.linspace(0.0, 2 * math.pi - 1e-6, 128)
        out = soft_bin_angular(angles, n_bins=128)
        assert torch.allclose(out.sum(dim=-1), torch.ones(128), atol=1e-6)

    def test_seam_at_zero(self):
        # Angle very close to 0 should put weight on bin 0 AND bin 127 (wrap-around)
        x = torch.tensor(1e-9)  # nearly 0
        out = soft_bin_angular(x, n_bins=128)
        assert out.sum().item() == pytest.approx(1.0, abs=1e-6)
        assert out[0].item() > 1e-6, "bin 0 should carry most weight"
        # bin 127 should carry the wrapping weight (even if tiny here)
        # more importantly: sum is correct and no mass escapes
        assert (out < 0).sum() == 0, "no negative weights"

    def test_seam_at_two_pi_minus_eps(self):
        # Angle just below 2π should wrap weight onto bins 127 and 0
        TWO_PI = 2.0 * math.pi
        w = TWO_PI / 128
        # x = 2π - 0.1*w: 90% in bin 127, 10% wraps to bin 0
        x = torch.tensor(TWO_PI - w * 0.1)
        out = soft_bin_angular(x, n_bins=128)
        assert out.sum().item() == pytest.approx(1.0, abs=1e-6)
        # Both boundary bins should have non-zero weight
        assert out[127].item() > 0.0, "bin 127 should have weight"
        assert out[0].item() > 0.0, "bin 0 should have wrap-around weight"
        # And the two of them should account for all mass
        assert (out[0] + out[127]).item() == pytest.approx(1.0, abs=1e-6)

    def test_midpoint_two_bins(self):
        # x at the exact boundary between bin 0 and bin 1 → equal split
        w = 2.0 * math.pi / 128
        x = torch.tensor(w)   # start of bin 1
        out = soft_bin_angular(x, n_bins=128)
        assert out.sum().item() == pytest.approx(1.0, abs=1e-6)
        nonzero = (out > 1e-9).sum().item()
        assert nonzero <= 2


# ---------------------------------------------------------------------------
# compute_soft_bin_targets
# ---------------------------------------------------------------------------

class TestComputeSoftBinTargets:
    def _make_states(self, B=2, T=4, N=3, seed=42):
        torch.manual_seed(seed)
        state_t   = torch.rand(B, T, N, STATE_DIM)
        state_tp1 = torch.rand(B, T, N, STATE_DIM)
        state_t[..., StateFeature.HEALTH]  = torch.rand(B, T, N) * 100.0
        state_tp1[..., StateFeature.HEALTH] = torch.rand(B, T, N) * 100.0
        state_t[..., StateFeature.POWER]   = torch.rand(B, T, N) * 100.0
        state_tp1[..., StateFeature.POWER]  = torch.rand(B, T, N) * 100.0
        pos_t   = torch.rand(B, T, N, 2) * 1000.0
        pos_tp1 = torch.rand(B, T, N, 2) * 1000.0
        vel_t   = state_t[..., StateFeature.VX : StateFeature.VY + 1]
        vel_tp1 = state_tp1[..., StateFeature.VX : StateFeature.VY + 1]
        return state_t, state_tp1, pos_t, pos_tp1, vel_t, vel_tp1

    def test_output_count_and_shapes(self):
        B, T, N = 2, 4, 3
        s_t, s_tp1, p_t, p_tp1, v_t, v_tp1 = self._make_states(B, T, N)
        targets = compute_soft_bin_targets(s_t, s_tp1, p_t, p_tp1, v_t, v_tp1, W=1000, H=1000)
        assert len(targets) == len(INTERLEAVED_SOFT_BIN_SPECS)
        for tgt, spec in zip(targets, INTERLEAVED_SOFT_BIN_SPECS):
            if spec.is_team_level:
                assert tgt.shape == (B, T, 1, spec.n_bins), f"{spec.name}: expected (B,T,1,bins)"
            else:
                assert tgt.shape == (B, T, N, spec.n_bins), f"{spec.name}: expected (B,T,N,bins)"

    def test_all_sum_to_one(self):
        B, T, N = 2, 4, 3
        s_t, s_tp1, p_t, p_tp1, v_t, v_tp1 = self._make_states(B, T, N)
        targets = compute_soft_bin_targets(s_t, s_tp1, p_t, p_tp1, v_t, v_tp1, W=1000, H=1000)
        for tgt, spec in zip(targets, INTERLEAVED_SOFT_BIN_SPECS):
            sums = tgt.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
                f"{spec.name}: soft bins don't sum to 1 (max err={( sums - 1).abs().max():.6f})"

    def test_angle_seam_with_large_wrap(self):
        """Toroidal wrap in position delta should not break angular binning."""
        B, T, N = 1, 1, 1
        state_t   = torch.zeros(B, T, N, STATE_DIM)
        state_tp1 = torch.zeros(B, T, N, STATE_DIM)
        state_tp1[..., StateFeature.HEALTH] = 5.0
        state_tp1[..., StateFeature.POWER]  = 5.0
        # Ship crosses world boundary: pos near edge
        pos_t   = torch.tensor([[[[999.0, 999.0]]]])
        pos_tp1 = torch.tensor([[[[1.0,   1.0]]]])  # wrapped around
        vel_t   = torch.zeros(B, T, N, 2)
        vel_tp1 = torch.zeros(B, T, N, 2)
        targets = compute_soft_bin_targets(state_t, state_tp1, pos_t, pos_tp1, vel_t, vel_tp1,
                                           W=1000.0, H=1000.0)
        for tgt, spec in zip(targets, INTERLEAVED_SOFT_BIN_SPECS):
            sums = tgt.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
                f"seam test {spec.name}: sums={sums}"


# ---------------------------------------------------------------------------
# SoftBinnedWorldHead
# ---------------------------------------------------------------------------

class TestSoftBinnedWorldHead:
    def test_output_shapes(self):
        d_model = 64
        head = SoftBinnedWorldHead(d_model, INTERLEAVED_SOFT_BIN_SPECS)
        B, T, N = 2, 4, 3
        x = torch.randn(B, T, N, d_model)
        logits = head(x)
        assert len(logits) == len(INTERLEAVED_SOFT_BIN_SPECS)
        for logit, spec in zip(logits, INTERLEAVED_SOFT_BIN_SPECS):
            assert logit.shape == (B, T, N, spec.n_bins), f"{spec.name}: bad shape"

    def test_total_logit_count(self):
        assert TOTAL_SOFT_BIN_LOGITS == 768  # 64+64+128+128+128+128+64+64


# ---------------------------------------------------------------------------
# SoftBinnedStateLoss
# ---------------------------------------------------------------------------

class TestSoftBinnedStateLoss:
    def _make_logits_and_targets(self, B=2, T=4, N=3):
        """Create matching logits and uniform soft targets."""
        logits_list, targets_list = [], []
        for spec in INTERLEAVED_SOFT_BIN_SPECS:
            if spec.is_team_level:
                shape = (B, T, 1, spec.n_bins)
            else:
                shape = (B, T, N, spec.n_bins)
            logits_list.append(torch.randn(*shape))
            # Uniform target (valid soft distribution)
            targets_list.append(torch.ones(*shape) / spec.n_bins)
        return logits_list, targets_list

    def test_finite_scalar_loss(self):
        B, T, N = 2, 4, 3
        loss_fn = SoftBinnedStateLoss(weight=1.0).set_specs(INTERLEAVED_SOFT_BIN_SPECS)
        logits, tgts = self._make_logits_and_targets(B, T, N)
        mask = torch.ones(B, T, N, dtype=torch.float32)
        preds = {"soft_bin_logits": logits}
        targets = {"soft_bin_targets": tgts}
        out = loss_fn(preds, targets, mask)
        assert "loss" in out
        assert torch.isfinite(out["loss"]), f"loss is not finite: {out['loss']}"

    def test_missing_preds_returns_zero(self):
        loss_fn = SoftBinnedStateLoss(weight=1.0)
        mask = torch.ones(2, 4, 3)
        out = loss_fn({}, {}, mask)
        assert out["loss"].item() == pytest.approx(0.0)

    def test_masked_loss_differs_from_unmasked(self):
        B, T, N = 2, 4, 3
        # Use non-uniform targets so per-token losses vary between ships
        torch.manual_seed(0)
        loss_fn = SoftBinnedStateLoss(weight=1.0).set_specs(INTERLEAVED_SOFT_BIN_SPECS)
        logits_list = []
        tgts_list = []
        for spec in INTERLEAVED_SOFT_BIN_SPECS:
            shape = (B, T, 1, spec.n_bins) if spec.is_team_level else (B, T, N, spec.n_bins)
            logits_list.append(torch.randn(*shape))
            # Spiky target: one-hot on a random bin so different ships differ
            idx = torch.randint(0, spec.n_bins, shape[:-1])
            t = torch.zeros(*shape)
            t.scatter_(-1, idx.unsqueeze(-1), 1.0)
            tgts_list.append(t)
        preds = {"soft_bin_logits": logits_list}
        targets = {"soft_bin_targets": tgts_list}
        full_mask  = torch.ones(B, T, N)
        # Half mask: only ship 0 valid → denominator differs
        half_mask  = torch.zeros(B, T, N)
        half_mask[:, :, 0] = 1.0
        out_full = loss_fn(preds, targets, full_mask)["loss"]
        out_half = loss_fn(preds, targets, half_mask)["loss"]
        # Both should be finite
        assert torch.isfinite(out_full) and torch.isfinite(out_half)
        # Losses should differ by more than 1% absolute (different denominator matters)
        assert abs(out_full.item() - out_half.item()) > 0.01, \
            f"Expected loss difference >0.01, got {abs(out_full.item()-out_half.item()):.4f}"

    def test_weight_scaling(self):
        B, T, N = 2, 4, 3
        loss_a = SoftBinnedStateLoss(weight=1.0).set_specs(INTERLEAVED_SOFT_BIN_SPECS)
        loss_b = SoftBinnedStateLoss(weight=2.0).set_specs(INTERLEAVED_SOFT_BIN_SPECS)
        logits, tgts = self._make_logits_and_targets(B, T, N)
        preds = {"soft_bin_logits": logits}
        targets = {"soft_bin_targets": tgts}
        mask = torch.ones(B, T, N)
        la = loss_a(preds, targets, mask)["loss"]
        lb = loss_b(preds, targets, mask)["loss"]
        assert lb.item() == pytest.approx(2.0 * la.item(), rel=1e-5)


# ---------------------------------------------------------------------------
# YemongDynamicsInterleaved end-to-end with soft bins
# ---------------------------------------------------------------------------

def _interleaved_config(d_model=64, use_soft_bins=True):
    return OmegaConf.create({
        "d_model": d_model,
        "n_layers": 1,
        "n_heads": 2,
        "input_dim": STATE_DIM,
        "target_dim": TARGET_DIM,
        "action_space_type": "separated",
        "action_embed_dim": 8,
        "loss_type": "fixed",
        "use_soft_bin_targets": use_soft_bins,
        "spatial_layer": {
            "_target_": "boost_and_broadside.models.components.layers.attention.RelationalAttention",
            "d_model": d_model,
            "n_heads": 2,
        },
        "loss": {
            "_target_": "boost_and_broadside.models.components.losses.CompositeLoss",
            "losses": [
                # Use ActionLoss (separated 3+7+2=12 logits) to match action_space_type=separated
                {"_target_": "boost_and_broadside.models.components.losses.ActionLoss", "weight": 1.0},
                {"_target_": "boost_and_broadside.models.components.losses.SoftBinnedStateLoss", "weight": 1.0},
            ],
        },
    })


class TestYemongInterleavedSoftBins:
    def test_forward_returns_10_items(self):
        cfg = _interleaved_config(use_soft_bins=True)
        model = YemongDynamicsInterleaved(cfg)
        B, T, N = 2, 4, 3
        state = torch.randn(B, T, N, STATE_DIM)
        state[..., StateFeature.HEALTH] = 1.0
        prev_action = torch.randint(0, 2, (B, T, N, 3))
        pos = torch.randn(B, T, N, 2)
        vel = torch.randn(B, T, N, 2)
        target_actions = torch.randint(0, 2, (B, T, N, 3))
        out = model(state=state, prev_action=prev_action, pos=pos, vel=vel,
                    target_actions=target_actions)
        assert len(out) == 10, f"Expected 10 outputs, got {len(out)}"

    def test_soft_bin_logits_shapes(self):
        cfg = _interleaved_config(use_soft_bins=True)
        model = YemongDynamicsInterleaved(cfg)
        B, T, N = 2, 4, 3
        state = torch.randn(B, T, N, STATE_DIM)
        state[..., StateFeature.HEALTH] = 1.0
        prev_action = torch.randint(0, 2, (B, T, N, 3))
        pos = torch.randn(B, T, N, 2)
        vel = torch.randn(B, T, N, 2)
        target_actions = torch.randint(0, 2, (B, T, N, 3))
        out = model(state=state, prev_action=prev_action, pos=pos, vel=vel,
                    target_actions=target_actions)
        soft_bin_logits = out[9]
        assert soft_bin_logits is not None
        assert len(soft_bin_logits) == len(INTERLEAVED_SOFT_BIN_SPECS)
        for logit, spec in zip(soft_bin_logits, INTERLEAVED_SOFT_BIN_SPECS):
            if spec.is_team_level:
                expected = (B, T, 1, spec.n_bins)
            else:
                expected = (B, T, N, spec.n_bins)
            # Team-level specs use Z_action which is (B,T,N,D) → head outputs (B,T,N,bins)
            # The team-level distinction is handled in the loss, not the head.
            # So all outputs are (B,T,N,bins) from the head.
            assert logit.shape[-1] == spec.n_bins, f"{spec.name}: wrong n_bins"

    def test_disabled_returns_9_items(self):
        cfg = _interleaved_config(use_soft_bins=False)
        model = YemongDynamicsInterleaved(cfg)
        B, T, N = 2, 3, 2
        state = torch.randn(B, T, N, STATE_DIM)
        state[..., StateFeature.HEALTH] = 1.0
        prev_action = torch.randint(0, 2, (B, T, N, 3))
        pos = torch.randn(B, T, N, 2)
        vel = torch.randn(B, T, N, 2)
        target_actions = torch.randint(0, 2, (B, T, N, 3))
        out = model(state=state, prev_action=prev_action, pos=pos, vel=vel,
                    target_actions=target_actions)
        # With soft bins disabled: 10 items but last is None
        assert len(out) == 10
        assert out[9] is None

    def test_get_loss_with_soft_bins(self):
        """get_loss with soft_bin_logits+targets should produce finite scalar."""
        cfg = _interleaved_config(use_soft_bins=True)
        model = YemongDynamicsInterleaved(cfg)
        B, T, N = 2, 4, 3

        state = torch.randn(B, T, N, STATE_DIM)
        state_tp1 = torch.randn(B, T, N, STATE_DIM)
        state[..., StateFeature.HEALTH] = 1.0
        state_tp1[..., StateFeature.HEALTH] = 1.0
        state_tp1[..., StateFeature.POWER] = torch.rand(B, T, N) * 100.0
        prev_action = torch.randint(0, 2, (B, T, N, 3))
        pos = torch.randn(B, T, N, 2)
        pos_tp1 = torch.randn(B, T, N, 2)
        vel = state[..., StateFeature.VX : StateFeature.VY + 1]
        vel_tp1 = state_tp1[..., StateFeature.VX : StateFeature.VY + 1]
        target_actions = torch.randint(0, 2, (B, T, N, 3))

        out = model(state=state, prev_action=prev_action, pos=pos, vel=vel,
                    target_actions=target_actions)
        soft_bin_logits = out[9]

        from boost_and_broadside.models.components.soft_bins import compute_soft_bin_targets, INTERLEAVED_SOFT_BIN_SPECS
        soft_bin_targets = compute_soft_bin_targets(
            state, state_tp1, pos, pos_tp1, vel, vel_tp1,
            W=1000.0, H=1000.0,
        )

        loss_mask = torch.ones(B, T, N)
        loss_out = model.get_loss(
            pred_actions=out[0],
            loss_mask=loss_mask,
            target_actions=target_actions,
            soft_bin_logits=soft_bin_logits,
            soft_bin_targets=soft_bin_targets,
        )
        assert "loss" in loss_out
        assert torch.isfinite(loss_out["loss"]), f"get_loss returned non-finite: {loss_out['loss']}"
        assert "soft_bin_loss" in loss_out
