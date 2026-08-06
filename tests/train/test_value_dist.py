"""The categorical value head's bin geometry, targets, and decode.

These pin the one property the whole critic rests on: a target built from a
scalar must decode back to that scalar. Everything downstream — GAE, the
bootstrap, every per-component diagnostic — reads the distribution's mean and
would be silently biased if the round trip drifted.
"""

import pytest
import torch

from boost_and_broadside.train.rl.value_dist import (
    bin_centers,
    cross_entropy,
    expected_value,
    two_hot_targets,
)

CPU = torch.device("cpu")


class TestBinGeometry:
    def test_endpoints_sit_exactly_on_the_support(self):
        centers = bin_centers(51, 5.0, device=CPU)
        assert centers[0].item() == pytest.approx(-5.0)
        assert centers[-1].item() == pytest.approx(5.0)
        assert centers.numel() == 51

    def test_bins_are_uniformly_spaced(self):
        centers = bin_centers(31, 4.0, device=CPU)
        gaps = centers[1:] - centers[:-1]
        assert torch.allclose(gaps, gaps[0].expand_as(gaps), atol=1e-6)


class TestTwoHotRoundTrip:
    """A two-hot target's mean is the value it was built from, exactly."""

    @pytest.mark.parametrize("num_bins", [2, 31, 51, 101])
    def test_decode_recovers_the_value(self, num_bins):
        centers = bin_centers(num_bins, 5.0, device=CPU)
        values = torch.linspace(-5.0, 5.0, 97).reshape(-1, 1)

        target = two_hot_targets(values, centers)
        # log of the target is the logit set whose softmax is the target itself.
        decoded = expected_value(target.clamp(min=1e-30).log(), centers)

        assert torch.allclose(decoded, values, atol=1e-4)

    def test_targets_are_proper_distributions(self):
        centers = bin_centers(51, 5.0, device=CPU)
        values = torch.randn(4, 3, 7) * 3.0

        target = two_hot_targets(values, centers)

        assert (target >= 0).all()
        assert torch.allclose(target.sum(-1), torch.ones_like(target.sum(-1)), atol=1e-5)

    def test_mass_sits_on_at_most_two_adjacent_bins(self):
        centers = bin_centers(51, 5.0, device=CPU)
        target = two_hot_targets(torch.tensor([[1.37]]), centers)

        nonzero = target[0, 0].nonzero().flatten()
        assert nonzero.numel() <= 2
        if nonzero.numel() == 2:
            assert nonzero[1] - nonzero[0] == 1

    def test_a_value_on_a_bin_centre_is_one_hot(self):
        centers = bin_centers(51, 5.0, device=CPU)
        target = two_hot_targets(centers[17].reshape(1, 1), centers)
        assert target[0, 0, 17].item() == pytest.approx(1.0)


class TestSupportClipping:
    """Out-of-support returns saturate rather than corrupting the target."""

    def test_values_beyond_the_support_land_on_the_end_bins(self):
        centers = bin_centers(51, 5.0, device=CPU)
        target = two_hot_targets(torch.tensor([[-40.0], [40.0]]), centers)

        assert target[0, 0, 0].item() == pytest.approx(1.0)
        assert target[1, 0, -1].item() == pytest.approx(1.0)

    def test_the_top_edge_does_not_wrap_to_the_bottom_bin(self):
        """pos lands exactly on num_bins-1 there, one past the last lower index."""
        centers = bin_centers(51, 5.0, device=CPU)
        target = two_hot_targets(torch.tensor([[5.0]]), centers)

        assert target[0, 0, -1].item() == pytest.approx(1.0)
        assert target[0, 0, 0].item() == pytest.approx(0.0)


class TestHLGauss:
    def test_smoothing_spreads_mass_but_keeps_the_mean(self):
        centers = bin_centers(51, 5.0, device=CPU)
        values = torch.tensor([[0.31], [-2.2], [1.75]])

        sharp = two_hot_targets(values, centers, sigma=0.0)
        smooth = two_hot_targets(values, centers, sigma=0.75)

        assert (smooth > 0).sum() > (sharp > 0).sum()
        assert torch.allclose(smooth.sum(-1), torch.ones_like(smooth.sum(-1)), atol=1e-5)
        decoded = expected_value(smooth.clamp(min=1e-30).log(), centers)
        assert torch.allclose(decoded, values, atol=1e-2)

    def test_tail_mass_is_kept_by_the_infinite_outer_edges(self):
        """Without unbounded outer edges the target would not sum to 1 near the rim."""
        centers = bin_centers(31, 5.0, device=CPU)
        target = two_hot_targets(torch.tensor([[5.0]]), centers, sigma=2.0)
        assert target.sum(-1).item() == pytest.approx(1.0, abs=1e-5)


class TestCrossEntropy:
    def test_is_minimized_when_the_prediction_matches_the_target(self):
        centers = bin_centers(51, 5.0, device=CPU)
        target = two_hot_targets(torch.tensor([[0.6]]), centers)
        logits = target.clamp(min=1e-30).log()

        exact = cross_entropy(logits, target)
        wrong = cross_entropy(torch.zeros_like(logits), target)

        assert exact.item() < wrong.item()

    def test_is_computed_in_fp32_under_a_bf16_input(self):
        centers = bin_centers(51, 5.0, device=CPU)
        target = two_hot_targets(torch.tensor([[0.6]]), centers)
        loss = cross_entropy(torch.zeros(1, 1, 51, dtype=torch.bfloat16), target)
        assert loss.dtype == torch.float32

    def test_shape_reduces_only_the_bin_axis(self):
        centers = bin_centers(31, 5.0, device=CPU)
        values = torch.randn(4, 2, 6, 13)
        target = two_hot_targets(values, centers)
        loss = cross_entropy(torch.randn(4, 2, 6, 13, 31), target)
        assert loss.shape == (4, 2, 6, 13)


class TestGradientBoundedness:
    """The property that lets return_min_span go back to being an epsilon.

    Squared error's gradient grows with the residual, so one far-out target
    dominates the batch. Cross-entropy's does not: however far outside the
    support the return lands, the gradient it produces is the same as one sitting
    on the end bin.
    """

    def test_an_extreme_target_produces_no_larger_gradient_than_an_edge_one(self):
        centers = bin_centers(51, 5.0, device=CPU)

        def grad_norm(value: float) -> float:
            logits = torch.zeros(1, 1, 51, requires_grad=True)
            target = two_hot_targets(torch.tensor([[value]]), centers)
            cross_entropy(logits, target).sum().backward()
            return logits.grad.norm().item()

        assert grad_norm(500.0) == pytest.approx(grad_norm(5.0), rel=1e-5)
