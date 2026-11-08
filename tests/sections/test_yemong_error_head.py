"""The error head (arms 3-4 brief, `yemong.delta_split`): the projection axes and the
positive semi-definite clamp, the capped step and the unapplied variance, the corrected
advance of hidden rows, the split masks, the stop-gradient, the switches (arms 1 and 2 bit
for bit as recorded, the arm's unroll a loop of steps), the term in the learners and a
policy-file round trip."""

import math
import os
import pathlib
import subprocess
import sys

import learner_cases as lc
import pytest
import torch
import yemong_cases as cases

from bnb.contracts.obs import SHIP
from bnb.controllers.scripted import Scripted
from bnb.game import api
from bnb.learn import bc, head_statistics, labels
from bnb.learn.ppo import COUNTS, TERMS
from bnb.models.yemong import codes, correction, delta
from bnb.models.yemong.inputs import believed, truth
from bnb.models.yemong.model import YemongConfig
from bnb.models.yemong.state_code import MEANS, SPREADS, state_code_ranges, unpack_covariance
from bnb.runs.checkpoint import load_policy, save_policy
from bnb.runtime import metrics
from bnb.runtime.rollout import Rollout

pytestmark = pytest.mark.usefixtures("global_generator")

OWN = 5
F64 = torch.float64
HERE = pathlib.Path(__file__).parent
# the arm's --set line
SETTINGS = dict(
    order="temporal_first",
    spread_fit="coarsest",
    acceleration_inputs=True,
    next_state="delta",
    delta_baseline="persistence",
    spread_inputs="ellipse",
    delta_split=True,
)
ARM_SETS = (
    "yemong.order=temporal_first",
    "yemong.acceleration_inputs=true",
    "yemong.next_state=delta",
    "yemong.delta_baseline=persistence",
    "yemong.spread_inputs=ellipse",
    "yemong.delta_split=true",
    "learner.next_state=0.0033",
    "learner.correction=0.0033",
)


def _model(**settings):
    return cases.model(**{**SETTINGS, **settings})


def _code() -> correction.CorrectionCode:
    rules, types = api.rules_of(cases.SQUAD), api.ship_types()
    return correction.CorrectionCode(state_code_ranges(rules, types)).double()


# ------------------------------------------------------------------ the projections


def test_a_vector_and_a_covariance_round_trip_through_the_three_axes():
    unit = correction.axes()
    vector = torch.tensor([[3.0, -4.0], [0.0, 0.0], [-1e3, 2.5e2]], dtype=F64)
    projected = correction.project(vector, unit)
    assert projected.shape == (3, 3)
    torch.testing.assert_close(projected[:, 0], vector[:, 0])  # the 0-degree axis is x
    back = correction.mean_of(projected, correction.least_squares())
    torch.testing.assert_close(back, vector, rtol=0, atol=1e-10)
    # a tilted ellipse: its variances on the axes recover it exactly
    angle, major, minor = 0.7, 9.0, 0.25
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=F64
    )
    matrix = rotation @ torch.diag(torch.tensor([major, minor], dtype=F64)) @ rotation.T
    variances = torch.einsum("ai,ij,aj->a", unit, matrix, unit)
    packed = correction.covariance_of(variances, correction.variance_solve())
    expected = torch.stack([matrix[0, 0], matrix[0, 1], matrix[1, 1]])
    torch.testing.assert_close(packed, expected, rtol=0, atol=1e-10)


def test_the_clamp_makes_the_covariance_positive_semidefinite_and_keeps_one_that_is():
    kept = torch.tensor([4.0, 1.0, 2.0], dtype=F64)
    torch.testing.assert_close(correction.positive_semidefinite(kept), kept)
    # three axis variances no ellipse has (0 on two axes, large on the third)
    variances = torch.tensor([0.0, 0.0, 5.0], dtype=F64)
    raw = variances @ correction.variance_solve().T
    xx, xy, yy = raw.unbind(-1)
    assert float(xx * yy - xy * xy) < 0  # indefinite before the clamp
    clamped = correction.covariance_of(variances, correction.variance_solve())
    xx, xy, yy = clamped.unbind(-1)
    eigenvalues = torch.linalg.eigvalsh(torch.stack([torch.stack([xx, xy]), torch.stack([xy, yy])]))
    assert float(eigenvalues.min()) >= -1e-12
    assert float(eigenvalues.max()) > 0  # the positive part is kept
    # a negative-definite triple clamps to zero
    zero = correction.positive_semidefinite(torch.tensor([-1.0, 0.0, -2.0], dtype=F64))
    assert zero.abs().max() == 0


# ------------------------------------------------------------------ the spread


def _component(name: str):
    return next(c for c in _code().components if c.name == name)


def _spread(component, probabilities: torch.Tensor) -> torch.Tensor:
    return codes.symlog_spread(
        probabilities, component.knee, component.bound, component.bins, below=component.below
    )


def _normal(count: int = 20_000) -> torch.Tensor:
    """A standard normal's quantile points (k + 1/2) / count."""
    levels = (torch.arange(count, dtype=F64) + 0.5) / count
    return math.sqrt(2.0) * torch.erfinv(2.0 * levels - 1.0)


@pytest.mark.parametrize("name", ["position_0", "velocity_60", "heading", "energy"])
def test_an_encoded_value_has_no_spread(name):
    """The probit-interpolated quantiles of an HL-Gauss are its own, so less the smoothing
    nothing is left, at any offset inside a bin and out in the wide bins."""
    component = _component(name)
    values = component.knee * torch.tensor([0.0, 0.13, 0.5, 1.0, -2.7, 9.0, 40.0], dtype=F64)
    values = values[values.abs() < component.bound]
    encoded, _ = component.encode(values)
    assert float(_spread(component, encoded).abs().max()) < 1e-6 * component.knee


@pytest.mark.parametrize("centre", [0.0, 2.0])
@pytest.mark.parametrize("width", [1.0, 3.0, 10.0, 50.0])
def test_an_encoded_gaussian_decodes_to_its_sigma(centre, width):
    """A Gaussian of sigma (in knees: across the knee and out into the wide bins), as the
    mean code of its quantile points: sigma to 6% (measured -5.1% to +1.5%, low where the
    symlog fold near 0 makes it two-humped in the bin coordinate)."""
    component = _component("position_0")
    sigma = width * component.knee
    encoded, _ = component.encode(centre * component.knee + sigma * _normal())
    assert math.isclose(float(_spread(component, encoded.mean(0))), sigma, rel_tol=0.06)


def test_the_spread_reads_each_side_of_an_asymmetric_code():
    """The position along the track (asymmetric sides): a Gaussian of 0.3 m at 2.6 m."""
    component = delta.POSITION_ALONG
    encoded, _ = component.encode(2.6 + 0.3 * _normal())
    assert math.isclose(float(_spread(component, encoded.mean(0))), 0.3, rel_tol=0.02)


def test_a_far_sliver_moves_the_spread_by_its_mass_not_its_distance():
    """The bug's case: 1e-3 of the mass in a bin 10 km out. The second moment's sigma grows
    past 250 m; the quantile spread of a sharp code stays about 0, and of a 20 m Gaussian
    at 20 m within 1% of its own."""
    component = _component("position_0")
    sharp, _ = component.encode(torch.tensor(7.0, dtype=F64))
    far, _ = component.encode(torch.tensor(10_000.0, dtype=F64))
    sliver = 0.999 * sharp + 0.001 * far
    assert float(_spread(component, sliver)) < 0.05
    _, second = codes.symlog_decode(sliver, component.knee, component.bound, component.bins)
    assert float(second) ** 0.5 > 250.0
    broad, _ = component.encode(20.0 + 20.0 * _normal())
    broad = broad.mean(0)
    clean = float(_spread(component, broad))
    assert math.isclose(float(_spread(component, 0.999 * broad + 0.001 * far)), clean, rel_tol=0.01)


def test_the_spread_has_a_finite_gradient_where_it_is_zero():
    component = _component("position_0")
    encoded, _ = component.encode(torch.tensor([0.0, 7.0], dtype=F64))
    logits = torch.log(encoded.clamp_min(1e-30)).float().requires_grad_()
    spread = _spread(component, torch.softmax(logits, -1))
    spread.sum().backward()
    assert torch.isfinite(spread).all() and torch.isfinite(logits.grad).all()


# ------------------------------------------------------------------ the code


@pytest.mark.parametrize("index", range(len(correction.COMPONENT_NAMES)))
def test_the_code_round_trips_with_an_exact_zero_and_odd_bins(index):
    code = _code()
    component = code.components[index]
    assert component.bins % 2 == 1 and component.below is None
    assert component.bins == correction.odd_bins(component.knee, component.bound)
    low, high = component.low, component.width - 1 - component.low
    coordinate = torch.linspace(-low, high, 2001, dtype=F64)
    values = codes.symlog_value(coordinate, component.knee, component.spacing)
    encoded, _ = component.encode(values)
    inverse = code.inverse(component.name)
    mean, _ = codes.symlog_decode(
        encoded, component.knee, component.bound, component.bins, inverse=inverse
    )
    back = codes.symlog_coordinate(mean, component.knee, component.spacing)
    assert (back - coordinate).abs().max() < 1e-3
    zero, _ = component.encode(torch.zeros(1, dtype=F64))
    assert torch.argmax(zero[0]) == component.low


def test_the_bounds_are_the_ranges_a_hidden_ships_error_can_reach():
    code = _code()
    bound = {c.name: c.bound for c in code.components}
    assert bound["position_0"] == 26244.0 and bound["heading"] == math.pi
    assert math.isclose(bound["velocity_60"], 2.0 * 319.4528, rel_tol=1e-4)
    assert bound["shield"] == bound["energy"] == 1.0
    assert math.isclose(bound["gun_cooldown"], 3.0 / 30.0)


def test_errors_are_the_truth_less_the_forecast_in_the_codes_units():
    code = _code()
    forecast = torch.zeros(1, len(MEANS), dtype=F64)
    truth_after = forecast.clone()
    truth_after[0, 0:2] = torch.tensor([10.0, -5.0])
    truth_after[0, 2:4] = torch.tensor([1.0, 2.0])
    truth_after[0, MEANS.index("heading")] = 3.0
    forecast[0, MEANS.index("heading")] = -3.0
    truth_after[0, MEANS.index("shield")] = 50.0
    errors = code.errors(forecast, truth_after, torch.zeros(1, dtype=torch.long))
    column = {name: i for i, name in enumerate(correction.COMPONENT_NAMES)}
    torch.testing.assert_close(
        errors[0, 0:3], correction.project(truth_after[0, 0:2], correction.axes())
    )
    assert math.isclose(float(errors[0, column["heading"]]), 6.0 - 2.0 * math.pi, rel_tol=1e-12)
    assert math.isclose(float(errors[0, column["shield"]]), 50.0 / float(code.max_shield[0]))


# ------------------------------------------------------------------ the advance


def test_the_step_is_the_whole_mean_below_r_and_r_far_above_it():
    small = torch.tensor([[0.03, -0.04]], dtype=F64)
    step = correction.capped_step(small, 8.0)
    torch.testing.assert_close(step, small, rtol=1e-4, atol=0)
    large = torch.tensor([[3000.0, 4000.0]], dtype=F64)
    step = correction.capped_step(large, 8.0)
    assert math.isclose(float(torch.linalg.vector_norm(step)), 8.0, rel_tol=1e-9)
    torch.testing.assert_close(step / 8.0, large / 5000.0)  # toward the mean
    assert correction.capped_step(torch.zeros(1, 2, dtype=F64), 8.0).abs().max() == 0


def _sharp_logits(code: correction.CorrectionCode, errors: torch.Tensor) -> torch.Tensor:
    encoded, _ = code.encode(errors)
    return torch.log(encoded.clamp_min(1e-300))


def test_a_known_error_steps_the_forecast_and_the_unapplied_part_widens_it():
    """Position 1 km off (capped at r), velocity 0.5 m/s off (taken whole), heading 0.01
    rad: the corrected mean moves by the step, and the variance holds (E e - step)^2. At a
    rate of 1, so the confident median (a sharp code: gain 1) is applied whole."""
    net = cases.double(_model(correction_rate=1.0))
    code, state_code = net.correction_code, net.state_code
    forecast = torch.zeros(1, len(MEANS), dtype=F64)
    forecast[0, MEANS.index("shield")] = 50.0
    forecast[0, MEANS.index("energy")] = 5000.0
    offset = torch.tensor([600.0, -800.0], dtype=F64)
    velocity = torch.tensor([0.3, 0.4], dtype=F64)
    errors = torch.zeros(1, len(correction.COMPONENT_NAMES), dtype=F64)
    errors[0, 0:3] = correction.project(offset, correction.axes())
    errors[0, 3:6] = correction.project(velocity, correction.axes())
    errors[0, correction.COMPONENT_NAMES.index("heading")] = 0.01
    means, spreads = code.correct(
        forecast, _sharp_logits(code, errors), torch.zeros(1, dtype=torch.long), state_code
    )
    limit = correction.STEP_LIMITS["position"]
    direction = offset / 1000.0
    step = limit * math.tanh(1000.0 / limit) * direction
    torch.testing.assert_close(means[0, 0:2], step, rtol=1e-3, atol=1e-3)
    # the windowed median of a coarse code resolves a value to about 1%
    torch.testing.assert_close(means[0, 2:4], velocity, rtol=2e-2, atol=1e-3)
    assert math.isclose(float(means[0, MEANS.index("heading")]), 0.01, rel_tol=2e-2)
    covariance = unpack_covariance(spreads[0, state_code.covariance_slice])
    rest = offset - step
    # the sharp code's own variance is ~0: the covariance is the unapplied part's outer product
    expected = torch.outer(rest, rest)
    torch.testing.assert_close(covariance[0:2, 0:2], expected, rtol=2e-2, atol=50.0)
    assert covariance[0:2, 2:4].abs().max() == 0  # block diagonal
    assert float(spreads[0, SPREADS.index("position")]) > 600.0


def test_the_step_is_the_rate_times_the_confident_median():
    """A sharp error moves by eta of itself; a broad one (sigma >> |m|) hardly at all."""
    net = cases.double(_model())
    code = net.correction_code
    assert code.rate == 0.1 == YemongConfig().correction_rate
    errors = torch.zeros(2, len(correction.COMPONENT_NAMES), dtype=F64)
    velocity = torch.tensor([0.3, 0.4], dtype=F64)
    errors[0, 3:6] = correction.project(velocity, correction.axes())
    encoded, _ = code.encode(errors)
    moments = code.moments(torch.log(encoded[:1].clamp_min(1e-300)))
    torch.testing.assert_close(moments.step["velocity"][0], 0.1 * velocity, rtol=2e-2, atol=1e-3)
    # the same median under a broad mixture: the gain collapses the step
    errors[1, 3:6] = correction.project(torch.tensor([300.0, -400.0], dtype=F64), correction.axes())
    encoded, _ = code.encode(errors)
    broad = torch.log((0.6 * encoded[0] + 0.4 * encoded[1]).clamp_min(1e-300))[None]
    moments = code.moments(broad)
    length = float(torch.linalg.vector_norm(moments.step["velocity"][0]))
    assert length < 0.1 * 0.5 * 0.1
    two, four, zero = (torch.tensor(value) for value in (2.0, 4.0, 0.0))
    assert correction.confident(two, four, zero) == (2.0, 1.0)
    with pytest.raises(ValueError, match="rate"):
        correction.CorrectionCode(code_ranges(), 0.0)


def code_ranges():
    return state_code_ranges(api.rules_of(cases.SQUAD), api.ship_types())


def test_the_applied_variance_adds_the_unapplied_median():
    median = {"position": torch.tensor([[3.0, 4.0]], dtype=F64), "heading": torch.tensor([0.5])}
    step = {"position": torch.tensor([[1.0, 1.0]], dtype=F64), "heading": torch.tensor([0.2])}
    variance = {
        "position": torch.tensor([[1.0, 0.0, 2.0]], dtype=F64),
        "heading": torch.tensor([0.1]),
    }
    moments = correction.Moments(median=median, variance=variance, gain={}, target=step, step=step)
    torch.testing.assert_close(
        moments.applied_variance("position"), torch.tensor([[5.0, 6.0, 11.0]], dtype=F64)
    )
    torch.testing.assert_close(moments.applied_variance("heading"), torch.tensor([0.19]))


@pytest.mark.parametrize("jump, low, high", [(0.2, 1.2, 2.2), (0.1, 0.0, 0.15)])
def test_the_rare_jumps_step_by_the_median(jump, low, high):
    """The wait's error: 0, or +4 s at p `jump`: no step. A jump past the 84th percentile's
    reach (p > 0.16) widens the spread by about half its size (the mixture's sigma is 1.6 s
    at 0.2); a rarer one leaves it near the zero's (0.07 s at 0.1: the quantile spread)."""
    net = cases.double(_model())
    code = net.correction_code
    errors = torch.zeros(2, len(correction.COMPONENT_NAMES), dtype=F64)
    column = correction.COMPONENT_NAMES.index("shield_recharge_wait")
    errors[1, column] = 4.0
    encoded, _ = code.encode(errors)
    mixture = (1.0 - jump) * encoded[0] + jump * encoded[1]
    forecast = torch.zeros(1, len(MEANS), dtype=F64)
    forecast[0, MEANS.index("shield_recharge_wait")] = 2.0
    means, spreads = code.correct(
        forecast,
        torch.log(mixture.clamp_min(1e-300))[None],
        torch.zeros(1, dtype=torch.long),
        net.state_code,
    )
    assert abs(float(means[0, MEANS.index("shield_recharge_wait")]) - 2.0) < 0.1 / 30.0
    assert low <= float(spreads[0, SPREADS.index("shield_recharge_wait")]) < high


def test_a_far_sliver_of_mass_keeps_the_sharp_spread():
    """The bug's case (2026-11-08): a sharp position error with 1e-3 of the mass 10 km out
    on every axis. The second moment put sigma near 300 m; the quantile spread keeps the
    sharp code's, and the gain stays near 1."""
    net = cases.double(_model())
    code = net.correction_code
    errors = torch.zeros(2, len(correction.COMPONENT_NAMES), dtype=F64)
    errors[0, 0:3] = correction.project(torch.tensor([20.0, -15.0], dtype=F64), code.axes.double())
    errors[1, 0:3] = 10_000.0
    encoded, _ = code.encode(errors)
    sliver = torch.log((0.999 * encoded[0] + 0.001 * encoded[1]).clamp_min(1e-300))[None]
    moments = code.moments(sliver)
    sharp = code.moments(torch.log(encoded[:1].clamp_min(1e-300)))
    assert float(sharp.variance["position"].abs().max()) < 1e-6
    assert float(moments.variance["position"].abs().max()) < 1.0  # m^2, against ~1e5
    assert float(moments.gain["position"][0]) > 0.99


def test_hidden_rows_take_the_correction_and_seen_rows_the_physics():
    net = cases.double(_model())
    rows = 4
    means = torch.zeros(rows, len(MEANS), dtype=F64)
    means[:, 2] = 50.0
    spreads = torch.zeros(rows, net.state_code.spread_count, dtype=F64)
    ship_type = torch.zeros(rows, dtype=torch.long)
    logits = torch.randn(rows, net.next_state_width, dtype=F64)
    correction_logits = torch.randn(rows, net.correction_code.width, dtype=F64)
    hidden = torch.tensor([False, True, False, True])
    physics = net.forecast(logits, means, spreads, ship_type)
    split = net.forecast(logits, means, spreads, ship_type, correction_logits, hidden)
    corrected = net.correction_code.correct(
        physics[0], correction_logits, ship_type, net.state_code
    )
    for got, seen, fixed in zip(split, physics, corrected, strict=True):
        torch.testing.assert_close(got[~hidden], seen[~hidden])
        torch.testing.assert_close(got[hidden], fixed[hidden])
    with pytest.raises(ValueError, match="hidden"):
        net.forecast(logits, means, spreads, ship_type, correction_logits)


# ------------------------------------------------------------------ the switches


def test_split_is_off_by_default_and_needs_the_delta_head():
    assert not YemongConfig().delta_split
    net = cases.model(**{**SETTINGS, "delta_split": False})
    assert net.correction_code is None and net.correction_head is None
    with pytest.raises(ValueError, match="delta_split"):
        YemongConfig(delta_split=True)


def test_arms_1_and_2_step_and_unroll_bit_for_bit_as_recorded():
    """In a fresh process on one thread, as recorded (`arm_goldens`)."""
    environment = dict(
        os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=""
    )
    code = "import torch; torch.set_num_threads(1); import arm_goldens; arm_goldens.reproduces()"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=HERE, env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_the_arms_set_line_builds_the_arm():
    from bnb.cli import parse_setting, with_settings
    from bnb.contracts import GameConfig
    from bnb.runtime.train import TrainConfig, model_build

    config = with_settings(
        TrainConfig(game=GameConfig(5, 10, 2600.0)), map(parse_setting, ARM_SETS)
    )
    build = model_build(config)
    assert build.config.delta_split and build.config.delta_baseline == "persistence"
    assert build.config.next_state == "delta" and build.config.spread_inputs == "ellipse"
    assert config.learner.next_state == config.learner.correction == 0.0033


FOURIER_SETS = (
    "yemong.order=temporal_first",
    "yemong.next_state=delta",
    "yemong.delta_baseline=extrapolation",
    "yemong.spread_inputs=scalar",
    "learner.next_state=0.0033",
    "yemong.delta_split=true",
    "learner.correction=0.0033",
    "yemong.position_inputs=fourier",
)


def test_fourier_inputs_are_off_by_default_and_need_the_delta_head():
    assert YemongConfig().position_inputs == "colour"
    with pytest.raises(ValueError, match="position_inputs 'fourier' needs next_state 'delta'"):
        YemongConfig(position_inputs="fourier")
    with pytest.raises(ValueError, match="position_inputs"):
        YemongConfig(next_state="delta", position_inputs="polar")


def test_the_fourier_arms_set_line_builds_it_and_old_records_read_as_colour():
    from bnb.cli import parse_setting, with_settings
    from bnb.contracts import GameConfig
    from bnb.models.yemong.model import YemongBuild
    from bnb.runs.serialise import from_json, to_json
    from bnb.runtime.train import TrainConfig, model_build

    config = with_settings(
        TrainConfig(game=GameConfig(5, 10, 2600.0)), map(parse_setting, FOURIER_SETS)
    )
    build = model_build(config)
    assert build.config.position_inputs == "fourier" and build.config.delta_split
    assert build.config.delta_baseline == "extrapolation"
    assert build.config.spread_inputs == "scalar" and not build.config.acceleration_inputs
    record = to_json(build)
    assert record["config"]["position_inputs"] == "fourier"
    assert from_json(YemongBuild, record)[0] == build
    del record["config"]["position_inputs"]  # a policy from before the switch
    old, filled = from_json(YemongBuild, record)
    assert old.config.position_inputs == "colour" and "config.position_inputs" in filled


def test_the_fourier_ship_token_reads_fourier_and_the_code_stays_colour():
    """Only the encoder's ship input changes (437 state columns for 490, the token 500 for
    553): the tokens' code, the next-state baseline, is the colour code."""
    colour = _model(**ARMS["arm4d"])
    fourier = _model(**ARMS["arm4d_fourier"])
    assert colour.state_code.input_width == colour.state_code.width == 490
    assert fourier.state_code.input_width == 437
    width = colour.encoder.ship.first.in_features
    assert fourier.encoder.ship.first.in_features == width - 490 + 437 == 500
    _, trajectory = cases.rollout(fourier, length=4)
    obs, replay = trajectory.obs, trajectory.replay
    reset = cases.resets(obs)
    with torch.no_grad():
        previous = fourier.preceding_motion(trajectory.initial_carry, obs, replay, reset)
        tokens = fourier.tokens(obs, replay, previous)
        plain = colour.tokens(obs, replay, previous)
    assert torch.equal(tokens.code, plain.code)
    assert tokens.ship.shape[-1] == 500 and plain.ship.shape[-1] == 553
    torch.testing.assert_close(tokens.ship[..., 437:], plain.ship[..., 490:], rtol=0, atol=0)
    valid, _, means, spreads = believed(obs.ships, replay)
    position = torch.where(valid[..., None], means[..., 0:2], replay.offset[..., None, :])
    means = torch.cat([position, means[..., 2:]], -1)
    expected = fourier.state_code.inputs(tokens.code, means, spreads)
    torch.testing.assert_close(tokens.ship[..., :437], expected, rtol=0, atol=0)


def test_the_error_head_starts_at_its_marginal_with_a_zero_last_layer():
    from bnb.models.yemong.model import Yemong, yemong_build

    net = Yemong(
        yemong_build(YemongConfig(**SETTINGS), api.rules_of(cases.SQUAD), api.ship_types())
    )
    last = net.correction_head.lasts[0]
    assert last.weight.abs().max() == 0
    torch.testing.assert_close(last.bias, net.correction_code.marginal_logits())
    parts = net.correction_code.split(torch.exp(last.bias.detach().double()))
    for name, part in parts.items():
        assert math.isclose(float(part.sum()), 1.0, rel_tol=1e-5), name


# arm 3 (persistence), arm 4 (extrapolation) and arm 4c (extrapolation without the
# acceleration inputs): every test on a segment runs on each
ARMS = dict(
    arm3=dict(),
    arm4=dict(delta_baseline="extrapolation"),
    arm4c=dict(delta_baseline="extrapolation", acceleration_inputs=False),
    # the split and extrapolation on C-fixed's inputs with the scalar spreads (arms 3d, 4d)
    arm3d=dict(acceleration_inputs=False, spread_inputs="scalar"),
    arm4d=dict(delta_baseline="extrapolation", acceleration_inputs=False, spread_inputs="scalar"),
    # arm 4d with the Fourier position and heading inputs
    arm4d_fourier=dict(
        delta_baseline="extrapolation",
        acceleration_inputs=False,
        spread_inputs="scalar",
        position_inputs="fourier",
    ),
)


@pytest.fixture(scope="module", params=tuple(ARMS))
def segment(request, module_global_generator):
    net = _model(**ARMS[request.param])
    _, trajectory = cases.rollout(net, length=32)
    return net, trajectory


def test_the_arms_unroll_equals_a_loop_of_steps(segment):
    net, trajectory = segment
    net = cases.double(net)
    obs = cases.cast(trajectory.obs, F64)
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        decisions, final = cases.step_loop(
            net, net.step, carry, obs, cases.joint_actions(trajectory)
        )
        unrolled = net.unroll(carry, obs, decisions.replay, cases.resets(obs))
    pairs = (
        ("policy_logits", unrolled.policy_logits),
        ("next_state_logits", unrolled.next_state_logits[..., OWN:, :]),
        ("correction_logits", unrolled.correction_logits[..., OWN:, :]),
        ("values", unrolled.values),
    )
    for name, actual in pairs:
        assert cases.scale_error(actual, getattr(decisions, name)) <= 1e-9, name
    assert int(final.nonfinite.sum()) == 0
    hidden = decisions.replay.valid & (decisions.replay.age > 0)
    assert hidden.any()
    visible = decisions.replay.valid & (decisions.replay.age == 0)
    assert (decisions.replay.spreads[visible] == 0).all()


def test_the_stored_belief_of_a_hidden_row_is_the_corrected_forecast(segment):
    """Each hidden row's belief at t + 1 is the corrected forecast of its belief at t."""
    net, trajectory = segment
    obs, replay = trajectory.obs, trajectory.replay
    with torch.no_grad():
        unrolled = net.unroll(trajectory.initial_carry, obs, replay, cases.resets(obs))
    _, _, means, spreads = believed(obs.ships, replay)
    ship_type = labels.privileged_rows(trajectory)[..., SHIP["ship_type"]].long()
    hidden = torch.cat([torch.zeros_like(replay.valid[..., :OWN]), replay.age > 0], -1)
    forecast, _ = net.forecast(
        unrolled.next_state_logits,
        means,
        spreads,
        ship_type,
        unrolled.correction_logits,
        hidden,
        labels.rates(net, trajectory),
    )
    # still hidden at t + 1, no reset between: the stored belief is the forecast
    still = (replay.age[1:] > 1) & replay.valid[1:] & replay.valid[:-1]
    still = still & ~cases.resets(obs)[1:, :, None]
    assert still.any()
    torch.testing.assert_close(
        replay.means[1:][still], forecast[:-1, :, OWN:][still], rtol=1e-5, atol=1e-3
    )


# ------------------------------------------------------------------ the learners


def _check_terms(net, learner, runner):
    prepared = learner.prepare(runner.store, runner.bootstrap())
    views = torch.arange(runner.store.views)
    trajectory, shot_labels = next(learner.staged(runner.store, [views]))
    label = learner.labels(trajectory, shot_labels)
    denominator = dict(zip(COUNTS, prepared["counts"].sum(0).clamp_min(1).unbind()))
    terms, _, _, out, _ = learner._terms(
        trajectory, label, prepared["derived"], denominator, prepared["scale"]
    )
    mask = label["next_state_mask"]
    hidden = torch.cat(
        [torch.zeros_like(trajectory.replay.age[..., :OWN]), trajectory.replay.age], -1
    )
    hidden = hidden > 0
    assert torch.equal(label["physics_mask"], mask & ~hidden)
    assert torch.equal(label["correction_mask"], mask & hidden)
    assert label["correction_mask"].any() and label["physics_mask"].any()
    count = float(mask.sum())
    # the physics target: the true change from the baseline (extrapolation: the rates read)
    rows = labels.privileged_rows(trajectory)
    following = labels.next_rows(trajectory, rows)
    offset = trajectory.replay.offset
    expected, _, _ = net.delta_code.targets(
        truth(rows, offset),
        truth(following, offset),
        rows[..., SHIP["ship_type"]].long(),
        labels.rates(net, trajectory),
    )
    torch.testing.assert_close(label["next_state"], expected)
    physics = codes.grouped_cross_entropy(
        out.next_state_logits, label["next_state"], net.delta_code.runs
    ).sum(-1)
    assert math.isclose(
        float(terms["next_state"]),
        float(physics[label["physics_mask"]].sum()) / count,
        rel_tol=1e-5,
    )
    # the error head's target: the truth at t + 1 less the physics forecast of the belief
    rows = labels.privileged_rows(trajectory)
    following = labels.next_rows(trajectory, rows)
    _, _, means, spreads = believed(trajectory.obs.ships, trajectory.replay)
    ship_type = rows[..., SHIP["ship_type"]].long()
    forecast, _ = net.delta_code.advance(
        means,
        spreads,
        out.next_state_logits.detach(),
        ship_type,
        net.state_code,
        labels.rates(net, trajectory),
    )
    errors = net.correction_code.errors(
        forecast, truth(following, trajectory.replay.offset), ship_type
    )
    target, _ = net.correction_code.encode(errors)
    error_entropy = codes.grouped_cross_entropy(
        out.correction_logits, target, net.correction_code.runs
    ).sum(-1)
    expected = float(error_entropy[label["correction_mask"]].sum()) / count
    assert math.isclose(float(terms["correction"]), expected, rel_tol=1e-5)
    # the stop-gradient: the error head's loss gives the physics head nothing
    physics_head = net.ship_heads.lasts[1].weight
    gradient = torch.autograd.grad(
        terms["correction"],
        [physics_head, net.correction_head.lasts[0].weight, net.encoder.ship.first.weight],
        allow_unused=True,
    )
    assert gradient[0] is None or gradient[0].abs().max() == 0
    assert gradient[1].abs().max() > 0 and gradient[2].abs().max() > 0
    return prepared, trajectory, label, denominator


def test_the_ppo_terms_and_their_series(segment):
    net, _ = segment
    runner = lc.rollout(net)
    learner = lc.learner(net, runner)
    prepared, trajectory, label, denominator = _check_terms(net, learner, runner)
    with torch.no_grad():
        total = learner._loss(
            trajectory, label, prepared["derived"], denominator, prepared["scale"], heads=True
        )[-1]
    series = metrics.read(head_statistics.series(total, net))
    bucket = head_statistics.AGE_BUCKETS[0][0]
    for name in correction.QUANTITIES:
        for kind in ("realised_error", "predicted_error", "within_one_sigma"):
            assert f"correction/{kind}/{name}/{bucket}" in series
    for name in correction.STEP_LIMITS:
        assert 0.0 <= series[f"correction/capped_share/{name}"] <= 1.0
        assert 0.0 <= series[f"correction/step_over_limit/{name}"] <= 1.0
        assert 0.0 <= series[f"correction/gain/{name}"] <= 1.0
    for name in correction.COMPONENT_NAMES:
        assert 0.0 <= series[f"correction/support_clip/{name}"] <= 1.0
    assert math.isfinite(series[f"next_state/hidden/error/position_metres/{bucket}"])
    assert math.isfinite(series[f"next_state/hidden/cross_entropy/{bucket}"])
    assert "correction" in TERMS


def test_the_bc_terms(segment):
    from test_bc import CHARACTERISTICS, PARAMETERS, POOLS, TEACHER

    net, _ = segment
    controllers = {
        "learner": cases.controller(net, cases.SHORT_SQUAD),
        "scripted": Scripted(CHARACTERISTICS, OWN, PARAMETERS),
    }
    runner = Rollout(
        cases.SHORT_SQUAD,
        api.ship_types(lc.CPU),
        POOLS,
        controllers,
        "learner",
        8,
        32,
        lc.CPU,
        compiled=False,
        extra_reads=Scripted.reads,
    )
    for _ in range(2):
        runner.run()
    views = runner.store.views
    learner = bc.BehaviourCloning(
        net,
        bc.config(minibatches=1, micro_batch_rows=views * runner.rules.num_ships),
        runner.rules,
        runner.carry["types"],
        lc.RewardConfig().outcome,
        lc.CPU,
        TEACHER,
        bc.TeacherConfig(),
        tuple(pool.name for pool in POOLS),
        runner.lineup.pool_of["learner"],
    )
    _check_terms(net, learner, runner)


def test_an_update_with_the_cadence_reports_both_heads_trunk_shares(segment):
    net, _ = segment
    runner = lc.rollout(net)
    learner = lc.learner(net, runner)
    prepared = learner.prepare(runner.store, runner.bootstrap())
    views = torch.arange(runner.store.views)
    series = metrics.read(learner._gradient_decomposition(runner.store, prepared, views))
    for term in ("next_state", "correction"):
        assert series[f"gradient/norm/trunk/{term}"] > 0
    assert series["gradient/norm/trunk/next_state_enemy_hidden"] == 0  # the physics mask


def test_the_arms_policy_file_round_trips(tmp_path, segment):
    net, trajectory = segment
    save_policy(tmp_path / "policy", net)
    loaded = load_policy(tmp_path / "policy", lc.CPU).eval()
    assert loaded.build == net.build and loaded.config.delta_split
    obs = trajectory.obs
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    joint = cases.joint_actions(trajectory)
    with torch.no_grad():  # each loop draws the same offsets at its resets
        torch.manual_seed(9)
        a, _ = cases.step_loop(net, net.step, carry, obs, joint)
        torch.manual_seed(9)
        b, _ = cases.step_loop(loaded, loaded.step, carry, obs, joint)
    assert torch.equal(a.correction_logits, b.correction_logits)
    assert torch.equal(a.carry.belief.means, b.carry.belief.means)
    assert torch.equal(a.carry.belief.spreads, b.carry.belief.spreads)


def test_the_baseline_rates_trace_whole_under_torch_compile(segment):
    """The learner compiles its labels (`Learner._codes`, fullgraph): the extrapolation
    baseline's segment rates must trace without a graph break. dataclasses.fields on Rates
    broke it (arm 4d's first GPU update, 2026-11-08); the eager backend runs dynamo's tracing
    alone."""
    net, trajectory = segment
    eager = labels.rates(net, trajectory)
    if eager is None:
        return  # persistence: no rates
    torch._dynamo.reset()
    with torch.no_grad():
        compiled = torch.compile(labels.rates, fullgraph=True, backend="eager")(net, trajectory)
        eager = labels.rates(net, trajectory)
    for name in delta.RATE_FIELDS:
        assert torch.equal(getattr(compiled, name), getattr(eager, name)), name
