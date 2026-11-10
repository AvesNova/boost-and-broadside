"""The persistence delta heads (arm-2 brief): the symlog HL-Gauss code (asymmetric for the
position along the track), the velocity frame and its fallback, the median rule, the
covariance advance, the ellipse inputs, the switches (absolute unchanged, the arm's
unroll a loop of steps), the term in the learners and a policy-file round trip."""

import math

import learner_cases as lc
import pytest
import torch
import yemong_cases as cases

from bnb.contracts.obs import SHIP
from bnb.controllers.scripted import Scripted
from bnb.game import api
from bnb.learn import bc, head_statistics, labels
from bnb.learn.ppo import COUNTS, TERMS
from bnb.models.yemong import codes, delta
from bnb.models.yemong.inputs import ship_inputs, truth
from bnb.models.yemong.model import Yemong, YemongConfig, yemong_build
from bnb.models.yemong.state_code import (
    COVARIANCE_ENTRIES,
    MEANS,
    SPREADS,
    ellipse,
    pack_covariance,
    state_code_ranges,
    unpack_covariance,
)
from bnb.runs.checkpoint import load_policy, save_policy
from bnb.runtime import metrics
from bnb.runtime.rollout import Rollout

pytestmark = pytest.mark.usefixtures("global_generator")

OWN = 5
F64 = torch.float64
DT = 1.0 / 30.0
# the arm's --set line
SETTINGS = dict(
    order="temporal_first",
    spread_fit="coarsest",
    acceleration_inputs=True,
    next_state="delta",
    delta_baseline="persistence",
    spread_inputs="ellipse",
)


def _model(**settings):
    return cases.model(**{**SETTINGS, **settings})


def _code() -> delta.DeltaCode:
    rules, types = api.rules_of(cases.SQUAD), api.ship_types()
    return delta.DeltaCode(state_code_ranges(rules, types)).double()


# ------------------------------------------------------------------ the code


@pytest.mark.parametrize("component", delta.COMPONENTS, ids=lambda c: c.name)
def test_the_code_round_trips_below_1e_3_bins_everywhere_including_the_padding(component):
    """Encode then decode (the mean, with the code's mean inverse) and the median: below
    1e-3 bins from the lowest padding centre to the highest, in float64 and float32."""
    code = _code()
    low, high = component.low, component.width - 1 - component.low
    coordinate = torch.linspace(-low, high, 6001, dtype=F64)
    for dtype in (F64, torch.float32):
        values = codes.symlog_value(coordinate, component.knee, component.spacing).to(dtype)
        encoded, clipped = component.encode(values)
        assert encoded.shape[-1] == component.width
        assert clipped.sum() <= 2  # the two ends, by rounding only
        arguments = (component.knee, component.bound, component.bins)
        inverse = tuple(
            part.to(dtype) if torch.is_tensor(part) else part
            for part in code.inverse(component.name)
        )
        mean, _ = codes.symlog_decode(encoded, *arguments, below=component.below, inverse=inverse)
        median = codes.symlog_median(encoded, *arguments, below=component.below, inverse=inverse)
        for decoded in (mean, median):
            back = codes.symlog_coordinate(decoded.double(), component.knee, component.spacing)
            assert (back - coordinate).abs().max() < 1e-3


def test_the_code_has_an_exact_zero_and_folds_only_beyond_the_padded_range():
    for component in delta.COMPONENTS:
        zero, clipped = component.encode(torch.zeros(1, dtype=F64))
        assert not clipped.any()
        centre = component.low
        assert torch.argmax(zero[0]) == centre
        assert torch.isclose(zero[0, centre - 1], zero[0, centre + 1])  # symmetric about 0
        mean, variance = codes.symlog_decode(
            zero, component.knee, component.bound, component.bins, below=component.below
        )
        assert mean.abs().max() < 1e-12 and variance.abs().max() < 1e-12
        top = codes.symlog_value(
            torch.tensor([float(component.width - 1 - component.low)], dtype=F64),
            component.knee,
            component.spacing,
        )
        inside, clip_inside = component.encode(top * (1 - 1e-9))
        beyond, clip_beyond = component.encode(top * 1.5)
        assert not clip_inside.any() and clip_beyond.all()
        torch.testing.assert_close(beyond, inside)  # clamped onto the outermost centre


def test_position_along_resolves_the_move_to_5_cm():
    """The whole move, |v| dt (median 2.6 m): a bin 4.93 cm wide at the median, 10 cm at
    the bound, and a reach of 0.5 m below zero."""
    component = delta.POSITION_ALONG
    index = codes.symlog_coordinate(torch.tensor(2.6, dtype=F64), component.knee, component.spacing)
    centres = codes.symlog_value(
        torch.stack([torch.floor(index), torch.floor(index) + 1]), component.knee, component.spacing
    )
    width = float(centres[1] - centres[0])
    assert 0.045 < width <= 0.05
    lowest = codes.symlog_value(torch.tensor(-19.0, dtype=F64), component.knee, component.spacing)
    assert -0.52 < float(lowest) < -0.5
    assert component.width == 179 and delta.DeltaCode.__doc__.count("624")
    assert _code().width == 624


def test_the_median_takes_the_mode_that_holds_it():
    """A countdown's steady -dt with a reset of p 0.3 to +5 s: the median is -dt (the most
    likely outcome); at p 0.7 it is the reset. The mean (the value at the mean coordinate)
    lies between the two."""
    component = _code().components["shield_recharge_wait"]
    arguments = (component.knee, component.bound, component.bins)
    tick, _ = component.encode(torch.tensor([-DT], dtype=F64))
    jump, _ = component.encode(torch.tensor([5.0], dtype=F64))
    inverse = _code().inverse(component.name)
    inverse = (inverse[0].double(), *inverse[1:])
    for p, expected in ((0.3, -DT), (0.7, 5.0)):
        mixture = (1 - p) * tick + p * jump
        median = codes.symlog_median(mixture, *arguments, inverse=inverse)
        mean, _ = codes.symlog_decode(mixture, *arguments, inverse=inverse)
        assert math.isclose(float(median), expected, rel_tol=1e-3)
        assert -DT < float(mean) < 5.0


# ------------------------------------------------------------------ the frame


def test_the_velocity_frame_and_its_inverse():
    generator = torch.Generator().manual_seed(1)
    velocity = 80.0 * torch.randn(64, 2, generator=generator, dtype=F64)
    heading = torch.rand(64, generator=generator, dtype=F64) * 6.0 - 3.0
    vector = torch.randn(64, 2, generator=generator, dtype=F64)
    unit_x, unit_y, slow = delta.velocity_frame(velocity, heading)
    assert not slow.any()
    framed = delta.to_frame(vector, unit_x, unit_y)
    torch.testing.assert_close(delta.from_frame(framed, unit_x, unit_y), vector)
    along = delta.to_frame(velocity, unit_x, unit_y)
    torch.testing.assert_close(along[:, 0], torch.linalg.vector_norm(velocity, dim=-1))
    torch.testing.assert_close(along[:, 1], torch.zeros(64, dtype=F64))
    # across is to the left: +y for a ship moving along +x
    left = delta.to_frame(
        torch.tensor([[0.0, 1.0]], dtype=F64),
        *delta.velocity_frame(torch.tensor([[50.0, 0.0]], dtype=F64), torch.zeros(1, dtype=F64))[
            :2
        ],
    )
    assert torch.equal(left, torch.tensor([[0.0, 1.0]], dtype=F64))


def test_below_one_metre_per_second_the_frame_is_the_heading():
    velocity = torch.tensor([[0.5, 0.0], [0.0, 0.999], [1.0, 0.0]], dtype=F64)
    heading = torch.full((3,), 0.5 * math.pi, dtype=F64)
    unit_x, unit_y, slow = delta.velocity_frame(velocity, heading)
    assert slow.tolist() == [True, True, False]
    torch.testing.assert_close(unit_x[:2], torch.zeros(2, dtype=F64), atol=1e-15, rtol=0)
    torch.testing.assert_close(unit_y, torch.tensor([1.0, 1.0, 0.0], dtype=F64))


def test_the_changes_are_the_truths_in_the_frame():
    code = _code()
    means = torch.zeros(2, len(MEANS), dtype=F64)
    means[:, 2] = torch.tensor([0.0, 60.0], dtype=F64)  # vx
    means[:, 3] = torch.tensor([60.0, 0.0], dtype=F64)  # vy
    means[:, MEANS.index("heading")] = 3.0
    following = means.clone()
    following[:, 0] += 0.1  # +x: across (right) for the first, along for the second
    following[:, 1] += 2.0  # +y: along for the first, across (left) for the second
    following[:, MEANS.index("heading")] = -3.0  # wraps to +0.283
    following[:, MEANS.index("gun_cooldown")] = -DT
    following[:, MEANS.index("shield")] = 5.0
    continuous, slow = code.changes(means, following, torch.zeros(2, dtype=torch.long))
    assert not slow.any()
    torch.testing.assert_close(continuous[0, :2], torch.tensor([2.0, -0.1], dtype=F64))
    torch.testing.assert_close(continuous[1, :2], torch.tensor([0.1, 2.0], dtype=F64))
    torch.testing.assert_close(continuous[:, 4], torch.full((2,), 2 * math.pi - 6.0, dtype=F64))
    assert torch.allclose(continuous[:, -1], torch.full((2,), -DT, dtype=F64))
    assert torch.allclose(continuous[:, delta.CONTINUOUS.index("shield")], 5.0 / code.max_shield[0])


# ------------------------------------------------------------------ the advance


def _logits_of(code: delta.DeltaCode, continuous: torch.Tensor) -> torch.Tensor:
    encoded, _ = code.encode(continuous)
    return torch.log(encoded.clamp_min(1e-300))


def _belief(net: Yemong, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    means = torch.zeros(rows, len(MEANS), dtype=F64)
    means[:, 0:2] = torch.tensor([100.0, -50.0], dtype=F64)
    means[:, 2:4] = torch.tensor([30.0, 40.0], dtype=F64)  # 50 m/s, unit (0.6, 0.8)
    means[:, MEANS.index("shield")] = 50.0
    means[:, MEANS.index("energy")] = 5000.0
    means[:, MEANS.index("shield_recharge_wait")] = 2.0
    return means, torch.zeros(rows, net.state_code.spread_count, dtype=F64)


def test_a_known_change_moves_the_mean_exactly():
    net = cases.double(_model())
    code = net.delta_code
    means, spreads = _belief(net, 1)
    change = torch.tensor(
        [[1.6, 0.02, 0.3, -0.5, 0.01, 0.2, 0.003, -DT, 0.004, 0.0006, 0.0, 0.0]], dtype=F64
    )
    forecast, _ = code.advance(
        means, spreads, _logits_of(code, change), torch.zeros(1, dtype=torch.long), net.state_code
    )
    unit_x, unit_y = torch.tensor([0.6], dtype=F64), torch.tensor([0.8], dtype=F64)
    expected = means.clone()
    expected[:, 0:2] += delta.from_frame(change[:, 0:2], unit_x, unit_y)
    expected[:, 2:4] += delta.from_frame(change[:, 2:4], unit_x, unit_y)
    expected[:, 4] += 0.01
    expected[:, 5] += 0.2
    expected[:, 6] += 0.003
    expected[:, 7] -= DT
    expected[:, 8] += 0.004 * float(code.max_shield[0])
    expected[:, 9] += 0.0006 * float(code.max_energy[0])
    torch.testing.assert_close(forecast, expected, rtol=1e-4, atol=1e-5)


def test_a_rare_reset_does_not_move_the_wait_but_widens_it():
    """The wait's change: -dt at p 0.8, a reset to 6 s at p 0.2. The forecast counts down
    (the median), its sigma grows by the mixture's spread."""
    net = cases.double(_model())
    code = net.delta_code
    means, spreads = _belief(net, 1)
    change = torch.zeros(2, len(delta.CONTINUOUS), dtype=F64)
    column = delta.CONTINUOUS.index("shield_recharge_wait")
    change[0, column], change[1, column] = -DT, 4.0
    encoded, _ = code.encode(change)
    mixture = 0.8 * encoded[0] + 0.2 * encoded[1]
    forecast, spread = code.advance(
        means,
        spreads,
        torch.log(mixture.clamp_min(1e-300))[None],
        torch.zeros(1, dtype=torch.long),
        net.state_code,
    )
    wait = MEANS.index("shield_recharge_wait")
    assert math.isclose(float(forecast[0, wait]), 2.0 - DT, rel_tol=1e-4)
    # about 0.4 x 4.03 (the spread is taken about the value at the mean coordinate)
    assert 1.5 < float(spread[0, SPREADS.index("shield_recharge_wait")]) < 2.0


def test_the_covariance_follows_f_p_f_t_plus_q_with_the_n_cubed_term():
    """Constant velocity with velocity noise q per tick on each axis and nothing else:
    after n ticks Var x = q dt^2 (n - 1) n (2n - 1) / 6, Cov(x, vx) = q dt n (n - 1) / 2."""
    net = cases.double(_model())
    code = net.delta_code
    means, spreads = _belief(net, 1)
    state_code = net.state_code
    # the velocity's change +-0.5 m/s at even odds on both axes: mean 0, variance 0.25
    change = torch.zeros(2, len(delta.CONTINUOUS), dtype=F64)
    change[0, 2:4], change[1, 2:4] = 0.5, -0.5
    encoded, _ = code.encode(change)
    logits = torch.log(encoded.mean(0).clamp_min(1e-300))[None]
    decoded, variance, _ = code.decode(logits)
    assert decoded[0, 2:4].abs().max() < 1e-12
    q_along, q_across = float(variance[0, 2]), float(variance[0, 3])
    position_q = float(variance[0, 0])
    assert q_along > 0 and q_across > 0
    n = 40
    for _ in range(n):
        means, spreads = code.advance(
            means, spreads, logits, torch.zeros(1, dtype=torch.long), state_code
        )
    matrix = unpack_covariance(spreads[..., state_code.covariance_slice])[0]
    unit = torch.tensor([0.6, 0.8], dtype=F64)
    along = unit @ matrix[0:2, 0:2] @ unit
    expected = q_along * DT**2 * (n - 1) * n * (2 * n - 1) / 6 + n * position_q
    assert math.isclose(float(along), expected, rel_tol=1e-6)
    assert math.isclose(
        float(unit @ matrix[0:2, 2:4] @ unit), q_along * DT * n * (n - 1) / 2, rel_tol=1e-6
    )
    assert math.isclose(float(unit @ matrix[2:4, 2:4] @ unit), n * q_along, rel_tol=1e-6)
    # the SPREADS slots are read off it
    sigma = dict(zip(SPREADS, spreads[0, : len(SPREADS)].tolist()))
    assert math.isclose(sigma["velocity_xx"], float(matrix[2, 2]), rel_tol=1e-12)
    assert math.isclose(
        sigma["position"] ** 2, 0.5 * float(matrix[0, 0] + matrix[1, 1]), rel_tol=1e-12
    )


def test_the_heading_grows_by_its_change_alone_and_is_capped_at_the_circle():
    """The nose rate's variance does not feed the heading (the heading change is the
    whole turn); the heading's variance stops at a uniform circle's, pi^2 / 3, its
    correlations kept."""
    net = cases.double(_model())
    code, state_code = net.delta_code, net.state_code
    means, spreads = _belief(net, 1)
    change = torch.zeros(2, len(delta.CONTINUOUS), dtype=F64)
    heading, nose = delta.CONTINUOUS.index("heading"), delta.CONTINUOUS.index("nose_rate")
    change[0, heading], change[1, heading] = 0.05, -0.05
    change[0, nose], change[1, nose] = 1.0, -1.0
    encoded, _ = code.encode(change)
    logits = torch.log(encoded.mean(0).clamp_min(1e-300))[None]
    _, variance, _ = code.decode(logits)
    q_heading = float(variance[0, heading])
    for _ in range(10):
        means, spreads = code.advance(
            means, spreads, logits, torch.zeros(1, dtype=torch.long), state_code
        )
    matrix = unpack_covariance(spreads[..., state_code.covariance_slice])[0]
    assert math.isclose(float(matrix[4, 4]), 10 * q_heading, rel_tol=1e-9)
    assert float(matrix[4, 5]) == 0.0
    big = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0, 40.0, 4.0], dtype=F64))
    big[4, 5] = big[5, 4] = 10.0
    capped = delta.cap_variance(big, 4, delta.UNIFORM_HEADING_VARIANCE)
    assert math.isclose(float(capped[4, 4]), math.pi**2 / 3)
    correlation = lambda m: float(m[4, 5] / torch.sqrt(m[4, 4] * m[5, 5]))  # noqa: E731
    assert math.isclose(correlation(capped), correlation(big))
    assert torch.equal(capped[:4, :4], big[:4, :4]) and float(capped[5, 5]) == 4.0
    assert delta.UNIFORM_NOSE_RATE_VARIANCE == 64.0 / 3.0


def test_the_covariance_packs_and_unpacks():
    matrix = torch.randn(3, 6, 6, dtype=F64)
    matrix = matrix @ matrix.transpose(-1, -2)
    packed = pack_covariance(matrix)
    assert packed.shape[-1] == COVARIANCE_ENTRIES == 21
    torch.testing.assert_close(unpack_covariance(packed), matrix)


# ------------------------------------------------------------------ the ellipse


def test_a_round_covariance_has_no_angle_and_a_rotated_one_its_angle():
    major, minor, cos, sin = ellipse(torch.tensor([[4.0, 0.0, 4.0], [0.0, 0.0, 0.0]], dtype=F64))
    assert torch.equal(cos, torch.zeros(2, dtype=F64)) and torch.equal(
        sin, torch.zeros(2, dtype=F64)
    )
    torch.testing.assert_close(major, torch.tensor([2.0, 0.0], dtype=F64))
    for angle in (0.3, 1.2, -0.7, 2.9):
        c, s = math.cos(angle), math.sin(angle)
        rotation = torch.tensor([[c, -s], [s, c]], dtype=F64)
        matrix = rotation @ torch.diag(torch.tensor([9.0, 1.0], dtype=F64)) @ rotation.T
        packed = torch.stack([matrix[0, 0], matrix[0, 1], matrix[1, 1]])
        major, minor, cos, sin = ellipse(packed)
        anisotropy = (3.0 - 1.0) / (3.0 + 1.0)
        assert math.isclose(float(major), 3.0) and math.isclose(float(minor), 1.0)
        assert math.isclose(float(cos), anisotropy * math.cos(2 * angle), abs_tol=1e-12)
        assert math.isclose(float(sin), anisotropy * math.sin(2 * angle), abs_tol=1e-12)


def test_the_ellipse_inputs_and_the_per_axis_blur():
    net = _model()
    state_code = net.state_code
    assert state_code.ellipse_width == 8 * 21
    assert net.encoder.ship.first.in_features == ship_inputs(
        state_code.width, net.ship_types, True, 8 * 21
    )
    means = torch.zeros(1, len(MEANS))
    spreads = torch.zeros(1, state_code.spread_count)
    matrix = torch.zeros(6, 6)
    matrix[0, 0], matrix[1, 1] = 400.0, 25.0  # sigma 20 m along x, 5 m along y
    spreads[:, state_code.covariance_slice] = pack_covariance(matrix)
    ship_type = torch.zeros(1, dtype=torch.long)
    code = state_code.encode(means, spreads, ship_type)
    position = code[:, :81].unflatten(-1, (9, 3, 3))
    attenuated_x = position.sum(-1)  # the x code's levels
    attenuated_y = position.sum(-2)
    # a level of period 4 m (the finest) is flat on both axes; at 36 m only y is still sharp
    level = 6  # period 26244 / 3^6 = 36 m
    assert attenuated_x[0, level].max() < attenuated_y[0, level].max()
    ellipse_code = state_code.ellipse_code(spreads)
    assert ellipse_code.shape[-1] == 168
    torch.testing.assert_close(ellipse_code.unflatten(-1, (8, 21)).sum(-1), torch.ones(1, 8))
    scalar = cases.model(**{**SETTINGS, "spread_inputs": "scalar"})
    assert scalar.state_code.ellipse_width == 0


# ------------------------------------------------------------------ the switches


def test_absolute_is_the_default_with_no_covariance_and_no_dynamics():
    config = YemongConfig()
    assert config.next_state == "absolute" and config.spread_inputs == "scalar"
    assert config.delta_baseline == "persistence"
    net = cases.model()
    assert net.delta_code is None and net.build.dynamics is None
    assert not net.state_code.covariance and net.state_code.spread_count == len(SPREADS) + 4
    assert net.ship_heads.lasts[1].out_features == net.state_code.width
    with pytest.raises(ValueError, match="next_state"):
        YemongConfig(next_state="relative")
    with pytest.raises(ValueError, match="delta_baseline"):
        YemongConfig(delta_baseline="rule")
    with pytest.raises(ValueError, match="spread_inputs"):
        YemongConfig(spread_inputs="circle")
    arm = _model()
    assert arm.build.dynamics is None
    assert arm.state_code.spread_count == len(SPREADS) + 4 + COVARIANCE_ENTRIES


def test_round_ones_build_record_is_refused():
    rules, types = api.rules_of(cases.SQUAD), api.ship_types()
    build = yemong_build(YemongConfig(next_state="delta"), rules, types)
    import dataclasses

    old = dataclasses.replace(build, dynamics=delta.state_dynamics(rules, types))
    with pytest.raises(ValueError, match="round 1"):
        Yemong(old)


ARM_SETS = (
    "yemong.order=temporal_first",
    "yemong.acceleration_inputs=true",
    "yemong.next_state=delta",
    "yemong.delta_baseline=persistence",
    "yemong.spread_inputs=ellipse",
)


def test_the_arms_set_line_builds_the_arm():
    from bnb.cli import parse_setting, with_settings
    from bnb.contracts import GameConfig
    from bnb.runtime.train import TrainConfig, model_build

    config = with_settings(
        TrainConfig(game=GameConfig(5, 10, 2600.0)), map(parse_setting, ARM_SETS)
    )
    build = model_build(config)
    assert build.config.next_state == "delta" and build.config.spread_inputs == "ellipse"
    assert build.config.acceleration_inputs and build.config.spread_fit == "coarsest"
    assert build.dynamics is None


def test_a_build_record_from_before_the_switches_loads_scalar(tmp_path):
    import json

    from bnb.runs import checkpoint
    from bnb.runs.checkpoint import FilledDefaults

    checkpoint.save_policy(tmp_path / "policy", cases.model(order="temporal_first"))
    path = tmp_path / "policy" / checkpoint.RECORD
    record = json.loads(path.read_text())
    for name in ("delta_baseline", "spread_inputs"):
        record["model"]["build"]["config"].pop(name)
    path.write_text(json.dumps(record))
    with pytest.warns(FilledDefaults, match="spread_inputs"):
        loaded = checkpoint.load_policy(tmp_path / "policy", cases.CPU)
    assert loaded.config.spread_inputs == "scalar" and loaded.delta_code is None


def test_the_marginal_start_is_near_the_measured_marginal():
    """The output bias is the measured marginal: every component's code sums to 1, the
    zero share sits on the zero bin, and the position along peaks near 2.5 m."""
    net = _model()
    bias = net.ship_heads.lasts[1].bias.detach().double()
    torch.testing.assert_close(bias.float(), net.delta_code.marginal)
    parts = net.delta_code.split(torch.exp(bias))
    for name, part in parts.items():
        assert math.isclose(float(part.sum()), 1.0, rel_tol=1e-5), name
    along = delta.POSITION_ALONG
    peak = int(torch.argmax(parts["position_along"])) - along.low
    value = codes.symlog_value(torch.tensor(float(peak)), along.knee, along.spacing)
    assert 1.5 < float(value) < 3.5
    wait = parts["shield_recharge_wait"]
    assert float(wait[delta.COMPONENTS[7].low]) > 0.35  # 80% exactly 0, half on its bin


@pytest.fixture(scope="module")
def segment(module_global_generator):
    net = _model()
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
        ("values", unrolled.values),
    )
    for name, actual in pairs:
        assert cases.scale_error(actual, getattr(decisions, name)) <= 1e-9, name
    assert int(final.nonfinite.sum()) == 0
    hidden = decisions.replay.valid & (decisions.replay.age > 0)
    assert hidden.any()  # the forecast was fed back
    covariance = decisions.replay.spreads[..., net.state_code.covariance_slice]
    assert (covariance[..., 0][hidden] > 0).all()  # and its covariance grew
    visible = decisions.replay.valid & (decisions.replay.age == 0)
    assert (decisions.replay.spreads[visible] == 0).all()  # a visible row: zero spread


def test_the_forecast_of_a_visible_row_is_its_truth_plus_the_change(segment):
    net, trajectory = segment
    obs, replay = trajectory.obs, trajectory.replay
    with torch.no_grad():
        unrolled = net.unroll(trajectory.initial_carry, obs, replay, cases.resets(obs))
    rows = labels.privileged_rows(trajectory)
    means = truth(rows, replay.offset)
    ship_type = rows[..., SHIP["ship_type"]].long()
    zero = torch.zeros(means.shape[:-1] + (net.state_code.spread_count,))
    forecast, _ = net.forecast(unrolled.next_state_logits, means, zero, ship_type)
    change, _, _ = net.delta_code.decode(unrolled.next_state_logits)
    unit_x, unit_y, _ = delta.velocity_frame(means[..., 2:4], means[..., MEANS.index("heading")])
    torch.testing.assert_close(
        forecast[..., 0:2], means[..., 0:2] + delta.from_frame(change[..., 0:2], unit_x, unit_y)
    )


# ------------------------------------------------------------------ the learners


def _grouped_cross_entropy(logits, target, runs) -> torch.Tensor:
    total, start = 0.0, 0
    for groups, size in runs:
        width = groups * size
        part = logits[..., start : start + width].unflatten(-1, (groups, size))
        goal = target[..., start : start + width].unflatten(-1, (groups, size))
        total = total - (goal * part.log_softmax(-1)).sum((-1, -2))
        start += width
    return total


def _check_term(net, learner, runner):
    prepared = learner.prepare(runner.store, runner.bootstrap())
    views = torch.arange(runner.store.views)
    trajectory, shot_labels = next(learner.staged(runner.store, [views]))
    label = learner.labels(trajectory, shot_labels)
    denominator = dict(zip(COUNTS, prepared["counts"].sum(0).clamp_min(1).unbind()))
    with torch.no_grad():
        _, terms, _, _, _, _ = learner.loss(
            trajectory, label, prepared["derived"], denominator, prepared["scale"]
        )
        out = learner.unroll(trajectory)
    got = dict(zip(TERMS, terms.tolist()))
    # the truth's change from t to t + 1, on the rows alive at both
    ships = trajectory.obs.ships
    now = torch.cat([ships[..., :OWN, :], trajectory.truth], -2)
    following = torch.cat([now[1:], trajectory.final_truth[None]], 0)
    offset = trajectory.replay.offset
    target, clipped, _ = net.delta_code.targets(
        truth(now, offset), truth(following, offset), now[..., SHIP["ship_type"]].long()
    )
    valid = torch.cat([torch.ones_like(trajectory.replay.valid), trajectory.replay.valid], -1)
    alive = (now[..., SHIP["alive"]] > 0.5) & (following[..., SHIP["alive"]] > 0.5)
    mask = valid & ~trajectory.done[..., None] & alive
    assert torch.equal(label["next_state_mask"], mask)
    ce = _grouped_cross_entropy(out.next_state_logits, target, net.delta_code.runs)
    assert math.isclose(got["next_state"], ce[mask].mean().item(), rel_tol=1e-5)
    assert torch.equal(label["next_state_clipped"], clipped)
    return label, trajectory, out


def test_the_ppo_term_is_the_delta_codes_cross_entropy_and_its_series(segment):
    net, _ = segment
    runner = lc.rollout(net)
    learner = lc.learner(net, runner)
    label, trajectory, out = _check_term(net, learner, runner)
    prepared = learner.prepare(runner.store, runner.bootstrap())
    denominator = dict(zip(COUNTS, prepared["counts"].sum(0).clamp_min(1).unbind()))
    with torch.no_grad():
        heads_read = learner._loss(
            trajectory, label, prepared["derived"], denominator, prepared["scale"]
        )[-1]
        total = learner._head_sums(heads_read, trajectory, label)
    series = metrics.read(head_statistics.series(total, net))
    for component in delta.CONTINUOUS:
        assert 0.0 <= series[f"next_state/support_clip/{component}"] <= 1.0
        assert 0.0 <= series[f"next_state/support_padding/{component}"] <= 1.0
    for predictor in ("head", "persistence"):
        assert math.isfinite(series[f"next_state/error/position_metres/{predictor}"])
    assert "next_state/error/position_metres/baseline" not in series
    for name in head_statistics.BENCH_MEDIANS:
        for baseline in head_statistics.BENCH_BASELINES:
            assert 0.0 <= series[f"next_state/beats_bench_median/{name}/{baseline}"] <= 1.0
    assert 0.0 <= series["next_state/slow_frame_share"] <= 1.0
    for situation in head_statistics.NEXT_STATE_SITUATIONS:
        assert f"next_state/cross_entropy/{situation}/position" in series
    assert math.isfinite(series["next_state/gain/ally"])
    belief = metrics.read(
        head_statistics.belief_series(head_statistics.belief_sums(trajectory, DT, net.state_code))
    )
    bucket = head_statistics.AGE_BUCKETS[1][0]
    for name in ("position", "velocity"):
        value = belief[f"belief/anisotropy/{name}/{bucket}"]
        assert math.isnan(value) or 0.0 <= value <= 1.0
    assert f"belief/within_one_sigma/position_metres/{bucket}" in belief


def test_the_bc_term_is_the_delta_codes_cross_entropy(segment):
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
    _check_term(net, learner, runner)


def test_the_arms_policy_file_round_trips(tmp_path, segment):
    net, trajectory = segment
    save_policy(tmp_path / "policy", net)
    loaded = load_policy(tmp_path / "policy", lc.CPU).eval()
    assert loaded.build == net.build and loaded.config.spread_inputs == "ellipse"
    obs = trajectory.obs
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    joint = cases.joint_actions(trajectory)
    with torch.no_grad():  # each loop draws the same offsets at its resets
        torch.manual_seed(9)
        a, _ = cases.step_loop(net, net.step, carry, obs, joint)
        torch.manual_seed(9)
        b, _ = cases.step_loop(loaded, loaded.step, carry, obs, joint)
    assert torch.equal(a.next_state_logits, b.next_state_logits)
    assert torch.equal(a.carry.belief.means, b.carry.belief.means)
    assert torch.equal(a.carry.belief.spreads, b.carry.belief.spreads)
