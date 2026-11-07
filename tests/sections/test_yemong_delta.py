"""The next-state delta heads (delta-heads brief): the symlog HL-Gauss code, the physics
baselines against the game, the frame, the belief advance, the switch (absolute unchanged,
delta's unroll a loop of steps), the term in the learners and a policy-file round trip."""

import math

import learner_cases as lc
import pytest
import torch
import yemong_cases as cases

from bnb.contracts.action import Action
from bnb.contracts.obs import SHIP
from bnb.controllers.scripted import Scripted
from bnb.game import api
from bnb.game.physics import flight
from bnb.game.physics.combat import apply_damage
from bnb.game.physics.numerics import flight_path_velocity
from bnb.game.state import Fields, apply, replace
from bnb.learn import bc, head_statistics, labels
from bnb.learn.ppo import COUNTS, TERMS
from bnb.models.yemong import codes, delta
from bnb.models.yemong.inputs import truth
from bnb.models.yemong.model import Yemong, YemongConfig
from bnb.models.yemong.state_code import MEANS, SPREADS
from bnb.runs.checkpoint import load_policy, save_policy
from bnb.runtime import metrics
from bnb.runtime.rollout import Rollout

pytestmark = pytest.mark.usefixtures("global_generator")

OWN = 5
F64 = torch.float64
SETTINGS = dict(order="temporal_first", spread_fit="coarsest", next_state="delta")


def _model(**settings):
    return cases.model(**{**SETTINGS, **settings})


# ------------------------------------------------------------------ the code


@pytest.mark.parametrize("component", delta.COMPONENTS, ids=lambda c: c.name)
def test_the_code_round_trips_without_bias_up_to_the_real_edges(component):
    """Encode then decode: the mean coordinate's bias is below 1e-3 bins across the real
    range up to 0.06 bins short of its last centre, and 1.3e-3 at it (the Gaussian's mass
    past the 2 padding bins, 4e-4, is dropped and the code renormalised; the coordinator's
    no-fold rule), in float64 and float32."""
    half = (component.bins - 1) // 2
    spacing = codes.symlog_spacing(component.knee, component.bound, component.bins)
    index = torch.arange(-half - 2, half + 3, dtype=F64)
    coordinate = torch.linspace(-half, half, 4001, dtype=F64)
    for dtype in (F64, torch.float32):
        values = codes.symlog_value(coordinate, component.knee, spacing).to(dtype)
        code, clipped = codes.symlog_encode(values, component.knee, component.bound, component.bins)
        assert not clipped.any()
        assert code.shape[-1] == component.width
        bias = ((code.double() * index).sum(-1) - coordinate).abs()
        inside = coordinate.abs() <= half - 0.06
        assert bias[inside].max() < 1e-3, bias[inside].max()
        assert bias.max() < 1.3e-3, bias.max()
        mean, _ = codes.symlog_decode(code, component.knee, component.bound, component.bins)
        tolerance = 1e-9 if dtype == F64 else 1e-5
        relative = (mean.double() - values.double()).abs() / (
            values.double().abs() + component.knee
        )
        assert relative[inside].max() < 1.3e-3 * spacing + tolerance


def test_the_code_has_an_exact_zero_and_symlog_spacing():
    component = delta.COMPONENTS[0]
    spacing = codes.symlog_spacing(component.knee, component.bound, component.bins)
    half = (component.bins - 1) // 2
    centres = codes.symlog_value(torch.arange(-half, half + 1, dtype=F64), component.knee, spacing)
    assert centres[half] == 0.0  # the middle centre is exactly 0
    assert centres[-1] == pytest.approx(component.bound, rel=1e-12)  # the bound: the last centre
    # symlog: ln(1 + x / knee) steps evenly; the first bin is about one knee at this count
    steps = torch.diff(torch.log1p(centres[half:] / component.knee))
    torch.testing.assert_close(steps, torch.full_like(steps, spacing))
    code, _ = codes.symlog_encode(torch.zeros(3, dtype=F64), component.knee, component.bound, 51)
    torch.testing.assert_close(code, code.flip(-1))  # symmetric about the centre
    mean, variance = codes.symlog_decode(code, component.knee, component.bound, 51)
    assert (mean == 0).all() and (variance == 0).all()


def test_only_a_change_past_the_padded_range_is_clamped_and_nothing_folds():
    component = delta.COMPONENTS[4]
    spacing = codes.symlog_spacing(component.knee, component.bound, component.bins)
    half = (component.bins - 1) // 2 + codes.PADDING_BINS
    edge = codes.symlog_value(torch.tensor([float(half)], dtype=F64), component.knee, spacing)
    values = torch.cat([edge * 0.999, edge, edge * 10.0, -edge * 3.0])
    code, clipped = codes.symlog_encode(values, component.knee, component.bound, component.bins)
    assert clipped.tolist() == [False, False, True, True]
    torch.testing.assert_close(code[2], code[1])  # clamped to the outermost centre
    torch.testing.assert_close(code[3], code[1].flip(-1))
    # no fold: the last bin holds only its own Gaussian share, the code renormalised over the
    # half of the Gaussian inside the padded range and the last half bin
    integral = 0.5 * math.erf(0.5 / (0.75 * math.sqrt(2.0)))
    assert code[1, -1] == pytest.approx(2.0 * integral / (0.5 + integral), rel=1e-9)
    assert code.sum(-1).sub(1).abs().max() < 1e-12


def test_the_code_widths_and_the_marginal_start():
    net = _model()
    code = net.delta_code
    assert code.width == 2 * 2 * 55 + 6 * 35 + 2 * 2 == 434
    assert net.ship_heads.lasts[1].out_features == 434
    fresh = Yemong(net.build)  # cases.model randomises the last layers
    assert (fresh.ship_heads.lasts[1].weight == 0).all()
    torch.testing.assert_close(fresh.ship_heads.lasts[1].bias.detach(), code.marginal)
    parts = code._split(torch.exp(codes.grouped_log_softmax(code.marginal, code.runs)))
    for component in delta.COMPONENTS:
        centre = parts[component.name][(component.width - 1) // 2]
        assert centre >= delta.ZERO_SHARE[component.name] * 0.5  # the zero's HL-Gauss share
    for name in delta.EVENTS:
        assert parts[name][1] == pytest.approx(delta.EVENT_RATE[name], rel=1e-5)


# ------------------------------------------------------------------ baselines


def _states(ticks: int = 20, batch: int = 8):
    """Real game states after a few ticks of random play, in float64, out of every field."""
    rules = api.rules_of(cases.SHORT_SQUAD)
    types = api.ship_types()
    generator = torch.Generator().manual_seed(3)
    state = api.reset(rules, types, api.uniforms(rules, batch, generator=generator))
    N = rules.num_ships
    for _ in range(ticks):
        action = Action(
            power=torch.randint(0, 3, (batch, N), generator=generator, dtype=torch.int32),
            stick=torch.randint(0, 7, (batch, N), generator=generator, dtype=torch.int32),
            shoot=torch.zeros((batch, N), dtype=torch.int32),
        )
        state, _ = api.step(
            rules,
            types,
            state,
            action,
            api.spread(rules, batch, generator=generator),
            api.uniforms(rules, batch, generator=generator),
        )
    ships = apply(lambda x: x.double() if x.is_floating_point() else x, state.ships)
    zero = torch.zeros_like(ships.x)
    norm = torch.rsqrt(ships.heading_cos**2 + ships.heading_sin**2)
    angle_norm = torch.rsqrt(ships.angle_of_attack_cos**2 + ships.angle_of_attack_sin**2)
    ships = replace(
        ships,
        heading_cos=ships.heading_cos * norm,
        heading_sin=ships.heading_sin * norm,
        angle_of_attack_cos=ships.angle_of_attack_cos * angle_norm,
        angle_of_attack_sin=ships.angle_of_attack_sin * angle_norm,
        log_refractive_index=zero,
        log_index_gradient_x=zero,
        log_index_gradient_y=zero,
    )
    return rules, types, ships


def _means(rules, ships) -> torch.Tensor:
    """The state in state_code.MEANS order, as the observation gives it."""
    speed = ships.proper_speed / torch.exp(ships.log_refractive_index)
    velocity_x, velocity_y = flight_path_velocity(
        speed,
        ships.heading_cos,
        ships.heading_sin,
        ships.angle_of_attack_cos,
        ships.angle_of_attack_sin,
    )
    return torch.stack(
        [
            ships.x,
            ships.y,
            velocity_x,
            velocity_y,
            torch.atan2(ships.heading_sin, ships.heading_cos),
            ships.nose_rate,
            ships.angle_of_attack,
            ships.shield_recharge_wait,
            ships.shield,
            ships.energy,
            ships.gun_cooldown.double() * rules.tick_length,
            ships.log_refractive_index,
        ],
        -1,
    )


FAR = Fields(
    x=torch.tensor([[1e9]], dtype=F64),
    y=torch.zeros((1, 1), dtype=F64),
    radius=torch.ones((1, 1), dtype=F64),
    band_width=torch.ones((1, 1), dtype=F64),
    level=torch.ones((1, 1), dtype=torch.int32),
)


def test_the_kinematic_baselines_are_the_games_drift(monkeypatch):
    """With no kick (no controls, no forces) and no field, the game's tick is the drift
    alone: position along the nose arc, velocity and heading turned by omega dt, everything
    else persisting. The baselines equal it."""
    rules, types, ships = _states()
    assert ships.nose_rate.abs().max() > 0.5  # the arcs are real turns
    monkeypatch.setattr(flight, "kick", lambda constants, airframe, ship, *rest: ship)
    airframe = flight.airframe_of(types, ships.ship_type)
    zero = torch.zeros_like(ships.x)
    moved = flight.flight_tick(rules.flight, airframe, rules.tick_length, ships, zero, zero, FAR)
    code = _model().delta_code.double()
    before, after = _means(rules, ships), _means(rules, moved)
    baseline = code.baseline(before, ships.ship_type.long())
    kinematic = [MEANS.index(name) for name in MEANS if name not in ("shield", "gun_cooldown")]
    kinematic.remove(MEANS.index("shield_recharge_wait"))
    error = (baseline - after)[..., kinematic].abs()
    assert error.max() < 1e-9 * (1.0 + after[..., kinematic].abs().max()), error.max()
    changes, events = code.changes(before, after, ships.ship_type.long())
    position = changes[..., :2].abs().max()
    assert position < 1e-9 and not events.any()


def test_the_shield_and_the_countdowns_are_the_games():
    rules, types, ships = _states()
    code = _model().delta_code.double()
    ship_type = ships.ship_type.long()
    # Some shields below the maximum, some recharge waits under a tick.
    generator = torch.Generator().manual_seed(4)
    maximum = code.max_shield.double()[ship_type]
    ships = replace(
        ships,
        shield=maximum * torch.rand(ships.x.shape, generator=generator, dtype=F64),
        shield_recharge_wait=torch.rand(ships.x.shape, generator=generator, dtype=F64) * 0.1,
    )
    unhurt = apply_damage(rules, types, ships, ships.alive, torch.zeros_like(ships.x)).ships
    baseline = code.baseline(_means(rules, ships), ship_type)
    for name in ("shield", "shield_recharge_wait"):
        expected = getattr(unhurt, name)
        torch.testing.assert_close(baseline[..., MEANS.index(name)], expected)
    hurt = apply_damage(rules, types, ships, ships.alive, torch.full_like(ships.x, 1.0)).ships
    _, events = code.changes(_means(rules, ships), _means(rules, hurt), ship_type)
    assert events[..., 0].all() and not events[..., 1].any()


def test_a_shot_is_the_cooldown_event_at_the_types_interval():
    rules = api.rules_of(cases.SHORT_SQUAD)
    types = api.ship_types()
    batch, N = 4, rules.num_ships
    generator = torch.Generator().manual_seed(5)
    state = api.reset(rules, types, api.uniforms(rules, batch, generator=generator))
    action = Action(
        power=torch.zeros((batch, N), dtype=torch.int32),
        stick=torch.zeros((batch, N), dtype=torch.int32),
        shoot=torch.ones((batch, N), dtype=torch.int32),
    )
    code = _model().delta_code
    means = []
    for _ in range(3):
        means.append(_means(rules, state.ships).float())
        state, _ = api.step(
            rules,
            types,
            state,
            action,
            api.spread(rules, batch, generator=generator),
            api.uniforms(rules, batch, generator=generator),
        )
    means.append(_means(rules, state.ships).float())
    ship_type = state.ships.ship_type.long()
    cooldown = MEANS.index("gun_cooldown")
    shots = 0
    for before, after in zip(means[:-1], means[1:]):
        _, events = code.changes(before, after, ship_type)
        fired = events[..., 1]
        shots += int(fired.sum())
        expected = code.fired_cooldown[ship_type]
        torch.testing.assert_close(after[..., cooldown][fired], expected[fired])
        countdown = torch.clamp_min(before[..., cooldown] - rules.tick_length, 0.0)
        torch.testing.assert_close(after[..., cooldown][~fired], countdown[~fired])
    assert shots > 0


def test_the_nose_frame_and_its_inverse():
    generator = torch.Generator().manual_seed(6)
    vector = torch.randn(100, 2, generator=generator, dtype=F64)
    heading = (torch.rand(100, generator=generator, dtype=F64) - 0.5) * 2.0 * math.pi
    torch.testing.assert_close(delta.from_nose(delta.to_nose(vector, heading), heading), vector)
    nose = torch.stack([torch.cos(heading), torch.sin(heading)], -1)
    along = delta.to_nose(3.0 * nose, heading)
    torch.testing.assert_close(along, torch.stack([torch.full_like(heading, 3.0), 0 * heading], -1))
    left = torch.stack([-torch.sin(heading), torch.cos(heading)], -1)
    torch.testing.assert_close(delta.to_nose(left, heading)[..., 1], torch.ones_like(heading))


# ------------------------------------------------------------------ the advance


def _belief(rows: int = 6):
    generator = torch.Generator().manual_seed(7)
    rules, _, ships = _states(batch=1)
    means = _means(rules, ships)[0, :rows]
    spreads = torch.rand(rows, len(SPREADS), generator=generator, dtype=F64)
    spreads[:, 2] = 0.0  # a positive definite velocity covariance
    cooldown_bins = 4
    spreads = torch.cat([spreads, torch.zeros(rows, cooldown_bins, dtype=F64)], -1)
    return means, spreads, ships.ship_type[0, :rows].long()


def _logits_of(code, continuous, events):
    target, _ = code.encode(continuous, events)
    return torch.log(target.clamp_min(1e-30))


def test_a_known_change_moves_the_mean_exactly():
    net = cases.double(_model())
    code = net.delta_code
    means, spreads, ship_type = _belief()
    generator = torch.Generator().manual_seed(8)
    continuous = torch.randn(means.shape[0], len(delta.CONTINUOUS), generator=generator, dtype=F64)
    continuous = continuous * torch.tensor([c.knee * 20 for c in delta.COMPONENTS], dtype=F64)
    continuous[:, 7:9] = -continuous[:, 7:9].abs()  # shield and energy start full
    events = torch.zeros(means.shape[0], 2, dtype=torch.bool)
    forecast, _ = code.advance(
        means, spreads, _logits_of(code, continuous, events), ship_type, net.state_code
    )
    base = code.baseline(means, ship_type)
    heading = means[:, MEANS.index("heading")]
    difference = forecast - base
    maximum = dict(
        shield=code.max_shield.double()[ship_type], energy=code.max_energy.double()[ship_type]
    )
    implied = torch.stack(
        [
            *delta.to_nose(difference[:, 0:2], heading).unbind(-1),
            *delta.to_nose(difference[:, 2:4], heading).unbind(-1),
            delta.wrap(difference[:, 4]),
            difference[:, 5],
            difference[:, 6],
            difference[:, 8] / maximum["shield"],
            difference[:, 9] / maximum["energy"],
            difference[:, 11],
        ],
        -1,
    )
    knee = torch.tensor([c.knee for c in delta.COMPONENTS], dtype=F64)
    error = (implied - continuous).abs() / (knee + continuous.abs())
    inside = torch.ones_like(error, dtype=torch.bool)
    for column, name in ((7, "shield"), (8, "energy")):  # not where the bounds clamp them
        value = forecast[:, MEANS.index(name)]
        inside[:, column] = (value > 0) & (value < maximum[name])
    assert inside.any(0).all()
    assert error[inside].max() < 1e-4, error[inside].max()
    # the countdowns with no event
    torch.testing.assert_close(forecast[:, 7], base[:, 7])
    torch.testing.assert_close(forecast[:, 10], base[:, 10])


def test_the_spread_grows_by_the_changes_variance():
    net = cases.double(_model())
    code = net.delta_code
    means, spreads, ship_type = _belief()
    rows = means.shape[0]
    # A two-point change: half at -c, half at +c, per component (variance c^2 less nothing)
    c = torch.tensor([component.knee * 50 for component in delta.COMPONENTS], dtype=F64)
    events = torch.zeros(rows, 2, dtype=torch.bool)
    low, _ = code.encode((-c).expand(rows, -1), events)
    high, _ = code.encode(c.expand(rows, -1), events)
    logits = torch.log((0.5 * (low + high)).clamp_min(1e-30))
    _, variance, _ = code.decode(logits)
    forecast, grown = code.advance(means, spreads, logits, ship_type, net.state_code)
    part = {name: i for i, name in enumerate(delta.CONTINUOUS)}
    sigma = dict(zip(SPREADS, spreads[:, : len(SPREADS)].unbind(-1)))
    after = dict(zip(SPREADS, grown[:, : len(SPREADS)].unbind(-1)))
    for name in ("nose_rate", "angle_of_attack", "log_refractive_index"):
        torch.testing.assert_close(after[name] ** 2, sigma[name] ** 2 + variance[:, part[name]])
    maximum_energy = code.max_energy.double()[ship_type]
    torch.testing.assert_close(
        after["energy"] ** 2, sigma["energy"] ** 2 + variance[:, part["energy"]] * maximum_energy**2
    )
    tick = code.tick_length
    torch.testing.assert_close(
        after["heading"] ** 2,
        sigma["heading"] ** 2 + tick**2 * sigma["nose_rate"] ** 2 + variance[:, part["heading"]],
    )
    # the variance is the two points' (c^2), the code's smoothing removed to within 10%
    torch.testing.assert_close(variance, (c**2).expand(rows, -1), rtol=0.1, atol=0.0)
    # the position: carried, the velocity's dt^2 and the nose rate's, plus the change's
    speed = means[:, 2] ** 2 + means[:, 3] ** 2
    expected = (
        sigma["position"] ** 2
        + tick**2 * 0.5 * (sigma["velocity_xx"] + sigma["velocity_yy"])
        + 0.5 * speed * (0.5 * tick**2) ** 2 * sigma["nose_rate"] ** 2
        + 0.5 * (variance[:, 0] + variance[:, 1])
    )
    torch.testing.assert_close(after["position"] ** 2, expected)


def test_the_events_advance_to_their_expected_values():
    net = cases.double(_model())
    code = net.delta_code
    means, spreads, ship_type = _belief()
    rows = means.shape[0]
    continuous = torch.zeros(rows, len(delta.CONTINUOUS), dtype=F64)
    logits = _logits_of(code, continuous, torch.zeros(rows, 2, dtype=torch.bool))
    rate = 0.3
    offsets = {}
    start = 0
    for name in code.order:
        offsets[name] = start
        start += code.sizes[name]
    for name in delta.EVENTS:
        logits[:, offsets[name]] = math.log(1.0 - rate)
        logits[:, offsets[name] + 1] = math.log(rate)
    forecast, grown = code.advance(means, spreads, logits, ship_type, net.state_code)
    base = code.baseline(means, ship_type)
    wait = MEANS.index("shield_recharge_wait")
    delay = code.recharge_delay.double()[ship_type]
    torch.testing.assert_close(forecast[:, wait], rate * delay + (1 - rate) * base[:, wait])
    cooldown = MEANS.index("gun_cooldown")
    fired = code.fired_cooldown.double()[ship_type]
    torch.testing.assert_close(
        forecast[:, cooldown], rate * fired + (1 - rate) * base[:, cooldown], rtol=0, atol=1e-12
    )


# ------------------------------------------------------------------ the switch


def test_absolute_is_the_default_and_its_build_record_has_no_dynamics():
    assert YemongConfig().next_state == "absolute"
    net = cases.model()
    assert net.delta_code is None and net.build.dynamics is None
    assert net.ship_heads.lasts[1].out_features == net.state_code.width
    with pytest.raises(ValueError, match="next_state"):
        YemongConfig(next_state="relative")
    assert _model().build.dynamics == delta.state_dynamics(
        api.rules_of(cases.SQUAD), api.ship_types()
    )


@pytest.fixture(scope="module")
def segment(module_global_generator):
    net = _model()
    _, trajectory = cases.rollout(net, length=32)
    return net, trajectory


def test_delta_unroll_equals_a_loop_of_steps(segment):
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
    assert (decisions.replay.spreads[..., 0][hidden] > 0).all()  # and its spread grew


def test_the_forecast_of_a_visible_row_is_its_baseline_plus_the_change(segment):
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
    base = net.delta_code.baseline(means, ship_type)
    heading = means[..., MEANS.index("heading")]
    torch.testing.assert_close(
        forecast[..., 0:2], base[..., 0:2] + delta.from_nose(change[..., 0:2], heading)
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
    # the truth's change from t to t + 1, from its baseline at t, on the absolute's mask
    ships = trajectory.obs.ships
    now = torch.cat([ships[..., :OWN, :], trajectory.truth], -2)
    following = torch.cat([now[1:], trajectory.final_truth[None]], 0)
    offset = trajectory.replay.offset
    target, clipped = net.delta_code.targets(
        truth(now, offset), truth(following, offset), now[..., SHIP["ship_type"]].long()
    )
    valid = torch.cat([torch.ones_like(trajectory.replay.valid), trajectory.replay.valid], -1)
    mask = valid & ~trajectory.done[..., None]
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
        total = learner._loss(
            trajectory, label, prepared["derived"], denominator, prepared["scale"], heads=True
        )[-1]
    series = metrics.read(head_statistics.series(total, net))
    for component in delta.CONTINUOUS:
        assert 0.0 <= series[f"next_state/support_clip/{component}"] <= 1.0
        assert 0.0 <= series[f"next_state/support_padding/{component}"] <= 1.0
    for predictor in ("head", "persistence", "baseline"):
        assert math.isfinite(series[f"next_state/error/position_metres/{predictor}"])
    # the baselines beat persistence on the kinematics of visible ships
    assert (
        series["next_state/error/position_metres/baseline"]
        < 0.1 * series["next_state/error/position_metres/persistence"]
    )
    for situation in head_statistics.NEXT_STATE_SITUATIONS:
        assert f"next_state/cross_entropy/{situation}/position" in series
    assert math.isfinite(series["next_state/gain/ally"])


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


def test_a_delta_policy_file_round_trips(tmp_path, segment):
    net, trajectory = segment
    save_policy(tmp_path / "policy", net)
    loaded = load_policy(tmp_path / "policy", lc.CPU).eval()
    assert loaded.build == net.build and loaded.config.next_state == "delta"
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
