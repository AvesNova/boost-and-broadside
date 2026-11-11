"""The extrapolation baseline (arms 3-4 brief, arm 4: `yemong.delta_baseline=extrapolation`):
equal to the baselines bench's E (`bench/baselines_core.py`) on recorded play, in float32;
the switches; its rates in the carry; arm 4c, the baseline without the acceleration inputs.
The arm's unroll, learner terms and policy-file round trip run in `test_yemong_error_head.py`
(its segment tests cover arms 3, 4 and 4c)."""

import dataclasses

import numpy as np
import pytest
import torch
import yemong_cases as cases

from bench import baselines_core as core
from bnb.contracts.obs import SHIP
from bnb.game import api
from bnb.learn import labels
from bnb.models.yemong import delta
from bnb.models.yemong.inputs import truth
from bnb.models.yemong.model import Yemong, YemongConfig, yemong_build
from bnb.models.yemong.state_code import MEANS, state_code_ranges

pytestmark = pytest.mark.usefixtures("global_generator")

SETTINGS = dict(
    order="temporal_first",
    spread_fit="coarsest",
    acceleration_inputs=True,
    next_state="delta",
    delta_baseline="extrapolation",
    spread_inputs="ellipse",
    delta_split=True,
)
ARM_SETS = (
    "yemong.order=temporal_first",
    "yemong.acceleration_inputs=true",
    "yemong.next_state=delta",
    "yemong.delta_baseline=extrapolation",
    "yemong.spread_inputs=ellipse",
    "yemong.delta_split=true",
)
# arm 4c: arm 4 on C-fixed's inputs, the rates computed for the baseline but not read
WITHOUT_INPUTS = dict(SETTINGS, acceleration_inputs=False)


@pytest.fixture(scope="module")
def recorded(module_global_generator):
    """A rollout's privileged rows (L, R, N, columns) and its resets."""
    net = cases.model(**SETTINGS)
    _, trajectory = cases.rollout(net, length=48)
    return labels.privileged_rows(trajectory), trajectory


def _bench_state(rows: np.ndarray, card) -> dict:
    """The bench's state dict of privileged rows (`bench/baselines_measure.load`)."""
    column = lambda name: rows[..., SHIP[name]]  # noqa: E731
    ship_type = column("ship_type").astype(int)
    max_shield = np.asarray(card.max_shield)[ship_type]
    max_energy = np.asarray(card.max_energy)[ship_type]
    return dict(
        x=column("x"),
        y=column("y"),
        vx=column("velocity_x"),
        vy=column("velocity_y"),
        heading=np.arctan2(column("heading_sin"), column("heading_cos")),
        omega=column("nose_rate"),
        attack=column("angle_of_attack"),
        energy=column("energy") / max_energy,
        shield=column("shield") / max_shield,
        wait=column("shield_recharge_wait"),
        cooldown=np.round(column("gun_cooldown") * 30.0),
        log_index=column("log_refractive_index"),
        rate=12.0 / max_shield,
    )


def test_the_baseline_equals_the_benchs_extrapolation_on_recorded_play(recorded):
    """Pairs alive at t - 1 and t with no reset between and a valid acceleration at t: the
    model's baseline (float32, rates as a visible row's) against the bench's E (float64)
    with E-arc's position; the nose rate is persistence (O3); ln n the gradient baseline.
    The bench's damage flag is the shield's fall here (a hit on a ship that lives always
    lowers its shield)."""
    rows, trajectory = recorded
    rules, types = api.rules_of(cases.SHORT_SQUAD), api.ship_types()
    code = delta.DeltaCode(
        state_code_ranges(rules, types), "extrapolation", delta.state_dynamics(rules, types)
    )
    zero = torch.zeros(rows.shape[1], 2)
    means = truth(rows, zero[None])
    now, before = rows[1:], rows[:-1]
    column = lambda name: now[..., SHIP[name]]  # noqa: E731
    reset = cases.resets(trajectory.obs)[1:, :, None]
    keep = (
        (now[..., SHIP["alive"]] > 0.5)
        & (before[..., SHIP["alive"]] > 0.5)
        & ~reset
        & (column("acceleration_valid") > 0.5)
    )
    assert keep.sum() > 500
    rates = delta.Rates(
        valid=keep,
        acceleration=torch.stack([column("acceleration_x"), column("acceleration_y")], -1),
        nose_acceleration=column("nose_acceleration"),
        change=means[1:] - means[:-1],
        rule=torch.zeros_like(keep),
        turn=torch.ones_like(keep, dtype=torch.float32),
        gradient=torch.stack([column("log_index_gradient_x"), column("log_index_gradient_y")], -1),
        gradient_known=torch.ones_like(keep),
        fired=column("fired") > 0.5,
    )
    ship_type = column("ship_type").long()
    baseline = code.baseline(means[1:].float(), rates, ship_type)[keep].double().numpy()
    state = _bench_state(now[keep].double().numpy(), types)
    previous = _bench_state(before[keep].double().numpy(), types)
    shield_fell = state["shield"] < previous["shield"]
    bench, (arc_x, arc_y) = core.extrapolate(
        previous,
        state,
        1.0 / 30.0,
        state_fired := (column("fired")[keep] > 0.5).numpy(),
        shield_fell,
        np.ones_like(state_fired),
    )
    index = {name: i for i, name in enumerate(MEANS)}
    ship = ship_type[keep].numpy()
    max_shield = np.asarray(types.max_shield)[ship]
    max_energy = np.asarray(types.max_energy)[ship]
    pairs = dict(
        x=(baseline[:, index["x"]], arc_x, 1e-2),  # float32 at up to km
        y=(baseline[:, index["y"]], arc_y, 1e-2),
        velocity_x=(baseline[:, index["velocity_x"]], bench["vx"], 1e-3),
        velocity_y=(baseline[:, index["velocity_y"]], bench["vy"], 1e-3),
        angle_of_attack=(baseline[:, index["angle_of_attack"]], bench["attack"], 1e-5),
        energy=(baseline[:, index["energy"]] / max_energy, bench["energy"], 1e-5),
        shield=(baseline[:, index["shield"]] / max_shield, bench["shield"], 1e-5),
        shield_recharge_wait=(baseline[:, index["shield_recharge_wait"]], bench["wait"], 1e-5),
        gun_cooldown=(baseline[:, index["gun_cooldown"]] * 30.0, bench["cooldown"], 1e-3),
    )
    for name, (got, expected, tolerance) in pairs.items():
        assert np.abs(got - expected).max() < tolerance, name
    heading = core.wrap(baseline[:, index["heading"]] - bench["heading"])
    assert np.abs(heading).max() < 1e-5
    np.testing.assert_allclose(
        baseline[:, index["nose_rate"]], state["omega"], rtol=0, atol=1e-6
    )  # persistence
    gradient = rates.gradient[keep].double().numpy()
    drift = (gradient * np.stack([state["vx"], state["vy"]], -1)).sum(-1) / 30.0
    np.testing.assert_allclose(
        baseline[:, index["log_refractive_index"]], state["log_index"] + drift, atol=1e-6
    )


def test_without_rates_the_baseline_is_constant_velocity_and_the_rules():
    rules, types = api.rules_of(cases.SQUAD), api.ship_types()
    code = delta.DeltaCode(
        state_code_ranges(rules, types), "extrapolation", delta.state_dynamics(rules, types)
    ).double()
    means = torch.zeros(1, len(MEANS), dtype=torch.float64)
    means[0, 2:4] = torch.tensor([30.0, 40.0])
    means[0, MEANS.index("nose_rate")] = 0.6
    means[0, MEANS.index("shield")] = 50.0
    means[0, MEANS.index("gun_cooldown")] = 2.0 / 30.0
    rates = delta.Rates(
        valid=torch.zeros(1, dtype=torch.bool),
        acceleration=torch.full((1, 2), 99.0, dtype=torch.float64),
        nose_acceleration=torch.full((1,), 99.0, dtype=torch.float64),
        change=torch.full((1, len(MEANS)), 7.0, dtype=torch.float64),
        rule=torch.zeros(1, dtype=torch.bool),
        turn=torch.ones(1, dtype=torch.float64),
        gradient=torch.zeros(1, 2, dtype=torch.float64),
        gradient_known=torch.zeros(1, dtype=torch.bool),
        fired=torch.zeros(1, dtype=torch.bool),
    )
    baseline = code.baseline(means, rates, torch.zeros(1, dtype=torch.long))
    torch.testing.assert_close(baseline[0, 0:2], means[0, 2:4] / 30.0)
    torch.testing.assert_close(baseline[0, 2:4], means[0, 2:4])
    assert torch.isclose(
        baseline[0, MEANS.index("heading")], torch.tensor(0.02, dtype=torch.float64)
    )
    assert torch.isclose(  # the shield's rule: the wait is 0, so it recharges a tick
        baseline[0, MEANS.index("shield")], torch.tensor(50.0 + 12.0 / 30.0, dtype=torch.float64)
    )
    assert torch.isclose(
        baseline[0, MEANS.index("gun_cooldown")], torch.tensor(1.0 / 30.0, dtype=torch.float64)
    )


def test_the_switches():
    rules, types = api.rules_of(cases.SQUAD), api.ship_types()
    build = yemong_build(YemongConfig(**SETTINGS), rules, types)
    assert build.dynamics == delta.state_dynamics(rules, types)
    net = Yemong(build)
    assert net.delta_code.baseline_name == "extrapolation"
    assert net.initial_carry(2, 5, 10).previous.shape == (2, 10, len(MEANS))
    persistence = yemong_build(
        YemongConfig(**{**SETTINGS, "delta_baseline": "persistence"}), rules, types
    )
    assert persistence.dynamics is None
    assert Yemong(persistence).initial_carry(2, 5, 10).previous is None
    with pytest.raises(ValueError, match="round 1"):
        Yemong(dataclasses.replace(persistence, dynamics=build.dynamics))
    with pytest.raises(ValueError, match="rates"):
        net.forecast(
            torch.zeros(1, net.next_state_width),
            torch.zeros(1, len(MEANS)),
            torch.zeros(1, net.state_code.spread_count),
            torch.zeros(1, dtype=torch.long),
        )


def test_the_arms_set_line_builds_the_arm():
    from bnb.cli import parse_setting, with_settings
    from bnb.contracts import GameConfig
    from bnb.runtime.train import TrainConfig, model_build

    config = with_settings(
        TrainConfig(game=GameConfig(5, 10, 2600.0)), map(parse_setting, ARM_SETS)
    )
    build = model_build(config)
    assert build.config.delta_baseline == "extrapolation" and build.config.delta_split
    assert build.dynamics is not None


def test_the_carry_holds_the_rows_as_read(recorded):
    _, trajectory = recorded
    net = cases.model(**SETTINGS)
    obs = trajectory.obs
    carry = net.initial_carry(obs.ships.shape[1], 5, 10)
    with torch.no_grad():
        decisions, final = cases.step_loop(
            net, net.step, carry, obs, cases.joint_actions(trajectory)
        )
    _, _, means, _ = labels.believed(obs.ships[-1:], _last(decisions.replay))
    torch.testing.assert_close(final.previous, means[0])


def _last(replay):
    from torch.utils import _pytree

    return _pytree.tree_map(lambda tensor: tensor[-1:], replay)


def test_a_hidden_row_holds_its_last_sightings_rates_decaying():
    """Seen with an acceleration of 6 m/s^2 along its track, then hidden two ticks: it reads
    6 e^{-dt/tau} then 6 e^{-2 dt/tau} along its (turned) track, takes the rule for its
    shield and countdowns, and drops its rates when the belief loses it."""
    import math

    from bnb.contracts.obs import SHIP_COLUMNS
    from bnb.models.yemong.inputs import Accelerations

    net = cases.double(cases.model(**SETTINGS))
    tau, dt = net.config.extrapolation_decay, net.tick_length
    ships = torch.zeros(1, len(SHIP_COLUMNS), dtype=torch.float64)
    ships[0, SHIP["visible"]] = 1.0
    means = torch.zeros(1, len(MEANS), dtype=torch.float64)
    means[0, 2:4] = torch.tensor([30.0, 40.0])  # track (0.6, 0.8)
    previous = means.clone()
    previous[0, MEANS.index("angle_of_attack")] = -0.01
    accelerations = lambda valid: Accelerations(  # noqa: E731
        values=torch.zeros(1, 3, dtype=torch.float64),
        valid=torch.tensor([valid]),
        slow=torch.zeros(1, dtype=torch.bool),
        world=torch.tensor([[3.6, 4.8, 2.0]], dtype=torch.float64),
    )
    from bnb.models.yemong.model import HELD

    held = torch.zeros(1, HELD, dtype=torch.float64)
    rates, held, held_valid = net.rates(
        ships, means, previous, accelerations(True), held, torch.zeros(1, dtype=torch.bool)
    )
    assert bool(held_valid[0]) and not bool(rates.rule[0])
    hidden = ships.clone()
    hidden[0, SHIP["visible"]] = 0.0
    turned = means.clone()
    turned[0, 2:4] = torch.tensor([40.0, -30.0])  # track (0.8, -0.6)
    for k in (1, 2):
        rates, held, held_valid = net.rates(
            hidden, turned, means, accelerations(True), held, held_valid
        )
        factor = math.exp(-k * dt / tau)
        torch.testing.assert_close(
            rates.acceleration[0], 6.0 * factor * torch.tensor([0.8, -0.6], dtype=torch.float64)
        )
        assert math.isclose(float(rates.nose_acceleration[0]), 2.0 * factor, rel_tol=1e-9)
        assert math.isclose(
            float(rates.change[0, MEANS.index("angle_of_attack")]), 0.01 * factor, rel_tol=1e-9
        )
        assert bool(rates.rule[0]) and bool(rates.valid[0])
        assert math.isclose(float(rates.turn[0]), factor, rel_tol=1e-9)
    rates, held, held_valid = net.rates(
        hidden, turned, means, accelerations(False), held, held_valid
    )
    assert not bool(rates.valid[0]) and not bool(held_valid[0])


# ------------------------------------------------------------------ arm 4c


def test_without_the_inputs_the_token_has_no_acceleration_columns(recorded):
    """Arm 4c's ship token: C-fixed's, plus the delta arms' own additions (the covariance
    spreads in the code, the ellipse inputs), as arm 3c's (persistence, inputs off); the
    inputs-on arm's less the acceleration inputs. The carry still holds the Motion."""
    from bnb.models.yemong.inputs import ACCELERATION_INPUTS, ship_inputs

    _, trajectory = recorded
    net = cases.model(**WITHOUT_INPUTS)
    width = net.encoder.ship.first.in_features
    assert width == ship_inputs(
        net.state_code.width, net.ship_types, False, net.state_code.ellipse_width
    )
    arm3c = cases.model(**{**WITHOUT_INPUTS, "delta_baseline": "persistence"})
    assert width == arm3c.encoder.ship.first.in_features
    assert width == cases.model(**SETTINGS).encoder.ship.first.in_features - ACCELERATION_INPUTS
    c_fixed = cases.model(order="temporal_first", spread_fit="coarsest")
    assert width == (
        c_fixed.encoder.ship.first.in_features
        + net.state_code.width
        - c_fixed.state_code.width
        + net.state_code.ellipse_width
    )
    assert net.initial_carry(2, 5, 10).motion is not None
    assert arm3c.initial_carry(2, 5, 10).motion is None
    obs = trajectory.obs
    reset = cases.resets(obs)
    previous = net.preceding_motion(trajectory.initial_carry, obs, trajectory.replay, reset)
    tokens = net.tokens(obs, trajectory.replay, previous)
    assert tokens.ship.shape[-1] == width and tokens.accelerations is not None
    with pytest.raises(ValueError, match="previous tick's motion"):
        net.tokens(obs, trajectory.replay)


def test_without_the_inputs_the_baseline_equals_the_inputs_ons(recorded):
    """On the same recorded segment (obs, replay, carry), arm 4c's rates, baseline and
    next-state targets equal the inputs-on arm's exactly."""
    rows, trajectory = recorded
    with_inputs, without = cases.model(**SETTINGS), cases.model(**WITHOUT_INPUTS)
    a, b = labels.rates(with_inputs, trajectory), labels.rates(without, trajectory)
    for field in dataclasses.fields(delta.Rates):
        assert torch.equal(getattr(a, field.name), getattr(b, field.name)), field.name
    assert a.valid.any() and a.rule.any()
    means = truth(rows, trajectory.replay.offset)
    ship_type = rows[..., SHIP["ship_type"]].long()
    assert torch.equal(
        with_inputs.delta_code.baseline(means, a, ship_type),
        without.delta_code.baseline(means, b, ship_type),
    )
    for got, expected in zip(
        labels.next_state_targets(without, trajectory, rows).values(),
        labels.next_state_targets(with_inputs, trajectory, rows).values(),
        strict=True,
    ):
        assert torch.equal(got, expected)


def test_without_the_inputs_the_step_reads_the_same_rates(recorded):
    """A step loop of arm 4c over the recorded play: its carry holds the Motion and the held
    rates as the inputs-on arm's would on the same replay (the segment's, from its start)."""
    _, trajectory = recorded
    net = cases.model(**WITHOUT_INPUTS)
    obs = trajectory.obs
    carry = net.initial_carry(obs.ships.shape[1], 5, 10)
    with torch.no_grad():
        decisions, final = cases.step_loop(
            net, net.step, carry, obs, cases.joint_actions(trajectory)
        )
    assert final.motion.present.any() and final.held_valid.any()
    reset = cases.resets(obs)
    inputs_on = cases.model(**SETTINGS)
    a = net.segment_rates(carry, obs, decisions.replay, reset)
    b = inputs_on.segment_rates(carry, obs, decisions.replay, reset)
    for field in dataclasses.fields(delta.Rates):
        assert torch.equal(getattr(a, field.name), getattr(b, field.name)), field.name
    _, _, means, _ = labels.believed(obs.ships[-1:], _last(decisions.replay))
    torch.testing.assert_close(final.previous, means[0])


def test_the_arm_4c_set_line_builds_the_arm():
    from bnb.cli import parse_setting, with_settings
    from bnb.contracts import GameConfig
    from bnb.runtime.train import TrainConfig, model_build

    sets = tuple(
        "yemong.acceleration_inputs=false" if line.startswith("yemong.acceleration") else line
        for line in ARM_SETS
    )
    config = with_settings(TrainConfig(game=GameConfig(5, 10, 2600.0)), map(parse_setting, sets))
    build = model_build(config)
    assert build.config.delta_baseline == "extrapolation" and build.config.delta_split
    assert not build.config.acceleration_inputs and build.dynamics is not None
