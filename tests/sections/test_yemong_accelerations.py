"""The acceleration inputs (acceleration-inputs brief): the quantities on scripted motion,
the invalid cases, the code, the switch (off unchanged, on: unroll a loop of
steps, the learner's segment start from the stored carry), a policy-file round trip and the
learners' steps with the diagnostics."""

import dataclasses
import json
import math

import arm_goldens
import learner_cases as lc
import pytest
import torch
import yemong_cases as cases
from torch.utils import _pytree

from bnb.contracts import GameConfig, Obs
from bnb.contracts.obs import SHIP, SHIP_COLUMNS
from bnb.controllers.scripted import Scripted
from bnb.game import api
from bnb.learn import bc, head_statistics
from bnb.models.yemong import codes
from bnb.models.yemong.inputs import (
    ACCELERATION_INPUTS,
    ACCELERATIONS,
    SLOW_SPEED,
    Accelerations,
    Motion,
    acceleration_code,
    accelerations,
    believed,
    ship_inputs,
)
from bnb.models.yemong.model import YemongConfig
from bnb.runs import checkpoint
from bnb.runs.checkpoint import FilledDefaults
from bnb.runtime import metrics
from bnb.runtime.rollout import Rollout
from bnb.runtime.train import TrainConfig, model_build

pytestmark = pytest.mark.usefixtures("global_generator")

OWN = 5
F64 = torch.float64
DT = 1.0 / 30.0
C_FIXED = dict(order="temporal_first", spread_fit="coarsest")
SETTINGS = dict(C_FIXED, acceleration_inputs=True)


def _model(**settings):
    return cases.model(**{**SETTINGS, **settings})


def _accelerations(
    now,
    before,
    now_rate=0.0,
    before_rate=0.0,
    heading=0.0,
    visible=True,
    flag=True,
    present=True,
    before_present=True,
    spawned=False,
):
    """The inputs of one enemy row (an own row before it): visible, it reads the
    observation's world-frame change (now - before) / dt with the flag `flag`; hidden, the
    belief's rows now and before (Motion). The row's velocity at t is `now`."""
    ships = torch.zeros((1, 2, len(SHIP_COLUMNS)), dtype=F64)
    ships[..., SHIP["visible"]] = torch.tensor([1.0, float(visible)], dtype=F64)
    ships[..., SHIP["alive"]] = torch.tensor([1.0, float(present)], dtype=F64)
    ships[0, 1, SHIP["spawned"]] = float(spawned)
    now_velocity, before_velocity = torch.tensor(now, dtype=F64), torch.tensor(before, dtype=F64)
    if visible and flag:
        change = (now_velocity - before_velocity) / DT
        ships[0, 1, SHIP["acceleration_x"]], ships[0, 1, SHIP["acceleration_y"]] = change
        ships[0, 1, SHIP["nose_acceleration"]] = (now_rate - before_rate) / DT
        ships[0, 1, SHIP["acceleration_valid"]] = 1.0
    means = torch.zeros((1, 2, 12), dtype=F64)
    means[0, 1, 2:4], means[0, 1, 4], means[0, 1, 5] = now_velocity, heading, now_rate

    def rows(velocity, rate, alive) -> Motion:
        return Motion(
            velocity=velocity.view(1, 1, 2),
            nose_rate=torch.tensor([[rate]], dtype=F64),
            present=torch.tensor([[alive]]),
        )

    out = accelerations(
        ships,
        means,
        rows(now_velocity, now_rate, present),
        rows(before_velocity, before_rate, before_present),
        DT,
    )
    return Accelerations(out.values[0, 1:], out.valid[0, 1:], out.slow[0, 1:])


SIGHT = pytest.mark.parametrize("visible", [True, False], ids=["visible", "hidden"])


@SIGHT
def test_a_straight_acceleration_is_along_the_velocity(visible):
    angle, speed, rate = 0.7, 80.0, 12.0  # rad, m/s, m/s^2
    direction = torch.tensor([math.cos(angle), math.sin(angle)], dtype=F64)
    before = speed * direction
    now = before + rate * DT * direction
    out = _accelerations(now.tolist(), before.tolist(), 1.5, 1.0, heading=3.0, visible=visible)
    along, across, angular = out.values[0].tolist()
    assert along == pytest.approx(rate, rel=1e-12)
    assert abs(across) < 1e-9
    assert angular == pytest.approx(0.5 / DT, rel=1e-12)
    assert out.valid.item() and not out.slow.item()


@SIGHT
@pytest.mark.parametrize("path_rate", [0.8, -0.8])
def test_a_steady_turn_is_across_the_velocity_left_positive(path_rate, visible):
    """Velocity of constant speed turning at path_rate rad/s: across = |v| sin(theta) / dt
    (about |v| x path rate), along = |v| (1 - cos theta) / dt (second order), theta the turn
    in one tick."""
    speed, angle = 90.0, 0.3
    turn = path_rate * DT
    before = [speed * math.cos(angle), speed * math.sin(angle)]
    now = [speed * math.cos(angle + turn), speed * math.sin(angle + turn)]
    out = _accelerations(now, before, visible=visible)
    along, across, _ = out.values[0].tolist()
    assert across == pytest.approx(speed * math.sin(turn) / DT, rel=1e-12)
    assert across == pytest.approx(speed * path_rate, rel=1e-3)
    assert along == pytest.approx(speed * (1.0 - math.cos(turn)) / DT, rel=1e-9)


@SIGHT
def test_below_the_slow_speed_the_frame_is_the_heading(visible):
    heading = math.pi / 2  # nose to +y
    before, now = [0.2, 0.0], [0.2, 0.6]  # |v_t| < SLOW_SPEED; the change is +y
    assert math.hypot(*now) < SLOW_SPEED
    out = _accelerations(now, before, heading=heading, visible=visible)
    along, across, _ = out.values[0].tolist()
    assert out.slow.item()
    assert along == pytest.approx(0.6 / DT, rel=1e-12)
    assert abs(across) < 1e-9
    fast = _accelerations([2.0, 0.6], [2.0, 0.0], heading=heading, visible=visible)
    assert not fast.slow.item()  # at 2 m/s the frame is the velocity's (+x): the change is left
    assert fast.values[0, 1].item() == pytest.approx(0.6 / DT * 2.0 / math.hypot(2.0, 0.6))


@pytest.mark.parametrize(
    "case",
    ["observed_invalid", "dead_now", "dead_before", "spawned", "first_tick"],
)
def test_an_invalid_difference_is_zero_with_flag_zero(case):
    """A visible row is invalid by the observation's flag (dead at either tick, spawned, a
    game's first tick, just spotted: tests/sections/test_observed_accelerations.py); a
    hidden row by its belief's rows: dead now, dead before or nothing before (the carry's
    first tick or a reset), or spawned now."""
    settings = dict(now=[50.0, 3.0], before=[40.0, 0.0], now_rate=2.0, visible=False)
    if case == "observed_invalid":
        settings.update(visible=True, flag=False)
    elif case == "dead_now":
        settings.update(present=False)
    elif case in ("dead_before", "first_tick"):
        settings.update(before_present=False)
    else:
        settings.update(spawned=True)
    out = _accelerations(**settings)
    assert not out.valid.item()
    assert (out.values == 0).all() and not out.slow.item()
    code = acceleration_code(out)
    assert code[0, -1].item() == 0.0
    zero = acceleration_code(dataclasses.replace(out, valid=torch.tensor([True])))
    assert torch.equal(code[..., :-1], zero[..., :-1])  # the zero's code


def test_hidden_at_t_is_valid_the_forecast_minus_the_row_before():
    """Hidden at t (just hidden or hidden at both ticks: the carry holds the row the model
    read, truth or forecast): the belief's difference, valid."""
    out = _accelerations([50.0, 3.0], [40.0, 0.0], 2.0, visible=False)
    assert out.valid.item() and out.values.abs().sum() > 0


def test_a_visible_row_reads_the_observation_not_the_belief():
    """A visible row's input is the observation's change even when the carried rows say
    otherwise (they are the belief's, and a just-spotted row's would be truth minus belief)."""
    out = _accelerations([50.0, 3.0], [40.0, 0.0], visible=True, before_present=False)
    assert out.valid.item()
    along = (10.0 * 50.0 + 3.0 * 3.0) / DT / math.hypot(50.0, 3.0)
    assert out.values[0, 0].item() == pytest.approx(along, rel=1e-12)


# ------------------------------------------------------------------ the code


@pytest.mark.parametrize("component", ACCELERATIONS, ids=lambda c: c.name)
def test_the_code_round_trips_with_an_exact_zero_up_to_the_bound(component):
    assert component.bins == 31 and component.width == 35
    half = (component.bins - 1) // 2
    spacing = codes.symlog_spacing(component.knee, component.bound, component.bins)
    coordinate = torch.linspace(-half, half, 2001, dtype=F64)
    values = codes.symlog_value(coordinate, component.knee, spacing)
    code, clipped = codes.symlog_encode(values, component.knee, component.bound, component.bins)
    assert not clipped.any()
    mean, _ = codes.symlog_decode(code, component.knee, component.bound, component.bins)
    inside = coordinate.abs() <= half - 0.06
    relative = (mean - values).abs() / (values.abs() + component.knee)
    assert relative[inside].max() < 1.3e-3 * spacing + 1e-9
    zero, _ = codes.symlog_encode(torch.zeros(1, dtype=F64), component.knee, component.bound, 31)
    torch.testing.assert_close(zero, zero.flip(-1))
    assert codes.symlog_value(
        torch.tensor(float(half), dtype=F64), component.knee, spacing
    ).item() == (pytest.approx(component.bound, rel=1e-12))


def test_the_inputs_widths():
    assert ACCELERATION_INPUTS == 3 * 35 + 1
    assert ship_inputs(100, 2, True) == ship_inputs(100, 2) + ACCELERATION_INPUTS
    net, plain = _model(), cases.model(**C_FIXED)
    assert (
        net.encoder.ship.first.in_features
        == plain.encoder.ship.first.in_features + ACCELERATION_INPUTS
    )


# ------------------------------------------------------------------ the switch, off


def test_off_is_the_default_and_carries_no_motion():
    assert not YemongConfig().acceleration_inputs
    net = cases.model(**C_FIXED)
    assert net.initial_carry(2, OWN, 10).motion is None
    assert net.encoder.ship.first.in_features == ship_inputs(net.state_code.width, net.ship_types)


def test_off_steps_and_unrolls_c_fixed_as_the_tree_before_the_switch():
    """C-fixed's step loop and unroll, recorded before the switch existed (`arm_goldens`),
    to float32 rounding."""
    arm_goldens.reproduces("c_fixed")


def test_a_build_record_from_before_the_option_loads_off(tmp_path):
    checkpoint.save_policy(tmp_path / "policy", cases.model(**C_FIXED))
    path = tmp_path / "policy" / checkpoint.RECORD
    record = json.loads(path.read_text())
    assert record["model"]["build"]["config"]["acceleration_inputs"] is False
    record["model"]["build"]["config"].pop("acceleration_inputs")
    path.write_text(json.dumps(record))
    with pytest.warns(FilledDefaults, match="config.acceleration_inputs"):
        loaded = checkpoint.load_policy(tmp_path / "policy", cases.CPU)
    assert not loaded.config.acceleration_inputs


def test_the_arms_set_line_builds_the_inputs():
    from bnb.cli import parse_setting, with_settings

    sets = ("yemong.order=temporal_first", "yemong.acceleration_inputs=true")
    config = with_settings(TrainConfig(game=GameConfig(5, 10, 2600.0)), map(parse_setting, sets))
    build = model_build(config).config
    assert build.acceleration_inputs and build.order == "temporal_first"
    assert build.spread_fit == "coarsest"


# ------------------------------------------------------------------ the switch, on


@pytest.fixture(scope="module")
def segment(module_global_generator):
    """A real fogged rollout played with the inputs on, run twice, so its stored initial
    carry has history."""
    net = _model()
    _, trajectory = cases.rollout(net, length=48, runs=2)
    return net, trajectory


def _views(obs: Obs, t: int) -> Obs:
    return Obs(ships=obs.ships[t], fields=obs.fields[t], globals=obs.globals[t], zones=obs.zones[t])


def test_the_inputs_on_a_real_segment_are_the_observed_differences(segment):
    """Visible at both ticks: the observation's own finite difference in the velocity frame.
    Invalid (zero, flag 0): dead at either tick, spawned, a reset (a game's first tick), just
    spotted. The flag ends the ship token."""
    net, trajectory = segment
    net = cases.double(net)
    obs = cases.cast(trajectory.obs, F64)
    reset = cases.resets(obs)
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        decisions, _ = cases.step_loop(net, net.step, carry, obs, cases.joint_actions(trajectory))
        out = net.acceleration_inputs(carry, obs, decisions.replay, reset)
        previous = net.preceding_motion(carry, obs, decisions.replay, reset)
        tokens = net.tokens(obs, decisions.replay, previous)
    ships = obs.ships
    column = {name: ships[..., SHIP[name]] for name in SHIP}
    alive, visible = column["alive"] > 0.5, column["visible"] > 0.5
    spawned = column["spawned"] > 0.5
    velocity = torch.stack([column["velocity_x"], column["velocity_y"]], -1)
    both = (alive & visible)[1:] & (alive & visible)[:-1] & ~spawned[1:] & ~reset[1:, :, None]
    change = (velocity[1:] - velocity[:-1]) / net.tick_length
    speed = velocity[1:].norm(dim=-1, keepdim=True)
    unit = velocity[1:] / speed.clamp_min(1e-12)
    along = (change * unit).sum(-1)
    across = change[..., 1] * unit[..., 0] - change[..., 0] * unit[..., 1]
    angular = (column["nose_rate"][1:] - column["nose_rate"][:-1]) / net.tick_length
    expected = torch.stack([along, across, angular], -1)
    assert both.sum() > 1000 and (speed[..., 0][both] >= SLOW_SPEED).all()
    assert out.valid[1:][both].all()
    # the observation divides in float32 (the sim's), the expectation in float64
    torch.testing.assert_close(out.values[1:][both], expected[both], rtol=1e-6, atol=1e-5)
    # the invalid cases occur in the segment (past its first tick, whose previous row is
    # the game's, not the segment's) and are zero with flag 0
    previous_alive, previous_visible = alive[:-1], visible[:-1]
    alive, visible, spawned = alive[1:], visible[1:], spawned[1:]
    cases_ = dict(
        dead=~alive | ~previous_alive,
        spawned=spawned,
        reset=reset[1:, :, None].expand_as(alive),
        just_spotted=visible & ~previous_visible & previous_alive,
    )
    for name, members in cases_.items():  # each case, staged: test_observed_accelerations
        if name in ("spawned", "reset"):
            assert members.any(), name
        assert not out.valid[1:][members].any(), name
        assert (out.values[1:][members] == 0).all(), name
    visible = column["visible"] > 0.5
    assert torch.equal(tokens.ship[..., -1], out.valid.to(F64))
    assert torch.equal(tokens.ship[..., -ACCELERATION_INPUTS:], acceleration_code(out))
    hidden = out.valid & ~visible
    assert hidden.any() and out.values[hidden].abs().sum() > 0  # the belief's forecast moves


def _arm_1_inputs(obs: Obs, replay, reset: torch.Tensor, tick_length: float) -> Accelerations:
    """Arm 1's inputs (the acceleration-inputs brief, commit fafab800), computed in the
    model from every row it read, kept here as the reference: the rows (truth where visible,
    the belief's forecast where hidden) differenced against the tick before's, which an
    initial carry holds absent; invalid when not present (alive with a belief) at both
    ticks, spawned at t, after a reset, or just spotted (hidden before, visible now)."""
    ships = obs.ships
    valid, _, means, _ = believed(ships, replay)
    present = (ships[..., SHIP["alive"]] > 0.5) & valid
    visible = ships[..., SHIP["visible"]] > 0.5

    def before(now: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.zeros_like(now[:1]), now[:-1]])

    present_before = before(present) & ~reset[..., None]
    rule = present & present_before & ~(ships[..., SHIP["spawned"]] > 0.5)
    rule = rule & (before(visible) | ~visible)
    velocity, rate = means[..., 2:4], means[..., 5]
    change = (velocity - before(velocity)) / tick_length
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    slow = speed < SLOW_SPEED
    safe = torch.where(slow, 1.0, speed)
    unit_x = torch.where(slow, torch.cos(means[..., 4]), velocity[..., 0] / safe)
    unit_y = torch.where(slow, torch.sin(means[..., 4]), velocity[..., 1] / safe)
    along = change[..., 0] * unit_x + change[..., 1] * unit_y
    across = change[..., 1] * unit_x - change[..., 0] * unit_y
    angular = (rate - before(rate)) / tick_length
    values = torch.where(rule[..., None], torch.stack([along, across, angular], -1), 0.0)
    return Accelerations(values=values, valid=rule, slow=slow & rule)


def test_the_inputs_equal_arm_1s_on_every_row(segment):
    """Equivalence: on a real fogged rollout, the observation's columns (visible rows) and
    the carried enemy rows (hidden rows) give arm 1's model-computed inputs on every row,
    in float32, to rounding (both divide the same float32 velocities by the same tick). The
    segment's first tick is left out: the step loop starts from an initial carry, which
    holds no previous row, while the observation knows the game's (a rollout's carry,
    stored with the segment, holds it: test_unroll_reproduces_the_stored_rollout)."""
    net, trajectory = segment
    obs = trajectory.obs
    reset = cases.resets(obs)
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        decisions, _ = cases.step_loop(net, net.step, carry, obs, cases.joint_actions(trajectory))
        out = net.acceleration_inputs(carry, obs, decisions.replay, reset)
    reference = _arm_1_inputs(obs, decisions.replay, reset, net.tick_length)
    visible = obs.ships[..., SHIP["visible"]] > 0.5
    assert (reference.valid & visible).sum() > 1000 and (reference.valid & ~visible).sum() > 100
    just_spotted = visible[1:] & ~visible[:-1] & (obs.ships[1:, ..., SHIP["alive"]] > 0.5)
    assert just_spotted.any()  # the case the observation's flag now carries
    assert torch.equal(out.valid[1:], reference.valid[1:])
    assert torch.equal(out.slow[1:], reference.slow[1:])
    # a difference of two nearby velocities over a tick: float32 rounds it at about 1e-6 of
    # the largest value, a wrong tick or frame by a good fraction of it
    assert cases.scale_error(out.values[1:], reference.values[1:]) <= 1e-5
    # at the first tick the observation's rows are valid where arm 1's were not, never less
    assert (out.valid[0] >= reference.valid[0]).all() and (out.valid[0] & visible[0]).any()


def test_unroll_equals_a_loop_of_steps(segment):
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
        ("enemy_action_logits", unrolled.enemy_action_logits[..., OWN:, :]),
        ("next_state_logits", unrolled.next_state_logits[..., OWN:, :]),
        ("values", unrolled.values),
    )
    for name, actual in pairs:
        assert cases.scale_error(actual, getattr(decisions, name)) <= 1e-9, name
    # the step's carry holds the rows it read: the last tick's motion
    assert final.motion.present.any()
    last = decisions.carry.motion
    assert torch.equal(final.motion.velocity, _pytree.tree_map(lambda x: x[-1], last).velocity)


def test_a_segment_start_reads_the_true_previous_row_from_the_carry(segment):
    """Split in two: the second half's unroll from the step loop's carry at the split equals
    the steps; with the carry's motion dropped (nothing present) the first tick's hidden
    rows lose their inputs (the visible rows read the observation) and the outputs
    differ."""
    net, trajectory = segment
    net = cases.double(net)
    obs = cases.cast(trajectory.obs, F64)
    joint = cases.joint_actions(trajectory)
    split = 24
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        head = _pytree.tree_map_only(torch.Tensor, lambda x: x[:split], obs)
        _, middle = cases.step_loop(net, net.step, carry, head, joint)
        decisions, _ = cases.step_loop(net, net.step, middle, obs, joint, start=split)
        tail = _pytree.tree_map_only(torch.Tensor, lambda x: x[split:], obs)
        reset = cases.resets(tail)
        unrolled = net.unroll(middle, tail, decisions.replay, reset)
        dropped = dataclasses.replace(
            middle,
            motion=dataclasses.replace(
                middle.motion, present=torch.zeros_like(middle.motion.present)
            ),
        )
        cold = net.unroll(dropped, tail, decisions.replay, reset)
        true = net.acceleration_inputs(middle, tail, decisions.replay, reset)
        blind = net.acceleration_inputs(dropped, tail, decisions.replay, reset)
    for name in ("policy_logits", "values"):
        assert cases.scale_error(getattr(unrolled, name), getattr(decisions, name)) <= 1e-9
    hidden = tail.ships[0, ..., SHIP["visible"]] <= 0.5
    assert (true.valid[0] & hidden).sum() > 0 and not (blind.valid[0] & hidden).any()
    assert torch.equal(true.valid[0] & ~hidden, blind.valid[0] & ~hidden)
    assert torch.equal(true.values[1:], blind.values[1:])
    assert cases.scale_error(cold.policy_logits[0], decisions.policy_logits[0]) > 1e-6


def test_unroll_reproduces_the_stored_rollout(segment):
    """The runtime stores the carry's motion with the initial carry: unroll from it gives
    what the rollout's steps gave."""
    net, trajectory = segment
    assert trajectory.initial_carry.motion.present.any()
    with torch.no_grad():
        obs = trajectory.obs
        unrolled = net.unroll(trajectory.initial_carry, obs, trajectory.replay, cases.resets(obs))
    assert cases.scale_error(unrolled.values, trajectory.values) <= 1e-4
    assert cases.scale_error(unrolled.values.sum(-1), trajectory.value) <= 1e-4


def test_a_policy_file_round_trips(tmp_path, segment):
    net, trajectory = segment
    checkpoint.save_policy(tmp_path / "policy", net)
    loaded = checkpoint.load_policy(tmp_path / "policy", cases.CPU).eval()
    assert loaded.build == net.build and loaded.config.acceleration_inputs
    obs = trajectory.obs
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    joint = cases.joint_actions(trajectory)
    with torch.no_grad():  # each loop draws the same offsets at its resets
        torch.manual_seed(9)
        a, _ = cases.step_loop(net, net.step, carry, obs, joint)
        torch.manual_seed(9)
        b, _ = cases.step_loop(loaded, loaded.step, carry, obs, joint)
    assert torch.equal(a.policy_logits, b.policy_logits)
    assert torch.equal(a.carry.motion.velocity, b.carry.motion.velocity)


# ------------------------------------------------------------------ the learners


def _check_series(series: dict) -> None:
    for component in ACCELERATIONS:
        for sight in head_statistics.SIGHT_AT_T:
            value = series[f"input/acceleration/mean_abs/{component.name}/{sight}"]
            assert math.isfinite(value) and value > 0, (component.name, sight)
        assert series[f"input/acceleration/padding_share/{component.name}"] == 0.0
    assert 0.0 < series["input/acceleration/invalid_share"] < 0.5
    assert series["input/acceleration/slow_frame"] < 0.1


def test_a_ppo_update_runs_and_logs_the_inputs():
    net = _model()
    runner = lc.rollout(net)
    learner = lc.learner(net, runner)
    out = metrics.read(learner.update(runner.store, runner.bootstrap()))
    assert all(math.isfinite(out[name]) for name in ("next_state/loss", "policy/kl"))
    _check_series(out)
    with_inputs = net.encoder.ship.first.weight
    assert with_inputs.grad is None or torch.isfinite(with_inputs.grad).all()


def test_the_acceleration_series_are_absent_when_off():
    net = cases.model(**C_FIXED)
    trajectory = lc.whole(lc.rollout(net, runs=1))
    assert head_statistics.acceleration_sums(net, trajectory) == {}
    assert head_statistics.acceleration_series({}) == {}


def test_a_bc_update_runs_and_logs_the_inputs():
    from test_bc import CHARACTERISTICS, PARAMETERS, POOLS, TEACHER

    net = _model()
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
    before = net.encoder.ship.first.weight.detach().clone()
    out = metrics.read(learner.update(runner.store, runner.bootstrap()))
    assert math.isfinite(out["next_state/loss"])
    _check_series(out)
    # the acceleration columns trained
    moved = net.encoder.ship.first.weight.detach() - before
    assert moved[:, -ACCELERATION_INPUTS:].abs().sum() > 0
