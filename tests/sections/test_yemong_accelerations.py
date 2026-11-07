"""The acceleration inputs (acceleration-inputs brief): the quantities on scripted motion,
the invalid cases, the code, the switch (off unchanged bit for bit, on: unroll a loop of
steps, the learner's segment start from the stored carry), a policy-file round trip and the
learners' steps with the diagnostics."""

import dataclasses
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys

import learner_cases as lc
import pytest
import torch
import yemong_cases as cases
from torch.utils import _pytree

from bnb.contracts import GameConfig, Obs
from bnb.contracts.obs import SHIP
from bnb.controllers.scripted import Scripted
from bnb.game import api
from bnb.learn import bc, head_statistics
from bnb.models.yemong import codes
from bnb.models.yemong.inputs import (
    ACCELERATION_INPUTS,
    ACCELERATIONS,
    SLOW_SPEED,
    Motion,
    acceleration_code,
    accelerations,
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
HERE = pathlib.Path(__file__).parent
SPREAD_GOLDEN = HERE / "golden" / "yemong_v1_spread_steps.pt"
# C-fixed's step loop and unroll before the switch existed (commit 334cad95), as SHA-256
# digests of each output's bytes: the spread-fit golden's inputs, offset seed 11, one thread.
C_FIXED_GOLDEN = HERE / "golden" / "yemong_c_fixed_steps.json"


def _model(**settings):
    return cases.model(**{**SETTINGS, **settings})


def _motion(velocity, nose_rate=0.0, present=True, visible=True) -> Motion:
    velocity = torch.tensor([velocity], dtype=F64)
    return Motion(
        velocity=velocity,
        nose_rate=torch.tensor([nose_rate], dtype=F64),
        present=torch.tensor([present]),
        visible=torch.tensor([visible]),
    )


def _accelerations(now: Motion, before: Motion, heading=0.0, spawned=False):
    return accelerations(
        now, before, torch.tensor([heading], dtype=F64), torch.tensor([spawned]), DT
    )


# ------------------------------------------------------------------ the quantities


def test_a_straight_acceleration_is_along_the_velocity():
    angle, speed, rate = 0.7, 80.0, 12.0  # rad, m/s, m/s^2
    direction = torch.tensor([math.cos(angle), math.sin(angle)], dtype=F64)
    before = speed * direction
    now = before + rate * DT * direction
    out = _accelerations(_motion(now.tolist(), 1.5), _motion(before.tolist(), 1.0), heading=3.0)
    along, across, angular = out.values[0].tolist()
    assert along == pytest.approx(rate, rel=1e-12)
    assert abs(across) < 1e-9
    assert angular == pytest.approx(0.5 / DT, rel=1e-12)
    assert out.valid.item() and not out.slow.item()


@pytest.mark.parametrize("path_rate", [0.8, -0.8])
def test_a_steady_turn_is_across_the_velocity_left_positive(path_rate):
    """Velocity of constant speed turning at path_rate rad/s: across = |v| sin(theta) / dt
    (about |v| x path rate), along = |v| (1 - cos theta) / dt (second order), theta the turn
    in one tick."""
    speed, angle = 90.0, 0.3
    turn = path_rate * DT
    before = [speed * math.cos(angle), speed * math.sin(angle)]
    now = [speed * math.cos(angle + turn), speed * math.sin(angle + turn)]
    out = _accelerations(_motion(now), _motion(before))
    along, across, _ = out.values[0].tolist()
    assert across == pytest.approx(speed * math.sin(turn) / DT, rel=1e-12)
    assert across == pytest.approx(speed * path_rate, rel=1e-3)
    assert along == pytest.approx(speed * (1.0 - math.cos(turn)) / DT, rel=1e-9)


def test_below_the_slow_speed_the_frame_is_the_heading():
    heading = math.pi / 2  # nose to +y
    before, now = [0.2, 0.0], [0.2, 0.6]  # |v_t| < SLOW_SPEED; the change is +y
    assert math.hypot(*now) < SLOW_SPEED
    out = _accelerations(_motion(now), _motion(before), heading=heading)
    along, across, _ = out.values[0].tolist()
    assert out.slow.item()
    assert along == pytest.approx(0.6 / DT, rel=1e-12)
    assert abs(across) < 1e-9
    fast = _accelerations(_motion([2.0, 0.6]), _motion([2.0, 0.0]), heading=heading)
    assert not fast.slow.item()  # at 2 m/s the frame is the velocity's (+x): the change is left
    assert fast.values[0, 1].item() == pytest.approx(0.6 / DT * 2.0 / math.hypot(2.0, 0.6))


@pytest.mark.parametrize(
    "case",
    ["dead_now", "dead_before", "spawned", "first_tick", "just_spotted"],
)
def test_an_invalid_difference_is_zero_with_flag_zero(case):
    now = dict(velocity=[50.0, 3.0], nose_rate=2.0)
    before = dict(velocity=[40.0, 0.0], nose_rate=0.0)
    spawned = False
    if case == "dead_now":
        now["present"] = False
    elif case in ("dead_before", "first_tick"):  # the initial carry holds nothing present
        before["present"] = False
    elif case == "spawned":
        spawned = True
    else:  # hidden at t - 1, visible at t
        before["visible"] = False
    out = _accelerations(_motion(**now), _motion(**before), spawned=spawned)
    assert not out.valid.item()
    assert (out.values == 0).all() and not out.slow.item()
    code = acceleration_code(out)
    assert code[0, -1].item() == 0.0
    zero = acceleration_code(dataclasses.replace(out, valid=torch.tensor([True])))
    assert torch.equal(code[..., :-1], zero[..., :-1])  # the zero's code


@pytest.mark.parametrize("sight", [(True, False), (False, False)], ids=["just_hidden", "hidden"])
def test_hidden_at_t_is_valid_the_forecast_minus_the_row_before(sight):
    before_visible, now_visible = sight
    out = _accelerations(
        _motion([50.0, 3.0], 2.0, visible=now_visible),
        _motion([40.0, 0.0], 0.0, visible=before_visible),
    )
    assert out.valid.item() and out.values.abs().sum() > 0


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


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def _c_fixed_reproduces_the_golden() -> None:
    golden = torch.load(SPREAD_GOLDEN, weights_only=True)
    obs = Obs(
        ships=golden["ships"],
        fields=golden["fields"],
        globals=golden["globals"],
        zones=golden["zones"],
    )
    net = cases.model(**C_FIXED, acceleration_inputs=False)
    torch.manual_seed(11)
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        decisions, final = cases.step_loop(net, net.step, carry, obs, golden["joint"])
        unrolled = net.unroll(carry, obs, decisions.replay, cases.resets(obs))
    outputs = dict(
        step_policy_logits=decisions.policy_logits,
        step_next_state_logits=decisions.next_state_logits,
        step_values=decisions.values,
        step_carry_state=final.state,
        step_belief_means=final.belief.means,
        unroll_policy_logits=unrolled.policy_logits,
        unroll_next_state_logits=unrolled.next_state_logits,
        unroll_values=unrolled.values,
    )
    expected = json.loads(C_FIXED_GOLDEN.read_text())
    assert {name: _digest(value) for name, value in outputs.items()} == expected


def test_off_steps_and_unrolls_c_fixed_bit_for_bit_like_the_tree_before_the_switch():
    """In a fresh process on one thread, as recorded (a CPU matmul's summation order follows
    the process's thread pool)."""
    environment = dict(
        os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=""
    )
    code = "import test_yemong_accelerations as t; t._c_fixed_reproduces_the_golden()"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=HERE, env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-2000:]


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
    torch.testing.assert_close(out.values[1:][both], expected[both], rtol=1e-9, atol=1e-6)
    # the invalid cases occur in the segment and are zero with flag 0
    previous_alive = torch.cat([torch.zeros_like(alive[:1]), alive[:-1]])
    previous_visible = torch.cat([torch.zeros_like(visible[:1]), visible[:-1]])
    cases_ = dict(
        dead=~alive | ~previous_alive,
        spawned=spawned,
        reset=reset[..., None].expand_as(alive),
        just_spotted=visible & ~previous_visible & previous_alive,
    )
    for name, members in cases_.items():
        if name != "just_spotted":
            assert members.any(), name
        assert not out.valid[members].any(), name
        assert (out.values[members] == 0).all(), name
    assert torch.equal(tokens.ship[..., -1], out.valid.to(F64))
    assert torch.equal(tokens.ship[..., -ACCELERATION_INPUTS:], acceleration_code(out))
    hidden = out.valid & ~visible
    assert hidden.any() and out.values[hidden].abs().sum() > 0  # the belief's forecast moves


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
    the steps; with the carry's motion dropped (nothing present) the first tick's inputs and
    the outputs differ."""
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
    assert true.valid[0].sum() > 0 and not blind.valid[0].any()
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
