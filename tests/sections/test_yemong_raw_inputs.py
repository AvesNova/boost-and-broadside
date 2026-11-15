"""Arm A17's raw inputs (`YemongConfig.input_encoding = "raw"`): every token reads plain
scalars where the default reads the colour and HL-Gauss codes. Off by default; the raw
columns are what `ShipStateCode.raw_inputs` documents; the tokens' code (the next-state
baseline) is still the state code; the view's position offset reaches no raw token; a raw
build steps as it unrolls, trains, saves and loads, and comes in through `--set`."""

import dataclasses
import math

import learner_cases as lc
import pytest
import torch
import yemong_cases as cases

from bnb.contracts import GameConfig
from bnb.game import api
from bnb.learn.ppo import COUNTS
from bnb.models.yemong.inputs import ACCELERATION_INPUTS, ARENA_CODE, POSITION_CODE, believed
from bnb.models.yemong.model import YemongConfig
from bnb.models.yemong.state_code import (
    MEANS,
    NOSE_RATE_LIMIT,
    RAW_POSITION_SCALE,
    RAW_SPEED_SCALE,
    RAW_STATE,
    SPREADS,
    ShipStateCode,
    state_code_ranges,
)
from bnb.runs.checkpoint import load_policy, save_policy
from bnb.runtime import metrics

pytestmark = pytest.mark.usefixtures("global_generator")

OWN = cases.SQUAD.ships_per_team
F64 = torch.float64
# a new run's next-state head (raw inputs need the delta head, as the Fourier inputs do)
RAW = dict(next_state="delta", input_encoding="raw")
# every optional input the raw encoding replaces
EVERY_INPUT = dict(
    acceleration_inputs=True, spread_inputs="ellipse", arena_code=True, frontline_timer_inputs=True
)


def _code(**settings) -> ShipStateCode:
    config = cases.SQUAD
    ranges = state_code_ranges(api.rules_of(config), api.ship_types())
    return ShipStateCode(ranges, **settings).double()


def test_the_codes_are_the_default_and_raw_needs_the_delta_head():
    assert YemongConfig().input_encoding == "codes"
    with pytest.raises(ValueError, match="input_encoding 'raw' needs next_state 'delta'"):
        YemongConfig(input_encoding="raw")
    with pytest.raises(ValueError, match="input_encoding 'raw' needs"):
        YemongConfig(next_state="delta", input_encoding="raw", position_inputs="fourier")
    with pytest.raises(ValueError, match="input_encoding"):
        YemongConfig(next_state="delta", input_encoding="scalars")


def test_the_raw_tokens_replace_each_code_by_its_scalars():
    """Widths against the codes' build: the state code's 580 columns become RAW_STATE's 25,
    the acceleration codes 3 symlog columns (the flag kept), each ellipse feature's code one
    column, a memory token's position code 2 columns, the arena code 1; the global token
    reads no centre code."""
    codes = cases.model(next_state="delta", **EVERY_INPUT)
    raw = cases.model(**RAW, **EVERY_INPUT)
    assert codes.encoder.ship.first.in_features == 644 + ACCELERATION_INPUTS + 8 * 25
    assert raw.state_code.input_width == len(RAW_STATE) == 25
    assert raw.state_code.ellipse_width == 8
    assert raw.encoder.ship.first.in_features == 644 - 580 + 25 + 4 + 8
    assert (
        raw.encoder.global_token.first.in_features
        == codes.encoder.global_token.first.in_features - POSITION_CODE - ARENA_CODE + 1
    )
    for kind in ("field", "zone"):
        width = getattr(codes.encoder, kind).first.in_features - POSITION_CODE + 2
        assert getattr(raw.encoder, kind).first.in_features == width, kind
    # the code itself, what the next-state head's targets and baseline read, is unchanged
    assert raw.state_code.width == codes.state_code.width == 580
    assert raw.next_state_width == codes.next_state_width


def test_a_sharp_row_reads_its_state_over_the_scales():
    code = _code(input_encoding="raw")
    ship_type = torch.tensor([0, 1])
    means = torch.zeros(2, len(MEANS), dtype=F64)
    column = dict(zip(MEANS, range(len(MEANS)), strict=True))
    centre = torch.tensor([1000.0, -2000.0], dtype=F64)
    means[:, column["x"]] = centre[0] + torch.tensor([780.0, -3900.0], dtype=F64)
    means[:, column["y"]] = centre[1] + 1560.0
    means[:, column["velocity_x"]] = torch.tensor([225.0, -90.0], dtype=F64)
    means[:, column["heading"]] = torch.tensor([0.3, -2.0], dtype=F64)
    means[:, column["nose_rate"]] = torch.tensor([2.0, -8.0], dtype=F64)
    means[:, column["shield"]] = torch.tensor([50.0, 160.0], dtype=F64)  # maxima 100 and 160
    means[:, column["energy"]] = torch.tensor([10000.0, 7000.0], dtype=F64)
    means[:, column["angle_of_attack"]] = code.angle_of_attack_limit
    means[:, column["gun_cooldown"]] = 2 * code.tick_length
    spreads = torch.zeros(2, code.spread_count, dtype=F64)
    raw = code.inputs(code.encode(means, spreads, ship_type), means, spreads, ship_type, centre)
    value = dict(zip(RAW_STATE, raw.unbind(-1), strict=True))
    expected = dict(
        x=torch.tensor([780.0, -3900.0]) / RAW_POSITION_SCALE,  # 0.1, -0.5
        y=torch.tensor([1560.0, 1560.0]) / RAW_POSITION_SCALE,
        velocity_x=torch.tensor([225.0, -90.0]) / RAW_SPEED_SCALE,
        heading_cos=torch.cos(torch.tensor([0.3, -2.0])),
        heading_sin=torch.sin(torch.tensor([0.3, -2.0])),
        nose_rate=torch.tensor([2.0 / NOSE_RATE_LIMIT, -1.0]),
        shield=torch.tensor([0.5, 1.0]),
        energy=torch.tensor([1.0, 0.5]),
        angle_of_attack=torch.tensor([1.0, 1.0]),
        gun_cooldown=torch.full((2,), 2.0 / 3.0),  # its longest: three ticks
    )
    for name, want in expected.items():
        torch.testing.assert_close(value[name], want.to(F64), msg=name)
    spread_columns = [name for name in RAW_STATE if "spread" in name or "correlation" in name]
    assert len(spread_columns) == 12
    for name in spread_columns:  # a sharp row: no spread, and its cooldown code one-hot
        torch.testing.assert_close(value[name], torch.zeros(2, dtype=F64), msg=name)


def test_a_spread_row_reads_its_spreads_on_a_log_scale_and_its_heading_fades():
    code = _code(input_encoding="raw")
    ship_type = torch.tensor([0])
    means = torch.zeros(1, len(MEANS), dtype=F64)
    means[0, MEANS.index("heading")] = 1.0
    spreads = torch.zeros(1, code.spread_count, dtype=F64)
    sigma = dict(position=120.0, heading=0.5, nose_rate=0.4, shield=10.0)
    for name, value in sigma.items():
        spreads[0, SPREADS.index(name)] = value
    spreads[0, SPREADS.index("velocity_xx")] = 25.0
    spreads[0, SPREADS.index("velocity_yy")] = 100.0
    spreads[0, SPREADS.index("velocity_xy")] = 25.0
    centre = torch.zeros(2, dtype=F64)
    raw = code.inputs(code.encode(means, spreads, ship_type), means, spreads, ship_type, centre)
    value = dict(zip(RAW_STATE, raw[0].unbind(-1), strict=True))
    position_bin = code.position_period / 3.0**9
    assert math.isclose(value["position_spread_x"], math.log1p(120.0 / position_bin))
    assert math.isclose(value["position_spread_y"], math.log1p(120.0 / position_bin))
    assert math.isclose(value["velocity_spread_x"], math.log1p(5.0 / 5.0))
    assert math.isclose(value["velocity_spread_y"], math.log1p(10.0 / 5.0))
    assert math.isclose(value["velocity_correlation"], 25.0 / 50.0)
    fade = math.exp(-0.5 * 0.5**2)
    assert math.isclose(value["heading_cos"], fade * math.cos(1.0))
    assert math.isclose(value["heading_sin"], fade * math.sin(1.0))
    assert math.isclose(value["nose_rate_spread"], math.log1p(1.0))  # one bin of 0.4 rad/s
    assert math.isclose(value["shield_spread"], math.log1p(0.1 / 0.05))  # a fraction, bins 1/20


@pytest.fixture(scope="module")
def segment(module_global_generator):
    net = cases.model(**RAW, **EVERY_INPUT)
    _, trajectory = cases.rollout(net, config=cases.SHORT_FRONTLINE, environments=4, length=24)
    return net, trajectory


def test_the_offset_reaches_no_raw_token(segment):
    """The same history in a frame shifted by delta (the replay's positions and offset
    moved alike) composes the same tokens, every column: positions are read from the map
    centre, and rotary attention and presence read differences."""
    net, trajectory = segment
    net = cases.double(net)
    obs, replay = cases.cast(trajectory.obs, F64), cases.cast(trajectory.replay, F64)
    reset = cases.resets(obs)
    delta = torch.rand(replay.offset.shape, dtype=F64) * net.config.position_period
    means = replay.means.clone()
    means[..., 0:2] += delta[..., None, :]
    shifted = dataclasses.replace(replay, offset=replay.offset + delta, means=means)
    initial = cases.cast(trajectory.initial_carry, F64)
    with torch.no_grad():
        plain = net.tokens(obs, replay, net.preceding_motion(initial, obs, replay, reset))
        moved = net.tokens(obs, shifted, net.preceding_motion(initial, obs, shifted, reset))
    for name in ("ship", "global_token", "field", "zone"):
        torch.testing.assert_close(getattr(moved, name), getattr(plain, name), rtol=0, atol=1e-9)
    # the code, the next-state baseline, is the colour code of the shifted positions
    assert not torch.allclose(moved.code, plain.code)


def test_the_raw_ship_token_opens_with_the_raw_state_of_the_belief(segment):
    net, trajectory = segment
    obs, replay = trajectory.obs, trajectory.replay
    reset = cases.resets(obs)
    with torch.no_grad():
        previous = net.preceding_motion(trajectory.initial_carry, obs, replay, reset)
        tokens = net.tokens(obs, replay, previous)
    valid, _, means, spreads = believed(obs.ships, replay)
    centre = replay.offset[..., None, :]
    position = torch.where(valid[..., None], means[..., 0:2], centre)
    means = torch.cat([position, means[..., 2:]], -1)
    expected = net.state_code.raw_inputs(tokens.code, means, spreads, tokens.ship_type, centre)
    torch.testing.assert_close(tokens.ship[..., : len(RAW_STATE)], expected, rtol=0, atol=0)
    assert torch.isfinite(tokens.ship).all() and torch.isfinite(tokens.global_token).all()
    hidden = replay.valid & (replay.age > 0)
    assert hidden.any(), "the segment should hold beliefs about hidden ships"


def test_a_raw_build_unrolls_as_a_loop_of_steps(segment):
    net, trajectory = segment
    net = cases.double(net)
    obs = cases.cast(trajectory.obs, F64)
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        decisions, final = cases.step_loop(
            net, net.step, carry, obs, cases.joint_actions(trajectory)
        )
        unrolled = net.unroll(carry, obs, decisions.replay, cases.resets(obs))
    for name in ("policy_logits", "values"):
        error = cases.scale_error(getattr(unrolled, name), getattr(decisions, name))
        assert error <= 1e-9, name
    error = cases.scale_error(unrolled.next_state_logits[..., OWN:, :], decisions.next_state_logits)
    assert error <= 1e-9
    assert int(final.nonfinite.sum()) == 0


def test_a_raw_build_trains():
    net = cases.model(**RAW)
    runner = lc.rollout(net, environments=4, length=16)
    learner = lc.learner(net, runner)
    out = metrics.read(learner.update(runner.store, runner.bootstrap()))
    assert math.isfinite(out["next_state/loss"]) and math.isfinite(out["shot/loss/outcome"])
    assert all(torch.isfinite(p).all() for p in net.parameters())
    prepared = learner.prepare(runner.store, runner.bootstrap())
    assert prepared["counts"].shape[-1] == len(COUNTS)


def _settings(lines: tuple[str, ...]):
    from bnb.cli import parse_setting, with_settings
    from bnb.runtime.train import TrainConfig, model_build

    config = with_settings(
        TrainConfig(game=GameConfig(5, 10, 2600.0)), [parse_setting(line) for line in lines]
    )
    return model_build(config)


def test_the_arms_settings_line_builds_it_and_old_records_read_as_codes(tmp_path):
    from bnb.models.yemong.model import YemongBuild
    from bnb.runs.serialise import from_json, to_json

    build = _settings(("yemong.input_encoding=raw",))
    assert build.config == dataclasses.replace(_settings(()).config, input_encoding="raw")
    record = to_json(build)
    assert record["config"]["input_encoding"] == "raw"
    assert from_json(YemongBuild, record)[0] == build
    del record["config"]["input_encoding"]  # a policy from before the setting
    old, filled = from_json(YemongBuild, record)
    assert old.config.input_encoding == "codes" and "config.input_encoding" in filled
    # a new run's raw build saves and loads, and plays as saved
    net = cases.model(**dataclasses.asdict(build.config))
    save_policy(tmp_path / "policy", net)
    loaded = load_policy(tmp_path / "policy", cases.CPU).eval()
    assert loaded.build == net.build
    for (name, a), (other, b) in zip(
        net.state_dict().items(), loaded.state_dict().items(), strict=True
    ):
        assert name == other and torch.equal(a, b), name
