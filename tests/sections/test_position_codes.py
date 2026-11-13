"""The position codes of 2026-11-12 (docs/rewrite/sections/position-codes/): the rotary
bands' ladder and per-band turn, the colour code's period, the arena radius's code, the
geometry head's targets and pairs, and the record defaults that keep older builds as they
were built."""

import math

import pytest
import torch
import yemong_cases as cases

from bnb.contracts.obs import SHIP
from bnb.game import api
from bnb.learn.ppo import LearnerConfig
from bnb.models.yemong import codes, geometry
from bnb.models.yemong.inputs import ARENA_CODE, ARENA_CODE_RADIUS, arena_code
from bnb.models.yemong.layers import rotary_tables
from bnb.models.yemong.model import ROTARY_TURN, Yemong, YemongConfig, yemong_build
from bnb.models.yemong.state_code import POSITION_LEVELS, POSITION_PERIOD
from bnb.runs.serialise import SerialiseError, from_json, to_json
from bnb.runtime.train import TrainConfig

pytestmark = pytest.mark.usefixtures("global_generator")

NEW = TrainConfig.__dataclass_fields__["yemong"].default  # a new run's build settings


def _angles(position, config: YemongConfig) -> torch.Tensor:
    axes = torch.tensor(config.rotary_axes(), dtype=torch.float64)
    periods = torch.tensor(config.rotary_periods(), dtype=torch.float64).repeat(2)
    heading = torch.zeros(position.shape[:-1], dtype=torch.float64)
    frequencies = torch.tensor(config.heading_harmonics, dtype=torch.float64)
    cos, sin = rotary_tables(position, heading, axes, periods, frequencies)
    return torch.atan2(sin, cos)[..., 0, : 2 * config.rotary_octaves]


def _wrapped(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def test_a_new_runs_ladder_is_the_decided_one():
    """Eight bands at ratio 3.405 from 64 m (48 m before the x2 game), each turned
    ROTARY_TURN from the coarser one, heading harmonics 1, 2, 4: 19 pairs, 38 of a head's 64
    dims."""
    periods = NEW.rotary_periods()
    # the decided values, m, coarsest first (64 x 3.405^k)
    decided = (339625, 99743, 29293, 8603, 2527, 742, 218, 64)
    assert len(periods) == 8 and periods[-1] == pytest.approx(64.0, rel=1e-12)
    for period, expected in zip(periods, decided, strict=True):
        assert period == pytest.approx(expected, rel=1e-3, abs=0.5)
    assert NEW.rotary_turn == ROTARY_TURN and NEW.heading_harmonics == (1, 2, 4)
    assert 2 * NEW.rotary_pairs == 38 <= NEW.width // NEW.heads
    axes = torch.tensor(NEW.rotary_axes(), dtype=torch.float64)
    for band in range(8):
        angle = math.radians(band * ROTARY_TURN)
        first, second = axes[band], axes[8 + band]
        torch.testing.assert_close(first, torch.tensor([math.cos(angle), math.sin(angle)]).double())
        assert float(first @ second) == pytest.approx(0.0, abs=1e-12)


def test_the_record_defaults_are_the_world_axes_octaves():
    """Older builds read the octaves 8192 m / 2^k on exactly the world's x and y."""
    config = YemongConfig()
    assert config.rotary_periods() == tuple(8192.0 / 2**k for k in range(8))
    assert config.rotary_axes() == ((1.0, 0.0),) * 8 + ((-0.0, 1.0),) * 8
    position = torch.tensor([[1234.5, -987.25], [70000.125, 3.5]], dtype=torch.float64)
    angles = _angles(position, config)
    periods = torch.tensor(config.rotary_periods(), dtype=torch.float64)
    expected = torch.cat(
        [
            2 * math.pi * torch.remainder(position[..., 0:1], periods) / periods,
            2 * math.pi * torch.remainder(position[..., 1:2], periods) / periods,
        ],
        -1,
    )
    torch.testing.assert_close(torch.remainder(angles, 2 * math.pi), expected)


@pytest.mark.parametrize("config", [YemongConfig(), NEW], ids=["record", "new"])
def test_the_rotary_angle_between_two_tokens_reads_only_their_difference(config):
    """With every band turned, the angles' difference between two positions is the same
    after a common shift (modulo 2 pi): attention reads relative position only."""
    generator = torch.Generator().manual_seed(3)
    a = torch.rand((64, 2), generator=generator, dtype=torch.float64) * 30_000.0
    b = a + (torch.rand((64, 2), generator=generator, dtype=torch.float64) - 0.5) * 20_000.0
    shift = torch.rand((64, 2), generator=generator, dtype=torch.float64) * 236_196.0
    before = _angles(a, config) - _angles(b, config)
    after = _angles(a + shift, config) - _angles(b + shift, config)
    assert _wrapped(after - before).abs().max() < 1e-8


def test_a_shift_by_the_record_coarsest_period_is_no_longer_invisible():
    """The record ladder aliases a shift of 8192 m along x exactly; the new one does not."""
    shift = torch.tensor([[8192.0, 0.0]], dtype=torch.float64)
    zero = torch.zeros((1, 2), dtype=torch.float64)
    for config, aliased in ((YemongConfig(), True), (NEW, False)):
        difference = _wrapped(_angles(shift, config) - _angles(zero, config)).abs()
        assert (difference.max() < 1e-9) == aliased


def test_a_new_runs_colour_code_spans_36_m_to_236_km():
    assert YemongConfig().position_period == POSITION_PERIOD == 4.0 * 3.0**8
    assert NEW.position_period == 36.0 * 3.0**8 == 236_196.0
    assert NEW.position_period / 3.0 ** (POSITION_LEVELS - 1) == 36.0
    net = cases.model(position_period=NEW.position_period).double()
    code = net.state_code
    assert code.position_period == NEW.position_period
    means = torch.zeros((3, 12), dtype=torch.float64)
    means[:, 0:2] = torch.tensor([[100.0, 2000.0], [150_000.0, 7.0], [-30_000.0, 236_000.0]])
    spreads = torch.zeros((3, code.spread_count), dtype=torch.float64)
    ship_type = torch.zeros(3, dtype=torch.int64)
    encoded = code.encode(means, spreads, ship_type)
    shifted = means.clone()
    shifted[:, 0] += NEW.position_period  # one period on: the same code
    torch.testing.assert_close(code.encode(shifted, spreads, ship_type), encoded)
    finer = means.clone()
    finer[:, 0] += 12.0  # a third of the finest level: another code
    assert (code.encode(finer, spreads, ship_type) - encoded).abs().max() > 0.1
    decoded, _ = code.decode(encoded, ship_type)
    expected = torch.remainder(means[:, 0:2], NEW.position_period)
    torch.testing.assert_close(decoded[:, 0:2], expected, rtol=0, atol=1e-6)


def test_a_new_views_offsets_span_the_new_period():
    net = cases.model(position_period=NEW.position_period)
    offsets = net.initial_carry(4096, 5, 10).offset
    assert ((offsets >= 0) & (offsets < NEW.position_period)).all()
    assert offsets.max() > 0.9 * NEW.position_period


def test_the_arena_code_peaks_at_its_radius_and_clamps_past_its_range():
    radius = ARENA_CODE_RADIUS * torch.tensor([0.25, 1.0, 8.0, 0.01, 100.0], dtype=torch.float64)
    code = arena_code(radius)
    assert code.shape == (5, ARENA_CODE) and ARENA_CODE == 21 + 2 * codes.PADDING_BINS
    torch.testing.assert_close(code.sum(-1), torch.ones(5, dtype=torch.float64))
    peaks = code.argmax(-1).tolist()
    pad = codes.PADDING_BINS
    assert peaks[:3] == [pad, pad + 8, pad + 20]  # a quarter octave a bin
    assert peaks[3] == 0 and peaks[4] == ARENA_CODE - 1  # clamped to the outermost padding


def test_the_arena_code_is_read_only_by_builds_that_ask_for_it():
    plain, coded = cases.model(), cases.model(arena_code=True)
    width = plain.encoder.global_token.first.weight.shape[1]
    assert coded.encoder.global_token.first.weight.shape[1] == width + ARENA_CODE


# --------------------------------------------------------------------------- geometry


def test_the_geometry_quantities_of_a_head_on_pair():
    """A at the origin nose east at 100 m/s, B 500 m east nose west at 150 m/s: range
    500 m, B dead ahead of A, A dead ahead of B (aspect pi), headings crossing at pi,
    closing at 250 m/s."""
    position = torch.tensor([[0.0, 0.0], [500.0, 0.0]], dtype=torch.float64)
    velocity = torch.tensor([[100.0, 0.0], [-150.0, 0.0]], dtype=torch.float64)
    heading = torch.tensor([0.0, math.pi], dtype=torch.float64)
    values = geometry.quantities(position, velocity, heading, torch.tensor([[1]]))[0, 0]
    assert values[0] == pytest.approx(500.0)
    assert values[1] == pytest.approx(0.0, abs=1e-12)
    assert abs(values[2]) == pytest.approx(math.pi)
    assert abs(values[3]) == pytest.approx(math.pi)
    assert values[4] == pytest.approx(250.0)


def test_the_geometry_quantities_of_a_tail_chase_off_the_beam():
    """A at the origin nose north; B 300 m east, nose north, flying away at 50 m/s more:
    B at -90 degrees off A's nose (to its right), A abeam of B (aspect -90 degrees: the
    line of sight A to B is east, B's heading north), headings parallel, closure 0."""
    position = torch.tensor([[0.0, 0.0], [300.0, 0.0]], dtype=torch.float64)
    velocity = torch.tensor([[0.0, 100.0], [0.0, 150.0]], dtype=torch.float64)
    heading = torch.tensor([math.pi / 2, math.pi / 2], dtype=torch.float64)
    values = geometry.quantities(position, velocity, heading, torch.tensor([[1]]))[0, 0]
    assert values[1] == pytest.approx(-math.pi / 2)
    assert values[2] == pytest.approx(-math.pi / 2)
    assert values[3] == pytest.approx(0.0, abs=1e-12)
    assert values[4] == pytest.approx(0.0, abs=1e-9)


def test_the_angle_codes_wrap_round_the_circle():
    bins = geometry.ANGLE_BINS
    near_pi = torch.tensor([math.pi - 1e-9, -math.pi + 1e-9], dtype=torch.float64)
    code = codes.wrapped_hl_gauss(near_pi, bins)
    torch.testing.assert_close(code[0], code[1], rtol=0, atol=1e-7)
    centre = bins // 2  # the bin at pi
    assert int(code[0].argmax()) == centre
    torch.testing.assert_close(code[0, centre - 1], code[0, centre + 1], rtol=0, atol=1e-7)
    at_zero = codes.wrapped_hl_gauss(torch.zeros(1, dtype=torch.float64), bins)[0]
    torch.testing.assert_close(at_zero[1], at_zero[-1], rtol=0, atol=1e-12)  # across the seam
    assert int(at_zero.argmax()) == 0
    torch.testing.assert_close(at_zero.sum(), torch.tensor(1.0, dtype=torch.float64))
    shifted = codes.wrapped_hl_gauss(torch.tensor([2 * math.pi / bins], dtype=torch.float64), bins)
    torch.testing.assert_close(shifted[0], torch.roll(at_zero, 1))  # one bin on, the same shape


def test_the_geometry_code_bins_range_and_closure():
    values = torch.tensor(
        [[10.0, 0.0, 0.0, 0.0, 0.0], [64_000.0, 0.0, 0.0, 0.0, 2000.0], [1e6, 0, 0, 0, -1e4]],
        dtype=torch.float64,
    )
    code = geometry.encode(values)
    assert code.shape == (3, geometry.WIDTH)
    parts = codes.split_runs(code, geometry.RUNS)
    for part in parts:
        torch.testing.assert_close(part.sum(-1), torch.ones_like(part.sum(-1)))
    range_code, closure_code = parts[0][:, 0], parts[2][:, 0]
    pad = codes.PADDING_BINS
    spacing = math.log1p(geometry.RANGE.bound / geometry.RANGE.knee) / (geometry.RANGE.bins - 1)
    assert int(range_code[0].argmax()) == pad + round(math.log(2.0) / spacing)  # 10 m: ln 2
    assert int(range_code[1].argmax()) == pad + geometry.RANGE.bins - 1  # the bound
    assert int(range_code[2].argmax()) == geometry.RANGE.width - 1  # past it: the padding
    middle = pad + geometry.CLOSURE.bins // 2
    assert int(closure_code[0].argmax()) == middle  # zero closure: the centre bin
    assert int(closure_code[1].argmax()) == middle + geometry.CLOSURE.bins // 2
    assert int(closure_code[2].argmax()) == 0  # opening past the range: the low padding


def test_partner_rows_cover_every_other_row_and_never_the_anchor():
    small = geometry.partner_rows(10, 5, 3)  # 9 others: 8 a tick
    assert small.shape == (3, 5, geometry.PARTNERS)
    for anchor in range(5):
        for tick in range(3):
            row = small[tick, anchor].tolist()
            assert anchor not in row and len(set(row)) == len(row)
    large = geometry.partner_rows(100, 50, 13)  # 99 others: covered in 13 ticks
    for anchor in (0, 17, 49):
        seen = set(large[:, anchor].flatten().tolist())
        assert seen == set(range(100)) - {anchor}
    all_rows = geometry.partner_rows(4, 2, 2)  # fewer rows than PARTNERS: every other row
    assert all_rows.shape == (2, 2, 3)
    assert set(all_rows[0, 0].tolist()) == {1, 2, 3}


def test_pairs_mask_hidden_partners_and_dead_ships():
    """Pairs read only ships the view sees: a hidden partner, a dead partner and every
    pair of a dead anchor are masked; the targets come from the observed columns."""
    _, trajectory = cases.rollout(cases.model(), length=24)
    ships, alive = trajectory.obs.ships, trajectory.alive
    partners, mask, values = geometry.pairs(ships, alive, 5)
    visible = ships[..., SHIP["visible"]] > 0.5
    seen = (alive & visible).gather(-1, partners.flatten(-2)).unflatten(-1, partners.shape[-2:])
    expected = alive[..., :5, None] & seen
    assert torch.equal(mask, expected)
    hidden = ~visible.gather(-1, partners.flatten(-2)).unflatten(-1, partners.shape[-2:])
    assert hidden.any() and not (mask & hidden).any()  # the fog hides some partners
    assert torch.isfinite(values[mask]).all()
    position = torch.stack([ships[..., SHIP["x"]], ships[..., SHIP["y"]]], -1)
    other = position.gather(-2, partners.flatten(-2)[..., None].expand(-1, -1, -1, 2))
    distance = (other.unflatten(-2, partners.shape[-2:]) - position[..., :5, None, :]).norm(dim=-1)
    torch.testing.assert_close(values[..., 0][mask], distance[mask].to(values.dtype))


def _initialised(**settings) -> dict:
    """The state dict of a freshly constructed Yemong, its constructor's draws seeded."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        build = yemong_build(YemongConfig(**settings), api.rules_of(cases.SQUAD), api.ship_types())
        return Yemong(build).state_dict()


def test_the_geometry_head_adds_only_its_own_parameters_and_draws_last():
    plain = _initialised()
    with_head = _initialised(geometry_head=True)
    added = set(with_head) - set(plain)
    assert added and all(name.startswith("geometry_head.") for name in added)
    assert all(torch.equal(plain[name], with_head[name]) for name in plain)


# --------------------------------------------------------------------------- records


def test_a_record_with_the_read_out_off_still_reads_and_one_with_it_on_fails():
    data = to_json(YemongConfig())
    config, filled = from_json(YemongConfig, {**data, "rotary_readout": False})
    assert config == YemongConfig() and not filled
    old = {key: value for key, value in data.items() if key not in ("rotary_ratio",)}
    config, filled = from_json(YemongConfig, old)
    assert config == YemongConfig() and filled == ["rotary_ratio"]
    with pytest.raises(SerialiseError, match="rotary_readout"):
        from_json(YemongConfig, {**data, "rotary_readout": True})
    learner, _ = from_json(LearnerConfig, {**to_json(LearnerConfig()), "rotary_readout": 0.5})
    assert learner == LearnerConfig()
