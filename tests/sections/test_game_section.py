"""bnb.game beyond the acceptance tests: compiled vs eager, the ship card as tensors, the
hit test against brute force (with its tie order), the medium against a float64
reference, the defaults against the spec's table, the batch-1 realtime path, and fresh
randomness in CUDA-graph replays."""

import json
import math
import pathlib
import sys

import pytest
import torch

from bnb.contracts import Action, GameConfig
from bnb.game import api
from bnb.game.physics.combat import hits
from bnb.game.physics.medium import medium
from bnb.game.physics.spawning import formation_rank
from bnb.game.rules import LN_SQRT2, FlightConstants, Rules
from bnb.game.ship_types import DEFAULTS
from bnb.game.state import Fields, replace

REFERENCE = pathlib.Path(__file__).resolve().parents[1] / "reference"
sys.path.insert(0, str(REFERENCE))
from race_arm import CARD, FLIGHT_SHARED, RULES  # noqa: E402

TYPES = api.ship_types()


def _actions(batch, ships, generator):
    choice = torch.rand(batch, ships, 3, generator=generator)
    return Action.unpack((choice * torch.tensor([3.0, 7.0, 2.0])).to(torch.int32))


def _warm(config=GameConfig(3, 5, 1200.0), batch=8, ticks=90, seed=0):
    """A mid-game state: some motion, firing, hits and respawns."""
    rules = api.rules_of(config)
    generator = torch.Generator().manual_seed(seed)
    state = api.reset(rules, TYPES, api.uniforms(rules, batch, generator=generator))
    for _ in range(ticks):
        state, _ = api.step(rules, TYPES, state, _actions(batch, rules.num_ships, generator),
                            api.spread(rules, batch, generator=generator),
                            api.uniforms(rules, batch, generator=generator))
    return rules, state, generator


def _leaves(tree):
    return torch.utils._pytree.tree_leaves(tree)


# ------------------------------------------------------------------ compiled vs eager

SIZES = [
    pytest.param(GameConfig(3, 5, 1200.0), id="odd-M"),
    pytest.param(GameConfig(2, 64, 1500.0), id="M64", marks=pytest.mark.slow),
    pytest.param(GameConfig(50, 40, 8200.0), id="battle", marks=pytest.mark.slow),
]


@pytest.mark.parametrize("config", SIZES)
def test_compiled_tick_and_perception_match_eager(config):
    """One tick plus perception from a mid-game state. Floats agree to float32 fusion
    rounding (2e-5 relative); flags, counters and indices exactly."""
    rules, state, generator = _warm(config, batch=4, ticks=60)
    B, N = state.ships.x.shape
    inputs = (_actions(B, N, generator), api.spread(rules, B, generator=generator),
              api.uniforms(rules, B, generator=generator))

    def tick(state, action, spread, uniforms):
        state, events = api.step(rules, TYPES, state, action, spread, uniforms)
        state, obs, truth = api.perceive(rules, state)
        return state, events, obs, truth

    want = tick(state, *inputs)
    got = torch.compile(tick, fullgraph=True, dynamic=False)(state, *inputs)
    for a, b in zip(_leaves(got), _leaves(want), strict=True):
        if b.dtype.is_floating_point:
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
        else:
            assert torch.equal(a, b)


# ------------------------------------------------------------------ the hit test

def test_hits_match_brute_force_with_friendly_fire():
    """First contact per bullet over every ship but its shooter, damage per (shooter,
    target) and the killer, against a direct loop. Bullets start near random ships."""
    rules, state, generator = _warm(GameConfig(4, 3, 600.0), batch=4, ticks=30, seed=1)
    B, N = state.ships.x.shape
    K = rules.bullet_slots
    ships, bullets = state.ships, state.bullets

    def noise(*shape, scale):
        return scale * (2.0 * torch.rand(*shape, generator=generator) - 1.0)

    near = torch.randint(0, N, (B, N * K), generator=generator)
    sweep = api.Sweep(
        ship_x=ships.x, ship_y=ships.y, ship_displacement_x=noise(B, N, scale=6.0),
        ship_displacement_y=noise(B, N, scale=6.0),
        bullet_x=ships.x.gather(1, near).reshape(B, N, K) + noise(B, N, K, scale=15.0),
        bullet_y=ships.y.gather(1, near).reshape(B, N, K) + noise(B, N, K, scale=15.0),
        bullet_displacement_x=noise(B, N, K, scale=20.0),
        bullet_displacement_y=noise(B, N, K, scale=20.0))
    angle = noise(B, N, K, scale=math.pi)
    bullets = replace(bullets, moves_left=torch.randint(0, 3, (B, N, K), generator=generator).int(),
                      direction_x=angle.cos(), direction_y=angle.sin())
    ships = replace(ships, alive=torch.rand(B, N, generator=generator) < 0.85)
    got = hits(rules, TYPES, sweep, ships, bullets, ships.alive)

    radius = TYPES.hit_radius[ships.ship_type.long()].double()
    full = TYPES.hit_damage[ships.ship_type.long()].double()
    damage = torch.zeros(B, N, N, dtype=torch.float64)
    first = torch.full((B, N), -1)
    best = torch.full((B, N), math.inf)
    count = allies = 0
    for b in range(B):
        for owner in range(N):
            for k in range(K):
                if bullets.moves_left[b, owner, k] <= 0:
                    continue
                hit_tau, target = math.inf, -1
                for t in range(N):
                    if t == owner or not ships.alive[b, t]:
                        continue
                    start = torch.stack([sweep.bullet_x[b, owner, k] - sweep.ship_x[b, t],
                                         sweep.bullet_y[b, owner, k] - sweep.ship_y[b, t]]).double()
                    relative = torch.stack(
                        [sweep.bullet_displacement_x[b, owner, k] - sweep.ship_displacement_x[b, t],
                         sweep.bullet_displacement_y[b, owner, k] - sweep.ship_displacement_y[b, t]]
                    ).double()
                    tau = float((-(start @ relative) / (relative @ relative).clamp_min(1e-12))
                                .clamp(0.0, 1.0))
                    miss = start + tau * relative
                    if float(miss @ miss) < float(radius[b, t]) ** 2 and tau < hit_tau:
                        hit_tau, target = tau, t
                assert bool(got.bullet_hit[b, owner, k]) == (target >= 0), (b, owner, k)
                if target < 0:
                    continue
                count += 1
                allies += (owner // rules.ships_per_team) == (target // rules.ships_per_team)
                dx = float(bullets.direction_x[b, owner, k])
                dy = float(bullets.direction_y[b, owner, k])
                nose_cos = float(ships.heading_cos[b, target])
                nose_sin = float(ships.heading_sin[b, target])
                theta = math.atan2(abs(dx * nose_sin - dy * nose_cos),
                                   -(dx * nose_cos + dy * nose_sin))
                damage[b, owner, target] += float(full[b, owner]) * (
                    1.0 - (1.0 - rules.head_on_fraction) * math.exp(-4.0 * theta * theta / math.pi))
                if hit_tau < best[b, target]:  # strict: ties keep the lower bullet index
                    best[b, target], first[b, target] = hit_tau, owner
    assert count > 20 and allies > 0, (count, allies)
    assert torch.equal(got.first_shooter.long(), first)
    torch.testing.assert_close(got.damage.double(), damage, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(got.taken, got.damage.sum(1))


@pytest.mark.parametrize("swap", [False, True])
def test_killer_ties_at_tau_zero_go_to_the_lower_bullet_eager_and_compiled(swap):
    """Two bullets touch ship 3 at tau = 0: one moving with it (tau = -0.0 before the
    clamp) and one moving away (negative tau). The lower bullet index wins either way."""
    rules = api.rules_of(GameConfig(2, 1, 1200.0))
    state = api.reset(rules, TYPES, api.uniforms(rules, 1))
    N, K = 4, rules.bullet_slots
    ships = replace(state.ships, x=torch.tensor([[0.0, 100.0, 200.0, 50.0]]), y=torch.zeros(1, 4),
                    alive=torch.ones(1, 4, dtype=torch.bool))
    start, step = torch.zeros(1, N, K), torch.zeros(1, N, K)
    moves_left = torch.zeros(1, N, K, dtype=torch.int32)
    with_ship, away = (1, 0) if swap else (0, 1)
    start[0, with_ship, 0], step[0, with_ship, 0] = 51.0, 10.0  # same motion as the target
    start[0, away, 0], step[0, away, 0] = 52.0, 20.0
    moves_left[0, :2, 0] = 1
    sweep = api.Sweep(ship_x=ships.x, ship_y=ships.y,
                      ship_displacement_x=torch.tensor([[0.0, 0.0, 0.0, 10.0]]),
                      ship_displacement_y=torch.zeros(1, 4), bullet_x=start,
                      bullet_y=torch.zeros(1, N, K), bullet_displacement_x=step,
                      bullet_displacement_y=torch.zeros(1, N, K))
    bullets = replace(state.bullets, moves_left=moves_left)

    def shooter(sweep, ships, bullets):
        return hits(rules, TYPES, sweep, ships, bullets, ships.alive).first_shooter

    assert int(shooter(sweep, ships, bullets)[0, 3]) == 0
    compiled = torch.compile(shooter, fullgraph=True, dynamic=False)
    assert int(compiled(sweep, ships, bullets)[0, 3]) == 0


# ------------------------------------------------------------------ the medium

def _reference_medium(fields: Fields, x, y):
    """Game spec 4 in float64, the leave-one-out products written out."""
    fx, fy, radius, band = (v.double()[:, None, :] for v in
                            (fields.x, fields.y, fields.radius, fields.band_width))
    target = fields.level.double()[:, None, :] * LN_SQRT2
    dx, dy = x.double()[..., None] - fx, y.double()[..., None] - fy
    d = torch.sqrt(dx * dx + dy * dy)
    zeta = 0.5 - (d - radius) / band
    z = zeta.clamp(0.0, 1.0)
    alpha = 6 * z ** 5 - 15 * z ** 4 + 10 * z ** 3
    in_band = (zeta > 0) & (zeta < 1) & (d > 1e-6)
    slope = torch.where(in_band, -30 * z ** 2 * (z - 1) ** 2 / band / d.clamp_min(1e-6), 0.0)
    gax, gay = slope * dx, slope * dy
    M = fields.x.shape[1]
    complement = 1.0 - alpha
    others = torch.stack([torch.cat([complement[..., :i], complement[..., i + 1:]], -1).prod(-1)
                          for i in range(M)], -1)
    coverage = 1.0 - complement.prod(-1)
    weight = alpha.sum(-1)
    covered = weight > 1e-12
    safe = torch.where(covered, weight, 1.0)
    mean = torch.where(covered, (alpha * target).sum(-1) / safe, 0.0)
    out = [coverage * mean]
    for g in (gax, gay):
        mean_gradient = torch.where(covered, ((g * target).sum(-1) - mean * g.sum(-1)) / safe, 0.0)
        out.append(mean * (g * others).sum(-1) + coverage * mean_gradient)
    return out


def _fields(x, y, radius, level):
    def row(values, dtype=torch.float32):
        return torch.tensor([values], dtype=dtype)

    return Fields(x=row(x), y=row(y), radius=row(radius), band_width=torch.full((1, len(x)), 40.0),
                  level=row(level, torch.int32))


INNER = 200.0 - 20.0  # the inner edge of a 200 m field's band (zeta = 1)
MEDIUM_CASES = {
    "inner edge": (_fields([0.0], [0.0], [200.0], [2]),
                   [INNER + e for e in (1e-1, 1e-3, 1e-5, 3e-6, 1e-6)]),
    "64 coincident": (_fields([0.0] * 64, [0.0] * 64, [200.0] * 64, [2] * 64),
                      [INNER + e for e in (1e-3, 1e-5, 2e-6)] + [190.0, 215.0]),
    "nested": (_fields([0.0, 0.0], [0.0, 0.0], [50.0, 400.0], [2, -2]), [30.0, 50.0, 60.0, 395.0]),
    "64 staggered": (_fields([0.0] * 64, [0.0] * 64, [INNER + 1e-4 * i for i in range(64)],
                             [1, -1] * 32), [INNER + 0.003, INNER + 10.0, INNER + 35.0]),
    "exact centre": (_fields([0.0, 0.0], [0.0, 0.0], [200.0, 15.0], [1, -1]), [0.0, 1e-7, 3.0]),
    "battle scale": (_fields([7000.0, -7000.0], [0.0, 0.0], [300.0, 300.0], [2, 2]),
                     [7000 + 280 + 0.01, 7000 + 300.0, 7000 + 310.0, -6700.0 - 5.0]),
}


@pytest.mark.parametrize("case", sorted(MEDIUM_CASES))
def test_medium_matches_a_float64_leave_one_out_reference(case):
    fields, points = MEDIUM_CASES[case]
    x = torch.tensor([points], dtype=torch.float32)
    y = torch.zeros_like(x)
    got = medium(fields, x, y)
    want = _reference_medium(fields, x, y)
    # float32 positions resolve |p| to ~eps |p|; the medium moves by the gradient (or the
    # second derivative, at most 3e-3 / m^2) times that.
    resolution = 4.0 * torch.finfo(torch.float32).eps * x.double().abs().clamp_min(1.0)
    slope = torch.hypot(want[1], want[2])
    for name, g, w, tolerance in zip(("logn", "gx", "gy"), got, want,
                                     (2e-6 + resolution * slope, 2e-6 + resolution * 3e-3,
                                      2e-6 + resolution * 3e-3)):
        assert torch.isfinite(g).all(), name
        error = (g.double() - w).abs()
        assert (error <= tolerance).all(), (case, name, error.max().item())


@pytest.mark.parametrize("case", sorted(MEDIUM_CASES))
def test_medium_gradients_are_finite_everywhere(case):
    fields, points = MEDIUM_CASES[case]
    x = torch.tensor([points], requires_grad=True)
    y = torch.zeros_like(x, requires_grad=True)
    fields = Fields(**{k: (v.clone().requires_grad_(True) if v.dtype.is_floating_point else v)
                       for k, v in fields.__dict__.items()})
    total = sum(out.sum() for out in medium(fields, x, y))
    inputs = [x, y, fields.x, fields.y, fields.radius, fields.band_width]
    for grad in torch.autograd.grad(total, inputs):
        assert torch.isfinite(grad).all(), case


# ------------------------------------------------------------------ constants and the card

def test_defaults_match_the_spec_table():
    spec = json.loads((REFERENCE / "golden" / "presets.json").read_text())
    preset = spec["presets"]["squad"]
    rules = api.rules_of(GameConfig(5, 10, 2600.0))
    for key, value in preset.items():
        if key in RULES:
            assert math.isclose(getattr(rules, RULES[key]), value), key
        else:
            sizes = ("num_ships", "num_fields", "arena_radius", "ship_types")
            assert key in CARD or key in sizes, key
    for key, value in spec["flight"].items():
        if key in FLIGHT_SHARED:
            assert math.isclose(getattr(FlightConstants(), FLIGHT_SHARED[key]), value), key
        else:
            assert key in CARD, key
    for t, name in enumerate(("fighter", "heavy")):
        override = spec["types"].get(name, {})
        for key, field in CARD.items():
            want = override.get("flight", {}).get(key, override.get(key))
            if want is None:
                want = spec["flight"].get(key, preset.get(key))
            assert math.isclose(float(getattr(TYPES, field)[t]), want, rel_tol=1e-6), (name, key)
    assert set(CARD.values()) == set(DEFAULTS)


def test_ship_card_is_data_per_environment():
    """A (B, num_types) card gives each environment its own values; changing values does
    not recompile."""
    rules, state, _ = _warm(batch=2)
    B, N = state.ships.x.shape
    action = Action.unpack(torch.zeros(B, N, 3, dtype=torch.int32))
    spread, uniforms = torch.zeros(B, N, 2), torch.zeros(B, rules.tick_uniform_count)
    lift = TYPES.max_lift_coefficient.expand(B, -1) * torch.tensor([[1.0], [0.5]])
    wide = api.ship_types(max_lift_coefficient=lift)
    assert wide.hit_radius.shape == (B, 2)
    one, _ = api.step(rules, TYPES, state, action, spread, uniforms)
    two, _ = api.step(rules, wide, state, action, spread, uniforms)
    alive = state.ships.alive
    assert torch.equal(one.ships.proper_speed[0], two.ships.proper_speed[0])
    assert not torch.equal(one.ships.proper_speed[1][alive[1]], two.ships.proper_speed[1][alive[1]])

    torch._dynamo.reset()
    step = torch.compile(lambda s, c: api.step(rules, c, s, action, spread, uniforms),
                         fullgraph=True)
    step(state, TYPES)
    graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    step(state, api.ship_types(hit_damage=(20.0, 30.0), ticks_between_shots=(1, 4)))
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == graphs


def test_formation_ranks_ties_to_the_lower_slot():
    rules = Rules(ships_per_team=4, num_fields=1, arena_radius=1000.0)
    keys = torch.tensor([[0.5, 0.5, 0.1, 0.9, 0.3, 0.3, 0.3, 0.0]])
    assert formation_rank(rules, keys).tolist() == [[1, 2, 0, 3, 1, 2, 3, 0]]


# ------------------------------------------------------------------ the batch-1 realtime path

@pytest.mark.parametrize("compiled", [False, pytest.param(True, marks=pytest.mark.slow)])
def test_realtime_steps_one_match(compiled):
    from bnb.contracts.obs import GLOBAL, SHIP, SHIP_COLUMNS
    from bnb.runtime.realtime import Realtime

    config = GameConfig(2, 3, 900.0, episode_s=2.0)
    for fogged in (True, False):
        realtime = Realtime(config, compile=compiled, fogged=fogged)
        actions = torch.zeros(config.num_ships, 3, dtype=torch.int32)
        actions[:, 2] = 1
        resets = 0
        for _ in range(70):  # past one 60-tick episode
            obs = realtime.step(actions)
            resets += int(obs.globals[0, 0, GLOBAL["reset"]])
        assert obs.ships.shape == (1, 2, config.num_ships, len(SHIP_COLUMNS))
        assert all(torch.isfinite(v).all() for v in (obs.ships, obs.fields, obs.globals))
        assert resets >= 1
        assert float(obs.globals[0, 0, GLOBAL["fogged"]]) == float(fogged)
        if not fogged:
            ships = obs.ships[0]
            assert torch.equal(ships[..., SHIP["visible"]], ships[..., SHIP["alive"]])


# ------------------------------------------------------------------ CUDA graphs

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_eager_cuda_graph_replays_draw_fresh_randomness():
    """The game's draws and its tick capture into a CUDA graph eagerly, and every replay
    draws anew."""
    rules = api.rules_of(GameConfig(2, 3, 1200.0))
    device = torch.device("cuda")
    card = api.ship_types(device)
    state = api.reset(rules, card, api.uniforms(rules, 16, device))
    action = Action.unpack(torch.ones(16, rules.num_ships, 3, dtype=torch.int32, device=device))
    spread, uniforms = api.spread(rules, 16, device), api.uniforms(rules, 16, device)

    def tick():
        spread.copy_(api.spread(rules, 16, device))
        uniforms.copy_(api.uniforms(rules, 16, device))
        new, _ = api.step(rules, card, state, action, spread, uniforms)
        torch.utils._pytree.tree_map(lambda d, s: d.copy_(s), state, new)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        tick()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tick()
    seen = []
    for _ in range(3):
        graph.replay()
        seen.append((spread.clone(), uniforms.clone()))
    for (s0, u0), (s1, u1) in zip(seen, seen[1:]):
        assert not torch.equal(s0, s1) and not torch.equal(u0, u1)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compiled_rollout_graph_replays_draw_fresh_randomness():
    from bnb.contracts import Pool
    from bnb.controllers.scripted import Random
    from bnb.runtime.rollout import Rollout

    rollout = Rollout(GameConfig(2, 3, 1200.0), (Pool("p", 1.0, ("r", "r")),), {"r": Random(2)},
                      "r", 16, 8, torch.device("cuda"), compile=True, graph_ticks=4)
    draws = []
    for _ in range(3):
        rollout.run()
        draws.append((rollout.carry["spread"].clone(), rollout.carry["uniforms"].clone()))
    assert rollout.graph is not None
    for (s0, u0), (s1, u1) in zip(draws, draws[1:]):
        assert not torch.equal(s0, s1) and not torch.equal(u0, u1)
    spread = torch.stack([s for s, _ in draws])
    uniforms = torch.stack([u for _, u in draws])
    assert abs(float(spread.mean())) < 0.2 and abs(float(spread.std()) - 1.0) < 0.2
    assert 0.0 <= float(uniforms.min()) and float(uniforms.max()) < 1.0
    assert abs(float(uniforms.mean()) - 0.5) < 0.1
