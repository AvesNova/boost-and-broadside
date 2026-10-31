"""bnb.game beyond the acceptance tests: compiled vs eager, the ship card as tensors,
the hit test against brute force, the defaults against the spec's machine-readable table,
and fresh randomness in CUDA-graph replays."""

import json
import math
import pathlib

import pytest
import torch

from bnb.contracts import Action, GameConfig
from bnb.game import match
from bnb.game.physics.combat import hits
from bnb.game.rules import FlightConstants, Rules
from bnb.game.ship_types import DEFAULTS, ship_types
from bnb.game.state import CANONICAL, flat

GOLDEN = pathlib.Path(__file__).resolve().parents[1] / "reference" / "golden" / "presets.json"


def _actions(B, N, g):
    u = torch.rand(B, N, 3, generator=g) * torch.tensor([3.0, 7.0, 2.0])
    return Action.unpack(u.to(torch.int32))


def _warm(cfg=GameConfig(3, 5, 1200.0), B=8, ticks=90, seed=0):
    """A mid-game state: some motion, firing, hits and respawns."""
    p = match.rules_of(cfg)
    g = torch.Generator().manual_seed(seed)
    st = match.reset(p, match.uniforms(p, B, generator=g))
    for _ in range(ticks):
        st, _ = match.step(p, st, _actions(B, p.num_ships, g), match.spread(p, B, generator=g),
                           match.uniforms(p, B, generator=g))
    return p, st, g


def _assert_close(a: dict, b: dict, context=""):
    for k in a:
        x, y = a[k], b[k]
        if x.dtype.is_floating_point:
            torch.testing.assert_close(x, y, rtol=1e-4, atol=1e-3, msg=f"{context}{k}")
        else:
            assert torch.equal(x, y), f"{context}{k}"


def test_compiled_tick_and_perception_match_eager():
    p, st, g = _warm()
    B, N = st.ships.x.shape
    act, spread = _actions(B, N, g), match.spread(p, B, generator=g)
    uni = match.uniforms(p, B, generator=g)

    def tick(state, action, spread, uniforms):
        state, events = match.step(p, state, action, spread, uniforms)
        state, obs, truth = match.perceive(p, state)
        return state, events, obs, truth

    want = tick(st, act, spread, uni)
    got = torch.compile(tick, fullgraph=True, dynamic=False)(st, act, spread, uni)
    _assert_close(flat(got[0]), flat(want[0]), "state ")
    for name in ("damage", "boundary_damage", "died", "fired", "spawned", "killer", "done"):
        _assert_close({name: getattr(got[1], name)}, {name: getattr(want[1], name)}, "events ")
    for part in ("ships", "fields", "globals"):
        _assert_close({part: getattr(got[2], part)}, {part: getattr(want[2], part)}, "obs ")


def test_defaults_match_the_spec_table():
    spec = json.loads(GOLDEN.read_text())
    preset = spec["presets"]["squad"]
    rules = match.rules_of(GameConfig(5, 10, 2600.0))
    renames = dict(h="tick", slots="slots")
    for key, value in preset.items():
        if key in DEFAULTS or key in ("num_ships", "num_fields", "ship_types"):
            continue
        assert math.isclose(getattr(rules, renames.get(key, key)), value), key
    for key, value in spec["flight"].items():
        if key in DEFAULTS:
            continue
        assert math.isclose(getattr(FlightConstants(), key), value), key
    card = ship_types()
    for t, name in enumerate(("fighter", "heavy")):
        override = spec["types"].get(name, {})
        for key in DEFAULTS:
            want = override.get("flight", {}).get(key, override.get(key))
            if want is None:
                want = spec["flight"].get(key, preset.get(key))
            assert math.isclose(float(getattr(card, key)[t]), want, rel_tol=1e-6), (name, key)


def test_ship_card_is_data_per_environment():
    """A (B, num_types) card gives each environment its own values; changing values does
    not recompile."""
    p, st, g = _warm(B=2)
    B, N = st.ships.x.shape
    act = Action.unpack(torch.zeros(B, N, 3, dtype=torch.int32))
    zeros_s, zeros_u = torch.zeros(B, N, 2), torch.zeros(B, p.num_uniforms)
    card = ship_types()
    wide = ship_types(cl_max=card.cl_max.expand(B, -1) * torch.tensor([[1.0], [0.5]]))
    for f in DEFAULTS:
        if f != "cl_max":
            setattr_value = getattr(card, f).expand(B, -1).clone()
            object.__setattr__(wide, f, setattr_value)
    one, _ = match.step(p, st, act, zeros_s, zeros_u, card)
    two, _ = match.step(p, st, act, zeros_s, zeros_u, wide)
    alive = st.ships.alive
    assert torch.equal(one.ships.w[0], two.ships.w[0])
    assert not torch.equal(one.ships.w[1][alive[1]], two.ships.w[1][alive[1]])

    torch._dynamo.reset()
    step = torch.compile(lambda s, c: match.step(p, s, act, zeros_s, zeros_u, c), fullgraph=True)
    step(st, card)
    frames = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    step(st, ship_types(damage=(20.0, 30.0), fire_interval=(1, 4)))
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == frames


def test_hits_match_brute_force_with_friendly_fire():
    """First contact per bullet over every ship but its shooter, damage per (shooter,
    target) and the killer, against a direct loop. Bullets start near random ships."""
    p, st, g = _warm(cfg=GameConfig(4, 3, 600.0), B=4, ticks=30, seed=1)
    B, N = st.ships.x.shape
    K = p.slots
    ships, bullets = st.ships, st.bullets

    def noise(*shape, scale):
        return scale * (2.0 * torch.rand(*shape, generator=g) - 1.0)

    near = torch.randint(0, N, (B, N, K), generator=g)
    sx, sy = ships.x.gather(1, near.reshape(B, -1)), ships.y.gather(1, near.reshape(B, -1))
    sweep = match.Sweep(x=ships.x, y=ships.y, dx=noise(B, N, scale=6.0), dy=noise(B, N, scale=6.0),
                        bx=sx.reshape(B, N, K) + noise(B, N, K, scale=15.0),
                        by=sy.reshape(B, N, K) + noise(B, N, K, scale=15.0),
                        bdx=noise(B, N, K, scale=20.0), bdy=noise(B, N, K, scale=20.0))
    ang = noise(B, N, K, scale=math.pi)
    bullets = match.replace(bullets, ttl=torch.randint(0, 3, (B, N, K), generator=g).int(),
                            dx=ang.cos(), dy=ang.sin())
    ships = match.replace(ships, alive=torch.rand(B, N, generator=g) < 0.85)
    card = ship_types()
    got = hits(p, card, sweep, ships, bullets, ships.alive)

    radius = card.hit_radius[ships.ship_type.long()].double()
    full = card.damage[ships.ship_type.long()].double()
    damage = torch.zeros(B, N, N, dtype=torch.float64)
    first = torch.full((B, N), -1)
    best = torch.full((B, N), math.inf)
    n_hits = n_ally = 0
    for b in range(B):
        for o in range(N):
            for k in range(K):
                if bullets.ttl[b, o, k] <= 0:
                    continue
                hit_tau, target = math.inf, -1
                for t in range(N):
                    if t == o or not ships.alive[b, t]:
                        continue
                    d0 = torch.stack([sweep.bx[b, o, k] - sweep.x[b, t],
                                      sweep.by[b, o, k] - sweep.y[b, t]]).double()
                    dl = torch.stack([sweep.bdx[b, o, k] - sweep.dx[b, t],
                                      sweep.bdy[b, o, k] - sweep.dy[b, t]]).double()
                    tau = float((-(d0 @ dl) / (dl @ dl).clamp_min(1e-12)).clamp(0.0, 1.0))
                    miss = d0 + tau * dl
                    if float(miss @ miss) < float(radius[b, t]) ** 2 and tau < hit_tau:
                        hit_tau, target = tau, t
                assert bool(got.bullet_hit[b, o, k]) == (target >= 0), (b, o, k)
                if target < 0:
                    continue
                n_hits += 1
                n_ally += (o // p.ships_per_team) == (target // p.ships_per_team)
                dx, dy = float(bullets.dx[b, o, k]), float(bullets.dy[b, o, k])
                nc, ns = float(ships.c[b, target]), float(ships.s[b, target])
                theta = math.atan2(abs(dx * ns - dy * nc), -(dx * nc + dy * ns))
                damage[b, o, target] += float(full[b, o]) * (
                    1.0 - (1.0 - p.head_on_fraction) * math.exp(-4.0 * theta * theta / math.pi))
                if hit_tau < best[b, target]:  # strict: ties keep the lower bullet index
                    best[b, target], first[b, target] = hit_tau, o
    assert n_hits > 20 and n_ally > 0, (n_hits, n_ally)
    assert torch.equal(got.first_shooter.long(), first)
    torch.testing.assert_close(got.damage.double(), damage, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(got.taken, got.damage.sum(1))


def test_formation_ranks_ties_to_the_lower_slot():
    from bnb.game.physics.spawning import formation_rank

    rules = Rules(ships_per_team=4, num_fields=1, arena_radius=1000.0)
    keys = torch.tensor([[0.5, 0.5, 0.1, 0.9, 0.3, 0.3, 0.3, 0.0]])
    assert formation_rank(rules, keys).tolist() == [[1, 2, 0, 3, 1, 2, 3, 0]]


def test_canonical_names_cover_the_state():
    p, st, _ = _warm(B=1, ticks=1)
    leaves = [x for x in torch.utils._pytree.tree_leaves(st)]
    assert len(CANONICAL) == len(leaves)
    assert st["bx"] is st.bullets.x and st["w"] is st.ships.w


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_eager_cuda_graph_replays_draw_fresh_randomness():
    """The game's draws and its tick capture into a CUDA graph eagerly, and every replay
    draws anew."""
    p = match.rules_of(GameConfig(2, 3, 1200.0))
    dev = torch.device("cuda")
    st = match.reset(p, match.uniforms(p, 16, dev))
    act = Action.unpack(torch.ones(16, p.num_ships, 3, dtype=torch.int32, device=dev))
    spread, uni = match.spread(p, 16, dev), match.uniforms(p, 16, dev)
    card = ship_types(dev)  # the default card is built from host values: pass it in

    def tick():
        spread.copy_(match.spread(p, 16, dev))
        uni.copy_(match.uniforms(p, 16, dev))
        new, _ = match.step(p, st, act, spread, uni, card)
        torch.utils._pytree.tree_map(lambda d, s: d.copy_(s), st, new)

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
        seen.append((spread.clone(), uni.clone()))
    for (s0, u0), (s1, u1) in zip(seen, seen[1:]):
        assert not torch.equal(s0, s1) and not torch.equal(u0, u1)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compiled_rollout_graph_replays_draw_fresh_randomness():
    from bnb.contracts import Pool
    from bnb.controllers.scripted import Random
    from bnb.runtime.rollout import Rollout

    ro = Rollout(GameConfig(2, 3, 1200.0), (Pool("p", 1.0, ("r", "r")),), {"r": Random(2)}, "r",
                 16, 8, torch.device("cuda"), compile=True, graph_ticks=4)
    draws = []
    for _ in range(3):
        ro.run()
        draws.append((ro.carry["spread"].clone(), ro.carry["uniforms"].clone()))
    assert ro.graph is not None
    for (s0, u0), (s1, u1) in zip(draws, draws[1:]):
        assert not torch.equal(s0, s1) and not torch.equal(u0, u1)
    spread = torch.stack([s for s, _ in draws])
    uniforms = torch.stack([u for _, u in draws])
    assert abs(float(spread.mean())) < 0.2 and abs(float(spread.std()) - 1.0) < 0.2
    assert 0.0 <= float(uniforms.min()) and float(uniforms.max()) < 1.0
    assert abs(float(uniforms.mean()) - 0.5) < 0.1
