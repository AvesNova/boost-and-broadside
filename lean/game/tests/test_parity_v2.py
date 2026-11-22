"""Further comparisons against v2 (game-audit.md section 5), beyond test_against_v2.py:
fields, games that end (wipes, the boundary, the clock), the clock's health rule, unequal
and large team sizes, the spawn rules' distributions, and every number of the card.

Each test names the assumption it guards. The known differences (docs/rewrite/game-parity.md)
are tests marked xfail(strict=True): they fail today because lean differs from v2, and the
mark turns into an error the day lean matches v2, so the list is kept honest.

Needs v2 on the path (PYTHONPATH=src); skipped otherwise. CPU, one thread.
"""

import dataclasses
import math

import pytest
import torch

api = pytest.importorskip("bnb.game.api")
from bnb.contracts import Action  # noqa: E402
from bnb.game.layout import boundary_fraction, environment  # noqa: E402
from bnb.game.physics import combat  # noqa: E402
from bnb.game.physics.medium import medium as v2_medium  # noqa: E402
from lean.contract import BULLET_SLOTS, State  # noqa: E402
from lean.game import GameConfig, card, reset, step  # noqa: E402
from lean.game.core import ship_constants  # noqa: E402

torch.set_num_threads(1)
replace = dataclasses.replace

MAP_SCALE_RADIUS = 2600.0  # v2's arena radius of map scale 1 (bnb.game.rules)
# (99.9% bound, bound on every sample), section 5.5 check A; bullets from the same table.
STRICT = {
    "x": (1e-3, 2e-3), "y": (1e-3, 2e-3), "heading_cos": (5e-7, 1e-4), "heading_sin": (5e-7, 1e-4),
    "angle_cos": (5e-7, 1e-4), "angle_sin": (5e-7, 1e-4), "speed": (1e-4, 2e-3), "angle": (3e-7, 5e-4),
    "nose_rate": (5e-6, 5e-3), "energy": (0.06, 0.1), "shield": (5e-3, 2e-2), "shield_wait": (1e-6, 1e-6),
}
BULLET_STRICT = {"bullet_x": (1e-3, 2e-3), "bullet_y": (1e-3, 2e-3), "bullet_speed": (1e-3, 1e-2)}
OPEN_LOOP = {30: dict(position=0.05, speed=0.01, energy=5.0, shield=0.1),
             150: dict(position=0.25, speed=0.05, energy=30.0, shield=0.3)}
V2_SHIP = {"speed": "proper_speed", "angle": "angle_of_attack", "angle_cos": "angle_of_attack_cos",
           "angle_sin": "angle_of_attack_sin", "shield_wait": "shield_recharge_wait"}
V2_BULLET = {"bullet_x": "x", "bullet_y": "y", "bullet_speed": "proper_speed",
             "bullet_dir_x": "direction_x", "bullet_dir_y": "direction_y"}
# A fixed map near the centre, where the pulled-in ships fly: overlapping fields (the union
# composition), every level, two band widths, and one field far from the action.
FIELDS = (
    (0.0, 0.0, 150.0, 160.0, 2.0),
    (250.0, 100.0, 200.0, 160.0, -1.0),
    (-200.0, -150.0, 120.0, 80.0, -2.0),
    (100.0, -300.0, 250.0, 160.0, 1.0),
    (2500.0, 2500.0, 400.0, 160.0, 2.0),
)


@dataclasses.dataclass
class Game:
    """A v2 game and the lean game that mirrors it. slots: the v2 slot of each lean slot."""

    rules: object
    types: object
    state: object
    generator: torch.Generator
    active: torch.Tensor | None
    slots: torch.Tensor
    owner_of: torch.Tensor  # lean slot of each v2 slot, -1 for a benched one
    config: GameConfig
    team: torch.Tensor


def v2_game(per_team, envs, seed, *, fields=(), arena=7800.0, clock_ticks=5400, active=None, pull=0.03):
    """v2's real_scale_elimination at this size, map and clock, the shrink off for the whole
    clock, ships pulled toward the centre by `pull`; active (3,) gives every environment those
    team sizes (v2 benches the rest)."""
    game = replace(api.PRESETS["real_scale_elimination"], ships_per_team=per_team, num_fields=len(fields),
                   arena_radius=arena, episode_seconds=clock_ticks / 30.0)
    types = api.ship_types(ship_set=game.ship_set)
    rules = api.rules_of(game, types=types)
    pool = rules.pools[0]
    # The shrink's end set 10 s after the clock: the boundary holds at R to the last tick.
    mode = replace(pool.mode, shrink_hold_seconds=1e9, shrink_end_lead_seconds=-10.0)
    rules = replace(rules, pools=(replace(pool, mode=mode),))
    ticks = torch.arange(clock_ticks + 1)
    assert (boundary_fraction(environment(rules, clock_ticks + 1), ticks) == 1.0).all()
    generator = torch.Generator().manual_seed(seed)
    active_tensor = None if active is None else torch.tensor([active] * envs, dtype=torch.int32)
    state = api.reset(rules, types, api.uniforms(rules, envs, generator=generator),
                      fogged=torch.zeros(envs, dtype=torch.bool), active=active_tensor)
    ships = replace(state.ships, x=state.ships.x * pull, y=state.ships.y * pull)
    if fields:
        table = torch.tensor(fields, dtype=torch.float32)
        field_state = replace(
            state.fields,
            x=table[:, 0].expand(envs, -1).clone(), y=table[:, 1].expand(envs, -1).clone(),
            radius=table[:, 2].expand(envs, -1).clone(), band_width=table[:, 3].expand(envs, -1).clone(),
            level=table[:, 4].to(state.fields.level.dtype).expand(envs, -1).clone(),
            present=torch.ones((envs, len(fields)), dtype=torch.bool),
        )
        state = replace(state, fields=field_state)
    log_index, gradient_x, gradient_y = v2_medium(state.fields, ships.x, ships.y)
    ships = replace(ships, log_refractive_index=log_index, log_index_gradient_x=gradient_x,
                    log_index_gradient_y=gradient_y)
    state = replace(state, ships=ships)
    sizes = (per_team, per_team) if active is None else active
    slots = torch.cat([torch.arange(sizes[0]), per_team + torch.arange(sizes[1])])
    owner_of = torch.full((2 * per_team,), -1, dtype=torch.long)
    owner_of[slots] = torch.arange(len(slots))
    team = (torch.arange(len(slots)) >= sizes[0]).long().expand(envs, -1)
    config = GameConfig(arena_radius=arena, boundary_ramp_width=rules.boundary_ramp_width * arena / MAP_SCALE_RADIUS,
                        clock_ticks=clock_ticks, fields=tuple(fields))
    return Game(rules, types, state, generator, active_tensor, slots, owner_of, config, team)


def to_lean(game: Game, v2_state) -> State:
    """The lean state of a v2 state (lean slots in `game.slots` order)."""
    ships, bullets, slots = v2_state.ships, v2_state.bullets, game.slots
    ship_type = ships.ship_type.long()[:, slots]
    values = {name: getattr(ships, V2_SHIP.get(name, name))[:, slots].clone()
              for name in ("x", "y", "heading_cos", "heading_sin", "speed", "angle", "angle_cos", "angle_sin",
                           "nose_rate", "energy", "shield", "shield_wait")}
    bullet_values = {name: getattr(bullets, V2_BULLET[name])[:, slots].clone()
                     for name in ("bullet_x", "bullet_y", "bullet_dir_x", "bullet_dir_y", "bullet_speed")}
    alive = ships.alive[:, slots].clone()
    return State(
        **values, **bullet_values,
        cooldown=ships.gun_cooldown.long()[:, slots], next_slot=ships.next_slot.long()[:, slots], alive=alive,
        exists=torch.ones_like(alive), team=game.team, type=ship_type,
        constants=ship_constants(game.config, ship_type),
        bullet_moves_left=bullets.moves_left.long()[:, slots], tick=v2_state.match.episode_tick.long(),
    )


def lean_result(v2_result, done):
    """v2's result (+1 team 0 wins, -1 team 1, 0 draw) in the lean coding (0, 1, 2; -1 not done)."""
    coded = torch.where(v2_result > 0, 0, torch.where(v2_result < 0, 1, 2))
    return torch.where(done, coded, -1)


def run(game: Game, ticks: int, *, shoot_share=0.5, actions=None):
    """Check A (one lean step from each v2 state) and check B (an open-loop lean rollout on
    v2's actions and spread) for `ticks` ticks, or until every environment has ended once.
    actions(tick) -> (E, N_v2, 3) long overrides the random sticky actions. Returns a report."""
    E, N2 = game.state.ships.x.shape
    slots, state = game.slots, game.state
    stall = card.STALL_ANGLE * torch.tensor(card.CARD["stall_angle_scale"])
    open_loop = to_lean(game, state)
    ended = torch.zeros(E, dtype=torch.bool)  # v2 ended its first episode (stop comparing)
    agree = torch.ones(E, dtype=torch.bool)  # open loop: no discrete difference yet
    v2_end = torch.full((E,), -1, dtype=torch.long)
    v2_end_result = torch.full((E,), -1, dtype=torch.long)
    lean_end = torch.full((E,), -1, dtype=torch.long)
    lean_end_result = torch.full((E,), -1, dtype=torch.long)
    errors = {name: [] for name in list(STRICT) + list(BULLET_STRICT)}
    post_stall = {"samples": 0, "angle": 0.0, "nose_rate": 0.0}
    mismatches = {name: 0 for name in ("alive", "died", "fired", "done", "result", "killer", "moves_left",
                                       "cooldown", "next_slot")}
    killer_ties = 0  # killer differs only by the known tie rule (equal tau: v2 the closer pass)
    stalled_out = torch.zeros(E, dtype=torch.bool)  # open loop: dropped once a ship is post-stall
    counts = dict(ship_ticks=0, ship_ticks_in_fields=0, bullet_moves=0, bullet_moves_in_fields=0,
                  shots=0, deaths=0, boundary_hurt_ticks=0, wipes=0, clock_ends=0)
    drift, held = {}, None
    for tick in range(ticks):
        drawn = torch.stack([torch.randint(0, n, (E, N2), generator=game.generator) for n in (7, 3)]
                            + [(torch.rand((E, N2), generator=game.generator) < shoot_share).long()], -1)
        keep = torch.rand((E, N2, 1), generator=game.generator) < 0.9
        held = drawn if held is None else torch.where(keep, held, drawn)
        action = held if actions is None else actions(tick)
        spread = torch.randn((E, N2, 2), generator=game.generator)
        v2_action = Action(power=action[..., 1].int(), stick=action[..., 0].int(), shoot=action[..., 2].int())
        before = to_lean(game, state)
        alive0 = before.alive & ~ended[:, None]
        log_index0, _, _ = v2_medium(state.fields, state.ships.x, state.ships.y)
        alive_v2 = state.ships.alive
        moved_v2, sweep = api.move(game.rules, game.types, state, v2_action, spread)
        pairs = combat.contacts(game.types, sweep, moved_v2.ships.ship_type, moved_v2.bullets.moves_left > 0, alive_v2)
        _, target, hit = combat.first_contact(pairs)
        bullet_tau = pairs.tau.gather(2, target[..., None])[..., 0]
        state, events = api.resolve(game.rules, game.types, moved_v2, sweep, v2_action,
                                    api.uniforms(game.rules, E, generator=game.generator), game.active)
        lean_action, lean_spread = action[:, slots], spread[:, slots]
        after, out = step(game.config, before, lean_action, {"spread": lean_spread})
        live = ~ended & ~events.done  # v2 returns the next episode's state where done
        died = events.died[:, slots]
        # Continuous: alive at tick start, not dying this tick (v2 writes the speed floor and a
        # zero nose rate on a death; lean keeps them), post-stall rows reported apart.
        rows = alive0 & live[:, None] & ~died
        stalled = before.angle.abs() > stall[before.type]
        for name in STRICT:
            difference = (getattr(after, name) - getattr(state.ships, V2_SHIP.get(name, name))[:, slots]).abs()
            errors[name].append(difference[rows & ~stalled])
            if name in post_stall and (rows & stalled).any():
                post_stall[name] = max(post_stall[name], float(difference[rows & stalled].max()))
        post_stall["samples"] += int((rows & stalled).sum())
        slot = torch.arange(BULLET_SLOTS)
        written = out.fired[..., None] & (slot == before.next_slot[..., None])
        moved = ((before.bullet_moves_left > 0) | written) & live[:, None, None]
        for name in BULLET_STRICT:
            reference = getattr(state.bullets, V2_BULLET[name])[:, slots]
            errors[name].append((getattr(after, name) - reference).abs()[moved])
        # Discrete, on every environment still in its first episode.
        now = ~ended
        killer = torch.where(events.killer >= 0, game.owner_of[events.killer.long().clamp_min(0)], -1)[:, slots]
        mismatches["died"] += int((out.died != died)[now].sum())
        mismatches["fired"] += int((out.fired != events.fired[:, slots])[now].sum())
        mismatches["done"] += int((out.done != events.done)[now].sum())
        mismatches["result"] += int((out.result != lean_result(events.result, events.done))[now].sum())
        for e, n in ((out.killer != killer) & now[:, None]).nonzero().tolist():
            # A tie: both killers own a bullet that hit this ship at its earliest tau.
            on_ship = hit[e] & (target[e] == slots[n])
            earliest = bullet_tau[e][on_ship].min()
            owners = (torch.nonzero(on_ship & (bullet_tau[e] == earliest))[:, 0] // BULLET_SLOTS).tolist()
            if int(slots[out.killer[e, n]]) in owners and int(slots[killer[e, n]]) in owners:
                killer_ties += 1
            else:
                mismatches["killer"] += 1
        mismatches["alive"] += int((after.alive != state.ships.alive[:, slots])[live].sum())
        mismatches["moves_left"] += int((after.bullet_moves_left != state.bullets.moves_left[:, slots])[live].sum())
        mismatches["cooldown"] += int((after.cooldown != state.ships.gun_cooldown[:, slots])[live].sum())
        mismatches["next_slot"] += int((after.next_slot != state.ships.next_slot[:, slots])[live].sum())
        # What the run exercised.
        counts["ship_ticks"] += int(alive0.sum())
        counts["ship_ticks_in_fields"] += int((alive0 & (log_index0[:, slots] != 0.0)).sum())
        bullet_log_index, _, _ = v2_medium(state.fields, state.bullets.x.flatten(1), state.bullets.y.flatten(1))
        in_field = (bullet_log_index.reshape(state.bullets.x.shape)[:, slots] != 0.0) & moved
        counts["bullet_moves"] += int(moved.sum())
        counts["bullet_moves_in_fields"] += int(in_field.sum())
        counts["shots"] += int(out.fired[now].sum())
        counts["deaths"] += int(died[now].sum())
        counts["boundary_hurt_ticks"] += int((events.boundary_damage[:, slots] > 0)[now].sum())
        finishing = now & events.done
        survivors = (alive0 & ~died)
        team_one = game.team == 1
        wiped = ((survivors & ~team_one).sum(-1) == 0) | ((survivors & team_one).sum(-1) == 0)
        counts["wipes"] += int((finishing & wiped).sum())
        counts["clock_ends"] += int((finishing & ~wiped).sum())
        v2_end = torch.where(finishing, tick + 1, v2_end)
        v2_end_result = torch.where(finishing, lean_result(events.result, events.done), v2_end_result)
        # Check B.
        open_loop, open_out = step(game.config, open_loop, lean_action, {"spread": lean_spread})
        first_lean_end = (lean_end < 0) & open_out.done
        lean_end = torch.where(first_lean_end, tick + 1, lean_end)
        lean_end_result = torch.where(first_lean_end, open_out.result, lean_end_result)
        agree &= ended | ((open_out.died == died).all(-1) & (open_out.fired == events.fired[:, slots]).all(-1))
        # The known post-stall difference (test_post_stall_matches_v2) grows in the open loop.
        stalled_out |= ~ended & (open_loop.alive & (open_loop.angle.abs() > stall[open_loop.type])).any(-1)
        ended |= events.done
        valid = agree & ~ended & ~stalled_out
        if tick + 1 in OPEN_LOOP and valid.any():
            rows = valid[:, None] & open_loop.alive
            ships = state.ships
            drift[tick + 1] = {
                "position": max(float((open_loop.x - ships.x[:, slots]).abs()[rows].max()),
                                float((open_loop.y - ships.y[:, slots]).abs()[rows].max())),
                "speed": float((open_loop.speed - ships.proper_speed[:, slots]).abs()[rows].max()),
                "energy": float((open_loop.energy - ships.energy[:, slots]).abs()[rows].max()),
                "shield": float((open_loop.shield - ships.shield[:, slots]).abs()[rows].max()),
                "kept": int(valid.sum()),
            }
        if bool(ended.all()):
            break
    report = {"ticks": tick + 1, "environments": E, "lean_slots": len(slots), "mismatches": mismatches,
              "killer_ties": killer_ties, "open_loop_dropped_post_stall": int(stalled_out.sum()),
              "counts": counts, "post_stall": post_stall, "drift": drift}
    report["errors"] = {}
    for name, parts in errors.items():
        values = torch.cat(parts)
        report["errors"][name] = (float(values.quantile(0.999)), float(values.max()), values.numel()) \
            if values.numel() else (0.0, 0.0, 0)
    compared = (v2_end > 0) & agree
    report["ends"] = {
        "v2_ended": int((v2_end > 0).sum()),
        "compared": int(compared.sum()),
        "same_tick": int((compared & (lean_end == v2_end)).sum()),
        "same_result": int((compared & (lean_end_result == v2_end_result)).sum()),
        "v2_results": torch.bincount(v2_end_result[v2_end > 0], minlength=3).tolist(),
        "end_ticks": sorted(v2_end[v2_end > 0].tolist()),
    }
    print(report)
    return report


def assert_parity(report, bullets=True):
    """The section 5.5 gates: check A's bounds, no discrete difference, check B's bounds."""
    for name, (most, every) in list(STRICT.items()) + list(BULLET_STRICT.items() if bullets else ()):
        quantile, worst, samples = report["errors"][name]
        assert samples > 0, name
        assert quantile <= most and worst <= every, (name, quantile, worst)
    assert all(count == 0 for count in report["mismatches"].values()), report["mismatches"]
    for at, limits in OPEN_LOOP.items():
        for name, limit in limits.items():
            if at in report["drift"]:
                assert report["drift"][at][name] <= limit, (at, name, report["drift"][at][name])
    ends = report["ends"]
    assert ends["same_tick"] == ends["compared"] and ends["same_result"] == ends["compared"], ends


# --------------------------------------------------------------------------- fields and sizes


def test_fields_match_v2():
    """Guards the medium in flight and bullets (bending, slowing, the union of overlapping
    fields, per-field bands): 5v5 on a fixed map of five fields, one step and open loop."""
    report = run(v2_game(5, 32, seed=11, fields=FIELDS), 150)
    assert_parity(report)
    counts = report["counts"]
    assert counts["ship_ticks_in_fields"] > 0.2 * counts["ship_ticks"]
    assert counts["bullet_moves_in_fields"] > 0.1 * counts["bullet_moves"]


def test_unequal_teams_match_v2():
    """Guards slot layout and the win count with unequal teams: 3v5 with fields (v2 benches
    team 0's last two slots; lean has 8 slots, team-contiguous)."""
    report = run(v2_game(5, 16, seed=12, fields=FIELDS, active=(3, 5)), 150)
    assert_parity(report)
    assert report["lean_slots"] == 8


def test_large_teams_match_v2():
    """Guards the hit test and first contact in a crowd: 20v20 with fields."""
    report = run(v2_game(20, 4, seed=13, fields=FIELDS), 60)
    assert_parity(report)
    assert report["counts"]["deaths"] > 0


# --------------------------------------------------------------------------- games that end


def _low_shields(game: Game, shield: float) -> Game:
    ships = game.state.ships
    game.state = replace(game.state, ships=replace(ships, shield=torch.full_like(ships.shield, shield)))
    return game


def test_wipes_end_games_as_v2():
    """Guards the win check and the result coding: low shields and heavy fire, so teams are
    wiped; done, the result and the tick must agree (and draws by a simultaneous wipe)."""
    game = _low_shields(v2_game(5, 48, seed=21, fields=FIELDS), 4.0)
    report = run(game, 400, shoot_share=0.8)
    assert_parity(report)
    assert report["counts"]["wipes"] >= 24, report["counts"]
    assert report["ends"]["compared"] >= 24


def test_boundary_deaths_end_games_as_v2():
    """Guards the boundary (radius, ramp scaled as v2 scales it, damage) and deaths by it:
    a 1,000 m arena (ramp 38.5 m), ships spawned at its edge, low shields."""
    game = _low_shields(v2_game(3, 32, seed=22, arena=1000.0, pull=1.15), 3.0)
    report = run(game, 300, shoot_share=0.2)
    assert_parity(report)
    assert report["counts"]["boundary_hurt_ticks"] > 100
    assert report["ends"]["compared"] >= 10


def test_simultaneous_wipe_is_a_draw_as_in_v2():
    """Guards the draw coding of a wipe: 3v3 in a 1,000 m arena, every ship 100 m beyond the
    edge with its shield broken, so both teams die on the first tick (v2's result 0, lean's 2)."""
    game = v2_game(3, 8, seed=27, arena=1000.0)
    ships = game.state.ships
    angle = torch.rand(ships.x.shape, generator=game.generator) * 2.0 * math.pi
    game.state = replace(game.state, ships=replace(
        ships, x=1100.0 * torch.cos(angle), y=1100.0 * torch.sin(angle), shield=torch.zeros_like(ships.shield)))
    report = run(game, 1, shoot_share=0.0)  # every ship dies: no continuous rows to compare
    assert all(count == 0 for count in report["mismatches"].values()), report["mismatches"]
    assert report["ends"]["v2_results"] == [0, 0, 8] and report["counts"]["deaths"] == 48
    assert report["ends"]["same_result"] == 8 and report["ends"]["same_tick"] == 8


def test_clock_ends_games_as_v2():
    """Guards the clock and its health rule: a 90-tick clock, ships far apart with shields
    drawn between a fifth and all of their full value; every game ends on the clock."""
    game = v2_game(5, 64, seed=23, clock_ticks=90, pull=1.0)
    ships = game.state.ships
    fraction = 0.2 + 0.8 * torch.rand(ships.shield.shape, generator=game.generator)
    game.state = replace(game.state, ships=replace(ships, shield=ships.shield * fraction))
    report = run(game, 90, shoot_share=0.0)
    assert_parity(report, bullets=False)
    assert report["counts"]["clock_ends"] == 64
    assert report["ends"]["end_ticks"] == [90] * 64


def _quiet_actions(envs, slots, quiet_team, per_team):
    """Team `quiet_team` coasts straight without firing (the shield's quiet input); the other
    boosts (not quiet)."""
    action = torch.zeros((envs, slots, 3), dtype=torch.long)
    loud = torch.ones(slots, dtype=torch.bool)
    loud[quiet_team * per_team:(quiet_team + 1) * per_team] = False
    action[:, loud, 1] = 1
    return lambda tick: action


@pytest.mark.xfail(strict=True, reason="lean's clock rule reads the shield before the last tick's "
                   "recharge; v2's after it (game-parity.md)")
def test_clock_counts_the_last_ticks_recharge():
    """Guards the tie-break's timing: 1v1 with a one-tick clock, both shields at half, team 0
    recharging on that tick (quiet, wait over), team 1 boosting. Health is equal before the
    recharge, so v2 (after it) gives team 0 the win; lean (before it) a draw."""
    game = v2_game(1, 4, seed=24, clock_ticks=1, pull=1.0)
    ships = game.state.ships
    game.state = replace(game.state, ships=replace(ships, shield=ships.shield * 0.5,
                                                   shield_recharge_wait=torch.zeros_like(ships.shield)))
    report = run(game, 1, actions=_quiet_actions(4, 2, 0, 1))
    assert report["ends"]["v2_results"] == [4, 0, 0]
    assert report["mismatches"]["result"] == 0, report["mismatches"]


@pytest.mark.xfail(strict=True, reason="lean sums the signed health of both teams in one sum, which "
                   "leaves a rounding residue on an exact tie; v2 compares the two team sums (game-parity.md)")
def test_clock_mirrored_health_is_a_draw():
    """Guards the tie at the clock: 5v5, each team-1 ship with its team-0 twin's shield
    fraction, nobody firing or recharging; v2 calls it a draw, and so must lean (its signed
    sum must not leave a rounding residue)."""
    game = v2_game(5, 64, seed=25, clock_ticks=30, pull=1.0)
    ships = game.state.ships
    fraction = 0.2 + 0.8 * torch.rand((64, 5), generator=game.generator)
    fraction = torch.cat([fraction, fraction], 1)
    game.state = replace(game.state, ships=replace(ships, shield=ships.shield * fraction))
    report = run(game, 30, actions=lambda tick: torch.tensor([3, 1, 0]).expand(64, 10, 3))
    assert report["ends"]["v2_results"] == [0, 0, 64], report["ends"]
    assert report["mismatches"]["result"] == 0, report["mismatches"]


# --------------------------------------------------------------------------- flight edge


@pytest.mark.xfail(strict=True, reason="lean clamps the lift slope at 0 past the stall angle; v2 keeps "
                   "it negative up to twice the stall angle (game-parity.md)")
def test_post_stall_matches_v2():
    """Guards the nose oscillator past the stall: angles of attack spread over -1.9 to 1.9
    stall angles, one step, coasting straight; check A's bounds on angle and nose rate."""
    game = v2_game(5, 32, seed=26)
    E, N = game.state.ships.x.shape
    stall = game.rules.flight.stall_angle * 1.0333333015441895
    angle = torch.linspace(-1.9, 1.9, E * N).reshape(E, N) * stall
    ships = replace(game.state.ships, angle_of_attack=angle, angle_of_attack_cos=torch.cos(angle),
                    angle_of_attack_sin=torch.sin(angle))
    before = to_lean(game, replace(game.state, ships=ships))
    action = torch.zeros((E, N, 3), dtype=torch.long)
    v2_state, _ = api.step(game.rules, game.types, replace(game.state, ships=ships),
                           Action(power=action[..., 1].int(), stick=action[..., 0].int(), shoot=action[..., 2].int()),
                           torch.zeros(E, N, 2), api.uniforms(game.rules, E, generator=game.generator))
    after, _ = step(game.config, before, action, {"spread": torch.zeros(E, N, 2)})
    angle_error = (after.angle - v2_state.ships.angle_of_attack).abs().max()
    rate_error = (after.nose_rate - v2_state.ships.nose_rate).abs().max()
    print("post-stall max error: angle", float(angle_error), "nose rate", float(rate_error))
    assert angle_error <= STRICT["angle"][1] and rate_error <= STRICT["nose_rate"][1]


# --------------------------------------------------------------------------- spawns


def _ks(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """(two-sample Kolmogorov-Smirnov statistic, its critical value at alpha = 0.001)."""
    a, b = a.double().sort().values, b.double().sort().values
    points = torch.cat([a, b])
    cdf_a = torch.searchsorted(a, points, right=True) / a.numel()
    cdf_b = torch.searchsorted(b, points, right=True) / b.numel()
    n, m = a.numel(), b.numel()
    return float((cdf_a - cdf_b).abs().max()), math.sqrt(-math.log(0.001 / 2) / 2) * math.sqrt((n + m) / (n * m))


def _spawn_statistics(x, y, heading_cos, heading_sin, team):
    """Rotation-invariant draws, independent across environments: each ship's distance from
    the centre and its heading's angle off the outward normal; per environment the distance
    from slot 0 to its first team-mate and to its first enemy, and the cosine between their
    headings."""
    distance = torch.sqrt(x * x + y * y)
    normal = torch.atan2(y, x)
    off = torch.remainder(torch.atan2(heading_sin, heading_cos) - normal + math.pi, 2 * math.pi) - math.pi
    enemy = (team == 1).long().argmax(-1)
    ex, ey = x.gather(1, enemy[:, None])[:, 0], y.gather(1, enemy[:, None])[:, 0]
    ecos, esin = heading_cos.gather(1, enemy[:, None])[:, 0], heading_sin.gather(1, enemy[:, None])[:, 0]
    return {
        "distance": distance.flatten(),
        "off_normal": off.abs().flatten(),
        "mate": torch.hypot(x[:, 0] - x[:, 1], y[:, 0] - y[:, 1]),
        "enemy": torch.hypot(x[:, 0] - ex, y[:, 0] - ey),
        "facing": heading_cos[:, 0] * ecos + heading_sin[:, 0] * esin,
    }


def _v2_spawns(preset, envs, arena, seed):
    game = replace(api.PRESETS[preset], num_fields=0, arena_radius=arena)
    types = api.ship_types(ship_set=game.ship_set)
    rules = api.rules_of(game, types=types)
    generator = torch.Generator().manual_seed(seed)
    state = api.reset(rules, types, api.uniforms(rules, envs, generator=generator),
                      fogged=torch.zeros(envs, dtype=torch.bool))
    ships = state.ships
    team = (torch.arange(ships.x.shape[1]) >= rules.ships_per_team).long().expand(envs, -1)
    return ships, team


def test_sides_spawn_matches_v2_in_distribution():
    """Guards lean's "sides" spawn against v2's elimination spawn (side_placement and the edge
    rule) at 4,200 m, 5v5, 4,096 environments each: rotation-invariant statistics by a
    two-sample KS test at alpha 0.001, and the spawn state exactly."""
    envs, arena = 4096, 4200.0
    v2_ships, v2_team = _v2_spawns("real_scale_elimination", envs, arena, seed=31)
    config = GameConfig(arena_radius=arena, spawn="sides")
    lean = reset(config, envs, torch.tensor([(5, 5)] * envs), torch.Generator().manual_seed(32), "cpu")
    ours = _spawn_statistics(lean.x, lean.y, lean.heading_cos, lean.heading_sin, lean.team)
    theirs = _spawn_statistics(v2_ships.x, v2_ships.y, v2_ships.heading_cos, v2_ships.heading_sin, v2_team)
    results = {name: _ks(ours[name], theirs[name]) for name in ours}
    print("sides spawn KS (statistic, critical):", results)
    for name, (statistic, critical) in results.items():
        assert statistic < critical, (name, statistic, critical)
    assert (lean.type == v2_ships.ship_type.long()).all()
    assert torch.equal(lean.speed, v2_ships.proper_speed)
    assert torch.equal(lean.energy, v2_ships.energy) and torch.equal(lean.shield, v2_ships.shield)
    assert (lean.angle == 0).all() and (lean.nose_rate == 0).all() and (lean.shield_wait == 0).all()


@pytest.mark.xfail(strict=True, reason="lean's \"mixed\" spawn (uniform in 0.85 R, uniform heading) is "
                   "not v2's team deathmatch spawn (best of 8 in R, farthest from the enemy) (game-parity.md)")
def test_mixed_spawn_matches_v2_deathmatch_in_distribution():
    """Records how far lean's "mixed" spawn is from v2's nearest rule, team deathmatch's
    uniform spawn at a reset (v2's elimination has no mixed spawn), at 4,200 m, 5v5."""
    envs, arena = 4096, 4200.0
    v2_ships, v2_team = _v2_spawns("real_scale_deathmatch", envs, arena, seed=33)
    config = GameConfig(arena_radius=arena, spawn="mixed")
    lean = reset(config, envs, torch.tensor([(5, 5)] * envs), torch.Generator().manual_seed(34), "cpu")
    ours = _spawn_statistics(lean.x, lean.y, lean.heading_cos, lean.heading_sin, lean.team)
    theirs = _spawn_statistics(v2_ships.x, v2_ships.y, v2_ships.heading_cos, v2_ships.heading_sin, v2_team)
    results = {name: _ks(ours[name], theirs[name]) for name in ours}
    print("mixed spawn KS against v2 deathmatch (statistic, critical):", results)
    for name, (statistic, critical) in results.items():
        assert statistic < critical, (name, statistic, critical)


# --------------------------------------------------------------------------- the card


def test_card_and_rules_match_v2():
    """Guards every number lean copied: each card entry against v2's real-scale card (float32,
    exactly), each shared constant against v2's rules, and the card entries lean leaves out at
    the values that make them inert."""
    game = api.PRESETS["real_scale_elimination"]
    types = api.ship_types(ship_set=game.ship_set)
    rules = api.rules_of(game, types=types)
    mode = rules.pools[0].mode
    assert tuple(name.lower() for name in api.type_names(game.ship_set)) == card.TYPE_NAMES
    for name, values in card.CARD.items():
        assert torch.equal(torch.tensor(values, dtype=torch.float32), getattr(types, name)), name
    # Left out of lean: inert at these values (bleed 2 is the plain square; arming 0 is off;
    # the coast exchange is unused with direct charge); vision belongs to fog.
    assert (types.bleed_exponent == 2.0).all() and (types.arming_distance == 0.0).all()
    assert rules.flight.direct_charge and rules.flight.vertical_exchange and rules.flight.mass == 1.0
    flight = rules.flight
    shared = [  # (lean, v2) pairs: a list, as several values are equal
        (card.GRAVITY, flight.gravity), (card.STALL_ANGLE, flight.stall_angle),
        (card.INDUCED_DRAG_FACTOR, flight.induced_drag_factor), (card.ENGINE_KNEE, flight.engine_knee),
        (card.STORE_FADE_BAND, flight.store_fade_band), (card.CLIMB_EXCHANGE, flight.climb_exchange),
        (card.DIVE_EXCHANGE, flight.dive_exchange), (card.LIFT_SOFTMIN_EXPONENT, flight.lift_softmin_exponent),
        (card.THRUST_SOFTMIN_EXPONENT, flight.thrust_softmin_exponent), (card.NOSE_DAMPING, flight.nose_damping),
        (card.NOSE_FREQUENCY_FLOOR, flight.nose_frequency_floor), (card.SPEED_FLOOR, flight.speed_floor),
        (card.CRITICAL_SPEED, flight.critical_speed), (card.DRAG_RISE_EXPONENT, flight.drag_rise_exponent),
        (card.SEPARATION_DRAG, flight.separation_drag), (card.SEPARATION_EXPONENT, flight.separation_exponent),
        (card.BULLET_DRAG, rules.bullet_drag), (card.BULLET_SPREAD, rules.bullet_spread),
        (card.HEAD_ON_FRACTION, rules.head_on_fraction), (card.HEAD_ON_ANGLE, rules.head_on_angle),
        (card.SHIELD_RECHARGE_DELAY, rules.shield_recharge_delay),
        (card.SHIELD_RECHARGE_RATE, rules.shield_recharge_rate),
        (card.BOUNDARY_MAX_DAMAGE, rules.boundary_max_damage), (card.SPAWN_SPEED, rules.spawn_speed),
        (card.SPAWN_OFFSET, rules.spawn_offset), (card.SPAWN_TURN_DISTANCE, rules.spawn_turn_distance),
        (card.SIDE_AREA_FRACTION, mode.side_area_fraction), (card.SIDE_HEADING_JITTER, mode.side_heading_jitter),
    ]
    assert len(shared) == 28
    for ours, theirs in shared:
        assert ours == theirs, (ours, theirs)
    from bnb.game.rules import POWER_MODES, QUIET_STICKS, STICK_LEVELS
    assert [math.copysign(1.0, a) if a else 0.0 for a in card.STICK_LEVELS] == \
        [math.copysign(1.0, b) if b else 0.0 for b in STICK_LEVELS]
    assert [abs(a) for a in card.STICK_LEVELS] == [0, 1, 1, 2, 2, 3, 3]  # fine, sustained, full
    assert POWER_MODES == (0.0, -1.0, 1.0)  # coast, boost (dive), reverse (climb)
    assert QUIET_STICKS == tuple(range(card.QUIET_STICK_MAX + 1))
    assert rules.bullet_slots == BULLET_SLOTS and rules.tick_length == GameConfig().tick_length
    assert rules.pools[0].episode_ticks == GameConfig().clock_ticks
    assert rules.boundary_ramp_width * 7800.0 / MAP_SCALE_RADIUS == 300.0
