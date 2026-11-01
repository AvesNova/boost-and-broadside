"""Shot labels (learner spec 6, acceptance item 4): a shot re-simulated from captured state
is the bullet the game fires from that state, tick by tick (position, velocity, hit),
including friendly fire, fields and flights that an episode end cuts short."""

import pytest
import torch

from bnb.contracts import Action, GameConfig
from bnb.game import api
from bnb.game.physics.bullets import launch_state
from bnb.game.physics.combat import hits
from bnb.learn.shots import ShotCapture, resimulate
from bnb.models.yemong.shots import HIT_ALLY, HIT_ENEMY, HIT_NONE

# Whole vector widths of environments, so eager CPU kernels take one code path everywhere
# (mixed-notes deviation 4) and the comparison can be exact.
ENVS, TICKS = 32, 100
CONFIG = GameConfig(3, 6, 500.0, episode_s=3.0)  # crowded, with fields and episode ends
OWN = CONFIG.ships_per_team


def _frame(launch, x, y, velocity_x, velocity_y):
    """World values in the launch heading's frame, as the re-simulation reports them."""
    cos, sin = launch.heading_cos, launch.heading_sin
    dx, dy = x - launch.x, y - launch.y
    return torch.stack([dx * cos + dy * sin, dy * cos - dx * sin,
                        velocity_x * cos + velocity_y * sin,
                        velocity_y * cos - velocity_x * sin], -1)


@pytest.fixture(scope="module")
def played():
    """The game, every ship shooting at will, recorded tick by tick (team 0's view order is
    sim order)."""
    rules, types = api.rules_of(CONFIG), api.ship_types()
    generator = torch.Generator().manual_seed(1)
    state = api.reset(rules, types, api.uniforms(rules, ENVS, generator=generator))
    state, obs, _ = api.perceive(rules, state)
    captures, fields, dones, spreads, game = [], [], [], [], []
    for _ in range(TICKS):
        choice = torch.rand(ENVS, rules.num_ships, 3, generator=generator)
        action = Action.unpack((choice * torch.tensor([3.0, 7.0, 2.0])).int())
        action = Action(action.power, action.stick, torch.ones_like(action.shoot))
        spread = api.spread(rules, ENVS, generator=generator)
        launch, alive = launch_state(state.ships), state.ships.alive
        slot = state.ships.next_slot
        moved, sweep = api.move(rules, types, state, action, spread)
        hit = hits(rules, types, sweep, moved.ships, moved.bullets, alive)
        state, events = api.resolve(rules, types, moved, sweep,
                                    api.uniforms(rules, ENVS, generator=generator))
        bullets = moved.bullets
        index = torch.exp(bullets.log_refractive_index)
        speed = bullets.proper_speed / index  # world speed
        game.append(dict(fired=moved.ships.fired, slot=slot, x=bullets.x, y=bullets.y,
                         velocity_x=bullets.direction_x * speed,
                         velocity_y=bullets.direction_y * speed,
                         index=index, hit=hit.bullet_hit, damage=hit.damage))
        captures.append(ShotCapture(launch=launch, displacement_x=sweep.ship_displacement_x,
                                    displacement_y=sweep.ship_displacement_y, alive=alive))
        fields.append(obs.fields[:, 0])
        dones.append(events.done)
        spreads.append(spread)
        state, obs, _ = api.perceive(rules, state)
    stack = lambda items: torch.utils._pytree.tree_map(lambda *x: torch.stack(x), *items)  # noqa: E731
    capture = stack(captures)
    shots = resimulate(rules, types, capture, torch.stack(fields), torch.stack(dones),
                       torch.stack(spreads), OWN)
    return capture, torch.stack(dones), shots, game


def _fired(game):
    for tick, record in enumerate(game):
        for env, ship in record["fired"].nonzero().tolist():
            yield tick, env, ship, record["slot"][env, ship].item()


def test_a_resimulated_shot_is_the_bullet_the_game_fires(played):
    capture, _, shots, game = played
    compared = 0
    for tick, env, ship, slot in _fired(game):
        launch = torch.utils._pytree.tree_map(lambda x: x[tick, env, ship], capture.launch)
        for age in range(12):
            if tick + age >= TICKS or not shots.age_valid[tick, env, ship, age]:
                break
            record = game[tick + age]
            expected = _frame(launch, *(record[name][env, ship, slot]
                                        for name in ("x", "y", "velocity_x", "velocity_y")))
            assert torch.equal(shots.snapshots[tick, env, ship, age, :4], expected)
            assert torch.equal(shots.snapshots[tick, env, ship, age, 4],
                               record["index"][env, ship, slot])
            compared += 1
    assert compared > 10_000


def test_a_resimulated_hit_is_the_games_hit_with_its_side(played):
    """Hits tick by tick: the game's bullet hits on the age where the shot's flight ends in
    a hit, on the side the re-simulation names (the game's damage matrix says whom)."""
    _, _, shots, game = played
    sides = {HIT_ENEMY: 0, HIT_ALLY: 0}
    for tick, env, ship, slot in _fired(game):
        valid = shots.age_valid[tick, env, ship]
        if not shots.valid[tick, env, ship]:
            continue
        last = int(valid.nonzero().max())
        for age in range(last + 1):
            game_hit = bool(game[tick + age]["hit"][env, ship, slot])
            ends_in_hit = age == last and shots.hit[tick, env, ship] != HIT_NONE
            assert game_hit == ends_in_hit, (tick, env, ship, age)
        side = int(shots.hit[tick, env, ship])
        if side != HIT_NONE:
            damage = game[tick + last]["damage"][env, ship]
            team = slice(0, OWN) if ship < OWN else slice(OWN, None)
            ally = damage[team].sum() > 0
            enemy = damage.sum() - damage[team].sum() > 0
            assert ally if side == HIT_ALLY else enemy
            sides[side] += 1
    assert sides[HIT_ENEMY] > 0 and sides[HIT_ALLY] > 0  # friendly fire happens


def test_flights_cross_fields(played):
    _, _, shots, _ = played
    index = shots.snapshots[..., 4][shots.age_valid]
    assert ((index - 1.0).abs() > 0.01).any()


def test_a_flight_an_episode_end_cuts_short_is_invalid(played):
    """The game clears every bullet at an episode end; a shot whose flight would continue
    past it is invalid, and one that hit before it is not."""
    capture, done, shots, _ = played
    cut = hit_first = 0
    for tick in range(TICKS - 12):
        for env in range(ENVS):
            ends = done[tick:tick + 11, env].nonzero()
            if not ends.numel():
                continue
            end = tick + int(ends[0])  # the episode ends after this tick
            for ship in range(2 * OWN):
                if not capture.alive[tick, env, ship]:
                    assert not shots.valid[tick, env, ship]  # a dead ship fires nothing
                    continue
                last_age = int(shots.age_valid[tick, env, ship].nonzero().max()) \
                    if shots.valid[tick, env, ship] else None
                if last_age is None:
                    cut += 1
                else:
                    assert tick + last_age <= end and shots.hit[tick, env, ship] != HIT_NONE
                    hit_first += 1
    assert cut > 0 and hit_first > 0
