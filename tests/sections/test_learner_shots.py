"""Shot labels (learner spec 6, acceptance item 4): a shot re-simulated from captured state
is the bullet the game fires from that state, tick by tick (position, velocity, hit),
including friendly fire, fields and flights that an episode end cuts short."""

import pytest
import torch

from bnb.contracts import Action, GameConfig
from bnb.contracts.obs import SHIP
from bnb.game import api
from bnb.game.physics.bullets import Launch, launch_state
from bnb.game.physics.combat import hits
from bnb.learn import labels
from bnb.learn.shots import ShotCapture, resimulate
from bnb.models.yemong.shots import HIT_ALLY, HIT_ENEMY, HIT_NONE, SIDE_ALLY, SIDE_ENEMY

# Whole vector widths of environments, so eager CPU kernels take one code path everywhere
# (mixed-notes deviation 4) and the comparison can be exact.
ENVS, TICKS = 32, 100
CONFIG = GameConfig(3, 6, 500.0, episode_seconds=3.0)  # crowded, with fields and episode ends
OWN = CONFIG.ships_per_team
MOVES = api.rules_of(CONFIG).bullet_moves  # the longest bullet flight, in ticks


def _frame(launch, x, y, velocity_x, velocity_y):
    """World values in the launch heading's frame, as the re-simulation reports them."""
    cos, sin = launch.heading_cos, launch.heading_sin
    dx, dy = x - launch.x, y - launch.y
    return torch.stack(
        [
            dx * cos + dy * sin,
            dy * cos - dx * sin,
            velocity_x * cos + velocity_y * sin,
            velocity_y * cos - velocity_x * sin,
        ],
        -1,
    )


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
        state, events = api.resolve(
            rules, types, moved, sweep, api.uniforms(rules, ENVS, generator=generator)
        )
        bullets = moved.bullets
        index = torch.exp(bullets.log_refractive_index)
        speed = bullets.proper_speed / index  # world speed
        game.append(
            dict(
                fired=moved.ships.fired,
                slot=slot,
                x=bullets.x,
                y=bullets.y,
                velocity_x=bullets.direction_x * speed,
                velocity_y=bullets.direction_y * speed,
                index=index,
                hit=hit.bullet_hit,
                damage=hit.damage,
            )
        )
        captures.append(
            ShotCapture(
                launch=launch,
                displacement_x=sweep.ship_displacement_x,
                displacement_y=sweep.ship_displacement_y,
                alive=alive,
            )
        )
        fields.append(obs.fields[:, 0])
        dones.append(events.done)
        spreads.append(spread)
        state, obs, _ = api.perceive(rules, state)
    stack = lambda items: torch.utils._pytree.tree_map(lambda *x: torch.stack(x), *items)  # noqa: E731
    capture = stack(captures)
    inputs = (capture, torch.stack(fields), torch.stack(dones), torch.stack(spreads))

    def segment(length: int):
        """The shots of the recording's first `length` ticks, as a segment of them."""
        return resimulate(rules, types, *stack_prefix(inputs, length), OWN)

    return capture, inputs[2], segment(TICKS), game, segment


def stack_prefix(tree, length: int):
    return torch.utils._pytree.tree_map(lambda x: x[:length], tree)


def _fired(game):
    for tick, record in enumerate(game):
        for environment, ship in record["fired"].nonzero().tolist():
            yield tick, environment, ship, record["slot"][environment, ship].item()


def test_a_resimulated_shot_is_the_bullet_the_game_fires(played):
    capture, _, shots, game, _ = played
    compared = 0
    for tick, environment, ship, slot in _fired(game):
        launch = torch.utils._pytree.tree_map(lambda x: x[tick, environment, ship], capture.launch)
        for age in range(12):
            if tick + age >= TICKS or not shots.age_valid[tick, environment, ship, age]:
                break
            record = game[tick + age]
            expected = _frame(
                launch,
                *(
                    record[name][environment, ship, slot]
                    for name in ("x", "y", "velocity_x", "velocity_y")
                ),
            )
            assert torch.equal(shots.snapshots[tick, environment, ship, age, :4], expected)
            assert torch.equal(
                shots.snapshots[tick, environment, ship, age, 4],
                record["index"][environment, ship, slot],
            )
            compared += 1
    assert compared > 10_000


def test_a_resimulated_hit_is_the_games_hit_with_its_side(played):
    """Hits tick by tick: the game's bullet hits on the age where the shot's flight ends in
    a hit, on the side the re-simulation names (the game's damage matrix says whom)."""
    _, _, shots, game, _ = played
    sides = {HIT_ENEMY: 0, HIT_ALLY: 0}
    for tick, environment, ship, slot in _fired(game):
        valid = shots.age_valid[tick, environment, ship]
        if not shots.valid[tick, environment, ship]:
            continue
        last = int(valid.nonzero().max())
        for age in range(last + 1):
            game_hit = bool(game[tick + age]["hit"][environment, ship, slot])
            ends_in_hit = age == last and shots.hit[tick, environment, ship] != HIT_NONE
            assert game_hit == ends_in_hit, (tick, environment, ship, age)
        side = int(shots.hit[tick, environment, ship])
        if side != HIT_NONE:
            damage = game[tick + last]["damage"][environment, ship]
            team = slice(0, OWN) if ship < OWN else slice(OWN, None)
            ally = damage[team].sum() > 0
            enemy = damage.sum() - damage[team].sum() > 0
            assert ally if side == HIT_ALLY else enemy
            sides[side] += 1
    assert sides[HIT_ENEMY] > 0 and sides[HIT_ALLY] > 0  # friendly fire happens


def test_flights_cross_fields(played):
    _, _, shots, _, _ = played
    index = shots.snapshots[..., 4][shots.age_valid]
    assert ((index - 1.0).abs() > 0.01).any()


def test_a_flight_an_episode_end_cuts_short_is_invalid(played):
    """The game clears every bullet at an episode end; a shot whose flight would continue
    past it is invalid, and one that hit before it is not."""
    capture, done, shots, _, _ = played
    cut = hit_first = 0
    for tick in range(TICKS - 12):
        for environment in range(ENVS):
            ends = done[tick : tick + 11, environment].nonzero()
            if not ends.numel():
                continue
            end = tick + int(ends[0])  # the episode ends after this tick
            for ship in range(2 * OWN):
                if not capture.alive[tick, environment, ship]:
                    assert not shots.valid[tick, environment, ship]  # a dead ship fires nothing
                    continue
                last_age = (
                    int(shots.age_valid[tick, environment, ship].nonzero().max())
                    if shots.valid[tick, environment, ship]
                    else None
                )
                if last_age is None:
                    cut += 1
                else:
                    assert tick + last_age <= end and shots.hit[tick, environment, ship] != HIT_NONE
                    hit_first += 1
    assert cut > 0 and hit_first > 0


# ------------------------------------------------------------------ the rollout's capture


@pytest.fixture(scope="module")
def stored():
    """A real rollout's stored segment: Yemong as the learner, in self-play (one team-0 and
    one team-1 view per environment) and against the chaser."""
    import learner_cases as lc
    import yemong_cases as cases

    runner = lc.rollout(cases.model(), runs=1)
    assert set(runner.lineup.role["learner"].tolist()) == {0, 1}  # team-1 views exist
    return runner.trajectory(0)


def test_the_rollouts_capture_is_the_tick_start_state_in_view_order(stored):
    """The capture at tick t is the state the stored privileged rows show at t, row for row:
    a capture taken after the move, or in sim order, differs (team-1 views reorder rows)."""
    rows = labels.privileged_rows(stored)
    capture, launch = stored.capture, stored.capture.launch
    assert torch.equal(capture.alive, rows[..., SHIP["alive"]] > 0.5)
    alive = capture.alive
    for name in ("x", "y", "heading_cos", "heading_sin", "proper_speed"):
        assert torch.equal(getattr(launch, name)[alive], rows[..., SHIP[name]][alive]), name


def test_the_rollouts_displacement_is_the_next_row_minus_this_one(stored):
    """Where a ship is alive at t and t + 1 within one episode, the stored displacement is
    its move from row t to row t + 1. Tolerance: float32 rounding of a position of a few
    thousand metres (an ulp is under 5e-4 m)."""
    rows = labels.privileged_rows(stored)
    following = labels.next_rows(stored, rows)
    moving = (
        stored.capture.alive
        & (following[..., SHIP["alive"]] > 0.5)
        & (following[..., SHIP["spawned"]] < 0.5)
        & ~stored.done[..., None]
    )
    assert moving.float().mean() > 0.5
    for axis, displacement in (
        ("x", stored.capture.displacement_x),
        ("y", stored.capture.displacement_y),
    ):
        moved = following[..., SHIP[axis]] - rows[..., SHIP[axis]]
        torch.testing.assert_close(displacement[moving], moved[moving], rtol=0, atol=1e-3)
    assert stored.capture.displacement_x[moving].abs().max() > 1.0  # the ships do move


def test_a_shot_launched_near_the_segments_end_is_invalid_whatever_it_hit(played):
    """A shot launched within the longest flight of the segment's end is invalid, hit or
    not, so the tail does not bias the labels toward hits; every other shot keeps the label
    it has in a longer segment (the body's hit rate is unchanged)."""
    _, _, long, _, segment = played
    length = 60
    short = segment(length)
    tail = length - (MOVES - 1)
    assert not short.valid[tail:].any()
    assert (
        long.valid[tail:length].any()
        and (long.hit[tail:length][long.valid[tail:length]] != HIT_NONE).any()
    )  # the tail held hits
    body = slice(0, tail)
    assert torch.equal(short.valid[body], long.valid[body])
    valid = short.valid[body]
    assert valid.any()
    for name in ("hit", "distance", "side"):
        assert torch.equal(getattr(short, name)[body][valid], getattr(long, name)[body][valid])
    assert torch.equal(short.age_valid[body], long.age_valid[body])


def _still(rows: int, length: int, positions: list[tuple[float, float]]) -> ShotCapture:
    """One view of `rows` ships standing at `positions` for `length` ticks, noses along +x,
    with no speed, angle of attack or medium gradient."""
    x = torch.tensor([p[0] for p in positions]).expand(length, 1, rows)
    y = torch.tensor([p[1] for p in positions]).expand(length, 1, rows)
    zero = torch.zeros(length, 1, rows)
    launch = Launch(
        x=x,
        y=y,
        heading_cos=zero + 1.0,
        heading_sin=zero,
        proper_speed=zero,
        angle_of_attack_cos=zero + 1.0,
        angle_of_attack_sin=zero,
        log_refractive_index=zero,
        log_index_gradient_x=zero,
        log_index_gradient_y=zero,
        ship_type=torch.zeros(length, 1, rows, dtype=torch.int32),
    )
    return ShotCapture(
        launch=launch,
        displacement_x=zero,
        displacement_y=zero,
        alive=torch.ones(length, 1, rows, dtype=torch.bool),
    )


@pytest.mark.parametrize(
    ("shooter", "positions", "distance", "side"),
    [
        # 2v2, rows 0-1 one side and 2-3 the other; the shooter at the origin fires along +x.
        (0, [(0, 0), (100, 30), (100, -50), (100, 2000)], 30.0, SIDE_ALLY),
        (0, [(0, 0), (100, 50), (100, -30), (100, 2000)], 30.0, SIDE_ENEMY),
        (2, [(100, -50), (100, 2000), (0, 0), (100, 30)], 30.0, SIDE_ALLY),
        (2, [(100, 2000), (100, -30), (0, 0), (100, 50)], 30.0, SIDE_ENEMY),
    ],
)
def test_a_miss_records_its_closest_pass_and_that_ships_side(shooter, positions, distance, side):
    """A straight shot past ships standing beside its line: the closest pass is the nearest
    one's offset from the line (computed here by hand, not by the re-simulation), on the
    side of that ship relative to the shooter. Tolerance: float32 rounding along a flight of
    about 200 m."""
    rules, types = api.rules_of(GameConfig(2, 2, 1500.0)), api.ship_types()
    length, rows = MOVES, 4
    capture = _still(rows, length, positions)
    far_field = torch.tensor([1e5, 1e5, 1.0, 1.0, 0.0]).expand(length, 1, 1, 5)
    shots = resimulate(
        rules,
        types,
        capture,
        far_field,
        torch.zeros(length, 1, dtype=torch.bool),
        torch.zeros(length, 1, rows, 2),
        2,
    )
    assert shots.valid[0, 0, shooter] and shots.hit[0, 0, shooter] == HIT_NONE
    assert shots.snapshots[0, 0, shooter, -1, 0] > 100.0  # the flight passes the ships
    torch.testing.assert_close(
        shots.distance[0, 0, shooter], torch.tensor(distance), rtol=0, atol=1e-3
    )
    assert shots.side[0, 0, shooter] == side
