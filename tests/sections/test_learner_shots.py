"""Shot labels (learner spec 6, acceptance item 4): a shot re-simulated from captured state
is the bullet the game fires from that state, tick by tick (position, velocity, hit),
including friendly fire, fields and flights that an episode end cuts short; and the ages the
trajectory head is queried at."""

import pytest
import torch

from bnb.contracts import Action, GameConfig
from bnb.contracts.action import NUM_POWER, NUM_SHOOT, NUM_STICK
from bnb.contracts.obs import SHIP
from bnb.game import api
from bnb.game.physics.bullets import Launch, launch_state
from bnb.game.physics.combat import hits
from bnb.learn import labels
from bnb.learn.shots import QUERIES, ShotCapture, query_ages, resimulate
from bnb.learn.shots import labels as shot_labels
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
        action = Action.unpack(
            (choice * torch.tensor([NUM_POWER, NUM_STICK, NUM_SHOOT], dtype=torch.float32)).int()
        )
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
        for age in range(MOVES):  # the game's bullet flies to its hit, the shot's on past it
            if (
                tick + age >= TICKS
                or not shots.valid[tick, environment, ship]
                or age > shots.hit_age[tick, environment, ship]
            ):
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
    """Hits tick by tick: the game's bullet hits on the shot's hit age, on the side the
    re-simulation names (the game's damage matrix says whom)."""
    _, _, shots, game, _ = played
    sides = {HIT_ENEMY: 0, HIT_ALLY: 0}
    for tick, environment, ship, slot in _fired(game):
        if not shots.valid[tick, environment, ship]:
            continue
        last = min(int(shots.hit_age[tick, environment, ship]), MOVES - 1)
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
    index = shots.snapshots[..., 4][shots.valid]
    assert ((index - 1.0).abs() > 0.01).any()


def test_a_flight_an_episode_end_cuts_short_is_invalid(played):
    """The game clears every bullet at an episode end; every age of a shot is labelled (past
    its hit, the straight-on flight), so a shot whose whole flight would continue past an
    end is invalid, whether or not it hit before it, and a living shooter's shot whose
    flight no end cuts is valid (inside the segment)."""
    capture, done, shots, _, _ = played
    cut = cut_after_hit = kept = 0
    for tick in range(TICKS - (MOVES - 1)):
        for environment in range(ENVS):
            ended = bool(done[tick : tick + MOVES - 1, environment].any())
            for ship in range(2 * OWN):
                if not capture.alive[tick, environment, ship]:
                    assert not shots.valid[tick, environment, ship]  # a dead ship fires nothing
                elif ended:
                    assert not shots.valid[tick, environment, ship]
                    cut += 1
                    cut_after_hit += int(shots.hit[tick, environment, ship] != HIT_NONE)
                else:
                    assert shots.valid[tick, environment, ship]
                    kept += 1
    assert cut > 0 and cut_after_hit > 0 and kept > 0


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
    for name in ("hit", "hit_age", "distance", "side", "closest_age", "bend_age"):
        assert torch.equal(getattr(short, name)[body][valid], getattr(long, name)[body][valid])


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


# ------------------------------------------------------------------ the queried ages


def _straight_shot(positions, field_rows, alive=None):
    """The shots of ships standing at `positions` (2v2) through `field_rows` (M, 5), the
    first fired along +x at tick 0; alive (rows,) which stand alive (all by default)."""
    rules, types = api.rules_of(GameConfig(2, 2, 1500.0)), api.ship_types()
    length, rows = MOVES, len(positions)
    capture = _still(rows, length, positions)
    if alive is not None:
        capture = ShotCapture(
            launch=capture.launch,
            displacement_x=capture.displacement_x,
            displacement_y=capture.displacement_y,
            alive=torch.tensor(alive).expand(length, 1, rows).clone(),
        )
    fields = torch.tensor(field_rows, dtype=torch.float32).expand(length, 1, -1, 5)
    return resimulate(
        rules,
        types,
        capture,
        fields,
        torch.zeros(length, 1, dtype=torch.bool),
        torch.zeros(length, 1, rows, 2),
        2,
    )


NO_BAND = [[1e5, 1e5, 1.0, 1.0, 0.0]]
STANDING = [(0, 0), (50, 30), (100, 2000), (-100, 2000)]  # 2v2, the shooter at the origin
BAND_ON_THE_LINE = [[150.0, 0.0, 15.0, 10.0, 2.0]]  # on the shooter's line, ahead of the ship


def test_the_closest_and_bend_ages_are_where_the_flight_passes_the_ship_and_meets_the_band():
    """A straight shot past a ship standing 30 m beside its line at x = 50 m: the closest
    pass is on the move that crosses x = 50 (from the shot's own positions). Through a
    field centred on the line at x = 150 m (radius 15 m, band 10 m): the first move that
    ends inside the band's outer edge (20 m from the centre) is the bend age, worked out
    from the positions alone; with no field there, none (-1)."""
    shots = _straight_shot(STANDING, BAND_ON_THE_LINE)
    forward = shots.snapshots[0, 0, 0, :, 0]
    assert shots.valid[0, 0, 0] and forward[-1] > 135.0  # the flight enters the field
    previous = torch.cat([torch.zeros(1), forward[:-1]])
    crossing = int(((previous < 50.0) & (forward >= 50.0)).nonzero()[0])
    assert shots.closest_age[0, 0, 0] == crossing
    inside_band = (forward - 150.0).abs() < 20.0  # on the line: distance to the centre
    assert shots.bend_age[0, 0, 0] == int(inside_band.nonzero()[0])
    straight = _straight_shot(STANDING, NO_BAND)
    assert straight.bend_age[0, 0, 0] == -1


def test_the_queried_ages_are_the_last_a_uniform_the_closest_and_the_bend():
    """`query_ages`: the last move, the first draw, the closest pass and the bend; with no
    other ship alive the closest pass falls back to the second draw, through no band the
    bend to the third."""
    drawn = torch.tensor([3, 5, 7], dtype=torch.uint8).expand(MOVES, 1, 4, 3)
    shots = _straight_shot(STANDING, BAND_ON_THE_LINE)
    ages = query_ages(shots, drawn)[0, 0, 0]
    assert ages.tolist() == [
        MOVES - 1,
        3,
        int(shots.closest_age[0, 0, 0]),
        int(shots.bend_age[0, 0, 0]),
    ]
    alone = _straight_shot(STANDING, NO_BAND, alive=[True, False, False, False])
    assert not torch.isfinite(alone.distance[0, 0, 0])
    assert query_ages(alone, drawn)[0, 0, 0].tolist() == [MOVES - 1, 3, 5, 7]


def test_the_labels_hold_the_snapshots_at_the_queried_ages(played):
    """`labels`: each query's snapshot is the shot's at that age, the outcome the shot's."""
    _, _, shots, _, _ = played
    generator = torch.Generator().manual_seed(3)
    drawn = torch.randint(MOVES, shots.valid.shape + (3,), dtype=torch.uint8, generator=generator)
    queried = shot_labels(shots, drawn)
    ages = query_ages(shots, drawn)
    assert torch.equal(queried.ages, ages.to(torch.uint8))
    tick, environment, ship = 5, 3, 1
    for query in range(QUERIES):
        age = int(ages[tick, environment, ship, query])
        assert torch.equal(
            queried.snapshots[tick, environment, ship, query],
            shots.snapshots[tick, environment, ship, age],
        )
    for name in ("hit", "distance", "side"):
        assert torch.equal(getattr(queried, name), getattr(shots, name))
