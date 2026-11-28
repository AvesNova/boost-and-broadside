"""The style counters (lean.eval.style): hand-built turning fights classify as two-circle,
one-circle or neither; the other measures read the state; the runner's sums start afresh at
each deal; the store keeps them per game."""

import math
import sqlite3

import pytest
import torch

from lean import game, scripted
from lean.eval import Evaluator, Queue, ScriptedAgent, Store, play, style
from lean.eval.store import STYLE_COLUMNS
from lean.scripted import ScriptedConfig
from lean.scripted.parameters import vector
from lean.scripted.tests.test_scripted import placed

GAME = game.GameConfig(clock_ticks=20)  # short games
COLUMN = {name: index for index, name in enumerate(style.SUMS)}


def pair(nose_rates):
    """A 1v1 at 600 m, both heading +x; nose rates in sustained rates."""
    state = placed(
        dict(x=0.0, y=0.0, heading=0.0, speed=200.0), dict(x=600.0, y=0.0, heading=0.0, speed=200.0)
    )
    sustained = state.constants["sustained_rate"]
    return state.replace(nose_rate=torch.tensor([nose_rates]) * sustained)


def rows_of(state, power=None, fired=None, hit_damage=None, bullet_damage=None, collision=None):
    shape = state.speed.shape
    none = torch.zeros(shape + shape[-1:])
    return style.per_ship(
        state,
        torch.zeros(shape, dtype=torch.long) if power is None else power,
        torch.zeros(shape, dtype=torch.bool) if fired is None else fired,
        none if hit_damage is None else hit_damage,
        none if bullet_damage is None else bullet_damage,
        none if collision is None else collision,
    )


@pytest.mark.parametrize(
    "rates, turning, two_circle",
    [
        ((0.6, 0.6), 1.0, 1.0),  # both turning left: the same rotation, two-circle
        ((-0.8, -0.5), 1.0, 1.0),  # both right
        ((0.6, -0.6), 1.0, 0.0),  # opposite senses: one-circle
        ((0.6, 0.0), 0.0, 0.0),  # the target flies straight: neither
        ((0.6, 0.1), 0.0, 0.0),  # under the floor counts as straight
        ((0.0, 0.0), 0.0, 0.0),
    ],
    ids=[
        "two-circle left",
        "two-circle right",
        "one-circle",
        "straight",
        "under the floor",
        "both straight",
    ],
)
def test_turning_fight_classification(rates, turning, two_circle):
    state = pair(rates)
    rows = rows_of(state)
    for ship in (0, 1):
        assert float(rows[0, ship, COLUMN["turning"]]) == turning
        assert float(rows[0, ship, COLUMN["two_circle"]]) == two_circle


def test_the_other_measures_read_the_state():
    state = pair((0.0, 0.0))
    c = state.constants
    state = state.replace(
        energy=torch.tensor([[0.25, 0.75]]) * c["max_energy"],
        speed=torch.tensor([[0.5, 2.0]]) * c["corner_speed"],
    )
    power = torch.tensor([[style.CLIMB, style.DIVE]])
    hit_damage = torch.zeros(1, 2, 2)
    hit_damage[0, 0, 1] = 3.0  # ship 0 hits ship 1
    rows = rows_of(state, power, torch.tensor([[True, False]]), hit_damage, hit_damage)
    expected = {
        "ticks": (1, 1),
        "store": (0.25, 0.75),
        "speed": (0.5, 2.0),
        "shots": (1, 0),
        "hits": (1, 0),
        "bullet_hits": (1, 0),
        "collision_hits": (0, 0),
        "tail_hits": (1, 0),  # both fly +x, ship 0 behind ship 1: in its rear cone
        "hit_range": (600.0 / float(c["gun_range"]), 0.0),
        "climb": (1, 0),
        "dive": (0, 1),
    }
    for name, values in expected.items():
        torch.testing.assert_close(
            rows[0, :, COLUMN[name]], torch.tensor(values, dtype=torch.float32)
        )
    # A ship not alive at tick start counts nothing.
    dead = state.replace(alive=torch.tensor([[True, False]]))
    assert float(rows_of(dead)[0, 1].abs().sum()) == 0.0
    # Seats: the tick's sums per seat, nothing for an environment not playing.
    seat_a = torch.tensor([[True, False]])
    sums = style.tick_sums(
        state,
        seat_a,
        torch.tensor([True]),
        power,
        torch.tensor([[True, False]]),
        hit_damage,
        hit_damage,
        torch.zeros(1, 2, 2),
    )
    torch.testing.assert_close(sums[0], rows[0])
    assert (
        float(
            style.tick_sums(
                state,
                seat_a,
                torch.tensor([False]),
                power,
                torch.tensor([[True, False]]),
                hit_damage,
                hit_damage,
                torch.zeros(1, 2, 2),
            )
            .abs()
            .sum()
        )
        == 0.0
    )


def test_measures_from_sums():
    sums = torch.zeros(1, 2, len(style.SUMS))
    sums[0, 0, COLUMN["ticks"]] = 1800  # one ship-minute at 30 ticks a second
    sums[0, 0, COLUMN["turning"]] = 900
    sums[0, 0, COLUMN["two_circle"]] = 300
    sums[0, 0, COLUMN["shots"]] = 12
    sums[0, 0, COLUMN["hits"]] = 2
    sums[0, 0, COLUMN["bullet_hits"]] = 1
    sums[0, 0, COLUMN["collision_hits"]] = 1
    sums[0, 0, COLUMN["tail_hits"]] = 1
    sums[0, 0, COLUMN["hit_range"]] = 1.0
    sums[0, 1, COLUMN["ticks"]] = 1800
    (row,) = style.measures(sums, 1.0 / 30.0)
    a, b = (
        dict(zip(style.MEASURES, row[: len(style.MEASURES)], strict=True)),
        dict(zip(style.MEASURES, row[len(style.MEASURES) :], strict=True)),
    )
    assert (
        a["minutes"] == pytest.approx(1.0)
        and a["turning"] == 0.5
        and a["two_circle"] == pytest.approx(1 / 3)
    )
    assert a["shots"] == pytest.approx(12.0) and a["hits"] == 2 and a["hit_range"] == 0.5
    assert (a["bullet_hits"], a["collision_hits"], a["tail_hits"]) == (1, 1, 1)
    assert b["two_circle"] is None and b["hit_range"] is None and b["hits"] == 0


def test_the_runner_starts_each_game_afresh():
    # 24 games on 8 slots: every slot plays several, and each game's sums are its own: its
    # ship-ticks are its length times each seat's two ships (no ship dies in 20 ticks).
    table = torch.tensor([vector(scripted.ScriptedConfig())] * 2)
    zeros = torch.zeros(24, dtype=torch.long)
    queue = Queue(zeros, zeros + 1, torch.arange(24) % 2, zeros + 2, zeros + 2)
    outcome = play(GAME, queue, table, envs=8, segment=8)
    ticks = outcome.style[..., COLUMN["ticks"]]
    torch.testing.assert_close(ticks, (2 * outcome.length[:, None]).float().expand(24, 2))
    assert (outcome.style[..., COLUMN["store"]] > 0).all()


def test_the_store_keeps_styles_per_game(tmp_path):
    store = Store(tmp_path / "games.sqlite")
    evaluator = Evaluator(store, GAME, device="cpu", envs=8, segment=8)
    yardstick, other = (
        ScriptedAgent.of(),
        ScriptedAgent.of(ScriptedConfig().replace(preferred_speed=1.1)),
    )
    evaluator.ensure([(yardstick, other, (2, 2), 10)])
    rows = store.connection.execute(
        "SELECT length, a_minutes, b_minutes, a_store FROM games"
    ).fetchall()
    assert len(rows) == 10
    for length, a_minutes, b_minutes, a_store in rows:
        assert a_minutes == pytest.approx(2 * length * GAME.tick_length / 60.0) == b_minutes
        assert 0.0 < a_store < 1.0
    conditions = store.add_conditions(GAME, (2, 2))
    styles = {row["agent"]: row for row in store.styles(conditions)}
    assert set(styles) == {yardstick.id, other.id}
    assert styles[yardstick.id]["games"] == 10 and set(style.MEASURES) <= set(styles[yardstick.id])


def test_an_older_store_takes_the_style_columns(tmp_path):
    path = tmp_path / "old.sqlite"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE games (agent_a TEXT NOT NULL, agent_b TEXT NOT NULL, side INTEGER NOT NULL, "
        "conditions TEXT NOT NULL, result INTEGER NOT NULL, length INTEGER NOT NULL)"
    )
    connection.execute("INSERT INTO games VALUES ('a', 'b', 0, 'c', 0, 100)")
    connection.commit()
    connection.close()
    store = Store(path)
    row = store.connection.execute(f"SELECT {', '.join(STYLE_COLUMNS)} FROM games").fetchone()
    assert row == (None,) * len(STYLE_COLUMNS)
    assert store.styles("c") == []  # no style: not in the view
    assert store.counts("a", "b", "c")["games"] == 1


def counters_after(state, fire=(1, 0)):
    """One real tick of the game from `state` (ship 0 fires with no spread when `fire`), then
    the style rows of that tick: (out, rows)."""
    action = torch.zeros(state.exists.shape + (3,), dtype=torch.long)
    action[0, :, 2] = torch.tensor(fire)
    state = state.replace(pending=action)
    config = game.GameConfig(spawn="mixed", field_count=0, fog=False)  # as `placed`
    new, out = game.step(config, state, action, {"spread": torch.zeros(state.x.shape + (2,))})
    rows = style.per_ship(
        state,
        torch.zeros(state.speed.shape, dtype=torch.long),
        new.fired,
        out.hit_damage,
        out.bullet_damage,
        out.collision_damage,
    )
    return out, rows


def target_ahead(target_heading):
    """Ship 0 at the origin flying +x, ship 1 (the enemy) 40 m ahead nearly still, nose along
    `target_heading` radians."""
    return placed(
        dict(x=0.0, y=0.0, heading=0.0, speed=200.0),
        dict(x=40.0, y=0.0, heading=target_heading, speed=1.0),
    )


@pytest.mark.parametrize(
    "heading, tail", [(0.0, 1.0), (0.5 * math.pi, 0.0)], ids=["from behind", "from the beam"]
)
def test_a_real_bullet_hit_is_a_bullet_hit_and_a_tail_hit_only_from_behind(heading, tail):
    """The target flying away from the shooter (the shooter in its rear cone) or across its
    path (beside its nose: beyond the nose armour, the bullet does its damage, not a tail
    hit)."""
    out, rows = counters_after(target_ahead(heading))
    assert float(out.bullet_damage[0, 0, 1]) > 0.0
    assert float(out.collision_damage.abs().sum()) == 0.0
    assert float(out.hit_damage[0, 0, 1]) == float(out.bullet_damage[0, 0, 1])
    got = {name: float(rows[0, 0, COLUMN[name]]) for name in style.SUMS}
    assert (got["hits"], got["bullet_hits"], got["collision_hits"]) == (1.0, 1.0, 0.0)
    assert got["tail_hits"] == tail
    assert float(rows[0, 1, COLUMN["bullet_hits"]]) == 0.0  # the target hit nothing


def test_a_hit_on_the_nose_armour_is_no_hit():
    """The target facing the shooter: the bullet lands inside the nose armour (on by default
    since 2026-11-28), does no damage and is not counted as a hit."""
    out, rows = counters_after(target_ahead(math.pi))
    assert float(out.bullet_damage[0, 0, 1]) == 0.0
    got = {name: float(rows[0, 0, COLUMN[name]]) for name in style.SUMS}
    assert (got["hits"], got["bullet_hits"], got["tail_hits"]) == (0.0, 0.0, 0.0)


def test_the_rear_cone_edge_is_the_named_angle():
    """The target (nose +x, at the origin) with the shooter on its line of sight 10 degrees
    either side of the 120 degree edge: a tail hit beyond it, not inside."""
    edge = math.degrees(style.TAIL_CHASE_ANGLE)
    assert edge == pytest.approx(120.0)
    for degrees, tail in ((edge - 10.0, 0.0), (edge + 10.0, 1.0), (-(edge + 10.0), 1.0)):
        angle = math.radians(degrees)
        state = placed(
            dict(x=300.0 * math.cos(angle), y=300.0 * math.sin(angle), heading=0.0, speed=200.0),
            dict(x=0.0, y=0.0, heading=0.0, speed=200.0),
        )
        damage = torch.zeros(1, 2, 2)
        damage[0, 0, 1] = 1.0
        rows = rows_of(state, hit_damage=damage, bullet_damage=damage)
        assert float(rows[0, 0, COLUMN["tail_hits"]]) == tail, degrees


def test_a_collision_is_a_collision_hit_for_both_ships_and_no_bullet_hit():
    """Head-on at cruise: each ship damages the other (credited to it), no bullet flew."""
    from lean.game.tests.test_collisions import constant, reach, run
    from lean.game.tests.test_collisions import placed as crash_placed

    cruise = 0.5 / constant("inverse_collision_speed")
    half = 0.5 * reach() + 0.1 * cruise * GAME.tick_length
    state = crash_placed([(-half, 0.0), (half, 0.0)], [0.0, math.pi], speeds=[cruise, cruise])
    _, out = run(state)
    assert float(out.bullet_damage.abs().sum()) == 0.0
    rows = style.per_ship(
        state,
        torch.zeros(state.speed.shape, dtype=torch.long),
        torch.zeros(state.speed.shape, dtype=torch.bool),
        out.hit_damage,
        out.bullet_damage,
        out.collision_damage,
    )
    for ship in (0, 1):
        got = {name: float(rows[0, ship, COLUMN[name]]) for name in style.SUMS}
        assert (got["hits"], got["bullet_hits"], got["collision_hits"], got["tail_hits"]) == (
            1.0,
            0.0,
            1.0,
            0.0,
        )
