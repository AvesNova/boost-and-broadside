"""Imugi's turning-fight commitment (`imugi_combat.turn_commitment`, the author's rules of
2026-11-16): the sense chosen once at the merge and held through the fight, a reacting ship
not flipping within the hold, the initiative turn's timing just before a head-on merge, and
the deterministic tie-break."""

import dataclasses
import math

import pytest
import torch
from imugi_cases import CHARACTERISTICS, absent, observation, ship, value

from bnb.contracts.obs import SHIP
from bnb.controllers import imugi_combat
from bnb.controllers.imugi import decide
from bnb.controllers.imugi_doctrine import Doctrine, table
from bnb.controllers.imugi_parameters import ImugiParameters
from bnb.game import api
from bnb.game.handling.characteristics import characteristics
from bnb.game.rules import Rules

pytestmark = pytest.mark.usefixtures("global_generator")
PARAMETERS = ImugiParameters()
TURN = CHARACTERISTICS.turn_rate[:, 2]
AGILE, STIFF = int(TURN.argmax()), int(TURN.argmin())
DOCTRINES = table([Doctrine(sense=1.0)] * TURN.shape[0])
RIGHT, LEFT = 5, 6  # full stick (`bnb.game.rules.STICK_LEVELS`)
RIGHTS = [1, 3, 5]  # every right stick level


def _decide(obs, carry=None, step: int = 0):
    keys = torch.arange(obs.ships.shape[0]) + 7 * step
    return decide(
        obs, CHARACTERISTICS, PARAMETERS, carry=carry, decision_key=keys, doctrines=DOCTRINES
    )


def _passed(
    nose_rate: float,
    own_speed: float | None = None,
    own_type: int = AGILE,
    target_speed: float | None = None,
):
    """A target just passed (behind, flying away), turning at `nose_rate` (positive left)."""
    distance = 0.5 * value("gun_range", own_type)
    enemy = ship(
        x=-distance, y=-30.0, heading=math.pi, speed=target_speed, ship_type=own_type,
        team_relative=1,
    )  # fmt: skip
    enemy[SHIP["nose_rate"]] = nose_rate
    own = ship(ship_type=own_type, speed=own_speed)
    return observation([own, absent(own_type, 0)], [enemy, absent(own_type)])


def test_a_reacting_ship_holds_its_sense_while_the_target_reverses():
    """Out-turned (slow), the ship reacts once to the target's left turn (one circle: right)
    and holds right for the hold time while the target's observed turn flips every tick."""
    parts, carry = _decide(_passed(0.3, own_speed=50.0))
    assert float(carry.fight_sense[0, 0]) == -1.0
    assert float(parts.stick[0, 0, RIGHT]) > 0.9
    ticks = int(imugi_combat.COMMIT_HOLD_SECONDS / Rules.tick_length) - 1
    for step in range(1, ticks):
        rate = -0.3 if step % 2 else 0.3
        parts, carry = _decide(_passed(rate, own_speed=50.0), carry, step)
        assert float(carry.fight_sense[0, 0]) == -1.0
        # right: full stick in one circle, the sustained level when the target's turn makes
        # it two circles (slow, below the corner speed: stick by energy)
        assert float(parts.stick[0, 0, RIGHTS].sum()) > 0.9


def test_an_even_pair_decides_by_geometry_and_holds_through_the_fight():
    """A mirror: the sense from the geometry (the target's tail on the right: right), held
    through the turning fight (well past the hold) whatever the target's turn does."""
    parts, carry = _decide(_passed(0.3))
    sense = float(carry.fight_sense[0, 0])
    assert sense == -1.0
    for step in range(1, 150):  # 5 s
        rate = 0.3 * math.sin(step)
        parts, carry = _decide(_passed(rate), carry, step)
        assert float(carry.fight_sense[0, 0]) == sense


def test_without_the_commitment_the_sense_follows_the_targets_turn(monkeypatch):
    """The old behaviour (BNB_TURN_COMMIT=0): re-decided every tick, it flips with the
    target's observed turn."""
    monkeypatch.setattr(imugi_combat, "TURN_COMMIT", False)
    parts, carry = _decide(_passed(0.3, own_speed=50.0))
    assert float(parts.stick[0, 0, RIGHT]) > 0.9
    parts, _ = _decide(_passed(-0.3, own_speed=50.0), carry, 1)
    assert float(parts.stick[0, 0, LEFT]) > 0.9


def test_a_re_reading_keeps_the_commitment():
    """Without a decision key the carry's commitment holds: the same marginals again on the
    same observation."""
    parts, carry = _decide(_passed(0.3, own_speed=50.0))
    again, after = decide(
        _passed(0.3, own_speed=50.0), CHARACTERISTICS, PARAMETERS, carry=carry,
        doctrines=DOCTRINES,
    )  # fmt: skip
    torch.testing.assert_close(again.stick, parts.stick)
    torch.testing.assert_close(after.fight_sense, carry.fight_sense)


def _head_on(own_x: float, own_y: float, lateral: float, crossing: float, target_speed=50.0):
    """Own ship (AGILE, cruise, nose +x) at (own_x, own_y) and a slow target (out-turned:
    the own ship holds the advantage) coming head-on, `lateral` m to the left of the own
    track, the closest approach `crossing` s away."""
    speed = value("cruise_speed", AGILE)
    ahead = crossing * (speed + target_speed)
    enemy = ship(
        x=own_x + ahead, y=own_y + lateral, heading=math.pi, speed=target_speed,
        ship_type=AGILE, team_relative=1,
    )  # fmt: skip
    own = ship(x=own_x, y=own_y, ship_type=AGILE)
    return observation([own, absent(AGILE, 0)], [enemy, absent(AGILE)])


def _onset() -> float:
    corner = value("corner_speed", AGILE)
    frequency = value("nose_frequency", AGILE) * min(value("cruise_speed", AGILE) / corner, 1.0)
    return imugi_combat.TURN_ONSET_NOSE_PERIODS / max(frequency, 1.0)


def test_the_initiative_turn_starts_at_the_latest_tick_with_full_rate_at_the_crossing():
    """The advantaged ship commits on a head-on approach only once the crossing is within
    its turn onset (to the decision's rounding), to the side the target passes on."""
    onset, tick = _onset(), Rules.tick_length
    _, carry = _decide(_head_on(0.0, 0.0, 60.0, onset + 1.5 * tick))
    assert float(carry.fight_sense[0, 0]) == 0.0  # too early: the nose stays on the target
    parts, carry = _decide(_head_on(0.0, 0.0, 60.0, onset - 0.25 * tick))
    assert float(carry.fight_sense[0, 0]) == 1.0
    assert float(parts.stick[0, 0, LEFT]) > 0.9
    _, carry = _decide(_head_on(0.0, 0.0, -60.0, onset - 0.25 * tick))
    assert float(carry.fight_sense[0, 0]) == -1.0


def test_the_initiative_turn_keeps_the_nose_in_the_armour_zone_until_the_crossing():
    """On the real-scale set, the turn the onset allows before the crossing (at most its
    full corner rate over the onset) stays inside the head-on armour's angle."""
    types = api.ship_types(ship_set="real-scale")
    game = dataclasses.replace(api.PRESETS["real_scale_deathmatch"], ships_per_team=1)
    card = characteristics(types, api.rules_of(game, types=types))
    corner_rate = card.max_acceleration / card.corner_speed
    onset = imugi_combat.TURN_ONSET_NOSE_PERIODS / card.nose_frequency
    head_on_angle = api.rules_of(game, types=types).head_on_angle * types.head_on_angle_scale
    assert bool((corner_rate * onset < head_on_angle).all())


def test_a_dead_on_merge_without_advantage_turns_toward_the_map_centre():
    """An even pair passing dead-on (no lateral offset): toward the map centre; on the line
    through the centre, left. No randomness: every decision key gives the same sense."""

    def passed(own_y: float, own_x: float = 0.0):
        enemy = ship(x=own_x - 200.0, y=own_y, heading=math.pi, ship_type=AGILE, team_relative=1)
        enemy[SHIP["nose_rate"]] = 0.0
        own = ship(x=own_x, y=own_y, ship_type=AGILE)
        return observation([own, absent(AGILE, 0)], [enemy, absent(AGILE)])

    for step in range(4):
        _, carry = _decide(passed(500.0), step=step)  # the centre on the right
        assert float(carry.fight_sense[0, 0]) == -1.0
        _, carry = _decide(passed(-500.0), step=step)
        assert float(carry.fight_sense[0, 0]) == 1.0
        _, carry = _decide(passed(0.0, own_x=-1000.0), step=step)  # flying at the centre
        assert float(carry.fight_sense[0, 0]) == 1.0


def test_an_advantaged_dead_on_initiative_turns_toward_the_map_centre():
    """The initiative turn at a dead-on merge breaks the tie the same way."""
    onset = _onset()
    _, carry = _decide(_head_on(0.0, 500.0, 0.0, 0.5 * onset))
    assert float(carry.fight_sense[0, 0]) == -1.0
    _, carry = _decide(_head_on(0.0, -500.0, 0.0, 0.5 * onset))
    assert float(carry.fight_sense[0, 0]) == 1.0


SUSTAINED_LEVELS = [3, 4]  # the 0.75 stick right and left


def test_stick_by_energy_eases_a_slow_two_circle_turn_to_the_sustained_level():
    """Committed into two circles (the target's own sense) below the corner speed: the
    sustained stick level; at the corner speed and above, full stick; one circle, full."""
    corner = value("corner_speed", AGILE)
    # an even pair (the same type and speed) at the merge turns right by the geometry; a
    # target turning right makes it two circles
    slow, _ = _decide(_passed(-0.3, 0.8 * corner, target_speed=0.8 * corner))
    fast, _ = _decide(_passed(-0.3, 1.2 * corner, target_speed=1.2 * corner))
    one, _ = _decide(_passed(0.3, 0.8 * corner, target_speed=0.8 * corner))
    assert float(slow.stick[0, 0, SUSTAINED_LEVELS].sum()) > 0.9
    assert float(fast.stick[0, 0, RIGHT]) > 0.9
    assert float(one.stick[0, 0, RIGHTS].sum()) > 0.9
    assert float(one.stick[0, 0, SUSTAINED_LEVELS].sum()) < 0.1
