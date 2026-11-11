"""The shield's recharge pause (game spec 7.2, game version 3): the recharge wait counts down
and the shield recharges only on a quiet tick (coasting, the stick straight or at a slight
turn, no shot); any other input holds the wait where it is and recharges nothing; a hit
resets the wait. Hand-computed cases at the default card's fighter (12 a second after 6 s)."""

import modes_cases as cases
import pytest
import torch

from bnb.contracts import Action, GameConfig
from bnb.contracts.action import NUM_JOINT, from_joint
from bnb.contracts.obs import SHIP
from bnb.game import api
from bnb.game.physics.combat import apply_damage, quiet
from bnb.game.rules import COAST, POWER_MODES, STICK_LEVELS
from bnb.game.state import replace

TICK = 1.0 / 30.0
RATE, DELAY = 12.0, 6.0  # the fighter's (rules.shield_recharge_rate, _delay at scale 1)
SLIGHT_LEFT, SLIGHT_RIGHT = STICK_LEVELS.index(0.4), STICK_LEVELS.index(-0.4)
NORMAL_LEFT, SHARP_RIGHT = STICK_LEVELS.index(0.75), STICK_LEVELS.index(-1.0)
BOOST, REVERSE = POWER_MODES.index(-1.0), POWER_MODES.index(1.0)
# (power, stick, shoot) of the eight ships of one environment, and whether each is quiet
MIXED = (
    ((COAST, 0, 0), True),  # straight
    ((COAST, SLIGHT_LEFT, 0), True),
    ((COAST, SLIGHT_RIGHT, 0), True),
    ((COAST, 0, 1), False),  # a shot
    ((BOOST, 0, 0), False),
    ((REVERSE, 0, 0), False),
    ((COAST, NORMAL_LEFT, 0), False),
    ((COAST, SHARP_RIGHT, 0), False),
)


def _mixed(actions=MIXED) -> tuple[Action, torch.Tensor]:
    packed = torch.tensor([[action for action, _ in actions]], dtype=torch.int32)
    return Action.unpack(packed), torch.tensor([[calm for _, calm in actions]])


def _ships(wait: float, shield: float = 50.0, count: int = len(MIXED)):
    rules = api.rules_of(GameConfig(count // 2, 0, 1000.0))
    state = api.reset(rules, api.ship_types(), api.uniforms(rules, 1, generator=torch.Generator()))
    ships = replace(
        state.ships,
        shield=torch.full((1, count), shield),
        shield_recharge_wait=torch.full((1, count), wait),
    )
    return rules, ships


def test_the_quiet_actions_are_coasting_straight_or_slightly_turning_without_a_shot():
    """Of the 42 joint actions exactly three are quiet: coast, the stick at 0 or +-0.4, no
    shot."""
    action = from_joint(torch.arange(NUM_JOINT, dtype=torch.int32))
    found = {
        (int(p), STICK_LEVELS[int(s)], int(f))
        for p, s, f, calm in zip(
            action.power, action.stick, action.shoot, quiet(action), strict=True
        )
        if calm
    }
    assert found == {(COAST, 0.0, 0), (COAST, 0.4, 0), (COAST, -0.4, 0)}
    assert POWER_MODES[COAST] == 0.0


@pytest.mark.parametrize("wait", [1.0, 0.01, 0.0])
def test_a_quiet_tick_counts_down_and_recharges_and_every_other_input_pauses_both(wait):
    """Eight ships, one per input, shield 50: a quiet one's wait falls by a tick and its
    shield gains 12 a second for the part of the tick past the wait; a shot, boost, reverse,
    a normal or a sharp turn keeps the wait and the shield as they were."""
    rules, ships = _ships(wait)
    action, calm = _mixed()
    alive = torch.ones_like(calm)
    damage = apply_damage(rules, api.ship_types(), ships, alive, torch.zeros(1, 8), quiet(action))
    assert torch.equal(quiet(action), calm)
    gained = RATE * min(max(TICK - wait, 0.0), TICK)  # 0, 12 (1/30 - 0.01), 0.4
    expected_wait = torch.where(calm, max(wait - TICK, 0.0), wait)
    expected_restored = torch.where(calm, gained, 0.0)
    torch.testing.assert_close(damage.ships.shield_recharge_wait, expected_wait)
    torch.testing.assert_close(damage.shield_restored, expected_restored)
    torch.testing.assert_close(damage.ships.shield, 50.0 + expected_restored)


def test_a_hit_resets_the_wait_whatever_the_input():
    """Damage on every one of the eight inputs: the wait is the delay and nothing
    recharges, quiet or not."""
    rules, ships = _ships(0.0)
    action, _ = _mixed()
    alive = torch.ones(1, 8, dtype=torch.bool)
    damage = apply_damage(
        rules, api.ship_types(), ships, alive, torch.full((1, 8), 5.0), quiet(action)
    )
    torch.testing.assert_close(damage.ships.shield_recharge_wait, torch.full((1, 8), DELAY))
    torch.testing.assert_close(damage.ships.shield, torch.full((1, 8), 45.0))
    assert not damage.shield_restored.any()


def test_the_wait_resumes_where_it_paused():
    """A wait of three ticks: one quiet tick (two left), five boosting (still two), two
    quiet (none left), then a quiet tick recharges a whole tick's 0.4."""
    rules, ships = _ships(3 * TICK, count=2)
    types, alive = api.ship_types(), torch.ones(1, 2, dtype=torch.bool)
    calm, boost = (torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2, dtype=torch.bool))
    waits, restored = [], []
    for step in [calm] + [boost] * 5 + [calm] * 3:
        damage = apply_damage(rules, types, ships, alive, torch.zeros(1, 2), step)
        ships = damage.ships
        waits.append(float(ships.shield_recharge_wait[0, 0]))
        restored.append(float(damage.shield_restored[0, 0]))
    assert waits == pytest.approx([2 * TICK] * 6 + [TICK, 0.0, 0.0], abs=1e-6)
    assert restored == pytest.approx([0.0] * 8 + [RATE * TICK], abs=1e-6)


def test_a_whole_tick_pauses_on_the_inputs_and_the_observation_shows_the_held_wait():
    """`api.step` with the eight inputs (no ship near another, so no hits): each pausing
    ship's wait and shield hold, each quiet one's count down, and the team's observation
    of its own ships carries the held wait."""
    rules = api.rules_of(GameConfig(4, 0, 1e5))
    state = api.reset(rules, cases.TYPES, api.uniforms(rules, 1, generator=torch.Generator()))
    spaced = torch.arange(8, dtype=torch.float32)[None] * 500.0
    state = cases.edit_ships(
        state,
        x=spaced,
        y=torch.zeros(1, 8),
        shield=torch.full((1, 8), 50.0),
        shield_recharge_wait=torch.full((1, 8), 1.0),
    )
    action, calm = _mixed()
    state, events = cases.tick(rules, state, action)
    assert not events.damage.any()
    wait = torch.where(calm, 1.0 - TICK, 1.0)
    torch.testing.assert_close(state.ships.shield_recharge_wait, wait)
    assert not events.shield_restored.any()
    _, obs, _ = api.perceive(rules, state)
    own = obs.ships[0, 0, :4, SHIP["shield_recharge_wait"]]  # team 0's own rows, in slot order
    torch.testing.assert_close(own, wait[0, :4])
