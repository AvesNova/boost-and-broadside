"""Hand-computed reward cases on paths the reference fixtures do not reach (rewards spec 2-3):
the objective levels' order, boundary damage healed by a shield restore, and a hit and a
restore counted in the same tick. Mutation-sweep items 13 to 15.

Two teams of two: ships 0 and 1 are team 0, ships 2 and 3 team 1. Damage-level rewards are
in points over `damage_unit`, times the level's weight.
"""

import pytest
import torch

from bnb.contracts import REWARD_LEVELS, Events
from bnb.learn import rewards

N = 4
TEAM = torch.tensor([0, 0, 1, 1])
CONFIG = rewards.RewardConfig()
LEVEL = {name: index for index, name in enumerate(REWARD_LEVELS)}


def _events(
    hits=(), boundary=(), lost=(), restored=(), died=(), progress=None, objective=None
) -> Events:
    """One environment's tick. hits: (shooter, target, raw); boundary, lost, restored:
    (ship, amount); died: ships."""

    def vector(pairs):
        values = torch.zeros(1, N)
        for ship, amount in pairs:
            values[0, ship] = amount
        return values

    damage = torch.zeros(1, N, N)
    for shooter, target, raw in hits:
        damage[0, shooter, target] = raw
    flags = torch.zeros(1, N, dtype=torch.bool)
    return Events(
        damage=damage,
        boundary_damage=vector(boundary),
        died=flags.clone().index_fill_(1, torch.tensor(list(died), dtype=torch.long), True),
        fired=flags,
        spawned=flags,
        killer=torch.full((1, N), -1, dtype=torch.int32),
        done=torch.zeros(1, dtype=torch.bool),
        shield_lost=vector(lost),
        shield_restored=vector(restored),
        objective_progress=torch.zeros(1, N) if progress is None else progress,
        objective=torch.zeros(1, N) if objective is None else objective,
        result=torch.zeros(1, dtype=torch.int32),
        enemy_damage_by_aspect=torch.zeros(1, N, 3),
        kill_aspect=torch.full((1, N), -1, dtype=torch.int32),
    )


def _step(ledger, events):
    return rewards.reward_step(CONFIG, ledger, events, TEAM)


def _points(values) -> torch.Tensor:
    """Damage-level rewards for these point totals per ship."""
    return torch.tensor([values]) * CONFIG.damage / CONFIG.damage_unit


def test_the_objective_progress_level_carries_objective_progress():
    progress = torch.tensor([[0.1, 0.1, -0.1, -0.1]])
    objective = torch.tensor([[0.5, 0.5, -0.5, -0.5]])
    events = _events(progress=progress, objective=objective)
    reward, _ = _step(rewards.blank_ledger(1, N), events)
    torch.testing.assert_close(
        reward[..., LEVEL["objective_progress"]], CONFIG.objective_progress * progress
    )
    torch.testing.assert_close(reward[..., LEVEL["objective"]], CONFIG.objective * objective)


def test_a_restore_after_boundary_damage_heals_the_boundary_row_by_its_share():
    """Tick 1: ship 2 hits ship 0 for 30 and the boundary takes 10 from it (shield 40
    lost). Tick 2: ship 0 restores 20 of its 40 deficit, so every row is half paid back:
    ship 2's row 30 -> 15 (charged back 15), the boundary row 10 -> 5, whose 5 unattributed
    points charge team 1 (2.5 a ship)."""
    ledger = rewards.blank_ledger(1, N)
    _, ledger = _step(ledger, _events(hits=[(2, 0, 30.0)], boundary=[(0, 10.0)], lost=[(0, 40.0)]))
    assert float(ledger.ships[0, 2, 0]) == 30.0 and float(ledger.boundary[0, 0]) == 10.0
    reward, ledger = _step(ledger, _events(restored=[(0, 20.0)]))
    assert float(ledger.ships[0, 2, 0]) == pytest.approx(15.0)
    assert float(ledger.boundary[0, 0]) == pytest.approx(5.0)
    torch.testing.assert_close(reward[..., LEVEL["damage"]], _points([20.0, 0.0, -17.5, -2.5]))


def test_a_hit_and_a_restore_in_one_tick_share_the_ticks_ledger_total():
    """Ship 2 hits ship 0 for 30 and ship 0 restores 10 in the same tick: the deficit is
    this tick's 30, so a third of ship 2's new row is paid back (30 -> 20, charged 10)."""
    reward, ledger = _step(
        rewards.blank_ledger(1, N),
        _events(hits=[(2, 0, 30.0)], lost=[(0, 30.0)], restored=[(0, 10.0)]),
    )
    assert float(ledger.ships[0, 2, 0]) == pytest.approx(20.0)
    # ship 0: -30 taken + 10 recharged; ship 2: +30 dealt - 10 charged back
    torch.testing.assert_close(reward[..., LEVEL["damage"]], _points([-20.0, 0.0, 20.0, 0.0]))


def test_a_kill_is_shared_by_the_ledger_including_the_killing_ticks_damage():
    """Ship 3 has 20 outstanding on ship 0; ship 2's 30 kills it. Half the kill goes by the
    final tick's hits (all ship 2's), half by the ledger with this tick's damage counted
    (20 and 30 of 50): ship 2 takes 0.5 + 0.3, ship 3 0.2."""
    ledger = rewards.blank_ledger(1, N)
    _, ledger = _step(ledger, _events(hits=[(3, 0, 20.0)], lost=[(0, 20.0)]))
    reward, _ = _step(ledger, _events(hits=[(2, 0, 30.0)], lost=[(0, 30.0)], died=[0]))
    want = torch.tensor([[-1.0, 0.0, 0.8, 0.2]]) * CONFIG.kill_death
    torch.testing.assert_close(reward[..., LEVEL["kill_death"]], want)
