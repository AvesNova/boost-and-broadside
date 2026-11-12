"""Rewards (rewards spec 5): zero-sum levels and ledger conservation on real play in every
mode, the scenario tests, the rollout's routing, and the step compiled."""

import modes_cases as cases
import pytest
import torch

from bnb.contracts import REWARD_LEVELS, Events, GameConfig, Pool
from bnb.game import api
from bnb.game.ship_types import per_ship
from bnb.learn import rewards

# The code under test draws from torch's default generator (model initialisation, the
# rollout, the scripted opponents): each test runs with it seeded and restored.
pytestmark = pytest.mark.usefixtures("global_generator")


CONFIG = rewards.RewardConfig()
LEVEL = {name: i for i, name in enumerate(REWARD_LEVELS)}
WEIGHTS = torch.tensor(CONFIG.weights)


def test_levels_and_weights():
    assert REWARD_LEVELS == ("damage", "kill_death", "objective_progress", "objective", "outcome")
    assert CONFIG.weights == (2.74, 2.83, 20.0, 20.0, 10.0) and CONFIG.damage_unit == 100.0


# ------------------------------------------------------------------ real play


def _check_conservation(ledger, state, types):
    """Each living ship's column (its ship rows plus its boundary entry) sums to its shield
    deficit within 1e-3 points; a dead ship's column was cleared at its death."""
    ships = state.ships
    deficit = per_ship(types.max_shield, ships.ship_type) - ships.shield
    column = ledger.ships.sum(1) + ledger.boundary
    assert (column[ships.alive] - deficit[ships.alive]).abs().max() <= 1e-3
    assert (column[~ships.alive] == 0).all()


def _check_tick(levels, ledger, after, types, scale_floor=1e-6):
    """Every level sums to zero over every environment (float32, relative to the tick's
    largest reward), and the ledger holds every deficit."""
    total = levels.sum(1)  # (B, 5)
    largest = levels.abs().amax((1, 2), keepdim=True)[:, 0].clamp_min(scale_floor)
    assert (total.abs() <= 1e-5 * largest).all(), (total.abs() / largest).max()
    _check_conservation(ledger, after, types)


def _largest(ledger) -> float:
    return max(float(ledger.ships.abs().max()), float(ledger.boundary.abs().max()))


@pytest.mark.parametrize("mode", list(cases.MODES))
def test_levels_are_zero_sum_and_the_ledger_holds_every_deficit(mode):
    config = cases.MODES[mode]
    batch = 16
    ledger = rewards.blank_ledger(batch, config.num_ships)
    team = torch.arange(config.num_ships) // config.ships_per_team
    seen = dict(friendly=0, boundary=0, respawns=0, resets=0, recharge=0, kills=0)
    for _, events, after in cases.play(config, batch=batch, ticks=300, seed=5):
        levels, ledger = rewards.reward_step(CONFIG, ledger, events, team)
        _check_tick(levels, ledger, after, cases.LIVELY_TYPES)
        assert (levels[..., LEVEL["outcome"]][~events.done] == 0).all()
        torch.testing.assert_close(levels[..., LEVEL["objective"]], 2.0 * events.objective)
        killer_team = torch.where(events.killer >= 0, events.killer // config.ships_per_team, -1)
        seen["friendly"] += int((events.died & (killer_team == team)).sum())
        seen["kills"] += int((events.died & (killer_team >= 0) & (killer_team != team)).sum())
        seen["boundary"] += int((events.died & (events.killer < 0)).sum())
        seen["respawns"] += int((after.ships.spawned & ~events.done[:, None]).sum())
        seen["resets"] += int(events.done.sum())
        seen["recharge"] += int((events.shield_restored > 0).sum())
    # every kind of death, in every mode (Frontline included), plus respawns (none in
    # elimination) and resets
    if mode == "elimination":
        assert seen.pop("respawns") == 0, seen
    assert all(count > 0 for count in seen.values()), seen


# ------------------------------------------------------------------ scenarios


class Match:
    """Synthetic ticks for a few ships: events built from the amounts given, the ledger
    carried, the rewards summed per ship and level."""

    def __init__(self, ships: int) -> None:
        self.ships = ships
        self.team = torch.arange(ships) // (ships // 2)
        self.ledger = rewards.blank_ledger(1, ships)
        self.total = torch.zeros(ships, len(REWARD_LEVELS))

    def tick(self, hits=(), boundary=(), lost=(), restored=(), died=(), done=False, result=0):
        """hits: (shooter, target, raw); boundary, lost, restored: (ship, amount); died: ships."""
        N = self.ships
        damage = torch.zeros(1, N, N)
        for shooter, target, raw in hits:
            damage[0, shooter, target] += raw
        vectors = {}
        for name, pairs in (("boundary", boundary), ("lost", lost), ("restored", restored)):
            vector = torch.zeros(1, N)
            for ship, amount in pairs:
                vector[0, ship] = amount
            vectors[name] = vector
        dead = torch.zeros(1, N, dtype=torch.bool)
        for ship in died:
            dead[0, ship] = True
        zero = torch.zeros(1, N)
        events = Events(
            damage=damage,
            boundary_damage=vectors["boundary"],
            died=dead,
            fired=dead & False,
            spawned=dead & False,
            killer=torch.full((1, N), -1, dtype=torch.int32),
            done=torch.tensor([done]),
            shield_lost=vectors["lost"],
            shield_restored=vectors["restored"],
            objective_progress=zero,
            objective=zero,
            result=torch.tensor([result], dtype=torch.int32),
            enemy_damage_by_aspect=torch.zeros(1, N, 3),
            kill_aspect=torch.full((1, N), -1, dtype=torch.int32),
        )
        levels, self.ledger = rewards.reward_step(CONFIG, self.ledger, events, self.team)
        assert levels.sum(1).abs().max() < 1e-6, "zero-sum"
        self.total += levels[0]
        return levels[0]

    def level(self, name: str) -> torch.Tensor:
        """The level's summed events per ship, before the weight."""
        return self.total[:, LEVEL[name]] / CONFIG.weights[LEVEL[name]]


def test_a_healed_duel_nets_zero_damage_reward():
    """A takes B to 10 of 100 shield; B recharges fully: both damage levels net 0."""
    match = Match(2)
    for _ in range(9):
        match.tick(hits=[(0, 1, 10.0)], lost=[(1, 10.0)])
    assert match.level("damage")[0] > 0.89
    for _ in range(9):
        match.tick(restored=[(1, 10.0)])
    torch.testing.assert_close(match.level("damage"), torch.zeros(2), atol=1e-6, rtol=0)
    assert _largest(match.ledger) < 1e-5


def test_a_kill_is_shared_by_the_final_hits_and_the_outstanding_damage():
    """Ship 0 (team 0) is emptied by ship 2 (60) and its ally ship 1 (40), then killed by
    ship 3's hit. Final hits: ship 3 half; outstanding: ship 2 0.3, ally 1 0.2, blamed and
    paid twice to team 1."""
    match = Match(4)
    match.tick(hits=[(2, 0, 60.0), (1, 0, 40.0)], lost=[(0, 100.0)])
    damage = match.level("damage").clone()
    torch.testing.assert_close(damage, torch.tensor([-1.0, -0.4, 0.6 + 0.4, 0.4]))
    match.tick(hits=[(3, 0, 12.0)], died=[0])
    torch.testing.assert_close(match.level("kill_death"), torch.tensor([-1.0, -0.2, 0.5, 0.7]))
    assert match.ledger.ships[0, :, 0].abs().max() == 0  # the dead ship's column is cleared
    assert match.ledger.boundary[0, 0] == 0


def test_a_boundary_death_pays_the_other_team_and_credits_no_attacker():
    match = Match(4)
    match.tick(hits=[(2, 0, 60.0)], boundary=[(0, 40.0)], lost=[(0, 100.0)])
    match.tick(boundary=[(0, 1.0)], died=[0])
    torch.testing.assert_close(match.level("kill_death"), torch.tensor([-1.0, 0.0, 0.5, 0.5]))
    # the boundary's share of the deficit paid team 1 when it was taken
    torch.testing.assert_close(match.level("damage"), torch.tensor([-1.0, 0.0, 0.6 + 0.2, 0.2]))


@pytest.mark.parametrize("then", ["dies", "heals"])
def test_a_dead_attacker_keeps_its_claims(then):
    """A (0) damages B (2) by 30 and dies (boundary) and respawns. Then B dies to C (1),
    and A gets its outstanding share; or B heals, and A is charged back."""
    match = Match(4)
    match.tick(hits=[(0, 2, 30.0)], lost=[(2, 30.0)])
    match.tick(boundary=[(0, 1.0)], died=[0])  # A had no shield left: an exposed core
    damage_before = match.level("damage")[0].item()
    kill_before = match.level("kill_death")[0].item()
    if then == "dies":
        match.tick(hits=[(1, 2, 70.0)], lost=[(2, 70.0)])
        match.tick(hits=[(1, 2, 12.0)], died=[2])
        assert abs(match.level("kill_death")[0].item() - kill_before - 0.15) < 1e-6
        assert abs(match.level("kill_death")[1].item() - 0.85) < 1e-6
    else:
        match.tick(restored=[(2, 30.0)])
        assert abs(match.level("damage")[0].item() - damage_before + 0.3) < 1e-6


def test_damage_to_a_respawned_target_starts_afresh():
    """B (2) is emptied by A (0) and dies; after it respawns, C (1) damages it. Its healing
    charges back only C, and its next death credits only C."""
    match = Match(4)
    match.tick(hits=[(0, 2, 100.0)], lost=[(2, 100.0)])
    match.tick(hits=[(0, 2, 12.0)], died=[2])
    a_damage, a_kill = match.level("damage")[0].item(), match.level("kill_death")[0].item()
    match.tick(hits=[(1, 2, 20.0)], lost=[(2, 20.0)])
    match.tick(restored=[(2, 20.0)])
    assert match.level("damage")[0].item() == a_damage
    torch.testing.assert_close(match.level("damage")[1], torch.tensor(0.0), atol=1e-6, rtol=0)
    match.tick(hits=[(1, 2, 100.0)], lost=[(2, 100.0)])
    match.tick(hits=[(1, 2, 12.0)], died=[2])
    assert match.level("kill_death")[0].item() == a_kill
    assert abs(match.level("kill_death")[1].item() - 1.0) < 1e-6


def test_the_outcome_pays_every_ship_of_both_teams():
    match = Match(4)
    levels = match.tick(done=True, result=-1)
    paid = CONFIG.outcome * torch.tensor([-1.0, -1.0, 1.0, 1.0])
    torch.testing.assert_close(levels[:, LEVEL["outcome"]], paid)
    match.tick(hits=[(0, 2, 10.0)], lost=[(2, 10.0)], done=True, result=0)
    assert _largest(match.ledger) == 0  # a finished match clears the ledger


# ------------------------------------------------------------------ the rollout


@pytest.mark.parametrize("mode", ["team_deathmatch", "frontline"])
def test_rollout_stores_every_level_zero_sum_per_environment(mode):
    from bnb.controllers.scripted import Random
    from bnb.runtime.rollout import Rollout

    config = cases.MODES[mode]
    rollout = Rollout(
        config,
        api.ship_types(),
        (Pool("self_play", 1.0, ("r", "r")),),
        {"r": Random(config.ships_per_team)},
        "r",
        8,
        64,
        torch.device("cpu"),
        compiled=False,
    )
    trajectory = rollout.run()
    L, R, S, K = trajectory.reward.shape
    assert (R, S, K) == (16, config.ships_per_team, len(REWARD_LEVELS))
    environment = rollout.lineup.environments["r"]
    total = torch.zeros(L, 8, K).index_add_(1, environment, trajectory.reward.sum(2))
    largest = trajectory.reward.abs().amax((1, 2, 3)).clamp_min(1e-6)[:, None, None]
    assert (total.abs() <= 1e-5 * largest).all()
    assert trajectory.reward.abs().sum() > 0
    outcome = trajectory.reward[..., LEVEL["outcome"]]
    assert (outcome[~trajectory.done] == 0).all()


@pytest.mark.parametrize("mode", list(cases.MODES))
def test_the_rollouts_ledger_holds_every_deficit(mode):
    """Through the runtime: the rollout's carried ledger against its carried state, after
    each of 16 short runs (staggered clocks, so resets fall inside)."""
    from bnb.controllers.scripted import Chaser, Random
    from bnb.game.handling.characteristics import characteristics
    from bnb.runtime.rollout import Rollout

    config = cases.MODES[mode]
    S = config.ships_per_team
    rollout = Rollout(
        config,
        api.ship_types(),
        (Pool("random", 0.5, ("r", "r")), Pool("chaser", 0.5, ("c", "r"))),
        {"r": Random(S), "c": Chaser(characteristics(api.ship_types(), api.rules_of(config)), S)},
        "r",
        8,
        8,
        torch.device("cpu"),
        compiled=False,
    )
    held = 0.0
    for _ in range(16):
        rollout.run()
        carry = rollout.carry
        _check_conservation(carry["ledger"], carry["state"], carry["types"])
        held = max(held, float(carry["ledger"].ships.sum()))
    assert held > 0, "some damage should be outstanding at some point"


@pytest.mark.slow
def test_compiled_reward_step_matches_eager():
    config = cases.MODES["frontline"]
    team = torch.arange(config.num_ships) // config.ships_per_team
    step = torch.compile(rewards.reward_step, fullgraph=True, dynamic=False)
    ledger = eager = rewards.blank_ledger(8, config.num_ships)
    for _, events, _ in cases.play(config, batch=8, ticks=60, seed=6):
        got, ledger = step(CONFIG, ledger, events, team)
        want, eager = rewards.reward_step(CONFIG, eager, events, team)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(ledger.ships, eager.ships, rtol=1e-5, atol=1e-4)
        torch.testing.assert_close(ledger.boundary, eager.boundary, rtol=1e-5, atol=1e-4)
        ledger = eager


def test_a_bigger_game_runs_too():
    """The rewards do not depend on the sizes: a 50v50 tick of zero events is all zeros."""
    N = 100
    zero = torch.zeros(2, N)
    events = Events(
        damage=torch.zeros(2, N, N),
        boundary_damage=zero,
        died=zero.bool(),
        fired=zero.bool(),
        spawned=zero.bool(),
        killer=zero.int() - 1,
        done=torch.zeros(2, dtype=torch.bool),
        shield_lost=zero,
        shield_restored=zero,
        objective_progress=zero,
        objective=zero,
        result=torch.zeros(2).int(),
        enemy_damage_by_aspect=torch.zeros(2, N, 3),
        kill_aspect=zero.int() - 1,
    )
    levels, ledger = rewards.reward_step(
        CONFIG, rewards.blank_ledger(2, N), events, torch.arange(N) // 50
    )
    assert levels.shape == (2, N, 5) and not levels.any()
    assert ledger.ships.shape == (2, N, N) and ledger.boundary.shape == (2, N)
    assert GameConfig(50, 40, 8200.0).num_ships == N
