"""v2's reward levels as lean computes them (v2 src/bnb/learn/rewards.py at 4cfd1e86), and
the value supports' range monitoring."""

import pytest
import torch

from lean import game, scripted
from lean.learn import LearnConfig, value_centres, value_knees
from lean.learn.config import DAMAGE, KILL_DEATH, LEVELS, OBJECTIVE, OBJECTIVE_PROGRESS, OUTCOME
from lean.learn.returns import usage_counts, usage_flags, usage_metrics
from lean.learn.rewards import TickEvents, blank_ledger, reward_step, tick_events

CONFIG = LearnConfig()
KILL = CONFIG.reward_weights[KILL_DEATH]
DAMAGE_POINT = CONFIG.reward_weights[DAMAGE] / CONFIG.damage_unit
# 2v2 in one environment: slots 0 and 1 are team 0, slots 2 and 3 team 1.
TEAM = torch.tensor([[0, 0, 1, 1]])


def events(
    raw=None, boundary=None, lost=None, restored=None, died=(), done=False, result=-1, present=None
):
    """One tick's events of the 2v2 environment; raw as {(attacker, target): damage}."""
    matrix = torch.zeros(1, 4, 4)
    for (attacker, target), amount in (raw or {}).items():
        matrix[0, attacker, target] = amount

    def per_ship(values):
        tensor = torch.zeros(1, 4)
        for slot, amount in (values or {}).items():
            tensor[0, slot] = amount
        return tensor

    dead = torch.zeros(1, 4, dtype=torch.bool)
    dead[0, list(died)] = True
    win_loss = torch.tensor([[float(result == 0), float(result == 1)]])
    return TickEvents(
        raw=matrix,
        boundary=per_ship(boundary),
        shield_lost=per_ship(lost),
        shield_restored=per_ship(restored),
        died=dead,
        done=torch.tensor([done]),
        win_loss=win_loss,
        present=torch.ones(1, 4, dtype=torch.bool) if present is None else present,
    )


def test_damage_and_kill_death_are_zero_sum_with_the_ledger_and_unequal_teams():
    """Guards v2's pairing: every amount one team earns is charged to the other (damage
    dealt and taken, recharge and charge-back, friendly fire and its refund, boundary
    losses, kills, ally kills, boundary deaths), over ticks that carry the ledger, with a
    benched slot (3v2 in five slots, the fifth out of the match)."""
    generator = torch.Generator().manual_seed(0)
    envs, slots = 64, 6
    team = torch.tensor([0, 0, 0, 1, 1, 1]).expand(envs, slots)
    present = torch.tensor([True, True, True, True, True, False]).expand(envs, slots)
    ledger = blank_ledger(envs, slots)
    for _ in range(20):
        raw = torch.rand(envs, slots, slots, generator=generator) * (
            torch.rand(envs, slots, slots, generator=generator) < 0.2
        )
        raw = raw * ~torch.eye(slots, dtype=torch.bool) * present[:, :, None] * present[:, None, :]
        boundary = (
            torch.rand(envs, slots, generator=generator)
            * (torch.rand(envs, slots, generator=generator) < 0.2)
            * present
        )
        total = raw.sum(1) + boundary
        lost = total * torch.rand(envs, slots, generator=generator)
        restored = torch.rand(envs, slots, generator=generator) * (total == 0) * present
        died = (torch.rand(envs, slots, generator=generator) < 0.05) & present
        done = torch.rand(envs, generator=generator) < 0.05
        tick = TickEvents(
            raw * 50,
            boundary * 50,
            lost * 50,
            restored * 20,
            died,
            done,
            torch.zeros(envs, 2),
            present,
        )
        reward, ledger = reward_step(CONFIG, ledger, tick, team)
        for level in (DAMAGE, KILL_DEATH):
            assert reward[..., level].sum(-1).abs().max() < 1e-4
        assert (reward[:, 5] == 0).all()  # the benched slot is rewarded nothing


def test_outcome_pays_every_ship_of_both_teams_on_the_last_tick():
    """Guards v2's outcome: +1 to the winners and -1 to the losers times the weight, the
    dead included, only on the tick the match ends; a draw pays nothing; the teams' sums
    cancel when the teams are equal."""
    ledger = blank_ledger(1, 4)
    won, _ = reward_step(CONFIG, ledger, events(done=True, result=1, died=(3,)), TEAM)
    assert torch.equal(
        won[0, :, OUTCOME], torch.tensor([-1.0, -1.0, 1.0, 1.0]) * CONFIG.reward_weights[OUTCOME]
    )
    assert won[0, :, OUTCOME].sum() == 0
    drawn, _ = reward_step(CONFIG, ledger, events(done=True, result=2), TEAM)
    assert (drawn[..., OUTCOME] == 0).all()
    running, _ = reward_step(CONFIG, ledger, events(done=False, result=-1), TEAM)
    assert (running[..., OUTCOME] == 0).all()
    absent = torch.tensor([[True, True, True, False]])
    benched, _ = reward_step(CONFIG, ledger, events(done=True, result=0, present=absent), TEAM)
    assert benched[0, 3, OUTCOME] == 0 and benched[0, 2, OUTCOME] == -CONFIG.reward_weights[OUTCOME]


def test_a_kill_is_shared_by_the_final_hit_and_the_outstanding_damage():
    """Guards v2's kill credit: half by the killing tick's raw hits, half by the ledger's
    outstanding damage, so an earlier attacker gets an assist; the dead ship pays one death."""
    ledger = blank_ledger(1, 4)
    first, ledger = reward_step(CONFIG, ledger, events(raw={(0, 2): 40.0}, lost={2: 40.0}), TEAM)
    assert torch.allclose(first[0, :, DAMAGE], torch.tensor([40.0, 0.0, -40.0, 0.0]) * DAMAGE_POINT)
    assert ledger.ships[0, 0, 2] == 40.0
    # Ship 2's shield is empty: ship 1's hit kills it and takes no shield.
    kill, ledger = reward_step(CONFIG, ledger, events(raw={(1, 2): 30.0}, died=(2,)), TEAM)
    assert torch.allclose(kill[0, :, KILL_DEATH], torch.tensor([0.5, 0.5, -1.0, 0.0]) * KILL)
    assert (ledger.ships[0, :, 2] == 0).all()  # the dead ship's column is cleared


def test_the_last_hit_takes_the_whole_kill_when_nothing_is_outstanding():
    ledger = blank_ledger(1, 4)
    kill, _ = reward_step(CONFIG, ledger, events(raw={(1, 2): 30.0}, died=(2,)), TEAM)
    assert torch.allclose(kill[0, :, KILL_DEATH], torch.tensor([0.0, 1.0, -1.0, 0.0]) * KILL)


def test_an_ally_kill_blames_the_shooter_and_pays_the_other_team_twice():
    ledger = blank_ledger(1, 4)
    kill, _ = reward_step(CONFIG, ledger, events(raw={(0, 1): 30.0}, died=(1,)), TEAM)
    # Ship 1 dies (-1), ship 0 is blamed (-1), the other team shares +2.
    assert torch.allclose(kill[0, :, KILL_DEATH], torch.tensor([-1.0, -1.0, 1.0, 1.0]) * KILL)


def test_a_boundary_death_pays_the_other_team():
    ledger = blank_ledger(1, 4)
    death, _ = reward_step(CONFIG, ledger, events(boundary={2: 5.0}, died=(2,)), TEAM)
    assert torch.allclose(death[0, :, KILL_DEATH], torch.tensor([0.5, 0.5, -1.0, 0.0]) * KILL)


def test_recharge_refunds_the_target_and_charges_back_the_attacker():
    """Guards v2's damage ledger: healed damage is taken back from whoever dealt it."""
    ledger = blank_ledger(1, 4)
    _, ledger = reward_step(CONFIG, ledger, events(raw={(0, 2): 50.0}, lost={2: 50.0}), TEAM)
    heal, ledger = reward_step(CONFIG, ledger, events(restored={2: 20.0}), TEAM)
    assert torch.allclose(heal[0, :, DAMAGE], torch.tensor([-20.0, 0.0, 20.0, 0.0]) * DAMAGE_POINT)
    assert torch.isclose(ledger.ships[0, 0, 2], torch.tensor(30.0))


def test_friendly_fire_blames_the_shooter_and_pays_the_other_team_twice():
    ledger = blank_ledger(1, 4)
    hit, _ = reward_step(CONFIG, ledger, events(raw={(0, 1): 20.0}, lost={1: 20.0}), TEAM)
    assert torch.allclose(
        hit[0, :, DAMAGE], torch.tensor([-20.0, -20.0, 20.0, 20.0]) * DAMAGE_POINT
    )


def test_objective_levels_are_kept_and_zero_in_elimination():
    ledger = blank_ledger(1, 4)
    reward, _ = reward_step(
        CONFIG, ledger, events(raw={(0, 2): 50.0}, lost={2: 50.0}, done=True, result=0), TEAM
    )
    assert reward.shape[-1] == LEVELS
    assert (reward[..., OBJECTIVE_PROGRESS] == 0).all() and (reward[..., OBJECTIVE] == 0).all()


def test_the_real_game_gives_zero_sum_levels():
    """Guards tick_events, the bridge from lean.game's StepOut: with the scripted agent on
    both teams the damage and kill/death levels still cancel over the ships every tick."""
    generator = torch.Generator().manual_seed(2)
    game_config = game.GameConfig(ghost_bullets=True)
    envs = 8
    state = game.reset(game_config, envs, torch.full((envs, 2), 2), generator, "cpu")
    slots = state.exists.shape[1]
    scripted_config = scripted.ScriptedConfig()
    persona = scripted.personas(scripted_config, envs, slots, generator, "cpu", 0.0)
    memory = scripted.fresh_memory(envs, slots, generator, "cpu")
    ledger = blank_ledger(envs, slots)
    paid = torch.zeros(LEVELS)
    for _ in range(300):
        uniforms = scripted.draw_uniforms(envs, slots, generator, "cpu")
        action, memory = scripted.act(scripted_config, state, None, persona, uniforms, memory)
        next_state, step_out = game.step(
            game_config, state, action, game.draw_noise(game_config, envs, slots, generator, "cpu")
        )
        reward, ledger = reward_step(CONFIG, ledger, tick_events(step_out, state), state.team)
        assert reward[..., DAMAGE].sum(-1).abs().max() < 1e-4
        assert reward[..., KILL_DEATH].sum(-1).abs().max() < 1e-4
        paid += reward.abs().sum((0, 1))
        fresh = game.deal(game_config, envs, slots, torch.full((envs, 2), 2), generator, "cpu")
        state = game.take_deal(next_state, step_out.done, fresh)
    assert paid[DAMAGE] > 0  # the scripted agents fought


def test_range_monitoring_flags_clamped_and_narrow_targets():
    """Guards the supports' watch: a target beyond the outermost centres counts as
    clamped and raises the warning; targets in a few middle bins raise the narrow one,
    except on a level that paid no reward."""
    config = LearnConfig()
    centres = value_centres(config, 63)
    mask = torch.ones(8, 2, 3, dtype=torch.bool)
    reward = torch.zeros(8, 2, 3, LEVELS)
    reward[..., DAMAGE] = 1.0
    reward[..., KILL_DEATH] = 1.0
    reward[..., OUTCOME] = 1.0
    spread = torch.linspace(0.0, 1.0, 48).view(8, 2, 3)
    target = torch.zeros(8, 2, 3, LEVELS)
    for level in (DAMAGE, KILL_DEATH):  # spread over the support, inside it
        target[..., level] = config.value_low[level] + spread * (
            config.value_high[level] - config.value_low[level]
        )
    target[0, 0, 0, KILL_DEATH] = 10 * config.value_high[KILL_DEATH]  # one far outside: 1 in 48
    target[..., OUTCOME] = 0.01 * spread  # all in the middle bin or two
    target[..., OBJECTIVE] = 0.01 * spread  # narrow, but the level paid nothing
    paid = (reward != 0).flatten(0, -2).any(0)  # as the update reads it from the rollout
    counts = usage_counts(target, mask, centres, value_knees(config))
    usage = usage_metrics(usage_flags(counts, paid, config).tolist())
    assert usage["range_clamped_kill_death"] == pytest.approx(1 / 48)
    assert usage["range_warning_clamped_kill_death"] == 1.0
    assert usage["range_clamped_damage"] == 0.0 and usage["range_warning_clamped_damage"] == 0.0
    assert usage["range_warning_narrow_damage"] == 0.0
    assert usage["range_warning_narrow_outcome"] == 1.0
    assert usage["range_warning_narrow_objective"] == 0.0
