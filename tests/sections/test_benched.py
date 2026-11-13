"""Benched ships (unequal teams at one static size, `bnb.game.physics.step`): with every ship
active the game is bit-identical to the game without the input; a benched ship never acts,
flies, fires, deals or takes damage, scores, spawns or is seen by the enemy, and its
observation rows are a dead ship's waiting to respawn; team deathmatch's winning score is
kills_per_ship times the enemy's active ships."""

import dataclasses

import modes_cases as cases
import pytest
import torch
from torch.utils import _pytree

from bnb.contracts import Pool
from bnb.contracts.obs import GLOBAL, SHIP, SHIP_HIDDEN_ZERO
from bnb.game import api
from bnb.game.modes import observation, phase
from bnb.game.modes.rules import ModeRules
from bnb.game.state import benched_of, by_team

GAMES = ("team_deathmatch", "grouped", "frontline", "sandbox")
BATCH = 8
TICKS = 250


def _active(ships_per_team: int, batch: int, seed: int) -> torch.Tensor:
    """(B, 2) int32 active counts from 1 to S, drawn; the first environment fully active."""
    generator = torch.Generator().manual_seed(seed)
    active = torch.randint(1, ships_per_team + 1, (batch, 2), generator=generator)
    active[0] = ships_per_team
    return active.to(torch.int32)


def _play(rules, batch: int, active: torch.Tensor | None, ticks: int = TICKS, seed: int = 0):
    """Lively random play with perception, every ship shooting often: yields (before,
    events, after, observation) each tick."""
    generator = torch.Generator().manual_seed(seed)
    types = cases.LIVELY_TYPES
    state = api.reset(
        rules, types, api.uniforms(rules, batch, generator=generator), None, active=active
    )
    state, obs, _ = api.perceive(rules, state, active)
    for _ in range(ticks):
        action = cases.random_actions(batch, rules.num_ships, generator, shoot=0.9)
        spread = api.spread(rules, batch, generator=generator)
        uniforms = api.uniforms(rules, batch, generator=generator)
        after, events = api.step(rules, types, state, action, spread, uniforms, active)
        after, obs_after, _ = api.perceive(rules, after, active)
        yield state, events, after, obs_after
        state = after


def _leaves(*trees) -> list[torch.Tensor]:
    return [x for x in _pytree.tree_leaves(trees) if isinstance(x, torch.Tensor)]


def _mixed_rules():
    pools = tuple(
        Pool(
            name,
            0.25,
            ("learner", "learner"),
            mode=game.mode,
            arena_radius=game.arena_radius,
            spawn_groups=game.spawn_groups,
            episode_seconds=game.episode_seconds,
            respawn_seconds=game.respawn_seconds,
        )
        for name, game in ((name, cases.MODES[name]) for name in GAMES)
    )
    return cases.lively_edits(
        api.rules_of(cases.MODES["team_deathmatch"], pools, BATCH, **cases.LIVELY_RULES)
    )


@pytest.mark.parametrize("name", [*GAMES, "mixed"])
def test_every_ship_active_is_bit_identical_to_the_game_without_the_input(name):
    rules = _mixed_rules() if name == "mixed" else cases.lively(cases.MODES[name])
    full = torch.full((BATCH, 2), rules.ships_per_team, dtype=torch.int32)
    without, with_full = _play(rules, BATCH, None), _play(rules, BATCH, full)
    deaths = resets = 0
    for (_, events, after, obs), (_, events_full, after_full, obs_full) in zip(
        without, with_full, strict=True
    ):
        deaths += int(events.died.sum())
        resets += int(events.done.sum())
        for one, two in zip(
            _leaves(events, after, obs), _leaves(events_full, after_full, obs_full), strict=True
        ):
            assert torch.equal(one, two)
    assert deaths > 0 and resets > 0  # the trajectories hold kills and episode ends


@pytest.mark.parametrize("name", [*GAMES, "mixed"])
def test_a_benched_ship_never_plays(name):
    rules = _mixed_rules() if name == "mixed" else cases.lively(cases.MODES[name])
    active = _active(rules.ships_per_team, BATCH, seed=1)
    benched = benched_of(rules.ships_per_team, active)
    playing = ~benched
    assert benched.any() and playing.all(-1).any()
    fired = resets = 0
    for before, events, after, _ in _play(rules, BATCH, active):
        assert not (after.ships.alive & benched).any()
        assert not (events.spawned & benched).any() and not (events.fired & benched).any()
        assert not (events.died & benched).any()
        assert not ((events.damage != 0) & benched[:, :, None]).any()  # deals none
        assert not ((events.damage != 0) & benched[:, None, :]).any()  # takes none
        assert not (events.boundary_damage[benched] != 0).any()
        killers = events.killer.clamp_min(0).long()
        assert not (benched.gather(1, killers) & (events.killer >= 0)).any()
        assert not ((after.bullets.moves_left > 0) & benched[..., None]).any()
        # it does not fly: its values change only when an episode's reset places it again
        kept = benched & ~events.done[:, None]
        assert torch.equal(after.ships.x[kept], before.ships.x[kept])
        assert torch.equal(after.ships.y[kept], before.ships.y[kept])
        assert torch.equal(after.ships.respawn_wait[benched], before.ships.respawn_wait[benched])
        fired += int((events.fired & playing).sum())
        resets += int(events.done.sum())
    assert fired > 0 and resets > 0


def test_a_benched_ship_gives_the_enemy_no_score():
    rules = cases.lively(cases.MODES["team_deathmatch"])
    active = _active(rules.ships_per_team, BATCH, seed=2)
    benched = by_team(benched_of(rules.ships_per_team, active))  # (B, 2, S)
    scored = 0
    for _, events, after, _ in _play(rules, BATCH, active):
        deaths = by_team(events.died).sum(-1)  # (B, 2) by the dying team
        scored += int(deaths.sum())
        assert not (by_team(events.died) & benched).any()
        assert (after.mode.score >= 0).all()
    assert scored > 0


@pytest.mark.parametrize("name", ["team_deathmatch", "frontline"])
def test_a_benched_ships_rows_are_a_dead_ship_waiting_to_respawn(name):
    """In every view and every tick: alive 0, fired 0, spawned 0, respawn_wait held at the
    full respawn length; the enemy sees nothing of it (visible 0, its hidden columns zero),
    and its own team sees it as it sees its own dead ships (visible, no accelerations)."""
    rules = cases.lively(cases.MODES[name])
    active = _active(rules.ships_per_team, BATCH, seed=3)
    benched = benched_of(rules.ships_per_team, active)  # (B, N) sim order
    order = api.view_order(rules.ships_per_team, active.device)  # (2, N) sim index per row
    rows = benched[:, order]  # (B, 2, N) by view row
    own = (api.ship_team(rules.ships_per_team)[order] == torch.arange(2)[:, None])[None]
    respawn = cases.MODES[name].respawn_seconds
    dead_own_seen = False
    for _, _, after, obs in _play(rules, BATCH, active, ticks=120):
        ships = obs.ships  # (B, 2, N, columns)
        benched_rows = ships[rows]
        assert (benched_rows[:, SHIP["alive"]] == 0).all()
        assert (benched_rows[:, SHIP["fired"]] == 0).all()
        assert (benched_rows[:, SHIP["spawned"]] == 0).all()
        assert (benched_rows[:, SHIP["acceleration_valid"]] == 0).all()
        assert torch.allclose(benched_rows[:, SHIP["respawn_wait"]], torch.tensor(respawn))
        enemy_rows = ships[rows & ~own]
        assert (enemy_rows[:, SHIP["visible"]] == 0).all()
        assert (enemy_rows[:, :SHIP_HIDDEN_ZERO] == 0).all()
        assert (ships[rows & own][:, SHIP["visible"]] == 1).all()
        # an ordinary dead ship of the viewer's own team shows the same flags
        dead = (ships[..., SHIP["alive"]] == 0) & ~rows & own
        if dead.any():
            dead_own_seen = True
            assert (ships[dead][:, SHIP["visible"]] == 1).all()
    assert dead_own_seen


def test_the_deathmatch_target_is_kills_per_ship_times_the_enemys_active_ships():
    """A 5-active team against a 3-active one: the first wins at 15 kills, the second at 25."""
    config = dataclasses.replace(cases.MODES["team_deathmatch"], ships_per_team=5)
    rules = api.rules_of(config)
    mode_rules = ModeRules("team_deathmatch", kills_per_ship=5)
    state = cases.rules_and_state(config, batch=4)[1]
    active = torch.tensor([[5, 3]] * 4, dtype=torch.int32)
    # environments: team 0 one kill short; team 1 one kill short; team 1 at 16 (not 25);
    # team 0 at 14 (not 15)
    score = torch.tensor([[14, 0], [0, 24], [0, 15], [13, 0]], dtype=torch.int32)
    mode = dataclasses.replace(state.mode, score=score)
    died = torch.zeros((4, 10), dtype=torch.bool)
    died[0, 5] = died[1, 0] = died[2, 0] = died[3, 5] = True  # one death each, the other side's
    alive = state.ships.alive & ~died
    result = phase.mode_phase(rules, mode_rules, mode, state.ships, alive, died, active)
    assert result.mode.score.tolist() == [[15, 0], [0, 25], [0, 16], [14, 0]]
    assert result.won.tolist() == [True, True, False, False]
    assert result.result.tolist() == [1, -1, 0, 0]
    globals_ = observation.mode_globals(rules, mode_rules, result.mode, active)  # (B, 2, ...)
    target = globals_[..., GLOBAL["score_target"] - GLOBAL["mode_team_deathmatch"]]
    assert target.tolist() == [[15.0, 25.0]] * 4
    # every ship active: both at 25, as without the input
    full = torch.full_like(active, 5)
    assert phase.deathmatch_targets(rules, mode_rules, full).tolist() == [[25, 25]] * 4
    assert phase.deathmatch_targets(rules, mode_rules, None) == 25


def test_a_5_against_3_game_ends_on_the_enemys_active_count():
    """Played out: every deathmatch game ended by its win rule ends with the winner at its
    own target, kills_per_ship times the enemy's active ships."""
    config = dataclasses.replace(cases.MODES["team_deathmatch"], kills_per_ship=1)
    rules = cases.lively(config)
    active = torch.tensor([[3, 1], [1, 3]] * (BATCH // 2), dtype=torch.int32)
    targets = active.flip(-1)  # kills_per_ship 1
    won = 0
    for before, events, _, _ in _play(rules, BATCH, active, ticks=400):
        score = before.mode.score + by_team(events.died).flip(1).sum(-1)  # after this tick
        reached = score >= targets
        assert torch.equal(events.done & reached.any(-1), reached.any(-1))  # a win ends it
        by_rule = events.done & reached.any(-1)
        expected = reached[:, 0].int() - reached[:, 1].int()
        assert torch.equal(events.result[by_rule], expected[by_rule])
        clock = events.done & ~reached.any(-1)
        assert torch.equal(events.result[clock], torch.sign(score[:, 0] - score[:, 1])[clock].int())
        won += int(by_rule.sum())
    assert won > 0


@pytest.mark.parametrize("name", ["team_deathmatch", "frontline"])
def test_the_benched_flag_marks_exactly_the_benched_slots_in_both_views_under_fog(name):
    """4v5, 3v5, 5v4 and 5v3 games under fog: from the reset on, through the active ships'
    deaths and respawns and the episode resets, the benched column is 1 on exactly the
    benched rows of both views (the enemy's included, which it cannot see) and of the
    truth; with every ship active it is 0 everywhere."""
    config = dataclasses.replace(cases.MODES[name], ships_per_team=5)
    rules = cases.lively(config)
    active = torch.tensor([[4, 5], [3, 5], [5, 4], [5, 3]] * 2, dtype=torch.int32)
    benched = benched_of(rules.ships_per_team, active)  # (B, N) sim order
    order = api.view_order(rules.ships_per_team, active.device)
    rows = benched[:, order].float()  # (B, 2, N) by view row
    own = (api.ship_team(rules.ships_per_team)[order] == torch.arange(2)[:, None])[None]
    generator = torch.Generator().manual_seed(4)
    types = cases.LIVELY_TYPES
    fogged = torch.ones(BATCH, dtype=torch.bool)
    state = api.reset(
        rules, types, api.uniforms(rules, BATCH, generator=generator), fogged, active=active
    )
    state, obs, truth = api.perceive(rules, state, active)
    respawned = resets = hidden_enemy = 0
    for _ in range(TICKS):
        assert torch.equal(obs.ships[..., SHIP["benched"]], rows)
        assert torch.equal(truth.ships[..., SHIP["benched"]], rows)
        hidden_enemy += int(((obs.ships[..., SHIP["visible"]] == 0) & (rows > 0) & ~own).sum())
        action = cases.random_actions(BATCH, rules.num_ships, generator, shoot=0.9)
        spread = api.spread(rules, BATCH, generator=generator)
        uniforms = api.uniforms(rules, BATCH, generator=generator)
        state, events = api.step(rules, types, state, action, spread, uniforms, active)
        state, obs, truth = api.perceive(rules, state, active)
        respawned += int((events.spawned & ~events.done[:, None]).sum())
        resets += int(events.done.sum())
    assert respawned > 0 and resets > 0 and hidden_enemy > 0
    _, full, full_truth = api.perceive(rules, state)
    assert not full.ships[..., SHIP["benched"]].any()
    assert not full_truth.ships[..., SHIP["benched"]].any()
