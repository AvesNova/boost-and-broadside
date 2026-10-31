"""Perception: views depend on team only through team_relative; fog of war per
environment."""

import math

import pytest
import torch

from bnb.contracts import Action, GameConfig, Pool
from bnb.contracts.obs import GLOBAL, SHIP, SHIP_HIDDEN_ZERO
from bnb.contracts.pools import pool_sizes
from bnb.game import api
from bnb.game.state import State, apply, replace

TYPES = api.ship_types()


def _game(ships_per_team=3, fields=6, radius=1800.0, batch=16, warm=60, fogged=None):
    rules = api.rules_of(GameConfig(ships_per_team, fields, radius))
    generator = torch.Generator().manual_seed(1)
    state = api.reset(rules, TYPES, api.uniforms(rules, batch, generator=generator), fogged)
    for _ in range(warm):  # some motion, firing and sight so the views are not trivial
        choice = torch.rand(batch, rules.num_ships, 3, generator=generator)
        action = Action.unpack((choice * torch.tensor([3.0, 7.0, 2.0])).to(torch.int32))
        state, _ = api.step(rules, TYPES, state, action,
                            api.spread(rules, batch, generator=generator),
                            api.uniforms(rules, batch, generator=generator))
        state, _, _ = api.perceive(rules, state)
    return rules, state


def _swap_teams(state: State, ships_per_team: int) -> State:
    """The same world with the two teams' sim indices exchanged."""
    def roll(values):
        return values.roll(ships_per_team, dims=1)

    since = state.match.ticks_since_seen.flip(1).roll(ships_per_team, dims=2)
    return replace(state, ships=apply(roll, state.ships), bullets=apply(roll, state.bullets),
                   match=replace(state.match, ticks_since_seen=since))


def test_views_swap_with_the_teams():
    rules, state = _game()
    _, obs, _ = api.perceive(rules, state)
    _, swapped, _ = api.perceive(rules, _swap_teams(state, rules.ships_per_team))
    for part in ("ships", "fields", "globals"):
        a, b = getattr(obs, part), getattr(swapped, part)
        assert torch.equal(a[:, 0], b[:, 1]) and torch.equal(a[:, 1], b[:, 0]), part


def test_own_rows_first_and_hidden_rows_blank():
    rules, state = _game()
    _, obs, truth = api.perceive(rules, state)
    S = rules.ships_per_team
    relative = obs.ships[..., SHIP["team_relative"]]
    assert (relative[..., :S] == 0).all() and (relative[..., S:] == 1).all()
    assert (obs.ships[..., :S, SHIP["visible"]] == 1).all()  # a team always sees itself
    hidden = obs.ships[..., SHIP["visible"]] == 0
    assert hidden.any(), "the scenario should hide some enemies"
    assert (obs.ships[hidden][:, :SHIP_HIDDEN_ZERO] == 0).all()
    assert torch.equal(truth.ships[..., :S, :], obs.ships[..., :S, :])


def test_spawn_layout_does_not_reveal_the_seat():
    rules = api.rules_of(GameConfig(3, 4, 2600.0))
    _, obs, _ = api.perceive(rules, api.reset(rules, TYPES, api.uniforms(rules, 512)))
    S = rules.ships_per_team
    own = obs.ships[..., :S, :]
    dx = own[..., 0, SHIP["x"]] - own[..., SHIP["x"]].mean(-1)
    dy = own[..., 0, SHIP["y"]] - own[..., SHIP["y"]].mean(-1)
    angle = torch.atan2(dy, dx)  # world-frame direction of slot 0 from its team's centre
    spread = 1.0 - torch.hypot(angle.cos().mean(0), angle.sin().mean(0))  # circular spread
    assert (spread > 0.5).all(), f"slot 0's direction is fixed per view: {spread} (uniform ~1)"
    assert math.isfinite(float(spread.sum()))


def test_unfogged_views_see_every_living_ship():
    fogged = torch.tensor([True, False] * 8)
    rules, state = _game(fogged=fogged)
    _, obs, truth = api.perceive(rules, state)
    assert torch.equal(obs.globals[..., GLOBAL["fogged"]], fogged[:, None].float().expand(16, 2))
    clear, foggy = obs.ships[~fogged], obs.ships[fogged]
    alive = clear[..., SHIP["alive"]] == 1
    assert torch.equal(clear[..., SHIP["visible"]] == 1, alive)
    assert (clear[..., SHIP["time_since_seen"]] == 0).all()
    assert torch.equal(clear[alive], truth.ships[~fogged][alive])
    assert (foggy[..., SHIP["visible"]] == 0).any(), "fogged environments still hide enemies"


def test_fog_survives_episode_resets():
    rules = api.rules_of(GameConfig(1, 1, 600.0, episode_s=1 / 15))  # two-tick episodes
    fogged = torch.tensor([True, False])
    state = api.reset(rules, TYPES, api.uniforms(rules, 2), fogged)
    action = Action.unpack(torch.zeros(2, 2, 3, dtype=torch.int32))
    resets = 0
    for _ in range(3):
        state, events = api.step(rules, TYPES, state, action, api.spread(rules, 2),
                                 api.uniforms(rules, 2))
        resets += int(events.done.sum())
    assert resets > 0 and torch.equal(state.match.fogged, fogged)


def test_fog_share_must_be_whole():
    assert pool_sizes((Pool("a", 1.0, ("x", "x"), fog_share=0.25),), 8) == (8,)
    with pytest.raises(ValueError):
        pool_sizes((Pool("a", 1.0, ("x", "x"), fog_share=0.3),), 8)


def test_rollout_fogs_the_first_share_of_each_pool():
    from bnb.controllers.scripted import Random
    from bnb.runtime.rollout import Rollout

    pools = (Pool("a", 0.5, ("r", "r"), fog_share=0.5), Pool("b", 0.5, ("r", "r"), fog_share=0.0))
    rollout = Rollout(GameConfig(1, 1, 600.0), pools, {"r": Random(1)}, "r", 8, 2,
                      torch.device("cpu"), compile=False)
    want = torch.tensor([True, True, False, False, False, False, False, False])
    assert torch.equal(rollout.carry["state"].match.fogged, want)
    rollout.run()
    assert torch.equal(rollout.carry["obs"]["globals"][:, 0, GLOBAL["fogged"]] == 1, want)
