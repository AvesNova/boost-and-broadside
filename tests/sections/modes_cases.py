"""Shared cases for the modes and rewards section tests: configs of every mode, random play
through the real game, and helpers that edit a state."""

import dataclasses

import torch

from bnb.contracts import Action, GameConfig
from bnb.game import api
from bnb.game.state import replace

TYPES = api.ship_types()
# Random play is made lively: small arenas, short clocks, weak shields that start
# recharging after half a second. Within a few hundred ticks it then has kills, friendly
# fire, boundary deaths, recharge, respawns and resets (and in Frontline, moving meters).
LIVELY_TYPES = api.ship_types(max_shield=(20.0, 30.0))
LIVELY_RULES = dict(shield_recharge_delay=0.5)
MODES = {
    "sandbox": GameConfig(3, 3, 600.0, mode="sandbox", episode_s=6.0, respawn_s=0.5),
    "team_deathmatch": GameConfig(3, 3, 900.0, episode_s=6.0, respawn_s=0.5),
    "grouped": GameConfig(3, 3, 1200.0, episode_s=6.0, respawn_s=0.5, spawn_groups=(2, 1)),
    "frontline": GameConfig(3, 3, 300.0, mode="frontline", episode_s=8.0, respawn_s=0.5),
}


def random_actions(batch: int, ships: int, generator: torch.Generator,
                   shoot: float = 0.7) -> Action:
    choice = torch.rand(batch, ships, 3, generator=generator)
    power = (choice[..., 0] * 3).to(torch.int32)
    stick = (choice[..., 1] * 7).to(torch.int32)
    return Action(power, stick, (choice[..., 2] < shoot).to(torch.int32))


def lively(config: GameConfig):
    return api.rules_of(config, **LIVELY_RULES)


def play(config: GameConfig, batch: int = 16, ticks: int = 300, seed: int = 0):
    """Lively random play: yields (before, events, after) for every tick, eager on the CPU."""
    rules = lively(config)
    generator = torch.Generator().manual_seed(seed)
    state = api.reset(rules, LIVELY_TYPES, api.uniforms(rules, batch, generator=generator))
    for _ in range(ticks):
        action = random_actions(batch, rules.num_ships, generator)
        after, events = api.step(rules, LIVELY_TYPES, state, action,
                                 api.spread(rules, batch, generator=generator),
                                 api.uniforms(rules, batch, generator=generator))
        yield state, events, after
        state = after


def rules_and_state(config: GameConfig, batch: int = 4, seed: int = 0):
    rules = api.rules_of(config)
    generator = torch.Generator().manual_seed(seed)
    return rules, api.reset(rules, TYPES, api.uniforms(rules, batch, generator=generator))


def edit_ships(state, **values):
    """The state with ship fields replaced (each a (B, N) tensor)."""
    return replace(state, ships=replace(state.ships, **values))


def edit_mode(state, **values):
    return replace(state, mode=replace(state.mode, **values))


def still(batch: int, ships: int) -> Action:
    zero = torch.zeros(batch, ships, dtype=torch.int32)
    return Action(zero, zero, zero)


def tick(rules, state, action=None, seed: int = 0):
    """One step with fresh random draws (the spread and uniforms barely matter here)."""
    B, N = state.ships.x.shape
    generator = torch.Generator().manual_seed(seed)
    action = still(B, N) if action is None else action
    return api.step(rules, TYPES, state, action, api.spread(rules, B, generator=generator),
                    api.uniforms(rules, B, generator=generator))


def fields_away(state, distance: float = 1e6):
    """Fields moved out of play, so ships fly in a uniform medium."""
    fields = dataclasses.replace(state.fields, x=torch.full_like(state.fields.x, distance))
    return replace(state, fields=fields)
