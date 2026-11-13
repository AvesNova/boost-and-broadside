"""The dtype and shape of every field of the `Events`, `Obs` and `Privileged` the game
produces (code style, Dtypes), in every mode and at two batch sizes; and that a tick keeps
every state field's dtype."""

import pytest
import torch
from torch.utils import _pytree

from bnb.contracts import ASPECTS, GameConfig
from bnb.contracts.action import Action
from bnb.contracts.config import MODES, TEAMS
from bnb.contracts.obs import FIELD_COLUMNS, GLOBAL_COLUMNS, SHIP_COLUMNS, ZONE_COLUMNS, ZONE_SLOTS
from bnb.game import api

SHIPS_PER_TEAM, FIELDS = 3, 4


def _tick(mode: str, batch: int):
    """One tick of a fresh game: (events, observation, privileged)."""
    generator = torch.Generator().manual_seed(7)
    rules = api.rules_of(GameConfig(SHIPS_PER_TEAM, FIELDS, 1500.0, mode=mode))
    types = api.ship_types()
    state = api.reset(rules, types, api.uniforms(rules, batch, generator=generator))
    shape = (batch, rules.num_ships)
    action = Action(
        torch.randint(0, 3, shape, dtype=torch.int32, generator=generator),
        torch.randint(0, 7, shape, dtype=torch.int32, generator=generator),
        torch.ones(shape, dtype=torch.int32),
    )
    state, events = api.step(
        rules,
        types,
        state,
        action,
        api.spread(rules, batch, generator=generator),
        api.uniforms(rules, batch, generator=generator),
    )
    _, observation, privileged = api.perceive(rules, state)
    return events, observation, privileged


def _expected(batch: int) -> dict:
    B, T, N = batch, TEAMS, TEAMS * SHIPS_PER_TEAM
    floats, flags, ints = torch.float32, torch.bool, torch.int32
    return {
        "events": dict(
            damage=(floats, (B, N, N)),
            boundary_damage=(floats, (B, N)),
            died=(flags, (B, N)),
            fired=(flags, (B, N)),
            spawned=(flags, (B, N)),
            killer=(ints, (B, N)),
            done=(flags, (B,)),
            shield_lost=(floats, (B, N)),
            shield_restored=(floats, (B, N)),
            objective_progress=(floats, (B, N)),
            objective=(floats, (B, N)),
            result=(ints, (B,)),
            win_loss=(floats, (B, 2)),  # a win's and a loss's weight
            enemy_damage_by_aspect=(floats, (B, N, len(ASPECTS))),
            kill_aspect=(ints, (B, N)),
        ),
        "observation": dict(
            ships=(floats, (B, T, N, len(SHIP_COLUMNS))),
            fields=(floats, (B, T, FIELDS, len(FIELD_COLUMNS))),
            globals=(floats, (B, T, len(GLOBAL_COLUMNS))),
            zones=(floats, (B, T, ZONE_SLOTS, len(ZONE_COLUMNS))),
        ),
        "privileged": dict(
            team=(ints, (B, N)),
            view_order=(torch.int64, (B, T, N)),
            ships=(floats, (B, T, N, len(SHIP_COLUMNS))),
        ),
    }


@pytest.mark.parametrize("batch", [1, 6])
@pytest.mark.parametrize("mode", MODES)
def test_the_game_produces_each_contract_fields_dtype_and_shape(mode, batch):
    events, observation, privileged = _tick(mode, batch)
    produced = dict(events=events, observation=observation, privileged=privileged)
    wrong = []
    for contract, fields in _expected(batch).items():
        tree = produced[contract]
        assert set(fields) == set(vars(tree)), contract  # every field is checked
        for name, (dtype, shape) in fields.items():
            value = getattr(tree, name)
            if value.dtype != dtype or tuple(value.shape) != shape:
                wrong.append(f"{contract}.{name}: {value.dtype} {tuple(value.shape)}")
    assert not wrong, "\n".join(wrong)


def _state_dtypes(state) -> dict[str, torch.dtype]:
    leaves, _ = _pytree.tree_flatten_with_path(state)
    return {_pytree.keystr(path): leaf.dtype for path, leaf in leaves}


@pytest.mark.parametrize("mode", MODES)
def test_a_tick_keeps_every_state_fields_dtype(mode):
    """Through ticks with deaths, respawns (none in elimination) and an episode's end (a
    one-second episode)."""
    generator = torch.Generator().manual_seed(11)
    respawn_seconds = None if mode == "elimination" else 0.5
    config = GameConfig(
        SHIPS_PER_TEAM,
        FIELDS,
        600.0,
        mode=mode,
        episode_seconds=1.0,
        respawn_seconds=respawn_seconds,
    )
    rules, types, batch = api.rules_of(config), api.ship_types(), 4
    state = api.reset(rules, types, api.uniforms(rules, batch, generator=generator))
    expected = _state_dtypes(state)
    assert set(expected.values()) <= {torch.float32, torch.int32, torch.bool}
    shape = (batch, rules.num_ships)
    for tick in range(2 * round(1.0 / rules.tick_length)):
        action = Action(
            torch.randint(0, 3, shape, dtype=torch.int32, generator=generator),
            torch.randint(0, 7, shape, dtype=torch.int32, generator=generator),
            torch.ones(shape, dtype=torch.int32),
        )
        state, _ = api.step(
            rules,
            types,
            state,
            action,
            api.spread(rules, batch, generator=generator),
            api.uniforms(rules, batch, generator=generator),
        )
        state, _, _ = api.perceive(rules, state)
        changed = {
            path: (expected[path], dtype)
            for path, dtype in _state_dtypes(state).items()
            if dtype != expected[path]
        }
        assert not changed, f"tick {tick}: {changed}"
