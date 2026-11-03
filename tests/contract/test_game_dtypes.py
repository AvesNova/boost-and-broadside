"""The dtype and shape of every field of the `Events`, `Obs` and `Privileged` the game
produces (code style, Dtypes), in every mode and at two batch sizes."""

import pytest
import torch

from bnb.contracts import GameConfig
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
