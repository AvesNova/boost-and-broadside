"""The cheese agents as league members: off by default, an entry per named agent when the
settings list them, a slot holding one plays it and tallies to the slot's own column."""

import league_cases as cases
import pytest
import torch

from bnb.controllers.scripted_cheese import NAMES, player_name
from bnb.league.pool import Entry
from bnb.runtime.opponents import OPPONENTS, SCRIPTED_KIND, LeagueConfig

pytestmark = pytest.mark.usefixtures("global_generator")


def test_cheese_is_off_by_default_and_checked():
    assert LeagueConfig().cheese == ()
    with pytest.raises(ValueError, match="cheese"):
        LeagueConfig(cheese=("nobody",))
    with pytest.raises(ValueError, match="at most once"):
        LeagueConfig(cheese=("kiter", "kiter"))
    with pytest.raises(ValueError, match="scripted:<agent>"):
        Entry("kiter", "cheese")
    assert not Entry(player_name("kiter"), "cheese").network


def test_the_default_pool_has_no_cheese_entry(tmp_path):
    trainer, league = cases.league(tmp_path)
    assert not any(entry.kind == "cheese" for entry in league.entries.values())
    assert trainer.opponents.cheese is None
    assert "cheese_of_slot" not in trainer.opponents.state_dict()
    assert not any("cheese" in name for name in league.metrics())


def test_each_named_agent_is_a_pool_entry_in_every_gauge(tmp_path):
    config = cases.with_league(cases.config("cpu"), cheese=("evader", "kiter")).league
    _, league = cases.league(tmp_path, league=config)
    names = [player_name("evader"), player_name("kiter")]
    assert all(league.entries[name].kind == "cheese" for name in names)
    for gauge in league.gauges.values():
        assert gauge.names() == list(league.entries)
    metrics = league.metrics()
    assert "pool/slot_share/cheese" in metrics


def test_a_slot_holding_a_cheese_agent_plays_it_and_keeps_its_own_column(tmp_path):
    config = LeagueConfig(cheese=NAMES)
    trainer, league = cases.league(tmp_path, league=config)
    opponents = trainer.opponents
    for slot, name in enumerate(NAMES[: opponents.slots]):
        league._load(slot, league.entries[player_name(name)])
        assert int(opponents.kind[slot]) == 2 and opponents.cheese_of_slot[slot] == NAMES.index(
            name
        )
    rows = opponents.modes.shape[0]
    view = trainer.rollout._view(trainer.rollout.carry, OPPONENTS, opponents.reads + ("globals",))
    carry = trainer.rollout.carry["controllers"][OPPONENTS]
    reset = torch.ones(rows, dtype=torch.bool)
    action, after, _ = opponents.step(carry, view, reset, torch.arange(rows))
    assert torch.equal(after.column, opponents.slot_of_row)  # no row tallies as the scripted agent
    for slot, name in enumerate(NAMES[: opponents.slots]):
        mine = opponents.slot_of_row == slot
        played = opponents.cheese.play(view, name).packed()
        assert torch.equal(action.packed()[mine], played[mine])
    # a slot taken back by the scripted agent plays no cheese
    opponents.set_kind(0, SCRIPTED_KIND)
    assert opponents.cheese_of_slot[0] == -1
    state = opponents.state_dict()
    opponents.load_state_dict({name: value.clone() for name, value in state.items()})


def test_a_league_with_cheese_runs_updates_and_tallies_by_entry(tmp_path):
    config = LeagueConfig(cheese=("zone_rusher", "jouster"), cut_every=1, cut_slots=4)
    trainer, league = cases.league(tmp_path, league=config)
    for _ in range(3):
        trainer.collect()
        league.collected(trainer)
        trainer.ppo.environment_steps += trainer.steps_per_update
        league.updated(trainer)
    assert set(league.held) <= set(league.entries)
