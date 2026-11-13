"""Ship sets (`bnb.game.ships`): a game names its set; the set's files give its default
card, its type names and the rule values its cards were solved under. The standard set is
the game as it was."""

import dataclasses
import shutil

import pytest
import torch
from play_cases import tiny_model

from bnb.contracts import GameConfig
from bnb.game import api, ships
from bnb.game.rules import FlightConstants, Rules
from bnb.league.matches import Player, card_of
from bnb.play.interface.menus import MAP_PRESETS, Draft, default_lineup, map_of, maps
from bnb.play.loop import MatchLoop
from bnb.play.maps import Map, Maps
from bnb.play.tuning import Tuning, changed_card
from bnb.runs.checkpoint import save_policy
from bnb.runtime.train import TrainConfig

TYPES = 12  # a set of many types
TRIAL_RULES = """
[rules]
bullet_drag = 0.0001
field_band_width = 120.0

[flight]
gravity = 9.81

[mode]
capture_seconds = 20.0
group_radius = 900.0
"""


@pytest.fixture
def trial(tmp_path, monkeypatch):
    """A copy of the ship files with a set "trial" of TYPES types (each a fighter with its
    own hit damage) and rules of its own; the game's own files are read back afterwards."""
    copy = tmp_path / "ships"
    shutil.copytree(ships.SHIPS, copy, ignore=shutil.ignore_patterns("real-scale"))
    monkeypatch.setattr(ships, "SHIPS", copy)
    api.reload_ships()
    fighter = ships.load()[0]
    trial_ships = [
        ships.Ship(
            f"type-{index}",
            f"Type {index}",
            dict(fighter.handles),
            dict(fighter.card, hit_damage=float(index + 1)),
            {"hit_damage": "a placeholder"} if index == 0 else {},
        )
        for index in range(TYPES)
    ]
    ships.write(trial_ships, "trial")
    (copy / "trial" / ships.RULES).write_text(TRIAL_RULES)
    api.reload_ships()
    yield "trial"
    monkeypatch.undo()
    api.reload_ships()


def test_the_standard_set_is_the_ship_directory_and_has_the_games_rules():
    assert ships.sets()[0] == api.STANDARD_SHIP_SET == GameConfig(1, 1, 1.0).ship_set
    assert ships.directory_of(api.STANDARD_SHIP_SET) == ships.SHIPS
    assert ships.rules(api.STANDARD_SHIP_SET) == {}
    config = api.PRESETS["frontline"]
    rules = api.rules_of(config)
    assert rules.flight == FlightConstants()
    assert rules.longest_bullet_moves == 12 and rules.field_band_width == Rules.field_band_width
    assert api.default_card() == api.default_card(api.STANDARD_SHIP_SET)
    with pytest.raises(ValueError, match="lowercase"):
        ships.directory_of("Real Scale")


def test_a_set_gives_its_card_names_and_rules_and_leaves_the_standard_alone(trial):
    standard = api.ship_types()
    types = api.ship_types(ship_set=trial)
    assert types.hit_damage.tolist() == [float(i + 1) for i in range(TYPES)]
    assert api.type_names(trial) == tuple(f"Type {i}" for i in range(TYPES))
    assert api.type_names() == ("Fighter", "Heavy")
    assert torch.equal(api.ship_types().hit_damage, standard.hit_damage)
    assert trial in ships.sets()

    config = GameConfig(5, 10, 7800.0, mode="frontline", ship_set=trial)
    rules = api.rules_of(config)
    assert rules.bullet_drag == 0.0001 and rules.field_band_width == 120.0
    assert rules.flight.gravity == 9.81
    assert rules.flight.coast_exchange == FlightConstants().coast_exchange
    assert rules.pools[0].mode.capture_seconds == 20.0
    assert rules.pools[0].mode.group_radius == 900.0
    assert api.rules_of(config, bullet_drag=1e-3).bullet_drag == 1e-3  # the caller's win


def test_a_sets_files_round_trip_with_their_notes(trial):
    found = ships.load(trial)
    assert found[0].notes == {"hit_damage": "a placeholder"} and found[1].notes == {}
    ships.write(found[:3], trial)
    again = ships.load(trial)
    assert again == found[:3]
    assert (ships.directory_of(trial) / ships.RULES).is_file()  # a save keeps the rules
    api.reload_ships()
    assert api.type_names(trial) == ("Type 0", "Type 1", "Type 2")


def test_training_takes_its_games_set_card(trial):
    game = GameConfig(5, 10, 7800.0, ship_set=trial)
    assert TrainConfig(game=game).ship_card == api.default_card(trial)
    assert TrainConfig(game=api.PRESETS["deathmatch"]).ship_card == api.default_card()


def test_a_tuning_and_a_saved_map_are_relative_to_their_set(trial, tmp_path):
    types = api.ship_types(ship_set=trial)
    assert changed_card(types, trial) == {}
    assert set(changed_card(types)) == set(api.default_card())  # 12 types against 2
    tuned = dataclasses.replace(types, hit_damage=types.hit_damage * 2.0)
    tuning = Tuning(card=changed_card(tuned, trial))
    assert set(tuning.card) == {"hit_damage"}
    assert torch.equal(tuning.types(trial).hit_damage, tuned.hit_damage)
    maps = Maps(tmp_path / "user", tmp_path / "project")
    saved = Map("doubled", GameConfig(2, 2, 1500.0, ship_set=trial), tuning=tuning, place="user")
    maps.save(saved)
    assert maps.find("doubled") == saved
    assert torch.equal(maps.find("doubled").tuning.types(trial).hit_damage, tuned.hit_damage)


def test_a_rated_match_plays_the_policies_card_else_the_sets(trial, tmp_path):
    game = GameConfig(2, 2, 1500.0, ship_set=trial)
    scripted = Player("scripted")
    assert torch.equal(
        card_of([scripted], game, torch.device("cpu")).hit_damage,
        api.ship_types(ship_set=trial).hit_damage,
    )
    standard_game = GameConfig(2, 2, 1500.0)
    trained = api.ship_types(hit_damage=(3.0, 4.0))
    first = save_policy(
        tmp_path / "first",
        tiny_model("relational", standard_game),
        game=standard_game,
        types=trained,
    )
    found = card_of([scripted, Player("first", first)], standard_game, torch.device("cpu"))
    assert found.hit_damage.tolist() == [3.0, 4.0]
    second = save_policy(
        tmp_path / "second",
        tiny_model("relational", standard_game),
        game=standard_game,
        types=api.ship_types(),
    )
    with pytest.raises(ValueError, match="different ship cards"):
        card_of(
            [Player("first", first), Player("second", second)], standard_game, torch.device("cpu")
        )


def test_the_new_match_maps_come_from_the_presets():
    keys = [game_map.key for game_map in maps()]
    assert keys == ["standard", "wide", "real_scale", "arena", "clover"]
    assert {map_of(name) for name in MAP_PRESETS} == set(keys)
    assert not set(api.FLEET_PRESETS) & set(MAP_PRESETS)  # the fleet sizes are no maps
    assert Draft().config() == api.PRESETS["real_scale_deathmatch"]
    assert Draft(map="standard").config() == api.PRESETS["deathmatch"]
    assert Draft(map="wide").config() == api.PRESETS["deathmatch_wide"]
    assert Draft(mode="frontline", map="standard").config() == api.PRESETS["frontline"]
    wide_frontline = Draft(mode="frontline", map="wide").config()
    assert (wide_frontline.mode, wide_frontline.arena_radius) == ("frontline", 8200.0)
    assert wide_frontline.spawn_groups == ()
    for name, config in MAP_PRESETS.items():
        setup = Draft(mode=config.mode, map=map_of(name)).setup()
        assert Draft.of(setup).map == map_of(name)


def test_choosing_the_arena_takes_its_fleet_fields_and_lineup():
    """The arena map is the arena preset: 2v2 at no fields, a Yongma and a Jujak a
    team; the other maps give back their own sizes."""
    draft = Draft()
    draft.choose_map("arena")
    assert (draft.ships_per_team, draft.fields) == (2, 0)
    assert draft.lineup == list(api.preset_lineup("arena"))
    setup = draft.setup()
    assert setup.config == api.PRESETS["arena"]
    assert MatchLoop(setup, compiled=False).rules.team_types == api.preset_lineup("arena")
    assert Draft.of(setup).map == "arena"
    draft.choose_map("real_scale")
    assert (draft.ships_per_team, draft.fields) == (5, 10)


def test_the_lineup_shown_is_the_lineup_played():
    """The New match page's default lineup (Yongma and Jujak alternating)
    reaches the match; only the game's own lineup is passed as None."""
    names = api.type_names("real-scale")
    draft = Draft()
    rules = MatchLoop(draft.setup(), compiled=False).rules
    assert [names[kind] for kind in rules.team_types] == [
        names[kind] for kind in default_lineup(5, "real-scale")
    ]
    draft.lineup = list(api.rules_of(draft.config()).team_types)
    assert draft.setup().lineup is None
