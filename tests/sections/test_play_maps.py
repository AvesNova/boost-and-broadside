"""Saved maps (`bnb.play.maps`): a map is a whole game (config, lineup, tuning) saved by
name in the user's or the project's directory; it round trips, is renamed, moved and
deleted, shows on the New match page with its mode, and starts with `bnb play --preset`."""

import dataclasses
import json

import pytest

from bnb.contracts import GameConfig
from bnb.game import api
from bnb.play import __main__ as play_main
from bnb.play.app import Options
from bnb.play.interface.menus import Draft, maps
from bnb.play.loop import MatchLoop, MatchSetup
from bnb.play.maps import Map, Maps, preset_map
from bnb.play.tuning import Tuning


@pytest.fixture
def library(tmp_path) -> Maps:
    return Maps(tmp_path / "user", tmp_path / "project")


def big_sky(place: str = "user") -> Map:
    """A frontline map of the real-scale set: 3 a side, 4 fields, a 4000 m arena, a 400 s
    match, a Jujak in every slot and a tuning (gravity, Yongma's damage, x2)."""
    names = api.type_names("real-scale")
    damage = list(api.default_card("real-scale")["hit_damage"])
    damage[0] = 9.0
    tuning = Tuning(
        rules={"flight.gravity": 10.0},
        card={"hit_damage": tuple(damage)},
        time_factor=2.0,
    )
    config = GameConfig(
        3, 4, 4000.0, episode_seconds=400.0, mode="frontline", ship_set="real-scale"
    )
    return Map("big-sky", config, (names[1],) * 3, tuning, place)


def test_a_map_round_trips_with_its_lineup_and_tuning(library):
    path = library.save(big_sky())
    assert path == library.directories["user"] / "big-sky.json"
    data = json.loads(path.read_text())
    assert data["lineup"] == ["Jujak"] * 3
    assert data["ships"] == {"yongma": {"hit_damage": 9.0}}  # by type key, only what changed
    assert data["time_factor"] == 2.0 and data["rules"] == {"flight.gravity": 10.0}
    assert library.find("big-sky") == big_sky()
    assert library.find("big-sky").lineup_types() == (1, 1, 1)
    assert "big-sky" in library.names() and "frontline" in library.names()


def test_a_saved_card_is_by_type_key_so_a_player_type_added_keeps_the_map(library, tmp_path):
    """The pool grows (a type appended): the map's card still reads, the new type at its
    file's card."""
    library.save(big_sky())
    path = library.directories["user"] / "big-sky.json"
    data = json.loads(path.read_text())
    assert set(data["ships"]) <= {ship.key for ship in api.ships.load("real-scale")}
    data["ships"]["nonesuch"] = {"hit_damage": 1.0}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="nonesuch"):
        library.find("big-sky")
    assert library.saved() == [] and "nonesuch" in library.problems()[0]


def test_names_are_hyphenated_unique_and_never_a_game_presets(library):
    library.save(big_sky())
    with pytest.raises(ValueError, match="saved already"):
        library.save(big_sky("project"))  # one name, one map
    with pytest.raises(ValueError, match="lowercase"):
        library.save(dataclasses.replace(big_sky(), name="Big Sky"))
    for reserved in ("arena", "frontline", "wide", "standard"):
        with pytest.raises(ValueError, match="game preset"):
            library.save(dataclasses.replace(big_sky(), name=reserved))
    with pytest.raises(ValueError, match="no game preset or saved map"):
        library.find("nowhere")


def test_a_map_is_edited_moved_renamed_and_deleted(library):
    library.save(big_sky())
    edited = dataclasses.replace(
        big_sky(), config=dataclasses.replace(big_sky().config, num_fields=6)
    )
    library.save(edited, replacing="big-sky")  # edited in place
    assert library.find("big-sky").config.num_fields == 6
    library.save(edited.renamed("big-sky", "project"), replacing="big-sky")  # moved
    assert library.place_of("big-sky") == "project"
    assert not (library.directories["user"] / "big-sky.json").exists()
    library.rename("big-sky", "bigger-sky")
    assert library.place_of("big-sky") is None and library.find("bigger-sky").place == "project"
    with pytest.raises(ValueError, match="game preset"):
        library.rename("arena", "my-arena")
    library.delete("bigger-sky")
    assert library.saved() == []


def test_a_match_becomes_a_map_with_its_tuned_clocks_in_the_config():
    """The tuned arena radius, clocks and kills to win are the config's own fields: they
    move there from the tuning (the design's values, at x1); the rest stays the tuning."""
    arena = preset_map("arena")
    tuning = Tuning(
        rules={
            "pool.arena_radius": 2500.0,
            "pool.episode_seconds": 300.0,
            "mode.kills_per_ship": 3.0,
            "flight.gravity": 10.0,
        },
        time_factor=2.0,
    )
    made = Map.of_match("duel", arena.config, arena.lineup_types(), tuning)
    assert (made.config.arena_radius, made.config.episode_seconds) == (2500.0, 300.0)
    # the arena's lineup (Yongma, Jujak) is the game's own: a map keeps None for it
    assert made.config.kills_per_ship == 3 and made.lineup is None
    assert made.tuning == Tuning(rules={"flight.gravity": 10.0}, time_factor=2.0)
    setup = MatchSetup(made.config, made.lineup_types(), tuning=made.tuning)
    played = MatchLoop(setup, compiled=False)
    assert played.rules.pools[0].episode_ticks == 300 * 30 // 2  # x2: half as long on the clock


def test_a_saved_map_is_offered_on_the_new_match_page_with_its_mode(library):
    library.save(big_sky())
    offered = maps(library.saved())
    assert [game_map.key for game_map in offered][-1] == "big-sky"
    draft = Draft(offered=offered)
    draft.choose_map("big-sky")
    assert draft.mode == "frontline" and (draft.ships_per_team, draft.fields) == (3, 4)
    assert draft.lineup == [1, 1, 1] and draft.tuned
    setup = draft.setup()
    assert setup.config == big_sky().config and setup.tuning == big_sky().tuning
    assert Draft.of(setup, offered).map == "big-sky"  # the page opens on it again
    draft.ships_per_team = 4
    assert draft.config().ships_per_team == 4  # the page's sizes apply to it


def test_bnb_play_preset_starts_a_saved_map(library):
    library.save(big_sky())
    setup = Options(preset="big-sky").setup(library)
    assert setup.config == big_sky().config and setup.lineup == (1, 1, 1)
    assert setup.tuning == big_sky().tuning
    loop = MatchLoop(setup, compiled=False)
    assert loop.rules.flight.gravity == pytest.approx(4.0 * 10.0)  # x2 of the tuned 10
    assert Options(preset="big-sky", ships_per_team=2).setup(library).config.ships_per_team == 2
    with pytest.raises(SystemExit, match="no game preset or saved map 'nowhere'"):
        play_main.main(["--preset", "nowhere"])
