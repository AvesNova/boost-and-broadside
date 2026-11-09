"""The prediction overlays (`bnb.play.overlays`, `bnb.play.predictions` and the tables of
`bnb.play.render.predictions`), on the CPU: which overlays each controller offers and
whose they draw, the decoded geometry they draw, and that watching an agent leaves its
play unchanged."""

import dataclasses

import numpy as np
import pytest
import torch
from play_cases import SMALL, tiny_model
from torch.utils import _pytree

from bnb.contracts.obs import SHIP
from bnb.game import api
from bnb.models.yemong import decode
from bnb.models.yemong.inputs import believed
from bnb.play import predictions
from bnb.play.loop import MatchLoop, MatchSetup, SideSetup
from bnb.play.overlays import (
    MEMORY,
    NAMES,
    NETWORK,
    PHYSICS,
    OverlaySettings,
    offered,
    provider,
)
from bnb.play.render.predictions import (
    density_vertices,
    hexagon_vertices,
    overlay_drawing,
    path_segments,
)
from bnb.play.sides import Agent, Scripted, ScriptedAgent, build_sides
from bnb.play.themes import Library

# The scripted agent's moves draw from torch's default generator inside its own fork.
pytestmark = pytest.mark.usefixtures("global_generator")

S, N = SMALL.ships_per_team, SMALL.num_ships


def _agent(seed: int = 0, name: str = "yemong") -> Agent:
    """A tiny agent on the CPU (load_agent would pick the GPU when there is one)."""
    return Agent(tiny_model(name, SMALL, seed), name, "tiny", S, N, seed=seed)


def _loop(sides: list, ticks: int, seed: int = 5) -> MatchLoop:
    loop = MatchLoop(MatchSetup(SMALL, seed=seed), sides=sides, compiled=False)
    for _ in range(ticks):
        loop.step()
    return loop


@pytest.fixture(scope="module")
def theme():
    return Library().load("mint-pink")


# Which overlays, whose -------------------------------------------------------------------


def test_scripted_against_scripted_offers_only_the_bullet_paths():
    assert (
        offered(
            [
                Scripted(api.rules_of(SMALL), api.ship_types()),
                Scripted(api.rules_of(SMALL), api.ship_types()),
            ]
        )
        == PHYSICS
        == ("bullet_paths",)
    )
    assert offered([]) == PHYSICS  # a replay runs no controllers


def test_each_controller_offers_what_it_can_provide():
    rules = api.rules_of(SMALL)
    memory = ScriptedAgent(rules, api.ship_types())
    assert offered([memory, Scripted(api.rules_of(SMALL), api.ship_types())]) == (
        "bullet_paths",
        "dead_reckoning",
    )
    assert (
        offered([_agent(), Scripted(api.rules_of(SMALL), api.ship_types())])
        == ("bullet_paths",) + NETWORK
    )
    assert (
        offered([_agent(name="relational"), Scripted(api.rules_of(SMALL), api.ship_types())])
        == PHYSICS
    )
    everything = offered([_agent(), memory])
    assert set(everything) == set(PHYSICS + NETWORK + MEMORY) == set(NAMES)
    assert everything == tuple(name for name in NAMES if name in everything)  # card order


def test_a_setup_names_the_scripted_agent():
    sides = build_sides(
        MatchSetup(SMALL, sides=(SideSetup("scripted_agent"), SideSetup())),
        api.rules_of(SMALL),
        api.ship_types(),
    )
    assert isinstance(sides[0], ScriptedAgent) and isinstance(sides[1], Scripted)


def test_an_overlay_draws_the_picked_side_else_the_selected_ships_else_the_other():
    agent, memory = _agent(), ScriptedAgent(api.rules_of(SMALL), api.ship_types())
    sides = [agent, memory]
    # nothing picked: the selected ship's side when it provides the overlay, else the other
    assert provider("belief", sides, None, selected_team=0, view_team=-1) == 0
    assert provider("belief", sides, None, selected_team=1, view_team=-1) == 0
    assert provider("dead_reckoning", sides, None, selected_team=0, view_team=-1) == 1
    # nothing selected: the view's team first, then team 0
    assert provider("dead_reckoning", sides, None, None, view_team=0) == 1
    assert provider("belief", sides, None, None, view_team=1) == 0
    # picked: that side, or nothing when it does not provide the overlay
    assert provider("belief", sides, 0, selected_team=1, view_team=-1) == 0
    assert provider("belief", sides, 1, selected_team=1, view_team=-1) is None
    both = [agent, _agent(seed=1)]
    assert provider("belief", both, None, selected_team=1, view_team=-1) == 1


# Watching an agent ------------------------------------------------------------------------


def test_watching_an_agent_leaves_its_play_unchanged():
    """A watched agent decides with Yemong.inspect: the same actions, tick for tick."""
    runs = []
    for watched in (False, True):
        agents = [_agent(0), _agent(1)]
        agents[0].watched = watched
        loop = MatchLoop(MatchSetup(SMALL, seed=6), sides=agents, compiled=False)
        actions = []
        for _ in range(40):
            loop.step()
            actions.append(loop.last_actions.clone())
        runs.append(torch.stack(actions))
        assert (agents[0].inspection.unrolled is not None) == watched
    assert torch.equal(runs[0], runs[1])


# The decoded geometry ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def match(module_global_generator):
    """An agent (team 1, so its rows are not sim order) against the chaser, watched, some
    seconds in; and its predictions."""
    agent = _agent(3)
    agent.watched = True
    loop = _loop([Scripted(api.rules_of(SMALL), api.ship_types()), agent], ticks=120)
    made = predictions.side_predictions(agent, 1, loop.previous, loop.current)
    return loop, agent, made


def _unrolled(agent):
    inspection = agent.inspection
    leading = _pytree.tree_map_only(
        torch.Tensor, lambda x: x[None], (inspection.view, inspection.replay)
    )
    with torch.no_grad():
        unrolled = agent.model.unroll(inspection.carry, *leading, inspection.reset[None])
    return _pytree.tree_map_only(torch.Tensor, lambda x: x[0, 0], unrolled)


def test_the_shot_path_is_the_decoded_path_from_the_ships_launch_point(match):
    loop, agent, made = match
    model, unrolled = agent.model, _unrolled(agent)
    order = api.view_order(S, torch.device("cpu"))[1]
    ages = torch.arange(1, model.bullet_moves + 1, dtype=torch.float32)
    with torch.no_grad():
        path = decode.shot_path(model.shot_code, model.shot_trajectory(unrolled.shot_latents, ages))
    ships = loop.previous.ships
    for row, ship in enumerate(order.tolist()):
        heading = torch.tensor(float(ships.heading[ship]))
        world = decode.launch_frame_to_world(
            path.displacement[row],
            torch.tensor(float(ships.x[ship])),
            torch.tensor(float(ships.y[ship])),
            torch.cos(heading),
            torch.sin(heading),
        )
        shot = made.shot_path
        assert np.allclose(shot.x[ship], world[:, 0], atol=1e-2)
        assert np.allclose(shot.y[ship], world[:, 1], atol=1e-2)
        assert np.allclose(shot.sigma[ship], path.displacement_sigma[row], atol=1e-3)
        assert (shot.launch_x[ship], shot.launch_y[ship]) == pytest.approx(
            (ships.x[ship], ships.y[ship])
        )


def test_the_ghosts_are_the_decoded_forecasts_in_sim_order(match):
    loop, agent, made = match
    model, unrolled = agent.model, _unrolled(agent)
    inspection = agent.inspection
    rows = inspection.view.ships[0]
    order = api.view_order(S, torch.device("cpu"))[1].numpy()
    _, _, means, spreads = believed(inspection.view.ships, inspection.replay)
    forecast = decode.next_state(
        model,
        unrolled.next_state_logits,
        means[0].float(),
        spreads[0].float(),
        rows[:, SHIP["ship_type"]].long(),
        inspection.replay.offset[0],
        torch.zeros(N, 2),
    )
    ghosts = made.next_state
    for index, ship in enumerate(ghosts.ship):
        row = int(np.flatnonzero(order == ship)[0])
        assert ghosts.x[index] == pytest.approx(float(forecast.position[row, 0]), abs=1e-2)
        assert ghosts.y[index] == pytest.approx(float(forecast.position[row, 1]), abs=1e-2)
        assert ghosts.true_x[index] == loop.current.ships.x[ship]
    assert set(made.enemy_action.ship.tolist()) == set(range(S))  # team 1's enemies: team 0
    assert set(made.value.ship.tolist()) == set(range(S, N))
    assert np.allclose(made.enemy_action.power.sum(-1), 1.0, atol=1e-5)
    critic = made.value
    assert critic.win + critic.tie + critic.loss + critic.unresolved == pytest.approx(1.0)


def test_the_unroll_fallback_gives_the_inspected_predictions(match):
    """Turned on while paused, the overlays unroll the last decision: the same arrays."""
    loop, agent, made = match
    unwatched = dataclasses.replace(agent.inspection, unrolled=None)
    saved, agent.inspection = agent.inspection, unwatched
    try:
        again = predictions.side_predictions(agent, 1, loop.previous, loop.current)
    finally:
        agent.inspection = saved
    assert np.allclose(again.shot_path.x, made.shot_path.x, atol=1e-2)
    assert np.allclose(again.density.predicted["enemy"], made.density.predicted["enemy"], atol=1e-4)


def test_the_true_density_splits_each_living_ship_over_its_cells(match):
    """Each living ship of a side is one unit over its grid triangle (a corner off the
    grid's edge drops its share), on the predicting side's ally or enemy cells."""
    loop, _, made = match
    ships = loop.previous.ships
    for side, team in (("ally", 1), ("enemy", 0)):
        living = float((ships.alive & (ships.team == team)).sum())
        total = float(made.density.true[side].sum())
        assert 0.0 < total <= living + 1e-4, side


def test_the_dead_reckoning_ghosts_are_the_memorys_recall():
    """A scripted agent that saw the enemy and then lost it: each ghost is its last
    sighting plus the velocity then times the time since, its opacity the weight."""
    memory = ScriptedAgent(api.rules_of(SMALL), api.ship_types(), seed=2)
    found = None
    loop = MatchLoop(
        MatchSetup(SMALL, seed=7),
        sides=[memory, Scripted(api.rules_of(SMALL), api.ship_types())],
        compiled=False,
    )
    for _ in range(600):
        loop.step()
        made = predictions.side_predictions(memory, 0, loop.previous, loop.current)
        if len(made.dead_reckoning.ghosts.ship):
            found = made
            break
    assert found is not None, "the scripted agent never lost sight of a remembered enemy"
    reckoning = found.dead_reckoning
    ghosts = reckoning.ghosts
    view = memory.inspection.view.ships[0]
    for index, ship in enumerate(ghosts.ship):
        row = int(ship)  # team 0's view lists the ships in sim order
        since = float(view[row, SHIP["time_since_seen"]])
        carry = memory.inspection.carry
        enemy = row - S
        expected_x = float(carry.enemy_x[0, enemy] + carry.enemy_velocity_x[0, enemy] * since)
        assert view[row, SHIP["visible"]] < 0.5
        assert 0.0 < ghosts.opacity[index] <= 1.0
        assert reckoning.last_x[index] == pytest.approx(float(carry.enemy_x[0, enemy]))
        radius = SMALL.arena_radius
        if abs(expected_x) < 0.9 * radius:  # held inside the arena otherwise
            assert ghosts.x[index] == pytest.approx(expected_x, abs=1e-2)


# The drawing tables -----------------------------------------------------------------------


def test_a_path_joins_only_consecutive_valid_points():
    x = np.array([[0.0, 1.0, 2.0, 3.0]])
    valid = np.array([[True, True, False, True]])
    table = path_segments(x, x, valid, np.ones((1, 4), np.float32), 1.0)
    assert len(table) == 1 and tuple(table[0, :4]) == (0.0, 0.0, 1.0, 1.0)


def test_neighbouring_density_cells_share_an_edge():
    """The hexagons sit on the density grid: two neighbours' corners meet."""
    field = decode.density(torch.zeros(2 * 7), 1, torch.tensor(900.0))
    centres, radius = field.centres.numpy(), float(field.cell_radius)
    corners = hexagon_vertices(centres, radius, np.ones((len(centres), 4), np.float32))
    corners = corners.reshape(len(centres), 6, 3, -1)[:, :, 1, :2]  # each cell's six corners
    assert np.allclose(np.linalg.norm(corners - centres[:, None], axis=-1), radius, atol=1e-3)
    shared = np.linalg.norm(corners[0][:, None] - corners[1][None], axis=-1) < 1e-2
    assert shared.sum() == 2


def test_the_density_error_is_coloured_by_its_sign(theme, match):
    _, _, made = match
    settings = OverlaySettings(truth=True, density_shown="error")
    vertices = density_vertices(theme, made.density, 1, settings)
    over, under = np.array(theme.prediction.floats()[:3]), np.array(theme.truth.floats()[:3])
    colours = vertices[::18, 2:5]
    assert all(np.allclose(c, over) or np.allclose(c, under) for c in colours)
    without = density_vertices(theme, made.density, 1, OverlaySettings(density_shown="error"))
    team0 = np.array(theme.team[0].floats()[:3])  # no truth: the prediction, enemy cells
    assert all(np.allclose(c, team0) for c in without[::18, 2:5])


def test_the_drawing_holds_every_overlay_on(theme, match):
    loop, agent, made = match
    paths = predictions.bullet_paths(loop.rules, loop.types, loop.realtime.state, loop.current)
    drawn = {"bullet_paths": (-1, paths)}
    drawn.update({name: (1, getattr(made, name)) for name in NETWORK})
    settings = OverlaySettings(set(drawn), truth=True)
    selected = S  # team 1's first ship
    drawing = overlay_drawing(theme, settings, drawn, loop.current, selected, 1.0)
    ghosts = len(made.next_state.ship) + len(made.belief.ship)
    assert len(drawing.ghosts["x"]) == ghosts
    assert len(drawing.circles) >= loop.rules.bullet_moves  # the shot's circles and dots
    assert len(drawing.density) % 18 == 0
    alone = overlay_drawing(theme, settings, {}, loop.current, None, 1.0)
    assert len(alone.segments) == len(alone.circles) == len(alone.ghosts["x"]) == 0
