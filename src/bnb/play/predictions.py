"""The prediction overlays' arrays (`bnb.play.overlays`) from the controllers and the match.

A Yemong agent's predictions are its model's outputs at its latest decision: every head
from that decision's own trunk pass (`Yemong.inspect`, while the agent is watched), or
else the learner's `unroll` over that one step, from the carry the step started at, the
view and the step's replay record (the beliefs after its sight); decoded by the decoder layer
(`bnb.models.yemong.decode`) as the figures decode them. The scripted agent's are its
memory's recall (`bnb.controllers.scripted_memory.remember`) of the same decision. The
true bullet paths are the game's own transport of the live bullets
(`bnb.game.api.bullet_paths`). Nothing here decodes a head itself.

The truth beside a prediction is the snapshot of the state the decision observed
(`observed`, the match loop's previous snapshot), or for the next state the one after it
(`following`, the current snapshot). Rows come back in sim order (`bnb.game.api.view_order`
maps a view's rows to sim indices).
"""

import numpy as np
import torch
from torch.utils import _pytree

from bnb.contracts.obs import GLOBAL, SHIP
from bnb.controllers.scripted import resolved
from bnb.controllers.scripted_memory import remember
from bnb.game import api
from bnb.models.yemong import decode
from bnb.models.yemong.density import HexGrid

from .overlays import (
    BulletPaths,
    Critic,
    Density,
    EnemyActions,
    Ghosts,
    Predictions,
    Reckoning,
    ShotPaths,
)
from .sides import Agent, ScriptedAgent
from .snapshot import Snapshot


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy().copy()


def bullet_paths(
    rules: api.Rules, types: api.ShipTypes, state: api.State, frame: Snapshot
) -> BulletPaths:
    """Each live bullet's flight ahead from the state's bullets (`frame` is the state's
    snapshot: the bullets' positions now and their shooters' teams)."""
    bullets = frame.bullets
    active = bullets.active
    if not active.any():
        empty = np.zeros((0, rules.bullet_moves + 1), np.float32)
        return BulletPaths(x=empty, y=empty, valid=empty.astype(bool), team=bullets.team[active])
    x, y, valid = api.bullet_paths(rules, types, state)

    def flat(values: torch.Tensor) -> np.ndarray:
        return values[0].reshape(-1, values.shape[-1]).cpu().numpy()[active]

    return BulletPaths(
        x=np.concatenate([bullets.x[active, None], flat(x)], -1),
        y=np.concatenate([bullets.y[active, None], flat(y)], -1),
        valid=np.concatenate([np.ones((int(active.sum()), 1), bool), flat(valid)], -1),
        team=bullets.team[active],
    )


def _ghosts(
    ship: np.ndarray,
    x: torch.Tensor,
    y: torch.Tensor,
    heading: torch.Tensor,
    sigma: torch.Tensor,
    opacity: torch.Tensor,
    truth: Snapshot,
) -> Ghosts:
    return Ghosts(
        ship=ship,
        x=_numpy(x),
        y=_numpy(y),
        heading=_numpy(heading),
        sigma=_numpy(sigma),
        opacity=_numpy(opacity),
        true_x=truth.ships.x[ship],
        true_y=truth.ships.y[ship],
        true_alive=truth.ships.alive[ship],
    )


@torch.no_grad()
def network(agent, team: int, observed: Snapshot, following: Snapshot) -> Predictions | None:
    """A Yemong agent's predictions at its latest decision (None before its first)."""
    inspection = agent.inspection
    if inspection is None:
        return None
    model = agent.model
    view, replay = inspection.view, inspection.replay

    def leading(tree):
        return _pytree.tree_map(lambda tensor: tensor[None], tree)

    unrolled = inspection.unrolled
    if unrolled is None:  # decided unwatched: the learner's unroll over that one step
        reset = inspection.reset[None]
        unrolled = model.unroll(inspection.carry, leading(view), leading(replay), reset)
    unrolled = _pytree.tree_map(lambda tensor: tensor[0, 0].float().cpu(), unrolled)
    rows = view.ships[0].float().cpu()  # (N, columns), the view's rows
    own = replay.previous_action.shape[-1]
    order = api.view_order(own, torch.device("cpu"))[team].numpy()  # the sim index of each row
    alive = rows[:, SHIP["alive"]] > 0.5
    visible = rows[:, SHIP["visible"]] > 0.5
    offset = replay.offset[0].float().cpu()
    origin = torch.zeros(rows.shape[0], 2)  # the arena's centre: every arena point's nearest

    forecast = decode.next_state(
        model.state_code,
        unrolled.next_state_logits,
        rows[:, SHIP["ship_type"]].long(),
        offset,
        origin,
    )
    shown = alive.numpy()
    next_state = _ghosts(
        order[shown],
        forecast.position[alive, 0],
        forecast.position[alive, 1],
        forecast.heading[alive],
        forecast.position_sigma[alive],
        torch.ones(int(alive.sum())),
        following,
    )

    belief = decode.ship_forecast(
        replay.means[0].float().cpu(), replay.spreads[0].float().cpu(), offset, origin[own:]
    )
    hidden = replay.valid[0].cpu() & alive[own:] & ~visible[own:]
    believed = _ghosts(
        order[own:][hidden.numpy()],
        belief.position[hidden, 0],
        belief.position[hidden, 1],
        belief.heading[hidden],
        belief.position_sigma[hidden],
        torch.ones(int(hidden.sum())),
        observed,
    )

    ages = torch.arange(1, model.bullet_moves + 1, dtype=torch.float32)
    path = decode.shot_path(model.shot_code, model.shot_trajectory(unrolled.shot_latents, ages))
    ships = observed.ships
    launch_x = torch.from_numpy(ships.x[order]).float()
    launch_y = torch.from_numpy(ships.y[order]).float()
    heading = torch.from_numpy(ships.heading[order]).float()
    world = decode.launch_frame_to_world(
        path.displacement, launch_x, launch_y, torch.cos(heading), torch.sin(heading)
    )
    hit = decode.shot_outcome(model.shot_code, unrolled.shot_outcome_logits)["enemy_hit"]
    by_ship = np.argsort(order)  # row of each sim index
    shot = ShotPaths(
        launch_x=launch_x.numpy()[by_ship],
        launch_y=launch_y.numpy()[by_ship],
        x=_numpy(world[..., 0])[by_ship],
        y=_numpy(world[..., 1])[by_ship],
        sigma=_numpy(path.displacement_sigma)[by_ship],
        enemy_hit=_numpy(hit)[by_ship],
    )

    arena_radius = view.globals[0, GLOBAL["arena_radius"]].float().cpu()
    rings = model.config.hex_rings
    field = decode.density(unrolled.density_log_rates, rings, arena_radius)
    is_ally = torch.from_numpy(ships.team == team)[None]
    counts = HexGrid(rings).counts(
        torch.from_numpy(ships.x)[None].float(),
        torch.from_numpy(ships.y)[None].float(),
        is_ally,
        torch.from_numpy(ships.alive)[None],
        arena_radius,
    )[0]
    cells = field.ally.shape[-1]
    density = Density(
        centres=_numpy(field.centres),
        cell_radius=float(field.cell_radius),
        predicted={"ally": _numpy(field.ally), "enemy": _numpy(field.enemy)},
        true={"ally": _numpy(counts[:cells]), "enemy": _numpy(counts[cells:])},
    )

    action = decode.action_distribution(unrolled.enemy_action_logits[own:])
    enemy_action = EnemyActions(
        ship=order[own:],
        power=_numpy(action.power),
        stick=_numpy(action.stick),
        shoot=_numpy(action.shoot),
    )
    outcome = decode.outcome(unrolled.outcome_logits)
    critic = Critic(
        ship=order[:own],
        levels=tuple(model.config.levels),
        values=_numpy(unrolled.values),
        win=float(outcome.win),
        tie=float(outcome.tie),
        loss=float(outcome.loss),
        unresolved=float(outcome.unresolved),
        value=float(outcome.value),
    )
    return Predictions(
        team=team,
        tick=observed.tick,
        shot_path=shot,
        next_state=next_state,
        belief=believed,
        density=density,
        enemy_action=enemy_action,
        value=critic,
    )


@torch.no_grad()
def memory(agent, team: int, observed: Snapshot) -> Predictions | None:
    """The scripted agent's dead-reckoned enemies at its latest decision (None before its
    first): each remembered enemy it cannot see, at its recalled position, heading along
    its last seen velocity, its weight as the ghost's opacity."""
    inspection = agent.inspection
    if inspection is None:
        return None
    controller, view = agent.controller, inspection.view
    after, recall = remember(
        inspection.carry,
        view,
        inspection.reset,
        controller.characteristics,
        resolved(controller.parameters, view),
        controller.temperature,
    )
    own = controller.ships_per_team
    enemy = view.ships[0, own:]
    seen = (enemy[:, SHIP["visible"]] > 0.5) & (enemy[:, SHIP["alive"]] > 0.5)
    remembered = recall.known[0] & ~seen
    order = api.view_order(own, torch.device("cpu"))[team].numpy()[own:]
    heading = torch.atan2(recall.velocity_y[0], recall.velocity_x[0])
    chosen = remembered.numpy()
    ghosts = _ghosts(
        order[chosen],
        recall.x[0, remembered],
        recall.y[0, remembered],
        heading[remembered],
        torch.zeros(int(remembered.sum())),
        recall.weight[0, remembered],
        observed,
    )
    reckoning = Reckoning(
        ghosts=ghosts,
        last_x=_numpy(after.enemy_x[0, remembered]),
        last_y=_numpy(after.enemy_y[0, remembered]),
    )
    return Predictions(team=team, tick=observed.tick, dead_reckoning=reckoning)


def side_predictions(
    controller, team: int, observed: Snapshot, following: Snapshot
) -> Predictions | None:
    """A side's predictions at its latest decision, by what its controller provides (None
    for a controller that provides none, or before its first decision)."""
    if isinstance(controller, Agent) and controller.overlays:
        return network(controller, team, observed, following)
    if isinstance(controller, ScriptedAgent):
        return memory(controller, team, observed)
    return None
