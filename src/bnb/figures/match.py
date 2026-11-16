"""The match record: one whole game played headless through the play shell's match loop
(`bnb.play.loop.MatchLoop`, uncompiled, on the CPU), every tick of it kept in sim order,
for the trail stills (`bnb.figures.trails`). Renderers read the record only.

Each side is Imugi (`bnb.play.sides.ImugiAgent`, as the play shell flies it: memory on)
or a policy (`bnb.play.sides.load_agent`), so a match needs no policy at all: Imugi
against Imugi renders before any model exists. Entry t of a per-tick array is the state
at the start of tick t (the snapshot the controllers decided on) with the events of tick
t: so the tick that ends the game is the last entry, with its kills, although the state
after it is already the next game's.

Every `ghost_ticks` ticks each side's view of the hidden enemies is kept: a policy's
belief ghost (its replay record decoded to world coordinates, mean and sigma, by
`bnb.play.predictions.network` as the shell's belief overlay draws it) or Imugi's
dead-reckoned memory (no sigma). It is NaN on the other ticks.

The artifact is `meta` (provenance) and `match` (the record, with `about`: the game's
numbers its renderers read), saved and read with `bnb.figures.capture.save` and `load`;
a policy's capture may hold a match record too (`bnb figures trails --policy P`).
"""

import dataclasses
import json
import sys
import time

import numpy as np
import torch

from bnb.contracts import GameConfig
from bnb.game import api
from bnb.runs import provenance
from bnb.runs.serialise import to_json

IMUGI = "imugi"  # a side flown by Imugi


@dataclasses.dataclass(frozen=True)
class MatchConfig:
    """preset: the game (`bnb.game.api.PRESETS`); ships_per_team: the fleet size (None:
    the preset's) on the map rule's arena (`bnb.league.crossover.sized_game`: "preset"
    keeps the preset's radius, "density" scales it with the fleet); seed: the match's
    generator and the sides' draws; seconds: stop after this much game time (None: at the
    game's end); ghost_ticks: ticks between the kept views of the hidden enemies (0:
    none); fogged: fog of war (None: the game's, none in team deathmatch); temperature:
    a policy side's sampling temperature; opponent: who flies team 1 when a policy flies
    team 0 ("imugi" or "policy": the same policy, `sides_of`)."""

    preset: str = "real_scale_deathmatch"
    ships_per_team: int | None = None
    map_rule: str = "preset"
    seed: int = 0
    seconds: float | None = None
    ghost_ticks: int = 15
    fogged: bool | None = None
    temperature: float = 1.0
    opponent: str = IMUGI


def sides_of(policy, config: MatchConfig) -> tuple[str, str]:
    """The two sides of a match: Imugi on both without a policy; else the policy on team 0
    and the opponent on team 1."""
    if policy is None:
        return IMUGI, IMUGI
    if config.opponent not in (IMUGI, "policy"):
        raise SystemExit(f"--match.opponent {config.opponent!r}: imugi or policy")
    return str(policy), IMUGI if config.opponent == IMUGI else str(policy)


def game_of(config: MatchConfig) -> GameConfig:
    """The match's game: the preset at the asked fleet size."""
    game = api.PRESETS[config.preset]
    if config.ships_per_team is None:
        return game
    from bnb.league.crossover import sized_game

    return sized_game(game, config.ships_per_team, config.map_rule)


def _sides(loop, sides: tuple[str, str], game: GameConfig, config: MatchConfig) -> list:
    """Each team's controller: Imugi, or the agent of a policy directory."""
    from bnb.play.sides import ImugiAgent, load_agent

    controllers = []
    for team, side in enumerate(sides):
        if side == IMUGI:
            controllers.append(ImugiAgent(loop.rules, loop.types, config.seed + team))
        else:
            controllers.append(
                load_agent(
                    side,
                    game,
                    loop.rules,
                    loop.types,
                    config.temperature,
                    config.seed + team,
                    graph=False,
                    any_mode=True,
                )
            )
    return controllers


def _ghosts(controller, team: int, observed, following) -> tuple | None:
    """(ship, x, y, sigma) of a side's view of the hidden enemies at its latest decision,
    or None (a side that keeps none)."""
    from bnb.play.predictions import side_predictions

    predictions = side_predictions(controller, team, observed, following)
    if predictions is None:
        return None
    ghosts = predictions.belief
    if ghosts is None and predictions.dead_reckoning is not None:
        ghosts = predictions.dead_reckoning.ghosts
    if ghosts is None:
        return None
    return ghosts.ship, ghosts.x, ghosts.y, ghosts.sigma


def meta_of(sides: tuple[str, str], config: MatchConfig, game) -> dict:
    """The match's provenance and the numbers its renderers need."""
    head, dirty = provenance.commit()
    policies = [side for side in sides if side != IMUGI]
    return dict(
        policy=" vs ".join(sides) if policies else "Imugi vs Imugi (scripted)",
        policy_commit=None,
        environment_steps=-1,
        commit=head,
        dirty=dirty,
        command=" ".join(["bnb", *sys.argv[1:]]),
        time=provenance.now(),
        config=dataclasses.asdict(config),
        game=json.dumps(to_json(game)),
    )


def about_of(sides: tuple[str, str], labels: list[str], config: MatchConfig, game, loop) -> dict:
    """The numbers the match's renderers need, kept in the record (`match["about"]`), so
    the record can join a policy's capture."""
    return dict(
        preset=config.preset,
        mode=loop.current.mode,
        ships_per_team=game.ships_per_team,
        tick_length=loop.tick_length,
        sides=list(sides),
        labels=labels,
        seed=config.seed,
    )


@torch.no_grad()
def record(sides: tuple[str, str], config: MatchConfig, game: GameConfig | None = None) -> dict:
    """Play one game with `sides` (each "imugi" or a policy directory) and return its
    artifact: `meta` and `match`. game: the game in place of the config's preset (a test's
    small one)."""
    from bnb.play.loop import MatchLoop, MatchSetup

    game = game_of(config) if game is None else game
    setup = MatchSetup(game, fogged=config.fogged, seed=config.seed)
    loop = MatchLoop(setup, sides=[None, None], compiled=False)
    loop.sides = _sides(loop, sides, game, config)
    events: list = []
    loop.on_tick = lambda actions, realtime, found: events.append(
        dict(
            fired=realtime.events.fired[0].numpy().copy(),
            hit=(realtime.events.damage[0].sum(0) > 0.0).numpy().copy(),
            died=realtime.events.died[0].numpy().copy(),
            killer=realtime.events.killer[0].numpy().copy(),
            done=bool(realtime.events.done[0]),
            result=int(realtime.events.result[0]),
        )
    )
    first = loop.current
    count = first.ships.x.shape[0]
    limit = None if config.seconds is None else round(config.seconds / loop.tick_length)
    frames: list = []
    ghosts: list = []
    started = time.perf_counter()
    while True:
        before = loop.current
        loop.step()
        frames.append(before)
        tick = len(frames) - 1
        sample = np.full((2, 3, count), np.nan, np.float32)
        if config.ghost_ticks and tick % config.ghost_ticks == 0:
            for team, controller in enumerate(loop.sides):
                found = _ghosts(controller, team, before, loop.current)
                if found is not None:
                    ship, x, y, sigma = found
                    sample[team, 0, ship], sample[team, 1, ship] = x, y
                    sample[team, 2, ship] = sigma
        ghosts.append(sample)
        if events[-1]["done"] or (limit is not None and len(frames) >= limit):
            break
    ships = [frame.ships for frame in frames]

    def stacked(name: str) -> np.ndarray:
        return np.stack([getattr(one, name) for one in ships])

    def happened(name: str) -> np.ndarray:
        return np.stack([event[name] for event in events])

    fields = first.fields
    ghost = np.stack(ghosts)  # (T, 2, 3, N)
    match = dict(
        x=stacked("x"),
        y=stacked("y"),
        heading=stacked("heading"),
        alive=stacked("alive"),
        shield=stacked("shield"),
        visible=stacked("visible"),  # (T, 2, N)
        team=first.ships.team.copy(),
        ship_type=first.ships.ship_type.copy(),
        fired=happened("fired"),
        hit=happened("hit"),
        died=happened("died"),
        killer=happened("killer"),
        done=np.array([event["done"] for event in events]),
        result=np.array([event["result"] for event in events]),
        clock=np.array([frame.clock for frame in frames], np.float32),
        boundary_radius=np.array([frame.boundary_radius for frame in frames], np.float32),
        score=np.stack([frame.score for frame in frames]),
        fogged=bool(first.fogged),
        arena_radius=float(first.arena_radius),
        fields=dict(
            x=fields.x,
            y=fields.y,
            radius=fields.radius,
            band_width=fields.band_width,
            level=fields.level,
            core_radius=fields.core_radius,
            outer_radius=fields.outer_radius,
        ),
        ghost_x=ghost[:, :, 0],
        ghost_y=ghost[:, :, 1],
        ghost_sigma=ghost[:, :, 2],
    )
    labels = [
        getattr(controller, "label", str(side)) for controller, side in zip(loop.sides, sides)
    ]
    match["about"] = about_of(sides, labels, config, game, loop)
    match["about"]["seconds_taken"] = round(time.perf_counter() - started, 1)
    meta = meta_of(sides, config, game)
    return dict(meta=meta, match=_tensors(match))


def _tensors(tree):
    """Every array of a nested dict as a tensor (an artifact holds tensors and plain values)."""
    if isinstance(tree, dict):
        return {name: _tensors(value) for name, value in tree.items()}
    if isinstance(tree, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(tree))
    return tree
