"""The prediction overlays: what each agent thinks, drawn on the battlefield beside what is
true, and the per-tick predictions they draw, as numpy arrays.

Each overlay is toggled on its own (`OverlaySettings`). The true bullet paths are physics
and every match offers them; the others come from a side's controller, which names the
ones it can provide (its `overlays`): a Yemong agent its heads' predictions, the scripted
agent its dead-reckoned enemies, the chaser nothing. An overlay draws one side's
predictions: the side the viewer picked, or by default the selected ship's side when it
provides that overlay, else the other side when it does (`provider`).

The predictions are those of the side's latest decision, from the observation the
previous snapshot shows (`bnb.play.predictions` makes them with the decoder layer,
`bnb.models.yemong.decode`); the truth beside them is that snapshot's. Rendering reads
these arrays and never imports torch.
"""

import dataclasses

import numpy as np

# Every overlay in the order the card lists them: name, label, what it shows.
OVERLAYS = (
    ("bullet_paths", "True bullet paths", "Each live bullet's flight ahead through the fields"),
    (
        "shot_path",
        "Predicted shot path",
        "The shot head's path for the selected ship's next shot, with its one-sigma circles",
    ),
    (
        "next_state",
        "Next-state ghosts",
        "Each ship's predicted state one tick later, with its one-sigma circle",
    ),
    (
        "belief",
        "Belief ghosts",
        "Where the side believes each enemy it cannot see is, with its one-sigma circle",
    ),
    (
        "dead_reckoning",
        "Dead-reckoning ghosts",
        "Each remembered enemy the scripted agent cannot see, dead reckoned from its last "
        "sighting; its opacity is the agent's weight on it",
    ),
    (
        "density",
        "Density",
        "The density head's expected ships per hex cell",
    ),
    (
        "enemy_action",
        "Enemy action",
        "The predicted action of the selected enemy, in the selection card",
    ),
    (
        "value",
        "Value and outcome",
        "The critic's value per reward level and the outcome head, in the selection card",
    ),
)
NAMES = tuple(name for name, _, _ in OVERLAYS)
LABELS = {name: label for name, label, _ in OVERLAYS}
HELP = {name: text for name, _, text in OVERLAYS}
PHYSICS = ("bullet_paths",)  # what every match offers
NETWORK = ("shot_path", "next_state", "belief", "density", "enemy_action", "value")
MEMORY = ("dead_reckoning",)
TRUTH = ("belief", "dead_reckoning", "density")  # the overlays the truth option adds to
DENSITY_SIDES = ("enemy", "ally")
DENSITY_SHOWN = ("predicted", "true", "error")


@dataclasses.dataclass
class OverlaySettings:
    """The viewer's choices: the overlays on, whose predictions they draw (a team, or None
    for the default, `provider`), the truth beside the predictions, and the density's
    side (the predicting side's "enemy" or "ally" cells) and what it shows ("predicted",
    and with the truth "true" or the signed "error", predicted minus true)."""

    enabled: set[str] = dataclasses.field(default_factory=set)
    side: int | None = None
    truth: bool = False
    density_side: str = "enemy"
    density_shown: str = "predicted"


def offered_by(controller) -> tuple[str, ...]:
    """The overlays a side's controller can provide (none for a controller that names none)."""
    return tuple(getattr(controller, "overlays", ()))


def offered(sides: list) -> tuple[str, ...]:
    """The overlays this match offers, in card order: the physics ones, and every overlay
    a side's controller provides (a replay runs no controllers: physics only)."""
    provided = set(PHYSICS).union(*(offered_by(side) for side in sides))
    return tuple(name for name in NAMES if name in provided)


def provider(
    name: str, sides: list, chosen: int | None, selected_team: int | None, view_team: int
) -> int | None:
    """The team whose predictions overlay `name` draws, or None when no side provides it.
    chosen: the viewer's pick; else the selected ship's team (selected_team) when it
    provides it, else the other team when it does; with nothing selected the view's team
    (view_team, -1 for the spectator) first, then team 0."""
    providing = [team for team, side in enumerate(sides) if name in offered_by(side)]
    if chosen is not None:
        return chosen if chosen in providing else None
    first = selected_team if selected_team is not None else max(view_team, 0)
    for team in (first, 1 - first):
        if team in providing:
            return team
    return None


@dataclasses.dataclass(frozen=True)
class BulletPaths:
    """Each live bullet's flight ahead (P bullets), from its position now: x, y (P, A + 1)
    m, valid (P, A + 1) (the bullet still flies there), and the shooter's team (P,)."""

    x: np.ndarray
    y: np.ndarray
    valid: np.ndarray
    team: np.ndarray


@dataclasses.dataclass(frozen=True)
class ShotPaths:
    """The shot head's path for the shot each ship (N, sim order) would fire at the
    decision: from the launch point launch_x, launch_y (N,) m, the mean at each age x, y
    (N, A) m in world coordinates, its sigma (N, A) m (per axis), and the outcome head's
    P(enemy hit) (N,)."""

    launch_x: np.ndarray
    launch_y: np.ndarray
    x: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    enemy_hit: np.ndarray


@dataclasses.dataclass(frozen=True)
class Ghosts:
    """Predicted ships (G of them): the ship (sim index), position x, y (m), heading (rad),
    sigma (m, per axis, 0 for none), opacity in [0, 1], and the truth at the same time
    (true_x, true_y, m; true_alive)."""

    ship: np.ndarray
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    sigma: np.ndarray
    opacity: np.ndarray
    true_x: np.ndarray
    true_y: np.ndarray
    true_alive: np.ndarray


@dataclasses.dataclass(frozen=True)
class Reckoning:
    """The scripted agent's remembered enemies it cannot see (G): its ghosts (`Ghosts`,
    the opacity its weight, no sigma) and each one's last sighting last_x, last_y (m)."""

    ghosts: Ghosts
    last_x: np.ndarray
    last_y: np.ndarray


@dataclasses.dataclass(frozen=True)
class Density:
    """The density head's field on the hex cells (C): centres (C, 2) m, cell_radius (m,
    centre to corner), and per side ("ally", "enemy" of the predicting team) the
    expected counts `predicted` and the true counts `true` (each a dict of (C,))."""

    centres: np.ndarray
    cell_radius: float
    predicted: dict[str, np.ndarray]
    true: dict[str, np.ndarray]


@dataclasses.dataclass(frozen=True)
class EnemyActions:
    """The enemy-action head's distribution for each enemy (E, sim index `ship`): the
    marginals power (E, 3), stick (E, 7) and shoot (E, 2)."""

    ship: np.ndarray
    power: np.ndarray
    stick: np.ndarray
    shoot: np.ndarray


@dataclasses.dataclass(frozen=True)
class Critic:
    """The critic for the own ships (S, sim index `ship`): values (S, levels) in the
    critic's units, by level name (`levels`), and the outcome head's P(win), P(tie),
    P(loss), P(unresolved) and its value P(win) - P(loss)."""

    ship: np.ndarray
    levels: tuple[str, ...]
    values: np.ndarray
    win: float
    tie: float
    loss: float
    unresolved: float
    value: float


@dataclasses.dataclass(frozen=True)
class Predictions:
    """One side's predictions at its latest decision: the team, the episode tick of the
    observation they read, and each overlay's arrays (None where the side provides none)."""

    team: int
    tick: int
    shot_path: ShotPaths | None = None
    next_state: Ghosts | None = None
    belief: Ghosts | None = None
    dead_reckoning: Reckoning | None = None
    density: Density | None = None
    enemy_action: EnemyActions | None = None
    value: Critic | None = None
