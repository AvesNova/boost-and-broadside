"""Scripted controllers over a team view: random (sticky), the chaser, and the scripted
agent (scripted spec A), which at its temperatures is also the stationary ladder's rungs,
and the BC teacher.

All read only the observation, never the game state.
"""

import dataclasses
import math

import torch

from bnb.contracts import Action, Obs, Outputs
from bnb.contracts.action import NUM_POWER, NUM_SHOOT, NUM_STICK
from bnb.contracts.obs import GLOBAL, SHIP
from bnb.game.handling.characteristics import Characteristics
from bnb.game.modes.rules import PRESETS
from bnb.game.rules import FlightConstants, Rules, rules_of

from .scripted_combat import Marginals, combat, context
from .scripted_defaults import MODE_DEFAULTS
from .scripted_doctrine import for_card
from .scripted_frontline import frontline
from .scripted_memory import ScriptedCarry, forgotten, fresh_carry, remember
from .scripted_parameters import MODES, ModeParameters, ScriptedParameters

REDRAW = 0.1  # sticky random: per-tick probability of a new action
STORED = "stored"  # the scripted agent's default doctrines: the stored ones of its card's set
# The standard card's lengths from its characteristics at 30 Hz, to the last bit
# (`gun_range` and `corner_radius`): its longest gun range (the fighter's) and its largest
# corner radius (the heavy's: the corner speed squared over the g limit's acceleration, the
# radius of a full-stick turn at the corner speed). Distances below are stated in these, so
# the standard card plays exactly the distances in metres they were set in, and any other
# card the same distances in its own lengths.
STANDARD_GUN_RANGE = 220.78790283203125  # m
STANDARD_CORNER_RADIUS = 45.45454775076505  # m


def uniform_actions(shape: tuple, device) -> torch.Tensor:
    """Actions (*shape, 3) int32 drawn uniformly from the global generator, in the order power,
    stick, shoot."""
    draws = torch.rand(shape + (3,), device=device)
    # scalar factors, not a tensor built from host values: a host copy cannot run inside a
    # CUDA graph capture (an eager rollout's, such as the head-to-head's)
    scaled = [
        draws[..., index] * count for index, count in enumerate((NUM_POWER, NUM_STICK, NUM_SHOOT))
    ]
    return torch.stack(scaled, -1).to(torch.int32)


class Random:
    """Uniform random actions, each ship redrawing with probability `REDRAW` per tick."""

    reads = ()

    def __init__(self, ships_per_team: int) -> None:
        self.ships_per_team = ships_per_team

    def init_carry(self, views: int, device: torch.device) -> torch.Tensor:
        """The held actions (R, S, 3) int32, drawn uniformly."""
        return uniform_actions((views, self.ships_per_team), device)

    def step(self, carry: torch.Tensor, obs: Obs, reset: torch.Tensor):
        """One decision for views (R, ...): each ship keeps its held action or, with probability
        REDRAW, draws a new one. The carry is the held actions."""
        redraw = torch.rand(carry.shape[:-1], device=carry.device) < REDRAW
        drawn = uniform_actions(carry.shape[:-1], carry.device)
        held = torch.where(redraw[..., None], drawn, carry)
        return Action.unpack(held), held, Outputs()


def gun_range(characteristics: Characteristics) -> float:
    """The card's longest gun range (m): the length the agents' gun distances are stated in."""
    return float(characteristics.gun_range.amax())


def corner_radius(characteristics: Characteristics) -> float:
    """The card's largest corner radius (m), corner speed^2 / a_max in float64: the length
    the chaser's flight distances are stated in."""
    speed = characteristics.corner_speed.double()
    return float((speed * speed / characteristics.max_acceleration.double()).amax())


# The chaser's distances (race BENCHMARK 3.1: boost beyond 600 m, reverse inside 120 m and
# shoot inside 250 m, on the standard card): the flight's in corner radii, the gun's in gun
# ranges.
BOOST_IN_CORNER_RADII = 600.0 / STANDARD_CORNER_RADIUS
REVERSE_IN_CORNER_RADII = 120.0 / STANDARD_CORNER_RADIUS
SHOOT_IN_GUN_RANGES = 250.0 / STANDARD_GUN_RANGE


class Chaser:
    """Turn toward and shoot at the nearest visible living enemy (race BENCHMARK 3.1 rules),
    at distances in the card's lengths (`BOOST_IN_CORNER_RADII`, `REVERSE_IN_CORNER_RADII`,
    `SHOOT_IN_GUN_RANGES`): the race's 600, 120 and 250 m on the standard card."""

    reads = ("ships",)

    def __init__(self, characteristics: Characteristics, ships_per_team: int) -> None:
        self.ships_per_team = ships_per_team
        self.retune(characteristics)

    def retune(self, characteristics: Characteristics) -> None:
        """Play at the distances of this card's characteristics."""
        radius, reach = corner_radius(characteristics), gun_range(characteristics)
        self.boost_distance = BOOST_IN_CORNER_RADII * radius  # m
        self.reverse_distance = REVERSE_IN_CORNER_RADII * radius  # m
        self.shoot_distance = SHOOT_IN_GUN_RANGES * reach  # m

    def init_carry(self, views: int, device: torch.device) -> None:
        """No carry: the chaser is stateless."""
        return None

    def step(self, carry: None, obs: Obs, reset: torch.Tensor):
        """Each own ship turns toward and shoots at its nearest visible living enemy."""
        S = self.ships_per_team
        own, enemy = obs.ships[:, :S], obs.ships[:, S:]  # rows are ordered by team_relative
        to_x = enemy[:, None, :, SHIP["x"]] - own[:, :, None, SHIP["x"]]  # (R, own, enemy)
        to_y = enemy[:, None, :, SHIP["y"]] - own[:, :, None, SHIP["y"]]
        target = (enemy[..., SHIP["visible"]] > 0.5) & (enemy[..., SHIP["alive"]] > 0.5)
        squared = torch.where(target[:, None, :], to_x * to_x + to_y * to_y, math.inf)
        pick = torch.nn.functional.one_hot(squared.argmin(-1), squared.shape[-1]).bool()
        # ties go to the lowest row
        to_x, to_y = torch.where(pick, to_x, 0.0).sum(-1), torch.where(pick, to_y, 0.0).sum(-1)
        cos, sin = own[..., SHIP["heading_cos"]], own[..., SHIP["heading_sin"]]
        signed = torch.atan2(cos * to_y - sin * to_x, cos * to_x + sin * to_y)
        bearing, distance = signed.abs(), torch.sqrt(to_x * to_x + to_y * to_y)
        level = sum((bearing >= math.radians(degrees)).int() for degrees in (3.0, 10.0, 30.0))
        stick = torch.where(level == 0, 0, 2 * level - (signed < 0.0).int())
        power = torch.where(
            distance > self.boost_distance, 1, torch.where(distance < self.reverse_distance, 2, 0)
        )
        shoot = ((bearing < math.radians(8.0)) & (distance < self.shoot_distance)).int()
        any_target = target.any(-1)[:, None]
        action = Action(
            *(torch.where(any_target, part, 0).to(torch.int32) for part in (power, stick, shoot))
        )
        return action, carry, Outputs()


# A team deathmatch view plays deathmatch_wide's set when its arena radius is more than this
# many of the card's longest gun ranges, the geometric mean of the two presets' on the
# standard card (2600 and 8200 m: 11.8 and 37.1), else deathmatch's.
WIDE_ARENA_IN_GUN_RANGES = (
    math.sqrt(PRESETS["deathmatch"].arena_radius * PRESETS["deathmatch_wide"].arena_radius)
    / STANDARD_GUN_RANGE
)


# A card plays the real-scale sets when its gravity is below this, the geometric mean of
# the standard game's and the real-scale set's (27 and 9.81 m/s^2).
REAL_SCALE_GRAVITY = math.sqrt(
    FlightConstants.gravity * rules_of(PRESETS["real_scale_deathmatch"]).flight.gravity
)


def mode_index(obs: Obs, characteristics: Characteristics) -> torch.Tensor:
    """(R,) int64 each view's mode, an index into `MODES`. On a real-scale card (its gravity
    below REAL_SCALE_GRAVITY): real_scale_frontline or real_scale_elimination by the mode's
    flag, else real_scale_deathmatch. Otherwise frontline by the mode's flag, else
    deathmatch_wide beyond WIDE_ARENA_IN_GUN_RANGES of the card's longest gun range, else
    deathmatch (a sandbox view and an elimination view too)."""
    globals_ = obs.globals
    wide_radius = WIDE_ARENA_IN_GUN_RANGES * characteristics.gun_range.amax()
    wide = globals_[:, GLOBAL["arena_radius"]] > wide_radius
    frontline_view = globals_[:, GLOBAL["mode_frontline"]] > 0.5
    index = torch.where(wide, MODES.index("deathmatch_wide"), MODES.index("deathmatch"))
    index = torch.where(frontline_view, MODES.index("frontline"), index)
    elimination_view = globals_[:, GLOBAL["mode_elimination"]] > 0.5
    real = torch.where(
        frontline_view,
        MODES.index("real_scale_frontline"),
        torch.where(
            elimination_view,
            MODES.index("real_scale_elimination"),
            MODES.index("real_scale_deathmatch"),
        ),
    )
    return torch.where(characteristics.gravity.amax() < REAL_SCALE_GRAVITY, real, index)


def resolved(
    parameters: ScriptedParameters | ModeParameters, obs: Obs, characteristics: Characteristics
) -> ScriptedParameters:
    """The parameters each view plays with: a per-mode mapping's set for the view's mode."""
    if isinstance(parameters, ModeParameters):
        return parameters.per_view(mode_index(obs, characteristics))
    return parameters


def decide(
    obs: Obs,
    characteristics: Characteristics,
    parameters: ScriptedParameters | ModeParameters,
    temperature: torch.Tensor | float = 1.0,
    carry: ScriptedCarry | None = None,
    reset: torch.Tensor | None = None,
    seconds: float = Rules.tick_length,
    switching: bool = False,
    doctrines: torch.Tensor | None = None,
) -> tuple[Marginals, ScriptedCarry]:
    """The scripted agent's power, stick and shoot marginals per own ship at a temperature (a
    float, or (R, 1) per view), given its carry, and the carry after the decision: the
    combat layer, with the Frontline layer on top in Frontline. A dead ship (or one waiting
    to respawn) takes the no-op action; at temperature +inf, or where the carry's agent flag
    is off, a living ship plays uniformly (random).

    carry None is a fresh one (nothing remembered: the memoryless agent); reset (R,) the
    views whose episode reset before this observation. With `switching` the plan choices may
    switch (a decision); without, only a fresh ship or a lost target re-chooses, so a carry
    `decide` returned gives the same marginals again (a label's re-reading). parameters: one
    set, per view, or per mode (`ModeParameters`: each view its mode's). doctrines: each
    ship type's doctrine (`bnb.controllers.scripted_doctrine`), a table (T, D) or per view
    (R, T, D); None flies none."""
    parameters = resolved(parameters, obs, characteristics)
    ships = obs.ships
    views, S = ships.shape[0], ships.shape[-2] // 2
    if carry is None:
        carry = fresh_carry(views, S, ships.shape[-2] - S, ships.device)
    if reset is None:
        reset = torch.zeros(views, dtype=torch.bool, device=ships.device)
    carry, recall = remember(carry, obs, reset, characteristics, parameters, temperature)
    fleet = context(
        obs, characteristics, parameters, temperature, carry, recall, seconds, switching
    )
    fighting = combat(fleet, characteristics, parameters, doctrines)
    strategy = frontline(fleet, obs.zones, characteristics, parameters, fighting)
    in_frontline = (obs.globals[:, None, GLOBAL["mode_frontline"]] > 0.5)[..., None]
    alive = (fleet.own[..., SHIP["alive"]] > 0.5)[..., None]
    uniform = ~carry.agent[..., None]
    # a float is compared on the host: a host copy cannot run inside a graph capture
    if isinstance(temperature, torch.Tensor):
        uniform = uniform | torch.isposinf(temperature)[..., None]  # (R, 1, 1)
    elif temperature == math.inf:
        uniform = torch.ones_like(uniform)

    def chosen(combat_part: torch.Tensor, frontline_part: torch.Tensor) -> torch.Tensor:
        part = torch.where(in_frontline, frontline_part, combat_part)
        part = torch.where(uniform, 1.0 / part.shape[-1], part)
        no_op = torch.nn.functional.one_hot(
            torch.zeros_like(part[..., 0], dtype=torch.long), part.shape[-1]
        ).float()
        return torch.where(alive, part, no_op)

    parts = Marginals(
        *(chosen(a, b) for a, b in zip(fighting.marginals, strategy.marginals, strict=True))
    )
    after = dataclasses.replace(
        carry,
        target=fighting.target,
        combat_left=fighting.left,
        frontline_left=strategy.left,
        defending=strategy.defending,
        recovering=strategy.recovering,
        fresh=torch.zeros_like(carry.fresh),
    )
    return parts, after


def marginals(
    obs: Obs,
    characteristics: Characteristics,
    parameters: ScriptedParameters | ModeParameters,
    temperature: torch.Tensor | float = 1.0,
    carry: ScriptedCarry | None = None,
    doctrines: torch.Tensor | None = None,
) -> Marginals:
    """The marginals `decide` gives (without switching): with carry None, the memoryless
    agent's, from the observation alone."""
    return decide(obs, characteristics, parameters, temperature, carry, doctrines=doctrines)[0]


def joint(parts: Marginals) -> torch.Tensor:
    """(R, S, NUM_JOINT) the outer product of the marginals, in `to_joint` order."""
    product = (
        parts.power[..., :, None, None]
        * parts.stick[..., None, :, None]
        * parts.shoot[..., None, None, :]
    )
    return product.flatten(-3)


def distribution(
    obs: Obs,
    characteristics: Characteristics,
    parameters: ScriptedParameters | ModeParameters,
    temperature: torch.Tensor | float = 1.0,
    carry: ScriptedCarry | None = None,
) -> torch.Tensor:
    """(R, S, NUM_JOINT) the scripted agent's joint action probabilities at a temperature,
    given its carry (None: memoryless), `to_joint` order."""
    return joint(marginals(obs, characteristics, parameters, temperature, carry))


def sample(probabilities: torch.Tensor) -> torch.Tensor:
    """One index per row of (..., K) probabilities, by the inverse CDF (a category of
    probability 0 is never drawn)."""
    cumulative = probabilities.cumsum(-1)
    draw = torch.rand(probabilities.shape[:-1] + (1,), device=probabilities.device)
    return (draw * cumulative[..., -1:] >= cumulative[..., :-1]).sum(-1)


class Scripted:
    """The scripted agent, and with it every rung of the stationary ladder (spec A6).

    temperature: a float, or a tensor (R,) per view: every logistic's and switch's scale is
    multiplied by it, so 1 is the agent and a larger one flattens every decision toward an
    even choice; +inf plays uniformly over the 42 joint actions (the ladder's random).
    share: a float, or a tensor (R,) per view: each ship plays the agent with this
    probability, drawn at each episode reset, else uniformly (a mixture rung; 1 is the
    agent). The league may change either tensor in place. ships_per_team: S, the carry's
    size. parameters: per mode (the default, `MODE_DEFAULTS`: each view plays its game
    mode's set), one set, or per view (`scripted_parameters.unpack` of a table). doctrines:
    each ship type's doctrine, a table (T, D) or per view (R, T, D)
    (`bnb.controllers.scripted_doctrine`); STORED, the default, the stored tuned ones of the
    card's ship set (`scripted_doctrine.for_card`); None flies none. memory:
    with False, every decision is the memoryless agent's (each plan choice made afresh,
    only visible enemies known); the carry still holds the ships' agent flags. seconds: a
    decision's length (the switches' rates are per second).
    """

    reads = ("ships", "zones", "globals")

    def __init__(
        self,
        characteristics: Characteristics,
        ships_per_team: int,
        parameters: ScriptedParameters | ModeParameters = MODE_DEFAULTS,
        temperature: float | torch.Tensor = 1.0,
        share: float | torch.Tensor = 1.0,
        memory: bool = True,
        seconds: float = Rules.tick_length,
        doctrines: torch.Tensor | str | None = STORED,
    ) -> None:
        if isinstance(doctrines, str):
            if doctrines != STORED:
                raise ValueError(f"doctrines {doctrines!r}: a table, None or {STORED!r}")
            doctrines = for_card(characteristics)
        if not isinstance(temperature, torch.Tensor) and not temperature > 0.0:
            raise ValueError(f"the scripted agent's temperature {temperature} must be positive")
        if not isinstance(share, torch.Tensor) and not 0.0 <= share <= 1.0:
            raise ValueError(f"the scripted agent's share {share} must be in [0, 1]")
        self.characteristics, self.parameters = characteristics, parameters
        if isinstance(parameters, ModeParameters):  # on the device before any graph capture
            parameters.table(characteristics.gun_range.device)
        self.ships_per_team = ships_per_team
        self.temperature, self.share = temperature, share
        self.memory, self.seconds = memory, seconds
        self.doctrines = doctrines

    def init_carry(self, views: int, device: torch.device) -> ScriptedCarry:
        """Nothing committed or remembered (the first observation's reset draws the agent
        flags)."""
        return fresh_carry(views, self.ships_per_team, self.ships_per_team, device)

    def _temperature(self) -> torch.Tensor | float:
        """The temperature per view, (R, 1) for a tensor."""
        temperature = self.temperature
        return temperature[:, None] if isinstance(temperature, torch.Tensor) else temperature

    def _agent_flags(self, carry: ScriptedCarry, reset: torch.Tensor) -> ScriptedCarry:
        """The agent flags, redrawn at each reset by the share."""
        share = self.share
        if not isinstance(share, torch.Tensor) and share == 1.0:
            return carry
        share = share[:, None] if isinstance(share, torch.Tensor) else share
        drawn = torch.rand(carry.agent.shape, device=carry.agent.device) < share
        return dataclasses.replace(carry, agent=torch.where(reset[:, None], drawn, carry.agent))

    def decide(
        self, carry: ScriptedCarry, obs: Obs, reset: torch.Tensor
    ) -> tuple[Marginals, ScriptedCarry]:
        """One decision's marginals and the carry after it (`decide`, switching)."""
        carry = self._agent_flags(carry, reset)
        if not self.memory:
            carry = forgotten(carry)
        return decide(
            obs,
            self.characteristics,
            self.parameters,
            self._temperature(),
            carry,
            reset,
            self.seconds,
            switching=True,
            doctrines=self.doctrines,
        )

    def marginals(self, obs: Obs, carry: ScriptedCarry | None = None) -> Marginals:
        """The marginals given a carry `decide` returned (None: the memoryless agent's)."""
        return marginals(
            obs,
            self.characteristics,
            self.parameters,
            self._temperature(),
            carry,
            self.doctrines,
        )

    def action_distribution(self, obs: Obs, carry: ScriptedCarry | None = None) -> torch.Tensor:
        """(R, S, NUM_JOINT) the distribution given a carry `decide` returned."""
        return joint(self.marginals(obs, carry))

    def step(self, carry: ScriptedCarry, obs: Obs, reset: torch.Tensor):
        """One decision for views (R, ...): an action sampled from each marginal."""
        parts, carry = self.decide(carry, obs, reset)
        action = torch.stack([sample(part) for part in parts], -1)
        return Action.unpack(action.to(torch.int32)), carry, Outputs()
