"""Imugi, Imugi (scripted spec A): the one controller that plays by the
per-mode parameter sets, which at its temperatures is also the stationary ladder's rungs,
and the BC teacher.

Reads only the observation, never the game state.
"""

import dataclasses
import math

import torch

from bnb.contracts import Action, Obs, Outputs
from bnb.contracts.obs import GLOBAL, SHIP
from bnb.game.handling.characteristics import Characteristics
from bnb.game.modes.rules import PRESETS
from bnb.game.rules import Rules
from bnb.streams import action_uniforms, inverse_cdf, purpose_key, row_uniforms

from .imugi_combat import Marginals, combat, context
from .imugi_defaults import MODE_DEFAULTS
from .imugi_doctrine import for_card, real_scale_card
from .imugi_frontline import frontline
from .imugi_memory import (
    ImugiCarry,
    explore_cells,
    forgotten,
    fresh_carry,
    remember,
    track_boundary,
    track_seen,
)
from .imugi_parameters import MODES, ImugiParameters, ModeParameters
from .scripted import STANDARD_GUN_RANGE

STORED = "stored"  # Imugi's default doctrines: the stored ones of its card's set
# The random-stream salt keeps its recorded text, so the draws are those of every earlier run.
AGENT_FLAGS = purpose_key("Imugi flags")


# A team deathmatch view plays deathmatch_wide's set when its arena radius is more than this
# many of the card's longest gun ranges, the geometric mean of the two presets' on the
# standard card (2600 and 8200 m: 11.8 and 37.1), else deathmatch's.
WIDE_ARENA_IN_GUN_RANGES = (
    math.sqrt(PRESETS["deathmatch"].arena_radius * PRESETS["deathmatch_wide"].arena_radius)
    / STANDARD_GUN_RANGE
)


def mode_index(obs: Obs, characteristics: Characteristics) -> torch.Tensor:
    """(R,) int64 each view's mode, an index into `MODES`. On a real-scale card
    (`real_scale_card`): real_scale_frontline or real_scale_elimination by the mode's flag,
    else real_scale_deathmatch. Otherwise frontline by the mode's flag, else
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
    return torch.where(real_scale_card(characteristics.gravity.amax()), real, index)


def resolved(
    parameters: ImugiParameters | ModeParameters, obs: Obs, characteristics: Characteristics
) -> ImugiParameters:
    """The parameters each view plays with: a per-mode mapping's set for the view's mode."""
    if isinstance(parameters, ModeParameters):
        return parameters.per_view(mode_index(obs, characteristics))
    return parameters


def decide(
    obs: Obs,
    characteristics: Characteristics,
    parameters: ImugiParameters | ModeParameters,
    temperature: torch.Tensor | float = 1.0,
    carry: ImugiCarry | None = None,
    reset: torch.Tensor | None = None,
    seconds: float = Rules.tick_length,
    decision_key: torch.Tensor | None = None,
    doctrines: torch.Tensor | None = None,
    rules: tuple | None = None,
) -> tuple[Marginals, ImugiCarry]:
    """Imugi's power, stick and shoot marginals per own ship at a temperature (a
    float, or (R, 1) per view), given its carry, and the carry after the decision: the
    combat layer, with the Frontline layer on top in Frontline. A dead ship (or one waiting
    to respawn) takes the no-op action; at temperature +inf, or where the carry's agent flag
    is off, a living ship plays uniformly (random).

    carry None is a fresh one (nothing remembered: the memoryless agent); reset (R,) the
    views whose episode reset before this observation. With a `decision_key` (R,) int64
    (`Controller.step`) the plan choices may switch, drawn from it (a decision); without,
    only a fresh ship or a lost target re-chooses, so a carry `decide` returned gives the
    same marginals again (a label's re-reading). parameters: one
    set, per view, or per mode (`ModeParameters`: each view its mode's). doctrines: each
    ship type's doctrine (`bnb.controllers.imugi_doctrine`), a table (T, D) or per view
    (R, T, D); None flies none. rules: (turn commit, stick by energy), each a bool or
    (R, 1) bool per view (`imugi_combat.Context`; None: the process defaults)."""
    parameters = resolved(parameters, obs, characteristics)
    ships = obs.ships
    views, S = ships.shape[0], ships.shape[-2] // 2
    if carry is None:
        carry = fresh_carry(views, S, ships.shape[-2] - S, ships.device)
    if reset is None:
        reset = torch.zeros(views, dtype=torch.bool, device=ships.device)
    carry, recall = remember(carry, obs, reset, characteristics, parameters, temperature)
    carry = track_seen(carry, obs, reset, characteristics, seconds)  # before track_boundary
    carry = track_boundary(carry, obs, reset, seconds)
    fleet = context(
        obs, characteristics, parameters, temperature, carry, recall, seconds, decision_key, rules
    )
    fighting = combat(fleet, characteristics, parameters, doctrines)
    strategy = frontline(fleet, obs.zones, characteristics, parameters, fighting, doctrines)
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
        target_doubt=fighting.target_doubt,
        extend_phase=fighting.extend_phase,
        fight_sense=fighting.fight_sense,
        fight_clock=fighting.fight_clock,
        frontline_left=strategy.left,
        defending=strategy.defending,
        fresh=torch.zeros_like(carry.fresh),
    )
    return parts, after


def marginals(
    obs: Obs,
    characteristics: Characteristics,
    parameters: ImugiParameters | ModeParameters,
    temperature: torch.Tensor | float = 1.0,
    carry: ImugiCarry | None = None,
    doctrines: torch.Tensor | None = None,
    rules: tuple | None = None,
) -> Marginals:
    """The marginals `decide` gives (without switching): with carry None, the memoryless
    agent's, from the observation alone."""
    return decide(
        obs, characteristics, parameters, temperature, carry, doctrines=doctrines, rules=rules
    )[0]


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
    parameters: ImugiParameters | ModeParameters,
    temperature: torch.Tensor | float = 1.0,
    carry: ImugiCarry | None = None,
) -> torch.Tensor:
    """(R, S, NUM_JOINT) Imugi's joint action probabilities at a temperature,
    given its carry (None: memoryless), `to_joint` order."""
    return joint(marginals(obs, characteristics, parameters, temperature, carry))


def sample(parts: Marginals, decision_key: torch.Tensor) -> torch.Tensor:
    """(R, S, 3) one action per own ship from the marginals (power, stick, shoot), each by
    the inverse CDF of its uniform (`action_uniforms`; a category of probability 0 is
    never drawn)."""
    uniforms = action_uniforms(decision_key, parts.power.shape[1])
    return torch.stack(
        [inverse_cdf(part, uniforms[..., index]) for index, part in enumerate(parts)], -1
    )


class Imugi:
    """Imugi, and with it every rung of the stationary ladder (spec A6).

    temperature: a float, or a tensor (R,) per view: every logistic's and switch's scale is
    multiplied by it, so 1 is the agent and a larger one flattens every decision toward an
    even choice; +inf plays uniformly over the 42 joint actions (the ladder's random).
    share: a float, or a tensor (R,) per view: each ship plays the agent with this
    probability, drawn at each episode reset, else uniformly (a mixture rung; 1 is the
    agent). The league may change either tensor in place. ships_per_team: S, the carry's
    size. parameters: per mode (the default, `MODE_DEFAULTS`: each view plays its game
    mode's set), one set, or per view (`imugi_parameters.unpack` of a table). doctrines:
    each ship type's doctrine, a table (T, D) or per view (R, T, D)
    (`bnb.controllers.imugi_doctrine`); STORED, the default, the stored tuned ones of the
    card's ship set (`imugi_doctrine.for_card`); None flies none. memory:
    with False, every decision is the memoryless agent's (each plan choice made afresh,
    only visible enemies known); the carry still holds the ships' agent flags. seconds: a
    decision's length (the switches' rates are per second).
    """

    reads = ("ships", "zones", "globals")

    def __init__(
        self,
        characteristics: Characteristics,
        ships_per_team: int,
        parameters: ImugiParameters | ModeParameters = MODE_DEFAULTS,
        temperature: float | torch.Tensor = 1.0,
        share: float | torch.Tensor = 1.0,
        memory: bool = True,
        seconds: float = Rules.tick_length,
        doctrines: torch.Tensor | str | None = STORED,
        rules: tuple | None = None,
    ) -> None:
        if isinstance(doctrines, str):
            if doctrines != STORED:
                raise ValueError(f"doctrines {doctrines!r}: a table, None or {STORED!r}")
            doctrines = for_card(characteristics)
        if not isinstance(temperature, torch.Tensor) and not temperature > 0.0:
            raise ValueError(f"Imugi's temperature {temperature} must be positive")
        if not isinstance(share, torch.Tensor) and not 0.0 <= share <= 1.0:
            raise ValueError(f"Imugi's share {share} must be in [0, 1]")
        self.characteristics, self.parameters = characteristics, parameters
        if isinstance(parameters, ModeParameters):  # on the device before any graph capture
            parameters.table(characteristics.gun_range.device)
        # the explore grid on the device before any compile or graph capture: copied inside a
        # traced step it would be a host-to-device copy that a capture refuses
        explore_cells(characteristics.gun_range.device)
        self.ships_per_team = ships_per_team
        self.temperature, self.share = temperature, share
        self.memory, self.seconds = memory, seconds
        self.doctrines = doctrines
        # (turn commit, stick by energy), each a bool or (R,) bool per view (a tensor may be
        # written in place between graph replays); None: the process defaults, from
        # BNB_TURN_COMMIT and BNB_STICK_BY_ENERGY (both on: v25's rules; both off: v24's)
        self.rules = rules

    def _rules(self) -> tuple | None:
        """The rules per view, (R, 1) for a tensor."""
        if self.rules is None:
            return None
        return tuple(
            one[:, None] if isinstance(one, torch.Tensor) else bool(one) for one in self.rules
        )

    def init_carry(self, views: int, device: torch.device) -> ImugiCarry:
        """Nothing committed or remembered (the first observation's reset draws the agent
        flags)."""
        return fresh_carry(views, self.ships_per_team, self.ships_per_team, device)

    def _temperature(self) -> torch.Tensor | float:
        """The temperature per view, (R, 1) for a tensor."""
        temperature = self.temperature
        return temperature[:, None] if isinstance(temperature, torch.Tensor) else temperature

    def _agent_flags(
        self, carry: ImugiCarry, reset: torch.Tensor, decision_key: torch.Tensor
    ) -> ImugiCarry:
        """The agent flags, redrawn at each reset by the share."""
        share = self.share
        if not isinstance(share, torch.Tensor) and share == 1.0:
            return carry
        share = share[:, None] if isinstance(share, torch.Tensor) else share
        drawn = row_uniforms(AGENT_FLAGS, decision_key, carry.agent.shape[1:]) < share
        return dataclasses.replace(carry, agent=torch.where(reset[:, None], drawn, carry.agent))

    def decide(
        self, carry: ImugiCarry, obs: Obs, reset: torch.Tensor, decision_key: torch.Tensor
    ) -> tuple[Marginals, ImugiCarry]:
        """One decision's marginals and the carry after it (`decide`, switching by the
        decision's key)."""
        carry = self._agent_flags(carry, reset, decision_key)
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
            decision_key=decision_key,
            doctrines=self.doctrines,
            rules=self._rules(),
        )

    def marginals(self, obs: Obs, carry: ImugiCarry | None = None) -> Marginals:
        """The marginals given a carry `decide` returned (None: the memoryless agent's)."""
        return marginals(
            obs,
            self.characteristics,
            self.parameters,
            self._temperature(),
            carry,
            self.doctrines,
            self._rules(),
        )

    def action_distribution(self, obs: Obs, carry: ImugiCarry | None = None) -> torch.Tensor:
        """(R, S, NUM_JOINT) the distribution given a carry `decide` returned."""
        return joint(self.marginals(obs, carry))

    def step(self, carry: ImugiCarry, obs: Obs, reset: torch.Tensor, decision_key: torch.Tensor):
        """One decision for views (R, ...): an action sampled from each marginal by the
        decision's key."""
        parts, carry = self.decide(carry, obs, reset, decision_key)
        action = sample(parts, decision_key)
        return Action.unpack(action.to(torch.int32)), carry, Outputs()
