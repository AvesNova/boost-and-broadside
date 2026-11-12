"""Yemong: one decision (`step`) and a stored segment (`unroll`) over one team's views.

The model owns its inputs: the belief, the pending actions and the view's position offset
live in its carry, and it composes its tokens from the raw observation. `step` returns a
replay record (what it read from the carry), so `unroll` recomputes a segment in parallel
without replaying the belief. `unroll` equals a loop of `step` over the same segment.

`step` skips the rows nothing reads: the enemy-action and next-state heads run on the enemy
rows only (own ships are always visible, so truth replaces their forecast on the next tick).
`unroll` computes every row, for the learner.
"""

import dataclasses
import math

import torch
import torch.nn as nn

from bnb.contracts import REWARD_LEVELS, Obs
from bnb.contracts.action import NUM_JOINT
from bnb.contracts.obs import SHIP
from bnb.contracts.tree import contract
from bnb.game.rules import Rules
from bnb.game.ship_types import ShipTypes

from . import belief as beliefs
from . import critic, density, shots
from .belief import Belief
from .codes import SPREAD_FITS
from .correction import CorrectionCode
from .delta import (
    DELTA_BASELINES,
    RATE_FIELDS,
    DeltaCode,
    Rates,
    StateDynamics,
    from_frame,
    state_dynamics,
    to_frame,
)
from .delta import fade as delta_fade
from .delta import velocity_frame as delta_frame
from .heads import HeadGroup, ShotTrajectoryHead, critic_values, next_state_logits
from .inputs import (
    ACCELERATION_INPUTS,
    FIELD_INPUTS,
    MODE_INPUTS,
    POSITION_CODE,
    ZONE_INPUTS,
    Accelerations,
    Motion,
    Replay,
    Tokens,
    accelerations,
    believed,
    compose,
    global_inputs,
    motion,
    preceding,
    ship_inputs,
    store_action,
    truth,
)
from .layers import rotary_tables
from .network import ORDERS, Encoder, Trunk
from .state_code import (
    MEANS,
    POSITION_INPUTS,
    POSITION_LEVELS,
    POSITION_PERIOD,
    SPREAD_INPUTS,
    ShipStateCode,
    StateCodeRanges,
    state_code_ranges,
    velocity_reach,
)

SHOT_HARMONICS = 4  # the trajectory head's age features
# a row's held rates: acceleration along, across, the nose's, the change, the turn's share
HELD = 4 + len(MEANS)
NEXT_STATES = ("absolute", "delta")  # `YemongConfig.next_state`


@dataclasses.dataclass(frozen=True)
class YemongConfig:
    """Production defaults (v1's known-good run). Nothing in the weights depends on the
    numbers of ships or fields or on the arena.

    levels: the critic's value levels in their configured order; outcome_level names the one
    read off the global token (the others are per-ship categorical values). None: no
    outcome head, every level per-ship. The learner reads each level's reward by its name
    (`bnb.learn.returns.value_rewards`): the reward levels' own (the default), or the single
    level `bnb.contracts.SUMMED_LEVEL` over their sum, with outcome_level None (the single
    value arm: one per-ship value over one reward stream, and no outcome head).
    rotary_period: the coarsest period (m) of the position frequencies (2 pi / p) 2^k,
    k < rotary_octaves; heading_harmonics: the heading's.
    bfloat16_matmuls: run the matrix products of the encoder, the trunk and the heads under
    bf16 autocast. Off by default. On, `step` and `unroll` agree only to bf16 rounding (the
    fix-round notes measure it), where float32 gives float32 rounding.
    forecast_feedback: after each decision store the forecast (the next-state head's decoded
    means and spreads, the enemy-action head's probabilities) as the belief the next tick
    reads. Off, nothing is written back: a hidden ship's row stays what the team last saw
    (its state at its last sighting, no spread, its age still counting) and every enemy
    action stays uniform, so the trunk's temporal layers carry the memory. The heads train
    as auxiliary losses either way.
    rotary: rotate the attention's queries and keys by the rotary tables. Off, the rotation
    is the identity everywhere (every angle 0: cos 1, sin 0, for the token and the map
    tables, in the plain and the fused kernel path alike), so attention reads no relative
    position or heading; the model trains and plays that way.
    order: per trunk block, the spatial sublayers then the temporal one ("spatial_first")
    or the temporal one then the spatial ones ("temporal_first"; `network.ORDERS`), or the
    spatial ones alone ("spatial_only", the ablation without temporal layers: the beliefs,
    the forecasts fed back, are then the model's only memory).
    rotary_readout (temporal_first and rotary only): in every block, a linear head of its
    own on that block's temporal sublayer's ship rows gives a (cos, sin) pair per rotary
    frequency, unnormalised (a pair shorter than 1 is the model unsure at that frequency).
    That block's spatial sublayers (which follow its temporal one) rotate a hidden ship's
    row by the block's read-out pairs in place of its belief's tables; visible ships keep
    their true tables and the global token and the map memory their exact ones, in every
    block. The read-outs enter the tables detached: they train only on their own loss (the
    learner's `rotary_readout` term, the mean over blocks, against the true phasors), which
    keeps each optimum the expected phasor and the fused attention kernel usable in the
    learner.
    spread_fit (`codes.SPREAD_FITS`): how the belief decodes a forecast's position and
    heading spreads. "coarsest" reads each at the coarsest level of its code that has
    broadened, and comes out honest on a forecast that is not Gaussian. "v1" is v1's fit,
    weighted so the finest informative level decides: on the codes a trained head predicts
    for a hidden ship it reports metres when the ship is hundreds of metres off. The default
    is "v1" so that every build record from before the option decodes as it was trained;
    training builds "coarsest" (`bnb.runtime.train.TrainConfig.yemong`). "v1" goes once no
    rated policy needs it.
    next_state (`NEXT_STATES`): what the next-state head predicts. "absolute", the state at
    t + 1 as the state code (`state_code.ShipStateCode`), its logits over the persistence
    baseline log(code_t + floor). "delta", each quantity's change from its baseline
    (`delta_baseline`) as a symlog HL-Gauss code (`delta.DeltaCode`), its output bias
    starting at the measured marginal; the forecast fed back is the belief plus the decoded
    change, and the belief carries the full covariance of the kinematic state
    (`state_code.COVARIANCE_STATE`, appended to its spreads), propagated by the kinematics'
    Jacobian. Round 1's rule baselines are gone: a policy trained with them does not load.
    delta_baseline (`delta.DELTA_BASELINES`, with next_state "delta"): "persistence",
    nothing changes, so the target is truth(t + 1) - truth(t); "extrapolation" (arm 4),
    last tick's rates carried a tick (`delta.DeltaCode.baseline`):
    the target is truth(t + 1) less that, and the carry holds every row as the model read
    it the tick before (`Carry.previous`). A hidden row holds its last sighting's rates,
    decaying with time constant extrapolation_decay (s; `Yemong.rates`). The rates are
    computed as the acceleration inputs are (`Carry.motion` rides in the carry either way),
    with or without acceleration_inputs: without it the ship token does not read them, and
    the head learns a residual from a baseline whose rates it cannot see (arm 4c).
    spread_inputs (`state_code.SPREAD_INPUTS`): how a ship row reads its belief's spread.
    "scalar": the position code blurred by one sigma for both axes, the velocity code by
    its covariance. "ellipse": the position code blurred per world axis by the position
    covariance's marginal sigmas, and the position and velocity ellipses (log sigma_major,
    log sigma_minor, the doubled angle times the anisotropy; `state_code.ellipse_code`) at
    the end of the ship token.
    position_inputs (`state_code.POSITION_INPUTS`, "fourier" needs next_state "delta"): what
    a ship row reads for its position and heading. "colour", the state code's runs.
    "fourier", (cos, sin) of the phase at each level of the colour code's periods (per world
    axis P / 3^l for l = 0..8, the finest 4 m; the heading 2 pi / 4^l for l = 0..3), each
    pair times the level's attenuation by the sigma as in the colour code: 36 and 8 columns
    in place of 81 and 16 (`ShipStateCode.inputs`). Only the encoder's input changes; the
    tokens' code, the global, field and zone tokens and every head stay the colour codes.
    The absolute head predicts the colour code from the colour inputs, so it is excluded.
    delta_split (with next_state "delta"): split the next-state work (arms 3-4). The
    next-state head becomes the physics head, trained on allies and visible enemies only
    (still run on hidden rows: their forecast reads it). A second head on the same tokens,
    the error head (`correction.CorrectionCode`), trained on hidden enemies only, predicts
    the physics forecast's error e = truth(t + 1) - forecast(t + 1); a hidden row's belief
    advances by the physics forecast plus a step toward e's median (capped at the one-tick physical
    bound) and takes its spread from e's distribution, in place of the physics' F P F^T + Q.
    Off by default. correction_rate (eta, with delta_split): the share of the error head's
    confident median applied per tick (`correction.CorrectionCode.moments`), before the cap.
    physics_hidden_decay (tau, s, with delta_split; None off): the physics head also trains
    on the hidden enemies, each row weighted exp(-age / tau) by the time since its ship was
    last seen, against the truth-to-truth target (`learn.labels.next_state_targets`: the
    true change from the baseline of the true state at t, under the extrapolation
    baseline's true rates, as a row seen at t - 1 and t would read them), its inputs still
    the belief it reads; the error head is unchanged. Only the labels read it.
    physics_hidden_fade (with physics_hidden_decay tau): use the physics head's change on a
    hidden row only where it is trained. A hidden row's physics forecast takes the share
    w = exp(-age / tau) of the head's change from the baseline (age the seconds since the
    ship was seen, at t): baseline + w (physics - baseline), on the means of position,
    velocity, heading, nose rate, angle of attack, energy and log refractive index
    (`delta.fade`); the rare jumps (shield, its recharge wait, the gun cooldown, advanced
    by their median) and every spread stay the physics forecast's (a hidden row's spread is
    the error head's). The error head's correction applies on top, and its label is the
    error of this faded forecast (`learn.labels.correction_targets`); the physics head's
    target is unchanged. Seen rows (age 0) are untouched, bit for bit. Off by default.
    acceleration_inputs: each ship row also reads its acceleration over the last tick, along
    and across its velocity and of its nose rate (`inputs.accelerations`), as symlog HL-Gauss
    codes and a validity flag at the end of the ship token. A visible row reads the
    observation's acceleration columns (world frame, turned into the velocity frame at t),
    so its validity is the observation's (seen alive at both ticks: just spotted is
    invalid). A hidden row differences the belief's forecast against the row the model read
    the tick before, which rides in the carry for the enemy rows (`Carry.motion`), and
    `unroll` shifts it within the segment. Off by default, so every build from before the
    option is unchanged.
    mode_inputs: the modes whose one-hot flags the global token reads, in this order
    (`inputs.MODE_INPUTS`). The default, team deathmatch and frontline, is every build's
    from before elimination (2026-11-12), which reads an elimination match as no mode (as
    sandbox); a new run's build reads all three (`bnb.runtime.train.TrainConfig`).
    """

    width: int = 128
    blocks: int = 2
    spatial_per_block: int = 2
    heads: int = 2
    map_width: int = 64
    value_bins: int = 77  # symlog bins 0.079 apart, one at 0 (as 51 over +-2 were)
    value_limit: float = 3.0  # symlog: +-19 in reward units
    levels: tuple[str, ...] = REWARD_LEVELS
    outcome_level: str | None = "outcome"
    hex_rings: int = 10
    presence_radius: float = 150.0  # m
    # Sized by the rule beside `state_code.POSITION_PERIOD` (Nyquist at the finest, the
    # largest map at the coarsest). 8192 m is under the real-scale map's 15.6 km, so far
    # pairs alias; proposed (2026-11-11, not yet adopted): 65 536 m down to 128 m.
    rotary_period: float = 8192.0  # m, the coarsest
    rotary_octaves: int = 8
    heading_harmonics: tuple[int, ...] = (1, 2, 4, 8)
    bfloat16_matmuls: bool = False
    forecast_feedback: bool = True  # store the forecast as the next tick's belief
    rotary: bool = True  # rotary attention (off: the identity rotation)
    order: str = "spatial_first"  # per block: "spatial_first", "temporal_first", "spatial_only"
    rotary_readout: bool = False  # hidden ships' rotary pairs from each block's temporal layer
    spread_fit: str = "v1"  # the belief's spread decoding; training builds "coarsest"
    next_state: str = "absolute"  # the next-state head's target: "absolute" or "delta"
    delta_baseline: str = "persistence"  # the delta head's baseline
    spread_inputs: str = "scalar"  # how a row reads its spread: "scalar" or "ellipse"
    position_inputs: str = "colour"  # a row's position and heading: "colour" or "fourier"
    acceleration_inputs: bool = False  # each ship row reads its acceleration over the last tick
    # the modes whose flags the global token reads (`inputs.MODE_INPUTS`); new runs read all
    mode_inputs: tuple[str, ...] = MODE_INPUTS[:2]
    delta_split: bool = False  # physics head on seen rows, an error head for hidden rows
    extrapolation_decay: float = 0.5  # s: a hidden row's held rates decay with this constant
    correction_rate: float = 0.1  # eta: the error head's step per tick, of its confident median
    physics_hidden_decay: float | None = None  # s: the physics head's hidden rows, exp(-age/tau)
    physics_hidden_fade: bool = False  # a hidden row's forecast: exp(-age/tau) of the change

    def __post_init__(self) -> None:
        if self.next_state not in NEXT_STATES:
            raise ValueError(f"next_state {self.next_state!r} is not one of {NEXT_STATES}")
        if self.delta_baseline not in DELTA_BASELINES:
            raise ValueError(
                f"delta_baseline {self.delta_baseline!r} is not one of {DELTA_BASELINES}"
            )
        if self.extrapolation_decay <= 0:
            raise ValueError(f"extrapolation_decay {self.extrapolation_decay} must be positive")
        if self.delta_split and self.next_state != "delta":
            raise ValueError("delta_split needs next_state 'delta'")
        if self.physics_hidden_decay is not None:
            if not self.delta_split:
                raise ValueError("physics_hidden_decay needs delta_split")
            if self.physics_hidden_decay <= 0:
                raise ValueError(
                    f"physics_hidden_decay {self.physics_hidden_decay} must be positive"
                )
        if self.physics_hidden_fade and (not self.delta_split or self.physics_hidden_decay is None):
            raise ValueError("physics_hidden_fade needs delta_split and physics_hidden_decay")
        if self.spread_inputs not in SPREAD_INPUTS:
            raise ValueError(f"spread_inputs {self.spread_inputs!r} is not one of {SPREAD_INPUTS}")
        if self.position_inputs not in POSITION_INPUTS:
            raise ValueError(
                f"position_inputs {self.position_inputs!r} is not one of {POSITION_INPUTS}"
            )
        if self.position_inputs == "fourier" and self.next_state != "delta":
            raise ValueError("position_inputs 'fourier' needs next_state 'delta'")
        if self.spread_fit not in SPREAD_FITS:
            raise ValueError(f"spread_fit {self.spread_fit!r} is not one of {SPREAD_FITS}")
        if self.order not in ORDERS:
            raise ValueError(f"order {self.order!r} is not one of {ORDERS}")
        if not set(self.mode_inputs) <= set(MODE_INPUTS) or len(set(self.mode_inputs)) != len(
            self.mode_inputs
        ):
            raise ValueError(f"mode_inputs {self.mode_inputs} must be distinct, of {MODE_INPUTS}")
        if self.rotary_readout and (self.order != "temporal_first" or not self.rotary):
            raise ValueError("rotary_readout needs order temporal_first and rotary on")
        if not self.levels or len(set(self.levels)) != len(self.levels):
            raise ValueError(f"the critic's levels {self.levels} must be distinct, and some")
        if self.outcome_level is not None:
            if self.outcome_level not in self.levels:
                raise ValueError(f"outcome level {self.outcome_level!r} is not in {self.levels}")
            if len(self.levels) < 2:
                raise ValueError("the critic needs at least one per-ship level besides the outcome")
        pairs = 2 * self.rotary_octaves + len(self.heading_harmonics)
        if 2 * pairs > self.width // self.heads:
            raise ValueError(
                f"{pairs} rotary pairs do not fit a head of {self.width // self.heads}"
            )

    @property
    def rotary_pairs(self) -> int:
        """The rotary frequencies: x and y per octave, then the heading harmonics."""
        return 2 * self.rotary_octaves + len(self.heading_harmonics)

    @property
    def local_levels(self) -> tuple[int, ...]:
        """The indices into levels of every level but outcome_level (every level without
        one): the levels with a value distribution."""
        return tuple(i for i, name in enumerate(self.levels) if name != self.outcome_level)


@contract
class Carry:
    """Per view (R leading). state (R, blocks, Q, D) and convolution_buffer
    (R, blocks, Q, 3, D): the trunk's; belief: per ship row, positions in the shifted frame;
    previous_action (R, S) int32, the own ships' last joint actions; offset (R, 2) m, the
    view's position offset, drawn uniformly in [0, P) at every reset; nonfinite (R,) int64,
    a running count of non-finite forecast entries (replaced by zeros); motion: the enemy
    rows' Motion at the last decision, what a hidden row's accelerations difference against
    (None without `YemongConfig.acceleration_inputs` or the extrapolation baseline);
    previous (R, N, len(MEANS)): every row as the model read it at the last decision
    (`inputs.believed`), what the extrapolation baseline's last-tick changes difference
    against (None without it); held
    (R, N, HELD) and held_valid (R, N): the rates each row holds from its last sighting
    (`Yemong.rates`; None without it)."""

    state: torch.Tensor
    convolution_buffer: torch.Tensor
    belief: Belief
    previous_action: torch.Tensor
    offset: torch.Tensor
    nonfinite: torch.Tensor
    motion: Motion | None = None
    previous: torch.Tensor | None = None
    held: torch.Tensor | None = None
    held_valid: torch.Tensor | None = None


ABLATIONS = ("rotary", "forecast")  # the inference-time ablations `Ablations` carries


@contract
class Ablations:
    """Inference-time ablations, per view (R,) bool (`Yemong.step`): rotary, the attention's
    rotary rotation is the identity on the view's rows (every angle 0: cos 1, sin 0, for the
    token and the map tables); forecast, nothing is written back to the view's belief, as
    with `YemongConfig.forecast_feedback` off (a hidden ship keeps its last sighting, every
    enemy action stays uniform). A view with neither computes what it would without them."""

    rotary: torch.Tensor
    forecast: torch.Tensor


@contract
class Decision:
    """One step's outputs, per view: policy_logits (R, S, NUM_JOINT); enemy_action_logits and
    next_state_logits (R, N - S, ...) for the enemy rows; values (R, S, levels) in the configured
    level order; outcome_probabilities (R, 4) (None without an outcome level); the next carry
    (previous_action not yet replaced by the sample); the replay record; correction_logits
    (R, N - S, ...), the error head's, for the enemy rows (None without
    `YemongConfig.delta_split`)."""

    policy_logits: torch.Tensor
    enemy_action_logits: torch.Tensor
    next_state_logits: torch.Tensor
    values: torch.Tensor
    outcome_probabilities: torch.Tensor | None
    carry: Carry
    replay: Replay
    correction_logits: torch.Tensor | None = None


@contract
class Unrolled:
    """A segment's outputs, (L, R, ...) leading: policy_logits (S rows), enemy_action_logits
    and next_state_logits (every row), value_logits (S, per-ship levels, bins), values
    (S, levels), outcome_logits (4; None without an outcome level), density_log_rates
    (2 cells: ally cells, then enemy cells), shot_outcome_logits (N, 19) and shot_latents
    (N, D) for `Yemong.shot_trajectory`; rotary_readout (N, blocks, 2, pairs), each block's
    read-out (cos, sin) rows per ship (None without `YemongConfig.rotary_readout`);
    correction_logits (N, ...), the error head's (every row; None without
    `YemongConfig.delta_split`)."""

    policy_logits: torch.Tensor
    enemy_action_logits: torch.Tensor
    next_state_logits: torch.Tensor
    value_logits: torch.Tensor
    values: torch.Tensor
    outcome_logits: torch.Tensor | None
    density_log_rates: torch.Tensor
    shot_outcome_logits: torch.Tensor
    shot_latents: torch.Tensor
    rotary_readout: torch.Tensor | None = None
    correction_logits: torch.Tensor | None = None


BELIEF_FIELDS = tuple(field.name for field in dataclasses.fields(Belief))  # (traced as constants)


def _rows(belief: Belief, rows: slice) -> Belief:
    return Belief(**{name: getattr(belief, name)[:, rows] for name in BELIEF_FIELDS})


def _where_views(off: torch.Tensor, ablated: torch.Tensor, kept: torch.Tensor) -> torch.Tensor:
    """ablated on the views (R,) where off is set, kept elsewhere."""
    return torch.where(off.view(off.shape + (1,) * (kept.dim() - 1)), ablated, kept)


def _nearest(position: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """The representative of position (mod P) nearest to reference: decoded positions come
    back in [0, P), and the belief's frame is continuous."""
    return position + POSITION_PERIOD * torch.round((reference - position) / POSITION_PERIOD)


def _cast(tensor: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor | None:
    return None if tensor is None else tensor.to(dtype)


def _blocks(store: dict) -> torch.Tensor | None:
    """The blocks' read-outs (..., N, 2, pairs) stacked as (..., N, blocks, 2, pairs)."""
    readouts = store.get("readout")
    return None if readouts is None else torch.stack(readouts, -3)


def _zero_angles(tables: tuple[torch.Tensor, torch.Tensor]):
    """The identity rotation's tables (cos 1, sin 0), shaped as the rotary tables."""
    cos, sin = tables
    return torch.ones_like(cos), torch.zeros_like(sin)


def _identity(tables: tuple[torch.Tensor, torch.Tensor], off: torch.Tensor):
    """The rotary tables (cos, sin), each (..., tokens, 1, pairs), with every angle 0 on the
    leading indices where off (...,) is set."""
    cos, sin = tables
    off = off.view(off.shape + (1,) * (cos.dim() - off.dim()))
    return torch.where(off, 1.0, cos), torch.where(off, 0.0, sin)


@dataclasses.dataclass(frozen=True)
class YemongBuild:
    """Everything Yemong's constructor takes (runs spec 4): its config and the numbers it
    derives from the rules and the ship card (`yemong_build`), so a saved model is rebuilt
    without the current game. tick_length (s); num_ships sets only the density head's
    starting rate; ship_types: the card's type count; bullet_moves: the card's longest
    bullet life in ticks (`Rules.longest_bullet_moves`); muzzle_speed (m/s): the card's
    fastest; hit_radius (m): its smallest; state_code: the ship-state code's ranges;
    dynamics: what a rule baseline reads (`delta.StateDynamics`): the extrapolation
    baseline's shield rule reads it; None for every other
    build (persistence reads nothing; round 1's rule baselines are gone)."""

    config: YemongConfig
    tick_length: float
    num_ships: int
    ship_types: int
    bullet_moves: int
    muzzle_speed: float
    hit_radius: float
    state_code: StateCodeRanges
    dynamics: StateDynamics | None = None


def yemong_build(config: YemongConfig, rules: Rules, types: ShipTypes) -> YemongBuild:
    """The build of a Yemong for the rules and a (types,) ship card."""
    return YemongBuild(
        config=config,
        tick_length=rules.tick_length,
        num_ships=rules.num_ships,
        ship_types=types.max_shield.shape[-1],
        bullet_moves=rules.longest_bullet_moves,
        muzzle_speed=float(types.muzzle_speed.max()),
        hit_radius=float(types.hit_radius.min()),
        state_code=state_code_ranges(rules, types),
        dynamics=(
            state_dynamics(rules, types)
            if config.next_state == "delta" and config.delta_baseline == "extrapolation"
            else None
        ),
    )


class Yemong(nn.Module):
    """Built from a `YemongBuild` alone (`yemong_build` derives one from the game). VERSION
    rises whenever a change alters what saved weights compute (runs spec 4)."""

    NAME = "yemong"
    VERSION = 1

    def __init__(self, build: YemongBuild) -> None:
        super().__init__()
        config = build.config
        self.build, self.config = build, config
        width, hidden = config.width, 2 * config.width
        self.ship_types = build.ship_types
        self.tick_length = build.tick_length
        delta = config.next_state == "delta"
        self.state_code = ShipStateCode(
            build.state_code,
            config.spread_fit,
            covariance=delta,
            spread_inputs=config.spread_inputs,
            position_inputs=config.position_inputs,
        )
        self.delta_code = None
        if delta:
            if config.delta_baseline == "persistence" and build.dynamics is not None:
                raise ValueError("a build with rule dynamics is round 1's delta head (removed)")
            self.delta_code = DeltaCode(build.state_code, config.delta_baseline, build.dynamics)
        self.shot_code = shots.shot_code_parameters(
            build.muzzle_speed,
            velocity_reach(),
            build.bullet_moves * build.tick_length,
            build.hit_radius,
        )
        self.bullet_moves = build.bullet_moves
        self.encoder = Encoder(
            ship_inputs(
                self.state_code.input_width,
                self.ship_types,
                config.acceleration_inputs,
                self.state_code.ellipse_width,
            ),
            global_inputs(config.mode_inputs),
            FIELD_INPUTS,
            width,
            config.map_width,
            ZONE_INPUTS,
        )
        self.trunk = Trunk(
            width,
            config.heads,
            config.map_width,
            config.blocks,
            config.spatial_per_block,
            config.order,
        )
        levels = len(config.local_levels)
        # Heads grouped by the token they read: own ships, every ship, the global token.
        self.own_heads = HeadGroup(width, hidden, (NUM_JOINT, levels * config.value_bins))
        self.ship_heads = HeadGroup(width, hidden, (NUM_JOINT, self.next_state_width))
        self.outcome = (
            None if config.outcome_level is None else HeadGroup(width, hidden, (critic.OUTCOMES,))
        )
        cells = density.cell_count(config.hex_rings)
        self.density = HeadGroup(width, hidden, (2 * cells,))
        self.shot_outcome = HeadGroup(width, hidden, (self.shot_code.outcome_width,))
        self.shot_trajectory_head = ShotTrajectoryHead(
            width, 1 + 2 * SHOT_HARMONICS, self.shot_code.trajectory_width
        )
        self._initialise(build.num_ships, cells)
        # Frequencies (2 pi / period) 2^k, held as their periods (exact powers of two metres).
        octaves = 2.0 ** torch.arange(config.rotary_octaves, dtype=torch.float32)
        self.register_buffer("position_periods", config.rotary_period / octaves, persistent=False)
        self.register_buffer(
            "heading_frequencies",
            torch.tensor(config.heading_harmonics, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "bins", critic.value_bins(config.value_bins, config.value_limit), persistent=False
        )
        # Last, so no other layer's initialisation depends on it: at the scale of a
        # normalised memory token (RMS 1), where the norm is well conditioned (at zero its
        # gradient is 1 / sqrt(eps)).
        with torch.no_grad():
            self.encoder.absent_zone.normal_()
        # After every other draw, so the rest initialise as they would without it.
        self.rotary_readout = (
            nn.ModuleList(nn.Linear(width, 2 * config.rotary_pairs) for _ in range(config.blocks))
            if config.rotary_readout
            else None
        )
        # The error head (`delta_split`), after every other draw, so the rest initialise as
        # they would without it: its last layer zero, its bias the measured marginal.
        self.correction_code, self.correction_head = None, None
        if config.delta_split:
            self.correction_code = CorrectionCode(build.state_code, config.correction_rate)
            self.correction_head = HeadGroup(width, hidden, (self.correction_code.width,))
            with torch.no_grad():
                nn.init.zeros_(self.correction_head.lasts[0].weight)
                self.correction_head.lasts[0].bias.copy_(self.correction_code.marginal_logits())

    def _initialise(self, num_ships: int, cells: int) -> None:
        """Code columns of each first projection ~ N(0, 1 / groups); the next-state head
        starts as "nothing changes" (absolute) or at the measured marginal (delta); the
        density head at an even spread of the ships."""
        with torch.no_grad():
            self.encoder.ship.first.weight[:, : self.state_code.input_width].normal_(
                0.0, self.state_code.input_groups**-0.5
            )
            ellipse = self.state_code.ellipse_width
            end = self.encoder.ship.first.weight.shape[1] - ellipse
            if self.config.acceleration_inputs:  # the codes as the state code's (not the flag)
                self.encoder.ship.first.weight[:, end - ACCELERATION_INPUTS : end - 1].normal_(
                    0.0, self.state_code.groups**-0.5
                )
            if ellipse:  # as the state code's columns
                self.encoder.ship.first.weight[:, end:].normal_(0.0, self.state_code.groups**-0.5)
            for kind in (self.encoder.field, self.encoder.global_token, self.encoder.zone):
                kind.first.weight[:, :POSITION_CODE].normal_(0.0, POSITION_LEVELS**-0.5)
            nn.init.zeros_(self.ship_heads.lasts[1].weight)
            if self.delta_code is not None:
                self.ship_heads.lasts[1].bias.copy_(self.delta_code.marginal_logits())
            self.density.lasts[0].bias.fill_(math.log(max(num_ships, 2) / 2 / cells))

    # ------------------------------------------------------------------ next state

    @property
    def next_state_runs(self):
        """The next-state head's softmax groups as (groups, size) runs."""
        code = self.state_code if self.delta_code is None else self.delta_code
        return code.runs

    @property
    def next_state_width(self) -> int:
        """The next-state head's outputs."""
        code = self.state_code if self.delta_code is None else self.delta_code
        return code.width

    def _next_state_logits(self, code: torch.Tensor, change: torch.Tensor) -> torch.Tensor:
        if self.delta_code is None:
            return next_state_logits(code, change)
        return change

    def forecast(
        self,
        logits: torch.Tensor,
        means: torch.Tensor,
        spreads: torch.Tensor,
        ship_type: torch.Tensor,
        correction_logits: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
        rates: Rates | None = None,
        age: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Next-state logits (..., N, next_state_width) to the forecast at t + 1 (means,
        spreads; positions in the shifted frame, as the representative nearest the state at
        t) of the rows' state at t (means (..., N, 12), spreads, as the model read them:
        `inputs.believed`); ship_type (..., N) int64. With `YemongConfig.delta_split` and
        the error head's logits (..., N, correction width), the rows where hidden (..., N)
        is set (a belief, not the truth, at t) take the corrected forecast
        (`correction.CorrectionCode.correct`); without them every row is the physics'.
        rates: the extrapolation baseline's (`rates`, required by it). age (..., N): the
        ticks since each row's ship was seen (the belief's), required with
        `YemongConfig.physics_hidden_fade`: the rows with age > 0 take the faded physics
        forecast (`delta.fade`) before any correction."""
        if self.delta_code is not None:
            if self._extrapolates and rates is None:
                raise ValueError("the extrapolation baseline needs the rows' rates")
            physics = self.delta_code.advance(
                means, spreads, logits, ship_type, self.state_code, rates
            )
            if self.config.physics_hidden_fade:
                physics = (self._faded(physics[0], means, ship_type, rates, age), physics[1])
            if correction_logits is None:
                return physics
            if hidden is None:
                raise ValueError("the corrected forecast needs the hidden rows")
            corrected = self.correction_code.correct(
                physics[0], correction_logits, ship_type, self.state_code
            )
            rows = hidden[..., None]
            return (
                torch.where(rows, corrected[0], physics[0]),
                torch.where(rows, corrected[1], physics[1]),
            )
        predicted, predicted_spreads = self.state_code.decode_logits(logits, ship_type)
        position = _nearest(predicted[..., 0:2], means[..., 0:2])
        return torch.cat([position, predicted[..., 2:]], -1), predicted_spreads

    def _faded(
        self,
        forecast: torch.Tensor,
        means: torch.Tensor,
        ship_type: torch.Tensor,
        rates: Rates | None,
        age: torch.Tensor | None,
    ) -> torch.Tensor:
        """The physics forecast's means (..., N, len(MEANS)) with each hidden row's (age > 0
        ticks) change from the baseline faded by exp(-age dt / tau) (`delta.fade`); the
        other rows the forecast itself."""
        if age is None:
            raise ValueError("the faded forecast needs the rows' age")
        seconds = age.to(forecast.dtype) * self.tick_length
        weight = torch.exp(-seconds / self.config.physics_hidden_decay)
        baseline = self.delta_code.baseline(means, rates, ship_type)
        faded = delta_fade(forecast, baseline, weight)
        return torch.where((age > 0)[..., None], faded, forecast)

    # ------------------------------------------------------------------ carry

    def initial_carry(self, views: int, ships_per_team: int, num_ships: int, device=None) -> Carry:
        """No history: the trunk carry zero, no beliefs, every previous action null, a
        fresh position offset per view."""
        config, dtype = self.config, self.bins.dtype
        queries = num_ships + 1
        state = torch.zeros(
            (views, self.trunk.temporal_blocks, queries, config.width), dtype=dtype, device=device
        )
        belief = beliefs.blank(
            (views, num_ships),
            len(MEANS),
            self.state_code.spread_count,
            NUM_JOINT,
            device=device,
            dtype=dtype,
        )
        rows = None
        if self._tracks_motion:  # nothing present: the first tick's inputs are invalid
            enemies = num_ships - ships_per_team
            rows = Motion(
                velocity=torch.zeros((views, enemies, 2), dtype=dtype, device=device),
                nose_rate=torch.zeros((views, enemies), dtype=dtype, device=device),
                present=torch.zeros((views, enemies), dtype=torch.bool, device=device),
            )
        return Carry(
            state=state,
            convolution_buffer=state.new_zeros(state.shape[:-1] + (3, config.width)),
            belief=belief,
            previous_action=torch.zeros((views, ships_per_team), dtype=torch.int32, device=device),
            offset=POSITION_PERIOD * torch.rand((views, 2), dtype=dtype, device=device),
            nonfinite=torch.zeros(views, dtype=torch.int64, device=device),
            motion=rows,
            previous=(
                torch.zeros((views, num_ships, len(MEANS)), dtype=dtype, device=device)
                if self._extrapolates
                else None
            ),
            held=(
                torch.zeros((views, num_ships, HELD), dtype=dtype, device=device)
                if self._extrapolates
                else None
            ),
            held_valid=(
                torch.zeros((views, num_ships), dtype=torch.bool, device=device)
                if self._extrapolates
                else None
            ),
        )

    @staticmethod
    def _cleared(carry: Carry, reset: torch.Tensor) -> Carry:
        """The carry with every view whose environment reset made fresh, with a new offset
        drawn (from the default generator: a captured graph draws afresh on each replay)."""
        keep = ~reset

        def masked(tensor: torch.Tensor, fill=0):
            shape = keep.shape + (1,) * (tensor.dim() - keep.dim())
            return torch.where(keep.view(shape), tensor, fill)

        prior = carry.belief
        belief = Belief(
            valid=prior.valid & keep[..., None],
            age=masked(prior.age),
            means=masked(prior.means),
            spreads=masked(prior.spreads),
            action=masked(prior.action, 1.0 / NUM_JOINT),
        )
        offset = torch.where(
            keep[..., None], carry.offset, POSITION_PERIOD * torch.rand_like(carry.offset)
        )
        rows = carry.motion
        if rows is not None:  # nothing is present before a reset
            rows = Motion(
                velocity=rows.velocity,
                nose_rate=rows.nose_rate,
                present=rows.present & keep[..., None],
            )
        return Carry(
            state=masked(carry.state),
            convolution_buffer=masked(carry.convolution_buffer),
            belief=belief,
            previous_action=masked(carry.previous_action),
            offset=offset,
            nonfinite=carry.nonfinite,
            motion=rows,
            previous=None if carry.previous is None else masked(carry.previous),
            held=None if carry.held is None else masked(carry.held),
            held_valid=None if carry.held_valid is None else carry.held_valid & keep[..., None],
        )

    # ------------------------------------------------------------------ shared

    @property
    def _extrapolates(self) -> bool:
        return self.delta_code is not None and self.delta_code.baseline_name == "extrapolation"

    @property
    def _tracks_motion(self) -> bool:
        """The carry holds the enemy rows' Motion (`Carry.motion`) and each tick computes the
        rows' accelerations: the acceleration inputs read them, the extrapolation baseline's
        rates too (with or without the inputs)."""
        return self.config.acceleration_inputs or self._extrapolates

    def rates(
        self,
        ships: torch.Tensor,
        means: torch.Tensor,
        previous: torch.Tensor,
        accelerations: Accelerations,
        held: torch.Tensor,
        held_valid: torch.Tensor,
    ) -> tuple[Rates, torch.Tensor, torch.Tensor]:
        """The extrapolation baseline's `delta.Rates` of rows (..., N) at one tick, and the
        held rates after it: ships the observation, means the rows as the model reads them,
        previous as it read them the tick before, accelerations the acceleration inputs,
        held (..., N, HELD) and held_valid (..., N) the carry's (`Carry.held`).

        A visible row reads the observation's acceleration and its validity, its change
        against previous, its gradient and its shot flag, and holds them: the acceleration
        along and across its velocity, the nose's, the change. A hidden row reads what it
        holds, decayed by exp(-dt / tau) per tick (tau `YemongConfig.extrapolation_decay`),
        the acceleration turned back from its velocity frame at t, valid while the belief
        holds the ship (`inputs.Accelerations.valid`); its shield and countdowns take the
        rule, and its heading turns by its nose rate times the same decay since the
        sighting (`delta.Rates.turn`), so a hidden ship does not turn forever. The
        belief's own change is not a rate: it holds the head's change, which would come
        back as next tick's acceleration and compound."""
        visible = ships[..., SHIP["visible"]] > 0.5
        dtype = means.dtype
        gradient = torch.stack(
            [ships[..., SHIP["log_index_gradient_x"]], ships[..., SHIP["log_index_gradient_y"]]],
            -1,
        )
        unit_x, unit_y, _ = delta_frame(means[..., 2:4], means[..., 4])
        world = accelerations.world.to(dtype)
        current = torch.cat(
            [
                to_frame(world[..., 0:2], unit_x, unit_y),
                world[..., 2:3],
                (means - previous.to(dtype)),
                torch.ones_like(world[..., 2:3]),
            ],
            -1,
        )
        seen = visible & accelerations.valid
        decayed = held.to(dtype) * math.exp(-self.tick_length / self.config.extrapolation_decay)
        hidden_valid = ~visible & held_valid & accelerations.valid
        source = torch.where(visible[..., None], current, decayed)
        valid = torch.where(visible, accelerations.valid, hidden_valid)
        rates = Rates(
            valid=valid,
            acceleration=from_frame(source[..., 0:2], unit_x, unit_y),
            nose_acceleration=source[..., 2],
            change=source[..., 3:-1],
            gradient=torch.where(visible[..., None], gradient, 0.0).to(dtype),
            gradient_known=visible,
            fired=visible & (ships[..., SHIP["fired"]] > 0.5),
            rule=~visible,
            turn=source[..., -1],
        )
        kept = torch.where(seen[..., None], current, torch.where(visible[..., None], 0.0, decayed))
        return rates, kept, seen | hidden_valid

    def segment_rates(
        self, carry: Carry, obs: Obs, replay: Replay, reset: torch.Tensor
    ) -> Rates | None:
        """The extrapolation baseline's rates over a stored segment (L, R, N), as `step`
        read them: the rows the tick before and the held rates from the carry at its start,
        then tick by tick; None for any other build."""
        if not self._extrapolates:
            return None
        _, _, means, _ = believed(obs.ships, replay)
        previous = torch.cat([carry.previous[None].to(means.dtype), means[:-1]], 0)
        accelerations = self.acceleration_inputs(carry, obs, replay, reset)
        keep = ~reset[0]
        held = torch.where(keep[:, None, None], carry.held, 0.0)
        held_valid = carry.held_valid & keep[:, None]
        out = []
        for t in range(means.shape[0]):
            if t > 0:  # step clears the carry at a reset
                held = torch.where(reset[t][:, None, None], 0.0, held)
                held_valid = held_valid & ~reset[t][:, None]
            tick = Accelerations(
                values=accelerations.values[t],
                valid=accelerations.valid[t],
                slow=accelerations.slow[t],
                world=accelerations.world[t],
            )
            rates, held, held_valid = self.rates(
                obs.ships[t], means[t], previous[t], tick, held, held_valid
            )
            out.append(rates)
        return Rates(
            **{name: torch.stack([getattr(rates, name) for rates in out]) for name in RATE_FIELDS}
        )

    def _matmuls(self, device: torch.device):
        return torch.autocast(device.type, torch.bfloat16, enabled=self.config.bfloat16_matmuls)

    def tokens(self, obs: Obs, replay: Replay, previous: Motion | None = None) -> Tokens:
        """The encoder inputs of views and their replay records; previous: the enemy rows'
        Motion at the tick before, with `YemongConfig.acceleration_inputs` or the
        extrapolation baseline (required then; the ship token reads the accelerations only
        with the inputs)."""
        if self._tracks_motion and previous is None:
            raise ValueError("the accelerations need the previous tick's motion")
        return compose(
            obs,
            replay,
            self.state_code,
            self.ship_types,
            self.config.presence_radius,
            self.tick_length,
            previous if self._tracks_motion else None,
            self.config.acceleration_inputs,
            self.config.mode_inputs,
        )

    def preceding_motion(
        self, carry: Carry, obs: Obs, replay: Replay, reset: torch.Tensor
    ) -> Motion | None:
        """The enemy rows' Motion before each tick of a stored segment (L, R, ...): the
        carry's at its start, then the tick before; reset (L, R) as `step` received it. None
        without `YemongConfig.acceleration_inputs` or the extrapolation baseline."""
        if not self._tracks_motion:
            return None
        valid, _, means, _ = believed(obs.ships, replay)
        own = replay.previous_action.shape[-1]
        return preceding(motion(obs.ships, valid, means, own), carry.motion, reset)

    def acceleration_inputs(
        self, carry: Carry, obs: Obs, replay: Replay, reset: torch.Tensor
    ) -> Accelerations | None:
        """The accelerations `unroll` computes over a stored segment (its arguments), for the
        diagnostics and the extrapolation baseline's rates (`segment_rates`); None without
        `YemongConfig.acceleration_inputs` or the extrapolation baseline."""
        previous = self.preceding_motion(carry, obs, replay, reset)
        if previous is None:
            return None
        valid, _, means, _ = believed(obs.ships, replay)
        now = motion(obs.ships, valid, means, replay.previous_action.shape[-1])
        return accelerations(obs.ships, means, now, previous, self.tick_length)

    def _encode(self, tokens: Tokens, rotary_off: torch.Tensor | None = None):
        """Queries (..., Q, D), map memory (..., M + Z, Dm) and the rotary tables; rotary_off
        (leading axes,) bool: the views whose tables are the identity (`Ablations.rotary`)."""
        queries, memory = self.encoder(
            tokens.ship, tokens.global_token, tokens.field, tokens.zone, tokens.zone_present
        )
        frequencies = (self.position_periods, self.position_periods, self.heading_frequencies)
        token_tables = rotary_tables(tokens.query_position, tokens.query_heading, *frequencies)
        map_tables = rotary_tables(
            tokens.memory_position, torch.zeros_like(tokens.memory_position[..., 0]), *frequencies
        )
        if not self.config.rotary:
            token_tables, map_tables = (
                _zero_angles(tables) for tables in (token_tables, map_tables)
            )
        if rotary_off is not None:
            token_tables, map_tables = (
                _identity(token_tables, rotary_off),
                _identity(map_tables, rotary_off),
            )
        return queries, memory, token_tables, map_tables

    def phasors(self, position: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
        """(..., tokens, 2, pairs): the (cos, sin) rows of the rotary angles of positions
        (..., tokens, 2) in the shifted frame and headings (..., tokens): the tables the
        rotary read-out is trained against."""
        cos, sin = rotary_tables(
            position,
            heading,
            self.position_periods,
            self.position_periods,
            self.heading_frequencies,
        )
        return torch.cat([cos, sin], -2)

    def _readout(
        self,
        ships: torch.Tensor,
        token_tables: tuple[torch.Tensor, torch.Tensor],
        rotary_off: torch.Tensor | None,
        store: dict,
    ):
        """The trunk's `retable` for the rotary read-out (None without it): from block k's
        temporal sublayer output, the token tables with every hidden ship's row replaced by
        block k's read-out pairs (detached), rotary_off's views made the identity again;
        block k's read-out (..., N, 2, pairs) goes into store["readout"][k]. ships (..., N,
        columns): the observation's rows."""
        if self.rotary_readout is None:
            return None
        hidden = ships[..., SHIP["visible"]] <= 0.5  # (..., N)
        count = hidden.shape[-1]

        store["readout"] = []

        def retable(index: int, tokens: torch.Tensor):
            readout = self.rotary_readout[index](tokens[..., :count, :]).unflatten(-1, (2, -1))
            store["readout"].append(readout)
            cos, sin = (part.detach().to(token_tables[0].dtype) for part in readout.unbind(-2))
            rows = torch.cat([hidden, torch.zeros_like(hidden[..., :1])], -1)[..., None, None]
            pad = token_tables[0][..., count:, :, :]
            tables = (
                torch.where(rows, torch.cat([cos[..., None, :], pad], -3), token_tables[0]),
                torch.where(
                    rows,
                    torch.cat([sin[..., None, :], token_tables[1][..., count:, :, :]], -3),
                    token_tables[1],
                ),
            )
            return tables if rotary_off is None else _identity(tables, rotary_off)

        return retable

    def _outcome_logits(self, global_hidden: torch.Tensor) -> torch.Tensor | None:
        """The outcome head's logits (..., 4) on the global token (None without one)."""
        if self.outcome is None:
            return None
        (outcome_logits,) = self.outcome(global_hidden)
        return outcome_logits

    def _values(self, value_logits: torch.Tensor, outcome_logits: torch.Tensor | None):
        value_logits = value_logits.unflatten(-1, (-1, self.config.value_bins))
        config = self.config
        values = critic_values(
            value_logits,
            outcome_logits,
            self.bins,
            config.local_levels,
            None if config.outcome_level is None else config.levels.index(config.outcome_level),
        )
        return value_logits, values

    # ------------------------------------------------------------------ step

    def step(
        self, carry: Carry, obs: Obs, reset: torch.Tensor, ablations: Ablations | None = None
    ) -> Decision:
        """One decision for views (R, ...); reset (R,) bool: the environment reset before
        this observation; ablations: the inference-time ablations per view (none by
        default)."""
        return self._step(carry, obs, reset, inspect=False, ablations=ablations)[0]

    def inspect(self, carry: Carry, obs: Obs, reset: torch.Tensor) -> tuple[Decision, Unrolled]:
        """One decision, as `step` makes it, and every head's output at it, as `unroll`
        gives them for a segment of this one step ((1, R, ...) leading), from the same
        trunk pass: what the play shell's prediction overlays read."""
        return self._step(carry, obs, reset, inspect=True)

    def _step(
        self,
        carry: Carry,
        obs: Obs,
        reset: torch.Tensor,
        inspect: bool,
        ablations: Ablations | None = None,
    ) -> tuple[Decision, Unrolled | None]:
        carry = self._cleared(carry, reset)
        ships, dtype = obs.ships, self.bins.dtype
        own = carry.previous_action.shape[-1]
        visible = ships[..., SHIP["visible"]] > 0.5
        spawned = ships[..., SHIP["spawned"]] > 0.5
        seen = beliefs.observe(carry.belief, truth(ships, carry.offset), visible, spawned)
        replay = Replay(
            valid=seen.valid[:, own:],
            age=seen.age[:, own:],
            means=seen.means[:, own:],
            spreads=seen.spreads[:, own:],
            enemy_action=store_action(seen.action[:, own:]),
            previous_action=carry.previous_action,
            offset=carry.offset,
        )
        tokens = self.tokens(obs, replay, carry.motion)
        ships_count = tokens.code.shape[-2]
        rotary_off = None if ablations is None else ablations.rotary
        readout: dict = {}
        with self._matmuls(ships.device):
            queries, memory, token_tables, map_tables = self._encode(tokens, rotary_off)
            hidden, state, buffer = self.trunk.step(
                queries,
                memory,
                token_tables,
                map_tables,
                carry.state,
                carry.convolution_buffer,
                self._readout(ships, token_tables, rotary_off, readout),
            )
            policy_logits, value_logits = self.own_heads(hidden[:, :own])
            enemy_action_logits, change = self.ship_heads(hidden[:, own:ships_count])
            correction_logits = every_correction = None
            if self.correction_head is not None:
                (correction_logits,) = self.correction_head(hidden[:, own:ships_count])
            outcome_logits = self._outcome_logits(hidden[:, ships_count])
            if inspect:
                every_action_logits, every_change = self.ship_heads(hidden[:, :ships_count])
                if self.correction_head is not None:
                    (every_correction,) = self.correction_head(hidden[:, :ships_count])
                (density_log_rates,) = self.density(hidden[:, ships_count])
                (shot_outcome_logits,) = self.shot_outcome(hidden[:, :ships_count])
                shot_latents = self.shot_trajectory_head.latent(hidden[:, :ships_count])
        policy_logits, enemy_action_logits, outcome_logits = (
            policy_logits.to(dtype),
            enemy_action_logits.to(dtype),
            _cast(outcome_logits, dtype),
        )
        value_logits, values = self._values(value_logits.to(dtype), outcome_logits)
        forecast_logits = self._next_state_logits(tokens.code[:, own:], change.to(dtype))
        enemy = _rows(seen, slice(own, None))
        correction_logits = _cast(correction_logits, dtype)
        rates = held = held_valid = None
        if self._extrapolates:
            rates, held, held_valid = self.rates(
                ships,
                seen.means,
                carry.previous,
                tokens.accelerations,
                carry.held,
                carry.held_valid,
            )
            rates = rates.rows(slice(own, None))
        if self.config.forecast_feedback:
            means, spreads = self.forecast(
                forecast_logits,
                enemy.means,
                enemy.spreads,
                tokens.ship_type[:, own:],
                correction_logits,
                enemy.age > 0,
                rates,
                enemy.age,
            )
            advanced, nonfinite = beliefs.advance(
                enemy, torch.cat([means, spreads], -1), enemy_action_logits
            )
            if ablations is not None:  # the ablated views keep what they saw, as below
                off = ablations.forecast
                advanced = Belief(
                    **{
                        name: _where_views(off, getattr(enemy, name), getattr(advanced, name))
                        for name in BELIEF_FIELDS
                    }
                )
                nonfinite = torch.where(off, 0, nonfinite)
        else:  # the belief keeps the last sighting (observe holds it, uniform actions)
            advanced, nonfinite = enemy, torch.zeros_like(carry.nonfinite)
        mine = _rows(seen, slice(None, own))
        belief = Belief(
            **{
                name: torch.cat([getattr(mine, name), getattr(advanced, name)], 1)
                for name in BELIEF_FIELDS
            }
        )
        carry = Carry(
            state=state.to(dtype),
            convolution_buffer=buffer.to(dtype),
            belief=belief,
            previous_action=carry.previous_action,
            offset=carry.offset,
            nonfinite=carry.nonfinite + nonfinite,
            motion=tokens.motion,
            previous=seen.means if self._extrapolates else None,
            held=held,
            held_valid=held_valid,
        )
        decision = Decision(
            policy_logits=policy_logits,
            enemy_action_logits=enemy_action_logits,
            next_state_logits=forecast_logits,
            values=values,
            outcome_probabilities=(
                None if outcome_logits is None else torch.softmax(outcome_logits, -1)
            ),
            carry=carry,
            replay=replay,
            correction_logits=correction_logits,
        )
        if not inspect:
            return decision, None
        unrolled = Unrolled(
            policy_logits=policy_logits,
            enemy_action_logits=every_action_logits.to(dtype),
            next_state_logits=self._next_state_logits(tokens.code, every_change.to(dtype)),
            value_logits=value_logits,
            values=values,
            outcome_logits=outcome_logits,
            density_log_rates=density_log_rates.to(dtype),
            shot_outcome_logits=shot_outcome_logits.to(dtype),
            shot_latents=shot_latents.to(dtype),
            rotary_readout=_cast(_blocks(readout), dtype),
            correction_logits=_cast(every_correction, dtype),
        )
        return decision, torch.utils._pytree.tree_map(lambda tensor: tensor[None], unrolled)

    # ------------------------------------------------------------------ unroll

    def unroll(self, carry: Carry, obs: Obs, replay: Replay, reset: torch.Tensor) -> Unrolled:
        """A stored segment (L, R, ...) from the carry at its start and its replay records;
        reset (L, R) as `step` received it. The encoder and the spatial sublayers run over
        every step at once, the temporal sublayers scan over time."""
        keep = ~reset[0]  # step clears the trunk carry at a reset; so does unroll
        state = torch.where(keep[:, None, None, None], carry.state, 0.0)
        buffer = torch.where(keep[:, None, None, None, None], carry.convolution_buffer, 0.0)
        tokens = self.tokens(obs, replay, self.preceding_motion(carry, obs, replay, reset))
        own, ships = replay.previous_action.shape[-1], tokens.code.shape[-2]
        dtype = self.bins.dtype
        readout: dict = {}
        with self._matmuls(obs.ships.device):
            queries, memory, token_tables, map_tables = self._encode(tokens)
            hidden, _, _ = self.trunk.sequence(
                queries,
                memory,
                token_tables,
                map_tables,
                state,
                buffer,
                reset,
                self._readout(obs.ships, token_tables, None, readout),
            )
            policy_logits, value_logits = self.own_heads(hidden[..., :own, :])
            ship_hidden = hidden[..., :ships, :]
            enemy_action_logits, change = self.ship_heads(ship_hidden)
            outcome_logits = self._outcome_logits(hidden[..., ships, :])
            (density_log_rates,) = self.density(hidden[..., ships, :])
            (shot_outcome_logits,) = self.shot_outcome(ship_hidden)
            shot_latents = self.shot_trajectory_head.latent(ship_hidden)
            correction_logits = None
            if self.correction_head is not None:
                (correction_logits,) = self.correction_head(ship_hidden)
        outcome_logits = _cast(outcome_logits, dtype)
        value_logits, values = self._values(value_logits.to(dtype), outcome_logits)
        return Unrolled(
            policy_logits=policy_logits.to(dtype),
            enemy_action_logits=enemy_action_logits.to(dtype),
            next_state_logits=self._next_state_logits(tokens.code, change.to(dtype)),
            value_logits=value_logits,
            values=values,
            outcome_logits=outcome_logits,
            density_log_rates=density_log_rates.to(dtype),
            shot_outcome_logits=shot_outcome_logits.to(dtype),
            shot_latents=shot_latents.to(dtype),
            rotary_readout=_cast(_blocks(readout), dtype),
            correction_logits=_cast(correction_logits, dtype),
        )

    def shot_trajectory(self, latents: torch.Tensor, ages: torch.Tensor) -> torch.Tensor:
        """Trajectory code logits (..., Q, trajectory_width) of latents (..., D) at bullet
        ages: whole moves 1 .. bullet_moves (int64), (Q,) for every row or (..., Q) each
        row's own.

        Its products run under the model's matmul autocast like every other head's (bf16
        when `bfloat16_matmuls`): the query is the head's whole cost, a 128 x 128 and a
        128 x outputs product at every ship, tick and queried age. The logits come back in
        the model's dtype, so the loss over them is float32 as before."""
        moves = torch.arange(1, self.bullet_moves + 1, dtype=self.bins.dtype, device=latents.device)
        features = shots.time_features(moves, float(self.bullet_moves), SHOT_HARMONICS)
        with self._matmuls(latents.device):
            logits = self.shot_trajectory_head.query(latents, features, ages - 1)
        return logits.to(self.bins.dtype)
