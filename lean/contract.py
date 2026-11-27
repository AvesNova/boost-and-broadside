"""The fixed interfaces between the lean packages (see lean/CONTRACT.md).

Sizes: E environments, N ship slots (both teams together), K bullet slots per ship.
Layout: batch first, features last. Units: metres, radians, seconds, ticks where stated.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass

import torch
from torch import Tensor

STICKS, POWERS, FIRES = 7, 3, 2
JOINT = STICKS * POWERS * FIRES
# A bullet's life in ticks: every ship writes one bullet a tick into its ring of BULLET_SLOTS
# slots (the slot of tick t is t mod BULLET_SLOTS), so each bullet flies exactly this many ticks.
BULLET_SLOTS = 12
SHIP_TYPES = 1  # len(lean.game.card.TYPE_NAMES)

# Per-ship observation row: 18 scaled scalars, 4 flags, then the ship-type one-hot.
SHIP_SCALARS = 18
SHIP_FLAGS = 4
SHIP_FEATURES = SHIP_SCALARS + SHIP_FLAGS + SHIP_TYPES
GLOBAL_FEATURES = 3
# Per-field observation row (v2's field inputs, src/bnb/models/yemong/inputs.py): radius over
# 1000 m, band width over 40 m, level (half-octaves of the refractive index) over 2.
FIELD_FEATURES = 3


NO_PENDING = JOINT  # Obs.pending of a row whose pending action is not observed

# The reward levels (v2's, in v2's order): one critic target per level, so the model's value
# head has LEVELS outputs of BINS each. The learner owns the rewards and their supports.
LEVEL_NAMES = ("damage", "kill_death", "objective_progress", "objective", "outcome")
LEVELS = len(LEVEL_NAMES)


def split_joint(joint: Tensor) -> Tensor:
    """Split a joint action index (...,) long in [0, 42) into (..., 3) long [stick, power, fire]."""
    stick = torch.div(joint, POWERS * FIRES, rounding_mode="floor")
    power = torch.div(joint, FIRES, rounding_mode="floor") % POWERS
    fire = joint % FIRES
    return torch.stack([stick, power, fire], dim=-1)


def join_action(action: Tensor) -> Tensor:
    """Join (..., 3) long [stick, power, fire] into the joint index stick * 6 + power * 2 + fire."""
    return action[..., 0] * (POWERS * FIRES) + action[..., 1] * FIRES + action[..., 2]


# The seat (environment) axis of lean.model's recurrent carry, (blocks, E, 1 + N, 4, width).
SEAT_AXIS = 1
# The observation fields constant within an episode (the fields drawn at the deal): a rollout
# store keeps them once per deal, not per tick (lean.learn.store).
EPISODE_CONSTANT = ("field", "field_pos")


def symlog(x: Tensor) -> Tensor:
    """sign(x) ln(1 + |x|) (DreamerV3's): the game's observation scales, the model's
    next-state bins."""
    return torch.sign(x) * torch.log1p(x.abs())


def map_fields(record, function):
    """A copy of a dataclass with `function` applied to every tensor field (and to every tensor
    of a dict field: Constants.per_type)."""
    changes = {}
    for field in dataclasses.fields(record):
        value = getattr(record, field.name)
        if isinstance(value, Tensor):
            changes[field.name] = function(value)
        elif (
            isinstance(value, dict) and value and all(isinstance(v, Tensor) for v in value.values())
        ):
            changes[field.name] = {name: function(item) for name, item in value.items()}
    return dataclasses.replace(record, **changes)


def whole(tensor: Tensor) -> Tensor:
    """`tensor` with contiguous strides (a copy only when some stride is 0, a broadcast)."""
    return tensor.clone(memory_format=torch.contiguous_format) if 0 in tensor.stride() else tensor


def whole_state(state):
    """A game state (State) with every tensor whole, its constants too: a compiled graph is
    specialised on strides, and a fresh deal holds broadcast rows where a re-dealt state holds
    whole ones."""
    state = map_fields(state, whole)
    return dataclasses.replace(state, constants=map_fields(state.constants, whole))


def compile_dynamic(function):
    """`function` under torch.compile with dynamic sizes, duck sizing turned off for the
    process (the one place it is set): with it on, two sizes equal at the first call (ship
    slots and field slots at 5v5) are tied and a new team size recompiles; off, each graph
    compiles once for every team size, environment count and batch size
    (docs/rewrite/scoreboard.md). The inductor and triton caches default to
    ~/.cache/bnb-inductor when TORCHINDUCTOR_CACHE_DIR is not set."""
    import torch.fx.experimental._config as fx_config

    # /tmp is wiped at reboot, which loses the warm compile cache (93 s against about 230 s
    # cold): default it to the home cache unless the caller (a test, a box) set its own.
    if "TORCHINDUCTOR_CACHE_DIR" not in os.environ:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.expanduser("~/.cache/bnb-inductor")
        os.environ.setdefault(
            "TRITON_CACHE_DIR", os.path.join(os.environ["TORCHINDUCTOR_CACHE_DIR"], "triton")
        )
    fx_config.use_duck_shape = False
    return torch.compile(function, dynamic=True)


def random_keywords(generator: torch.Generator | None, device) -> dict:
    """Keywords for a random factory (torch.rand, randn, randint) drawing on `device` from
    `generator`, or from torch's default generator of the device when it is None. The
    `generator` keyword is left out when it is None: torch.compile (PyTorch 2.15) cannot
    trace a factory called with `generator=None` and a symbolic size, and a compiled graph
    always draws from the default generator."""
    if generator is None:
        return {"device": device}
    return {"generator": generator, "device": device}


@dataclass(eq=False)  # compared and hashed by identity (built-once caches key on it)
class Constants:
    """Per-ship constants of one run, read by name: `constants["lift"]` or `constants.lift`.

    A constant equal for every ship of the run (every type in the lineup, or fixed by the
    rules) is a Python number in `uniform`, compiled into the graph. One that differs by ship
    type is a row of `table` (E, C, N) float32, gathered once per slot at reset; `columns`
    gives its first row and width (width 1 reads as (E, N), a wider one, an action-indexed
    table, as (E, width, N)). Action-indexed tables are always rows of `table`. The run's card
    (lean.game.core.ship_card) is the same with one column per ship type: table (1, C, T).
    """

    uniform: dict[str, float | int]
    columns: dict[str, tuple[int, int]]
    table: Tensor
    # Tables by card type (T, ...), read with a ship's type (State.type), never dealt to slots:
    # lean.game.core.PER_TYPE_TABLES (the steady-flight table, too wide for every slot's row).
    per_type: dict[str, Tensor] = dataclasses.field(default_factory=dict)

    def __getitem__(self, name: str) -> Tensor | float | int:
        if name in self.uniform:
            return self.uniform[name]
        if name in self.per_type:
            return self.per_type[name]
        start, width = self.columns[name]
        return self.table[:, start] if width == 1 else self.table[:, start : start + width]

    def __getattr__(self, name: str) -> Tensor | float | int:
        if name.startswith("__") or name in ("uniform", "columns", "table", "per_type"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __contains__(self, name: str) -> bool:
        return name in self.uniform or name in self.columns or name in self.per_type


# The packed state's rows, by name. Each name reads as an (E, N) view (bullets (E, N, K)).
# Per-ship quantities that change in an episode, float32 (counts held as whole numbers):
SHIP_COLUMNS = (
    "x",  # metres, arena centre at 0
    "y",
    "heading_cos",  # nose direction
    "heading_sin",
    "speed",  # metres per second (proper speed)
    "angle",  # angle of attack, radians
    "angle_cos",
    "angle_sin",
    "nose_rate",  # radians per second
    "energy",  # health and fuel in one store
    "shield",
    "shield_wait",  # seconds until the shield may recharge
    "cooldown",  # whole ticks until the gun may fire
    # The medium (ln n, d/dx ln n, d/dy ln n) at the ship, cached from the end of the last tick
    # (fields are fixed within an episode), as v2 caches it.
    "log_index",
    "gradient_x",
    "gradient_y",
    # The velocity at proper speed along the flight path (the nose turned back by the angle of
    # attack), stored by the step once (the world velocity is this over the refractive index).
    "velocity_x",
    "velocity_y",
)
FLAG_COLUMNS = (
    "alive",
    "exists",  # fixed for an episode
    "fired",  # the ship fired a real bullet on the last tick (stored, not inferred)
)
IDENTITY_COLUMNS = ("team", "type")  # long, fixed for an episode: 0 or 1; index into the ship card
BULLET_COLUMNS = (
    "bullet_x",
    "bullet_y",
    "bullet_dir_x",  # unit direction of travel
    "bullet_dir_y",
    "bullet_speed",  # metres per second (proper speed)
    "bullet_log_index",  # the medium at the bullet, cached from the end of its last move
    "bullet_gradient_x",
    "bullet_gradient_y",
)
# A bullet's state (State.bullet_state, uint8): a ghost (no damage; labels the hit head), a
# real shot, or contacted (it touched a ship: no further contact or damage; it flies on to the
# end of its life).
GHOST, REAL, CONTACTED = 0, 1, 2
BLOCKS = {
    "ships": SHIP_COLUMNS,
    "flags": FLAG_COLUMNS,
    "identity": IDENTITY_COLUMNS,
    "bullets": BULLET_COLUMNS,
}
COLUMN = {
    name: (block, index) for block, names in BLOCKS.items() for index, name in enumerate(names)
}


@dataclass
class State:
    """The whole game state, packed so that per-field work (a reset's select, holding a dead
    ship) is one operation per block. Every quantity in the *_COLUMNS tuples reads by name as a
    view: `state.x` is `state.ships[:, 0]`, (E, N); `state.bullet_x` is (E, N, K).

    Stick index: 0 straight, then (1, 2) fine, (3, 4) sustained, (5, 6) full, odd indices
    one direction and even the other. Power index: 0 coast, 1 dive, 2 climb.
    """

    ships: Tensor  # (E, SHIP_COLUMNS, N) float32
    flags: Tensor  # (E, FLAG_COLUMNS, N) bool
    identity: Tensor  # (E, IDENTITY_COLUMNS, N) long
    constants: Constants  # per-ship constants: Python numbers and one table gathered at reset
    bullets: Tensor  # (E, BULLET_COLUMNS, N, K) float32
    bullet_state: Tensor  # (E, N, K) uint8: GHOST, REAL or CONTACTED
    tick: Tensor  # (E,) long
    # The environment's refractive fields, (E, 5, F) float32 rows [x, y, radius, band width,
    # target ln n]; F may be 0 (no medium).
    fields: Tensor
    # Each ship's pending action, (E, N, 3) long [stick, power, fire]: the action chosen on the
    # last tick, which the next tick applies (the one-tick action delay). Neutral at a deal.
    pending: Tensor
    # Fog (None with fog off). Each ship as its enemy team last saw it: (E, SEEN_COLUMNS, N)
    # float32 raw values in the order of lean.game.core.SEEN_COLUMNS (the last two are "alive"
    # at that sighting, 1 until seen dead, and "known", 1 once ever seen); and the ticks since
    # the enemy team last saw it, (E, N) long, 0 while visible, capped (v2's ticks_since_seen).
    seen: Tensor | None = None
    ticks_since_seen: Tensor | None = None

    @classmethod
    def pack(
        cls,
        *,
        constants: Constants,
        tick: Tensor,
        fields: Tensor,
        bullet_state: Tensor,
        pending: Tensor | None = None,
        seen: Tensor | None = None,
        ticks_since_seen: Tensor | None = None,
        **named: Tensor,
    ) -> State:
        """A state from every named quantity of the *_COLUMNS tuples, each (E, N) or (E, N, K);
        `pending` defaults to the neutral action for every ship."""
        missing = set(COLUMN) - set(named)
        if missing or len(named) != len(COLUMN):
            raise TypeError(f"State.pack needs every column once; missing {sorted(missing)}")
        dtypes = {
            "ships": torch.float32,
            "flags": torch.bool,
            "identity": torch.long,
            "bullets": torch.float32,
        }
        blocks = {
            block: torch.stack([named[name].to(dtypes[block]) for name in names], 1)
            for block, names in BLOCKS.items()
        }
        if pending is None:
            pending = torch.zeros(
                named["x"].shape + (3,), dtype=torch.long, device=named["x"].device
            )
        return cls(
            **blocks,
            constants=constants,
            tick=tick,
            fields=fields,
            bullet_state=bullet_state,
            pending=pending,
            seen=seen,
            ticks_since_seen=ticks_since_seen,
        )

    def replace(self, **changes) -> State:
        """A copy with named quantities (and any dataclass fields) replaced; a changed block is
        copied, the others shared."""
        blocks: dict[str, Tensor] = {}
        fields = {}
        for name, value in changes.items():
            if name in COLUMN:
                block, index = COLUMN[name]
                if block not in blocks:
                    blocks[block] = getattr(self, block).clone()
                blocks[block][:, index] = value
            else:
                fields[name] = value
        return dataclasses.replace(self, **fields, **blocks)


def _column(block: str, index: int) -> property:
    return property(lambda state: getattr(state, block)[:, index])


for _name, (_block, _index) in COLUMN.items():
    setattr(State, _name, _column(_block, _index))


@dataclass
class StepOut:
    """What happened in one tick."""

    damage_taken: Tensor  # (E, N) float32, all damage: bullets, the boundary, collisions
    died: Tensor  # (E, N) bool
    done: Tensor  # (E,) bool
    result: Tensor  # (E,) long: 0 or 1 = winning team, 2 = draw, -1 = not done
    # Each bullet's outcome, once, per slot (E, N, K) (real and ghost alike): at its first
    # contact (1 hit an enemy, 2 hit a team-mate) or at its last tick without one (0 missed);
    # -1 none this tick (a contacted bullet flies on but never reports again);
    # shot_tick the episode tick that bullet left the gun (the tick the hit head labels).
    shot_outcome: Tensor
    shot_tick: Tensor
    # Damage detail for the rewards (v2's ledger), each before the shield's clamp:
    # (E, N, N) float32 [attacker, target]: real bullets' damage this tick, and collisions' (each
    # ship's collision damage credited to the other ship of the pair)
    hit_damage: Tensor
    boundary_damage: Tensor  # (E, N) float32: damage from the arena boundary this tick
    shield_restored: Tensor  # (E, N) float32: shield recharged this tick


@dataclass
class Obs:
    """One team's view of the game: full information with fog off; with fog on, enemy rows are
    their last-seen values and nothing depends on what the team cannot see."""

    ship: Tensor  # (E, N, SHIP_FEATURES) float32
    pos: Tensor  # (E, N, 2) float32 metres
    heading: Tensor  # (E, N) float32 radians
    key_mask: Tensor  # (E, N) bool: row may be attended (exists and alive)
    own: Tensor  # (E, N) bool: rows this view controls (own team, alive)
    globals: Tensor  # (E, GLOBAL_FEATURES) float32
    field: Tensor  # (E, F, FIELD_FEATURES) float32; F may be 0
    field_pos: Tensor  # (E, F, 2) float32 metres, the field's centre
    visible: Tensor  # (E, N) bool: the row is seen this tick (own rows always; all with fog off)
    since_seen: Tensor  # (E, N) float32: symlog of the seconds since the row was seen (0 visible)
    # (E, N) long: the joint pending action (the one the next tick applies) of the view's own
    # living ships; NO_PENDING for every other row (enemies' pending actions are not observable).
    pending: Tensor
