"""Designing real-scale ship types for balance and cycles
(docs/explanation/game/ship-design.md): the knobs a design moves, the WW2 envelope that
bounds them, and the instant card of a design.

A design is a base type of the real-scale set (a propeller type, real or invented) and the
values of its knobs. The knobs are existing handles, or ratios of them chosen so that the
knobs are nearly independent (`KNOBS`, each with the reason it is in the search):

- flight: the stall speed, the cruise speed, the sustained turn rate, the climb rate (the
  store's initial charge, as a ratio to the set-wide level) and the store (`ceiling`);
- combat: the damage per second (the hit damage is it over the fire rate), the fire rate (a
  whole number of hertz dividing the 30 Hz tick), the shield, the muzzle speed and the
  dispersion angle (the bullet spread over the muzzle speed);
- size: the hit radius (the half span).

Every other handle is the base type's; the gun range follows from the cruise and the muzzle
speed, the climb time from the store and the climb rate, the spawn energy is the base's
share of the store, and the shot keeps the base's store cost per second of fire.

The envelope is the 16 real propeller fighters of the set (`ENVELOPE`; the Me 262 and Me 163
are left out, as are the invented designs): each knob's log has their mean and standard
deviation, and a design's realism is the Mahalanobis distance of its knobs' logs from that
mean, under the 16 types' covariance shrunk half way to its diagonal (`Envelope`), so that
a combination the real fighters do not show (fast and weakly armed) costs more than its
parts. The bounds are the 16 types' range, with margins only where the data is soft or the
author widened them (`BOUNDS`). The two steepest knobs, the muzzle speed and the shield (about
1,400 and 1,200 Elo a doubling), are held to a narrower band around the real values
(`NARROWED`, the author's decision of 2026-11-10): the differences between designs may be
smaller than the real fighters', not larger.

`design_cards` is the instant card of designs (`bnb.game.handling.estimate.Estimate`,
anchored on each base's solved card); `bnb.game.real_scale.solve_type` gives the exact one.
"""

import dataclasses
import math

import torch

from . import ships
from .handling import Handles
from .handling.estimate import Estimate
from .handling.weapons import gun_range
from .real_scale import CLIMB_FILL, GRAVITY, SHIP_SET
from .rules import Rules
from .ship_types import FIELDS, ShipTypes, ticks_between_shots

# The real propeller fighters of the set: the envelope a design stays in.
ENVELOPE = (
    "p51d",
    "p47d",
    "p38l",
    "spitfire-ix",
    "spitfire-xiv",
    "tempest-v",
    "mosquito",
    "fw190-a8",
    "fw190-d9",
    "bf109-k4",
    "a6m5",
    "ki84",
    "yak3",
    "f4u-1d",
    "la7",
    "f6f-5",
)
# The bases a design may start from: the envelope's types and the invented propeller designs
# (not the rocket Firecracker, nor the jets).
BASES = ENVELOPE + (
    "heavy",
    "hammerhead",
    "wren",
    "anvil",
    "albatross",
    "zerstorer",
    "shrike",
    "lantern",
)
# The volley rates a design may fire: whole hertz dividing the 30 Hz tick, from the
# invented heavy cannon's 5 Hz to the .50 batteries' 15 (2 and 3 Hz, the Anvil's, are left
# out: a WW2 fighter's main battery fires faster).
FIRE_RATES = (5.0, 6.0, 10.0, 15.0)


@dataclasses.dataclass(frozen=True)
class Knob:
    """One knob: its name, its group and why it is in the search."""

    name: str
    group: str
    reason: str


KNOBS = (
    Knob("stall_speed", "flight", "the wing loading: the corner speed and the slow fight"),
    Knob("cruise_speed", "flight", "the speed: who chooses the fight"),
    Knob("sustained_turn_rate", "flight", "the turn fight"),
    Knob(
        "climb_rate",
        "flight",
        "the store's charge, over the set-wide level: the energy fight and the reset",
    ),
    Knob("ceiling", "flight", "the store: how long a ship can boost and fire"),
    Knob("damage_per_second", "combat", "the weight of fire"),
    Knob("fire_rate", "combat", "how the damage arrives: the hits a kill takes against a shield"),
    Knob("max_shield", "combat", "toughness"),
    Knob("muzzle_speed", "combat", "the gun's reach and the lead a shot needs"),
    Knob("dispersion", "combat", "the cone: hits on a manoeuvring target at range"),
    Knob("hit_radius", "size", "the target a design presents (the half span)"),
)
NAMES = tuple(knob.name for knob in KNOBS)
CONTINUOUS = tuple(name for name in NAMES if name != "fire_rate")
# Bounds (physical units) a knob may not leave: None takes the envelope's range. The
# damage per second is not researched (the balancing set it): +-20% of the P-51D's 360. The
# shield may rise above the set's 160 (the author's decision of 2026-11-10). The turn has
# +-10% beyond the envelope (no turn rate is measured).
BOUNDS = dict(
    damage_per_second=(288.0, 432.0),
    max_shield=(80.0, 200.0),
)
SOFT_MARGINS = dict(sustained_turn_rate=0.10)
# The steep knobs' narrowed band (physical units): a design's value stays within it (the
# envelope's statistics, and so the realism distance, are unchanged). The muzzle speed's is
# the real guns' cluster (the Bf 109K's 520 m/s cannon left out) with room above the 880 m/s
# bases for a trim; the shield's is the researched types' 90 to 130 with room above for the
# armoured designs.
NARROWED = dict(
    muzzle_speed=(720.0, 900.0),
    max_shield=(90.0, 150.0),
)


def climb_rate_of(handles: dict[str, float]) -> float:
    """A card's climb rate (m/s): the one its climb time was solved for (to the solve's 1%)."""
    return CLIMB_FILL * handles["ceiling"] / (GRAVITY * handles["climb_time"])


def knobs_of(handles: dict[str, float]) -> dict[str, float]:
    """A type's knob values from its handles (physical units)."""
    return dict(
        stall_speed=handles["stall_speed"],
        cruise_speed=handles["cruise_speed"],
        sustained_turn_rate=handles["sustained_turn_rate"],
        climb_rate=climb_rate_of(handles),
        ceiling=handles["ceiling"],
        damage_per_second=handles["hit_damage"] * handles["fire_rate"],
        fire_rate=handles["fire_rate"],
        max_shield=handles["max_shield"],
        muzzle_speed=handles["muzzle_speed"],
        dispersion=handles["bullet_spread"] / handles["muzzle_speed"],
        hit_radius=handles["hit_radius"],
    )


def handles_of(base: dict[str, float], knobs: dict[str, float], rules: Rules) -> dict[str, float]:
    """The handles of a design: the base's, with these knobs (any subset) applied and the
    handles that follow from them recomputed (the module docstring)."""
    values = knobs_of(base) | knobs
    if values["fire_rate"] not in FIRE_RATES:
        raise ValueError(f"a fire rate of {values['fire_rate']:g} Hz: use one of {FIRE_RATES}")
    handles = dict(base)
    for name in ("stall_speed", "cruise_speed", "sustained_turn_rate", "ceiling"):
        handles[name] = values[name]
    handles["max_shield"] = values["max_shield"]
    handles["hit_radius"] = values["hit_radius"]
    handles["muzzle_speed"] = values["muzzle_speed"]
    handles["fire_rate"] = values["fire_rate"]
    handles["hit_damage"] = values["damage_per_second"] / values["fire_rate"]
    handles["bullet_spread"] = values["dispersion"] * values["muzzle_speed"]
    handles["climb_time"] = CLIMB_FILL * values["ceiling"] / (GRAVITY * values["climb_rate"])
    handles["spawn_energy"] = base["spawn_energy"] / base["ceiling"] * values["ceiling"]
    cost_per_second = base["ceiling"] / base["shots_per_store"] * base["fire_rate"]
    handles["shots_per_store"] = values["ceiling"] * values["fire_rate"] / cost_per_second
    drag = torch.tensor(rules.bullet_drag, dtype=torch.float64)
    launch = torch.tensor(values["cruise_speed"] + values["muzzle_speed"], dtype=torch.float64)
    handles["gun_range"] = float(gun_range(rules, drag, launch))
    return handles


def set_handles(ship_set: str = SHIP_SET) -> dict[str, dict[str, float]]:
    """Every type's handles in a set, by key."""
    return {ship.key: ship.handles for ship in ships.load(ship_set)}


@dataclasses.dataclass
class Envelope:
    """The envelope's statistics over its types' knobs (continuous ones, in logs): mean,
    standard deviation, bounds (logs), and the shrunk covariance's inverse."""

    mean: torch.Tensor
    deviation: torch.Tensor
    low: torch.Tensor
    high: torch.Tensor
    precision: torch.Tensor

    @classmethod
    def of(cls, handles: dict[str, dict[str, float]], keys=ENVELOPE, shrink: float = 0.5):
        """The envelope of these types' knobs (`keys`), the correlation shrunk by `shrink`."""
        rows = torch.tensor(
            [[math.log(knobs_of(handles[key])[name]) for name in CONTINUOUS] for key in keys],
            dtype=torch.float64,
        )
        mean, deviation = rows.mean(0), rows.std(0)
        low, high = rows.min(0).values.clone(), rows.max(0).values.clone()
        for index, name in enumerate(CONTINUOUS):
            if name in SOFT_MARGINS:
                low[index] += math.log(1.0 - SOFT_MARGINS[name])
                high[index] += math.log(1.0 + SOFT_MARGINS[name])
            if name in BOUNDS:
                low[index], high[index] = (math.log(bound) for bound in BOUNDS[name])
                # a knob the envelope does not vary (the balanced damage) spreads over its
                # bounds: four deviations
                deviation[index] = max(float(deviation[index]), float(high[index] - low[index]) / 4)
        for index, name in enumerate(CONTINUOUS):  # the steep knobs' band (bounds only)
            if name in NARROWED:
                low[index], high[index] = (math.log(bound) for bound in NARROWED[name])
        centred = (rows - mean) / deviation
        correlation = centred.T @ centred / (len(keys) - 1)
        shrunk = (
            shrink * torch.eye(len(CONTINUOUS), dtype=torch.float64) + (1 - shrink) * correlation
        )
        return cls(mean, deviation, low, high, torch.linalg.inv(shrunk))

    def z(self, knobs: dict[str, float]) -> torch.Tensor:
        """The knobs' logs in deviations from the envelope's mean (CONTINUOUS order)."""
        logs = torch.tensor([math.log(knobs[name]) for name in CONTINUOUS], dtype=torch.float64)
        return (logs - self.mean) / self.deviation

    def distance(self, knobs: dict[str, float]) -> float:
        """The Mahalanobis distance of a design's knobs from the envelope's mean."""
        z = self.z(knobs)
        return float(z @ self.precision @ z) ** 0.5

    def clamp(self, knobs: dict[str, float]) -> dict[str, float]:
        """The knobs held within the bounds."""
        out = dict(knobs)
        for index, name in enumerate(CONTINUOUS):
            low, high = math.exp(float(self.low[index])), math.exp(float(self.high[index]))
            out[name] = min(max(knobs[name], low), high)
        return out


def design_cards(
    rules: Rules, bases: list[str], designs: list[dict[str, float]], ship_set: str = SHIP_SET
) -> ShipTypes:
    """The instant cards ((len(designs),) fields, float32) of designs given as their base
    keys and full handles, each estimated from its base's solved card (`Estimate`)."""
    found = {ship.key: ship for ship in ships.load(ship_set)}
    anchor = ShipTypes(
        **{
            name: torch.tensor([found[key].card[name] for key in bases], dtype=torch.float64)
            for name in FIELDS
        }
    )

    def stacked(rows: list[dict[str, float]]) -> Handles:
        return Handles(
            **{
                field.name: torch.tensor([row[field.name] for row in rows], dtype=torch.float64)
                for field in dataclasses.fields(Handles)
            }
        )

    estimate = Estimate(rules, anchor, stacked([found[key].handles for key in bases]))
    card = estimate.card(stacked(designs))
    ticks_between_shots(card.fire_rate, rules.tick_length)
    return card
