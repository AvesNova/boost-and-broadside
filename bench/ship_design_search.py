"""Design real-scale ship types whose head-to-head results form cycles
(docs/explanation/game/ship-design.md): a CMA-ES search over the designs' knobs
(`bnb.game.ship_design`) and their doctrine (a few of the scripted agent's parameters, per
design), each generation's candidates played on one fixed arena whose card and doctrine
table are written in place, so nothing recompiles.

    uv run --no-sync python -m bench.ship_design_search search --spec SPEC.json --out DIR
    uv run --no-sync python -m bench.ship_design_search export --spec SPEC.json --out DIR \\
        --ship-set design-NAME
    uv run --no-sync python -m bench.ship_design_search trim --spec SPEC.json --out DIR \\
        --card design-NAME
    uv run --no-sync python -m bench.ship_design_search check --spec SPEC.json --out DIR \\
        [--candidate GENERATION:INDEX | trim] [--games 1024] [--card SET]
    uv run --no-sync python -m bench.ship_design_search robust --spec SPEC.json --out DIR ...
    uv run --no-sync python -m bench.ship_design_search summary --spec SPEC.json --out DIR ...

SPEC.json (see `docs/explanation/game/ship-design/`): the designs (name, base type, the
knobs the search moves, optional start values; the doctrine dimensions it moves,
"doctrine", and optional fixed or start values, "doctrine_start"), the edges (each [winner,
loser]) or a cycle (names in order; the search picks its direction at the first
generation), the band an edge's score should lie in, the trim ("trim": the knob, its Elo per
doubling or the case study it is read from) and the search's sizes. Each design's knobs
start at its base's values; a knob's search coordinate is its log in envelope deviations
from there (the fire rate: one unit a step between FIRE_RATES), held within the envelope's
bounds (the steep knobs within their narrowed band, `NARROWED`). A doctrine dimension is a
scripted parameter (`energy_fight_strength`) or a part of one (`shoot_in_gun_range.centre`);
its coordinate is DOCTRINE_STEP of its log a unit (an additive one: DOCTRINE_STEP of its
bounds' width), on top of the preset's set.

The loss of a candidate (a value for every design's searched knobs and doctrine) is

    sum over edges of (the edge's log-odds' distance outside the band)^2
    + realism_weight x sum over designs of max(0, Mahalanobis distance - realism_radius)^2
    + distinct_weight x sum over design pairs of max(0, distinct - their z distance)^2
    + curl_weight x (for a cycle) max(0, edges x logit(band low) - the curl)^2
    + role_weight x sum over designs of the roles' shortfalls^2 (with "axes", below),

the edges played `games_per_edge` games each (`games_late` once the step size has halved)
(half with each design on each side), every ship of a team one design flown by its own
doctrine, both teams the scripted agent (its real-scale set for the preset, T = 1). With a
trim, the edges are the trimmed prediction (`predicted_trim`): the ratings the edges imply
are taken out, as far as the trim knob can within its band, and the realism is the trimmed
designs'; so the search shapes the cycle and leaves the ratings to the trim. Each generation
plays the CMA-ES mean and `population - 1` samples; its rows (every candidate's knobs,
doctrine, scores, trim and loss) are appended to DIR/search.jsonl and the optimiser's state
to DIR/state.json as it ends, so a stopped search resumes. The search stops at
`generations` or `--hours`; each generation prints its ETA.

Attribute axes ("axes": {axis: {member: +1 or -1}}, a member a knob or "doctrine:" and a
doctrine dimension, the sign +1 where more of it is more of the attribute) let a spec say what
each type is: a design's "strong" axis and its "weak" one (archers strong in reach and weak in
armour). An axis' score of a design is the mean of its members' signed coordinates (a knob's
in envelope deviations, a doctrine dimension's in search units from the preset's value);
the roles' term asks each design to lead every other by `role_margin` on its strong axis and
trail every other by as much on its weak one. A design that lists no knobs or doctrine of its
own moves every axis' members.

`export` solves a candidate's designs exactly (`bnb.game.real_scale.solve_type`, one process
each) and writes them as a ship set (`--ship-set`, with the designs' doctrine in its
doctrine.json), so `--card SET` plays the exact cards.

`trim` brings the designs' ratings in line with the trim knob: rounds of a round robin
(`--games` a pair), each design's knob times 2^(-its Elo / the slope), within its band, until
every rating is within `--target` Elo of the mean or `--rounds`; the slope is the spec's,
refreshed from the rounds' own changes once two have played. Each round is a row of
DIR/trim.jsonl (resumed if rerun); `--candidate trim` is the round of the smallest spread.

`check` plays a candidate (by default the best mean of the search) at `--games` games per
edge on every pair of designs (a round robin), and the mixed-team test: for every pair
{A, B} and every design D that the spec says beats both, a team of A and B (alternating,
3 + 2 and 2 + 3 in half the games each) against a team of D; in the spec's preset and in
each of `--presets` (Frontline by default: an evaluation, not a gate). Rows to
DIR/check-<preset>.jsonl.

`robust` replays the round robin with every design flying its own random perturbation of
its doctrine, several draws (the acceptance test of the agent's dependence), and writes its
verdict to DIR/robust-summary-<preset>.json.

`compose` writes the seven-type spec from three cycle searches (the Paley tournament's
three planes, one per cycle; see the function).

`summary` turns a check into the verdict against the targets: the Bradley-Terry ratings, the
edges against the band, the uniform pick's exploitability, the Nash equilibrium (regret
matching) and its support, and the mixed-team test (DIR/summary-<preset>.json); only the
spec's preset is a gate.

A card is "exact" when it is a ship set's solved card for the candidate's own handles; any
other card played (the instant estimate, or a set's card estimated at changed handles) is
labelled as such in every row (`card`), and only exact cards are accepted.
"""

import argparse
import dataclasses
import datetime
import itertools
import json
import math
import pathlib
import time

import numpy as np
import torch

from bench.lock import bench_lock
from bench.scripted_search import clamp as clamp_parameters
from bnb.controllers.scripted_defaults import MODE_DEFAULTS
from bnb.controllers.scripted_parameters import (
    ADDITIVE_BOUNDS,
    ScriptedParameters,
)
from bnb.game import api
from bnb.game.ship_design import (
    CONTINUOUS,
    FIRE_RATES,
    Envelope,
    design_cards,
    handles_of,
    knobs_of,
    set_handles,
)
from bnb.game.ship_types import FIELDS, ShipTypes
from bnb.league.matches import Arena, ArenaConfig, Player
from bnb.runtime.opponents import Rung

SLOTS = 36  # the arena's card: a fixed number of type slots, so a new card never recompiles
ARENA_BLOCKS = 36  # matchups a batch
BLOCK_GAMES = 64
ELO = 400.0 / math.log(10.0)  # Elo a unit of natural log-odds
FRONTLINE = "real_scale_frontline"  # evaluated by `check`, never a gate
# A doctrine dimension's step a search unit: a quarter of its log (the perturbation size at
# which the robustness study's single edges moved by up to 0.45), or a quarter of an additive
# one's bounds; a multiplicative one stays within DOCTRINE_RANGE of its start either way.
DOCTRINE_STEP = 0.25
DOCTRINE_RANGE = 4.0
DOCTRINE_PREFIX = "doctrine:"  # an axis member that is a doctrine dimension


def clock(seconds: float) -> str:
    """A clock time this many seconds from now."""
    return (datetime.datetime.now() + datetime.timedelta(seconds=seconds)).strftime("%H:%M")


# ---------------------------------------------------------------- the designs


def doctrine_value(parameters: ScriptedParameters, dimension: str) -> float:
    """A doctrine dimension's value in a parameter set: a field, or a part of one
    (`shoot_in_gun_range.centre`)."""
    field, _, part = dimension.partition(".")
    value = getattr(parameters, field)
    return float(getattr(value, part) if part else value)


def with_doctrine(parameters: ScriptedParameters, values: dict[str, float]) -> ScriptedParameters:
    """A parameter set with these doctrine dimensions' values, within the scripted search's
    bounds (`bench.scripted_search.clamp`: the logistics' scale floors, the additive bounds)."""
    fields = {name: getattr(parameters, name) for name in vars(parameters)}
    for dimension, value in values.items():
        field, _, part = dimension.partition(".")
        if field not in fields:
            raise ValueError(f"{dimension}: no such scripted parameter")
        if part:
            fields[field] = fields[field]._replace(**{part: float(value)})
        else:
            fields[field] = float(value)
    return clamp_parameters(ScriptedParameters(**fields))


def is_additive(dimension: str) -> bool:
    """Whether a doctrine dimension moves by addition (`ADDITIVE_BOUNDS`) rather than by a
    factor."""
    return dimension.partition(".")[0] in ADDITIVE_BOUNDS


@dataclasses.dataclass
class Space:
    """The search's coordinates: per design, its base's handles and knobs, the knobs it
    moves and its doctrine dimensions; `x` is every design's moved knobs' coordinates, in the
    spec's order, then every design's doctrine dimensions'."""

    spec: dict
    handles: dict
    envelope: Envelope
    rules: object

    def __post_init__(self) -> None:
        axes = self.spec.get("axes", {})
        members = [m for axis in axes.values() for m in axis]
        knobs = list(dict.fromkeys(m for m in members if not m.startswith(DOCTRINE_PREFIX)))
        dimensions = list(
            dict.fromkeys(
                m[len(DOCTRINE_PREFIX) :] for m in members if m.startswith(DOCTRINE_PREFIX)
            )
        )
        # a design with no knobs or doctrine of its own moves every axis' members
        self.designs = [
            {"knobs": knobs, "doctrine": dimensions} | design if axes else design
            for design in self.spec["designs"]
        ]
        for design in self.designs:
            for role in ("strong", "weak"):
                if role in design and design[role] not in axes:
                    raise ValueError(f"{design['name']}: {role} axis {design[role]!r} not in axes")
        self.columns = [
            (index, knob) for index, design in enumerate(self.designs) for knob in design["knobs"]
        ]
        self.doctrine_columns = [
            (index, dimension)
            for index, design in enumerate(self.designs)
            for dimension in design.get("doctrine", [])
        ]
        self.preset = self.spec.get("preset", "real_scale_deathmatch")

    @property
    def size(self) -> int:
        """The number of search coordinates."""
        return len(self.columns) + len(self.doctrine_columns)

    def base(self, index: int) -> dict:
        """A design's start: its base's knobs with the spec's start values."""
        design = self.designs[index]
        knobs = knobs_of(self.handles[design["base"]]) | design.get("start", {})
        if knobs["fire_rate"] not in FIRE_RATES:  # a base outside the rates: the nearest
            knobs["fire_rate"] = min(
                FIRE_RATES, key=lambda r: abs(math.log(r / knobs["fire_rate"]))
            )
        return knobs

    def knobs(self, x) -> list[dict[str, float]]:
        """Every design's knobs at coordinates x."""
        out = [self.base(index) for index in range(len(self.designs))]
        for value, (index, name) in zip(x[: len(self.columns)], self.columns, strict=True):
            if name == "fire_rate":
                start = FIRE_RATES.index(out[index]["fire_rate"])
                step = min(max(start + int(round(float(value))), 0), len(FIRE_RATES) - 1)
                out[index]["fire_rate"] = FIRE_RATES[step]
            else:
                deviation = float(self.envelope.deviation[CONTINUOUS.index(name)])
                out[index][name] = out[index][name] * math.exp(float(value) * deviation)
        return [self.envelope.clamp(knobs) for knobs in out]

    def doctrine_start(self, index: int) -> dict[str, float]:
        """A design's doctrine at the search's start: its searched dimensions at the preset's
        values, with the spec's `doctrine_start` (which may also fix dimensions it does not
        search)."""
        design = self.designs[index]
        preset = MODE_DEFAULTS.sets[self.preset]
        start = {
            dimension: doctrine_value(preset, dimension) for dimension in design.get("doctrine", [])
        }
        return start | design.get("doctrine_start", {})

    def doctrine(self, x) -> list[dict[str, float]]:
        """Every design's doctrine values at coordinates x (each dimension within its bounds)."""
        out = [self.doctrine_start(index) for index in range(len(self.designs))]
        values = x[len(self.columns) :]
        for value, (index, dimension) in zip(values, self.doctrine_columns, strict=True):
            start = out[index][dimension]
            if is_additive(dimension):
                low, high = ADDITIVE_BOUNDS[dimension.partition(".")[0]]
                moved = start + float(value) * DOCTRINE_STEP * (high - low)
                out[index][dimension] = min(max(moved, low), high)
            else:
                factor = math.exp(float(value) * DOCTRINE_STEP)
                factor = min(max(factor, 1.0 / DOCTRINE_RANGE), DOCTRINE_RANGE)
                out[index][dimension] = start * factor
        return out

    def parameters(self, values: dict[str, float], preset: str | None = None) -> ScriptedParameters:
        """A design's parameter set in a preset (the spec's by default): the preset's set with
        the design's doctrine values."""
        return with_doctrine(MODE_DEFAULTS.sets[preset or self.preset], values)

    def handles_of(self, knobs: list[dict[str, float]]) -> list[dict[str, float]]:
        """Every design's handles from its knobs."""
        return [
            handles_of(self.handles[design["base"]], one, self.rules)
            for design, one in zip(self.designs, knobs, strict=True)
        ]

    def penalty(self, knobs: list[dict[str, float]]) -> dict:
        """The realism and distinctness terms of the loss."""
        spec = self.spec
        distances = [self.envelope.distance(one) for one in knobs]
        realism = sum(max(0.0, d - spec["realism_radius"]) ** 2 for d in distances)
        zs = [self.envelope.z(one) for one in knobs]
        close = 0.0
        for a, b in itertools.combinations(range(len(knobs)), 2):
            gap = float((zs[a] - zs[b]).norm())
            close += max(0.0, spec["distinct"] - gap) ** 2
        return dict(
            distances=distances,
            realism=spec["realism_weight"] * realism,
            distinct=spec["distinct_weight"] * close,
        )

    def axis_scores(
        self, knobs: list[dict[str, float]], doctrine: list[dict[str, float]]
    ) -> dict[str, list[float]]:
        """Each axis' score per design: the mean of its members' signed coordinates, a knob's
        in envelope deviations from the envelope's mean, a doctrine dimension's in search units
        from the preset's value (`axes`: {axis: {member: +1 or -1}}, +1 where more of the
        member is more of the attribute)."""
        preset = MODE_DEFAULTS.sets[self.preset]
        out = {}
        for axis, members in self.spec.get("axes", {}).items():
            scores = []
            for one, values in zip(knobs, doctrine, strict=True):
                z = self.envelope.z(one)
                total = 0.0
                for member, sign in members.items():
                    if member.startswith(DOCTRINE_PREFIX):
                        dimension = member[len(DOCTRINE_PREFIX) :]
                        at = doctrine_value(preset, dimension)
                        value = values.get(dimension, at)
                        if is_additive(dimension):
                            low, high = ADDITIVE_BOUNDS[dimension.partition(".")[0]]
                            coordinate = (value - at) / (DOCTRINE_STEP * (high - low))
                        else:
                            coordinate = math.log(value / at) / DOCTRINE_STEP
                    elif member == "fire_rate":
                        coordinate = math.log(one["fire_rate"] / 10.0) / math.log(1.5)
                    else:
                        coordinate = float(z[CONTINUOUS.index(member)])
                    total += sign * coordinate
                scores.append(total / len(members))
            out[axis] = scores
        return out

    def roles(self, knobs: list[dict[str, float]], doctrine: list[dict[str, float]]) -> float:
        """The roles' term of the loss: role_weight x the sum over designs of max(0, role_margin
        - its lead over every other design on its strong axis)^2 and max(0, role_margin - its
        deficit below every other on its weak axis)^2 (`axis_scores`); 0 without axes."""
        if "axes" not in self.spec:
            return 0.0
        scores = self.axis_scores(knobs, doctrine)
        margin = self.spec.get("role_margin", 0.5)
        total = 0.0
        for index, design in enumerate(self.designs):
            others = [k for k in range(len(self.designs)) if k != index]
            if "strong" in design:
                axis = scores[design["strong"]]
                lead = axis[index] - max(axis[k] for k in others)
                total += max(0.0, margin - lead) ** 2
            if "weak" in design:
                axis = scores[design["weak"]]
                deficit = min(axis[k] for k in others) - axis[index]
                total += max(0.0, margin - deficit) ** 2
        return self.spec.get("role_weight", 1.0) * total


# ---------------------------------------------------------------- the trim


def trim_slope(spec: dict) -> float:
    """The trim knob's Elo per doubling: the spec's "slope", or a case study's line
    ("slope_from": its report.json, the model of "airframe", pooled by default)."""
    trim = spec["trim"]
    if "slope" in trim:
        return float(trim["slope"])
    root = pathlib.Path(__file__).resolve().parents[1]
    report = json.loads((root / trim["slope_from"]).read_text())
    model = report["models"][trim.get("airframe", "pooled")]
    return float(model["elo_per_doubling"][trim["knob"]])


def logit_of(score: float, games: int) -> float:
    """A score's log-odds, the score held off 0 and 1 by half a game."""
    p = min(max(score, 0.5 / games), 1.0 - 0.5 / games)
    return math.log(p / (1.0 - p))


def edge_ratings(edges: list[tuple[int, int]], logits: list[float], count: int) -> np.ndarray:
    """The ratings (natural log-odds, mean 0) that best explain the edges' log-odds (least
    squares; with every pair played, the Bradley-Terry ratings to first order). What they leave
    is the edges' cyclic part."""
    design = np.zeros((len(edges), count))
    for row, (a, b) in enumerate(edges):
        design[row, a], design[row, b] = 1.0, -1.0
    ratings = np.linalg.pinv(design) @ np.array(logits)
    return ratings - ratings.mean()


def trimmed_knobs(
    space: Space, knobs: list[dict[str, float]], factors: list[float]
) -> list[dict[str, float]]:
    """The designs' knobs with the trim knob times each factor, within the knob's bounds."""
    knob = space.spec["trim"]["knob"]
    out = []
    for one, factor in zip(knobs, factors, strict=True):
        out.append(space.envelope.clamp(one | {knob: one[knob] * factor}))
    return out


def predicted_trim(
    space: Space,
    knobs: list[dict[str, float]],
    edges: list[tuple[int, int]],
    scores: list[float],
    games: int,
) -> dict:
    """The trim the edges call for and the edges it predicts: each design's ratings from the
    edges (`edge_ratings`, in Elo), its trim knob times 2^(-Elo / slope) within the knob's
    band, and every edge's log-odds less the ratings' difference plus the applied trims'
    (the slope's Elo). A trim within the band leaves only the cyclic part."""
    count = len(knobs)
    slope = trim_slope(space.spec)
    knob = space.spec["trim"]["knob"]
    logits = [logit_of(p, games) for p in scores]
    ratings = edge_ratings(edges, logits, count)
    wanted = [2.0 ** (-r * ELO / slope) for r in ratings]
    trimmed = trimmed_knobs(space, knobs, wanted)
    applied = [  # in natural log-odds
        slope * math.log2(after[knob] / before[knob]) / ELO
        for before, after in zip(knobs, trimmed, strict=True)
    ]
    predicted = []
    for (a, b), logit in zip(edges, logits, strict=True):
        moved = logit - (ratings[a] - ratings[b]) + (applied[a] - applied[b])
        predicted.append(1.0 / (1.0 + math.exp(-moved)))
    return dict(
        ratings=[float(r * ELO) for r in ratings],
        factors=[after[knob] / before[knob] for before, after in zip(knobs, trimmed, strict=True)],
        knobs=trimmed,
        scores=predicted,
    )


def curl_shortfall(scores: list[float], games: int, spec: dict) -> float:
    """For a cycle (a "cycle" spec, or "edges" with "edges_are_cycle": the edges in order
    round one): the squared shortfall of its curl (the sum of its edges' log-odds, the
    part no rating difference can make) below the band's lower edge times the edges' count;
    0 for a spec of edges, or a curl at or above it."""
    if "cycle" not in spec and not spec.get("edges_are_cycle", False):
        return 0.0
    clipped = [min(max(p, 0.5 / games), 1.0 - 0.5 / games) for p in scores]
    curl = sum(math.log(p / (1.0 - p)) for p in clipped)
    low = spec["band"][0]
    return max(0.0, len(scores) * math.log(low / (1.0 - low)) - curl) ** 2


def band_loss(score: float, games: int, band) -> float:
    """An edge's squared log-odds distance outside the band (the score held off 0 and 1
    by half a game)."""
    logit = logit_of(score, games)
    low, high = (math.log(b / (1.0 - b)) for b in band)
    return max(0.0, low - logit, logit - high) ** 2


# ---------------------------------------------------------------- the arena


class DesignArena:
    """A fixed arena of SLOTS type slots and ARENA_BLOCKS matchups a batch: one player per slot
    (every ship that slot's type), and mixed teams added as needed. `play(card, matchups)`
    writes the card in place and plays the matchups, ARENA_BLOCKS at a time."""

    def __init__(self, preset: str, device: torch.device, block_games: int = BLOCK_GAMES):
        from bnb.controllers.scripted_defaults import MODE_DEFAULTS

        self.game = api.PRESETS[preset]
        self.players = [Player(f"slot {i}", rung=Rung(1.0), ship_type=i) for i in range(SLOTS)]
        # the build card: the P-51D (15 Hz, the fastest rate a design may fire) in every
        # slot, so the bullet slots fit any design
        base = api.ship_types(device, ship_set=self.game.ship_set)
        card = ShipTypes(**{k: v[:1].expand(SLOTS).clone() for k, v in vars(base).items()})
        self.arena = Arena(
            self.players,
            self.game,
            device,
            ArenaConfig(blocks=ARENA_BLOCKS, block_environments=block_games),
            card=card,
            doctrine=MODE_DEFAULTS.sets[preset],
        )
        self.device = device

    def set_doctrine(self, slot: int, parameters) -> None:
        """The scripted parameters a slot's player flies with (its type's doctrine; None: the
        preset's set), used from the next batch on (written in place: no recompile)."""
        self.arena.players[slot] = dataclasses.replace(
            self.arena.players[slot], parameters=parameters
        )

    def player(self, lineup: tuple[int, ...]) -> int:
        """The index of a player whose team flies this lineup (added the first time)."""
        for index, one in enumerate(self.arena.players):
            if one.ship_type == lineup:
                return index
        one = Player(f"lineup {lineup}", rung=Rung(1.0), ship_type=lineup)
        self.arena.players.append(one)
        self.arena.team_types.append(lineup)
        return len(self.arena.players) - 1

    def play(self, card: ShipTypes | None, matchups: list[tuple[int, int]]) -> np.ndarray:
        """(len(matchups), 3) each matchup's first player's wins, ties and losses (with this
        card written first; None: the card already written)."""
        if card is not None:
            self.arena.set_card(card)
        out = []
        for start in range(0, len(matchups), ARENA_BLOCKS):
            chunk = list(matchups[start : start + ARENA_BLOCKS])
            real = len(chunk)
            chunk += [(0, 1)] * (ARENA_BLOCKS - real)  # padding, not read
            counts = self.arena.play(chunk).numpy()
            out.append(counts[:real])
        return np.concatenate(out)


def padded(card: ShipTypes) -> ShipTypes:
    """A card of up to SLOTS types, padded to SLOTS with copies of its first type, on the
    arena's float32."""
    count = card.max_energy.shape[0]
    if count > SLOTS:
        raise ValueError(f"{count} designs for {SLOTS} slots")
    index = torch.cat([torch.arange(count), torch.zeros(SLOTS - count, dtype=torch.long)])
    return ShipTypes(**{k: v.float()[index] for k, v in vars(card).items()})


# ---------------------------------------------------------------- CMA-ES


class CMA:
    """The (mu/mu_w, lambda) CMA-ES (Hansen's tutorial, 2016), with its state as JSON."""

    def __init__(self, size: int, population: int, sigma: float, seed: int, state=None):
        n = size
        self.n, self.population = n, population
        self.mu = population // 2
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / weights.sum()
        self.mueff = 1.0 / (self.weights**2).sum()
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff)
        )
        self.damps = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chi = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        if state is None:
            self.mean, self.sigma = np.zeros(n), sigma
            self.cov, self.pc, self.ps = np.eye(n), np.zeros(n), np.zeros(n)
            self.generation = 0
        else:
            self.mean, self.sigma = np.array(state["mean"]), state["sigma"]
            self.cov = np.array(state["cov"])
            self.pc, self.ps = np.array(state["pc"]), np.array(state["ps"])
            self.generation = state["generation"]
        self.rng = np.random.default_rng(seed + 7919 * self.generation)

    def state(self) -> dict:
        """The optimiser's state, as JSON (`CMA(..., state=...)` resumes it)."""
        return dict(
            mean=self.mean.tolist(),
            sigma=self.sigma,
            cov=self.cov.tolist(),
            pc=self.pc.tolist(),
            ps=self.ps.tolist(),
            generation=self.generation,
        )

    def ask(self) -> list[np.ndarray]:
        """The mean, then population - 1 samples."""
        values, vectors = np.linalg.eigh(self.cov)
        root = vectors @ np.diag(np.sqrt(np.maximum(values, 1e-20)))
        samples = [self.mean.copy()]
        for _ in range(self.population - 1):
            samples.append(self.mean + self.sigma * root @ self.rng.standard_normal(self.n))
        return samples

    def tell(self, samples: list[np.ndarray], losses: list[float]) -> None:
        """Update the mean, the step size and the covariance from the samples' losses."""
        order = np.argsort(losses)[: self.mu]
        chosen = np.array([samples[i] for i in order])
        old = self.mean
        self.mean = self.weights @ chosen
        values, vectors = np.linalg.eigh(self.cov)
        inverse_root = vectors @ np.diag(1 / np.sqrt(np.maximum(values, 1e-20))) @ vectors.T
        step = (self.mean - old) / self.sigma
        self.ps = (1 - self.cs) * self.ps + math.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * inverse_root @ step
        hsig = (
            np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1)))
            < (1.4 + 2 / (self.n + 1)) * self.chi
        )
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        ) * step
        spread = (chosen - old) / self.sigma
        self.cov = (
            (1 - self.c1 - self.cmu) * self.cov
            + self.c1
            * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.cov)
            + self.cmu * spread.T @ np.diag(self.weights) @ spread
        )
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chi - 1))
        self.generation += 1


# ---------------------------------------------------------------- the search


def setup(spec: dict):
    """The search space of a spec: the set's handles, the envelope and the rules."""
    handles = set_handles()
    game = api.PRESETS[spec.get("preset", "real_scale_deathmatch")]
    rules = api.rules_of(game)
    envelope = Envelope.of(handles)
    return Space(spec, handles, envelope, rules)


def spec_edges(spec: dict, names: list[str]) -> tuple[list[tuple[int, int]], bool]:
    """The edges played (index pairs), and whether they are a cycle whose direction the
    search picks: "edges" lists [winner, loser] pairs; "cycle" lists names in order, played
    each against the next (and the last against the first), the direction chosen at the
    first generation as the one whose mean is nearer the band."""
    if "cycle" in spec:
        order = [names.index(name) for name in spec["cycle"]]
        return [(order[k], order[(k + 1) % len(order)]) for k in range(len(order))], True
    return [(names.index(a), names.index(b)) for a, b in spec["edges"]], False


def edge_matchups(designs: int, edges: list[tuple[int, int]], offset: int):
    """The matchups of one candidate's edges (its designs in slots offset...)."""
    return [(offset + a, offset + b) for a, b in edges]


def cycle_orientation(scores: list[float], games: int, spec: dict) -> int:
    """The way round a cycle runs (1: as listed, -1: backward), from the first mean's
    scores: with a trim, the sign of the curl (the trim takes the rest away); else the way
    whose edges are nearer the band."""
    if "trim" in spec:
        return 1 if sum(logit_of(p, games) for p in scores) >= 0.0 else -1
    forward = sum(band_loss(p, games, spec["band"]) for p in scores)
    backward = sum(band_loss(1 - p, games, spec["band"]) for p in scores)
    return 1 if forward <= backward else -1


def scored(
    space: Space,
    knobs: list[dict[str, float]],
    edges: list[tuple[int, int]],
    scores: list[float],
    games: int,
    doctrine: list[dict[str, float]] | None = None,
) -> dict:
    """A candidate's loss and its parts from its edges' scores (oriented: each edge's first
    design the one the spec says wins), with the trim's prediction when the spec has one."""
    spec = space.spec
    out = dict(scores=scores)
    judged, realism_knobs = scores, knobs
    if "trim" in spec:
        trim = predicted_trim(space, knobs, edges, scores, games)
        judged, realism_knobs = trim["scores"], trim["knobs"]
        out |= dict(
            trimmed_scores=trim["scores"],
            trim=trim["factors"],
            ratings=trim["ratings"],
            trimmed_knobs=trim["knobs"],
        )
    edge_loss = sum(band_loss(p, games, spec["band"]) for p in judged)
    curl_loss = spec.get("curl_weight", 0.0) * curl_shortfall(scores, games, spec)
    penalty = space.penalty(realism_knobs)
    doctrine = doctrine if doctrine is not None else [{} for _ in knobs]
    roles = space.roles(knobs, doctrine)
    loss = edge_loss + curl_loss + penalty["realism"] + penalty["distinct"] + roles
    return out | dict(edge_loss=edge_loss, curl_loss=curl_loss, roles=roles, loss=loss, **penalty)


def search(args) -> None:
    """The `search` command (the module docstring)."""
    spec = json.loads(pathlib.Path(args.spec).read_text())
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    space = setup(spec)
    names = [design["name"] for design in space.designs]
    edges, cycle = spec_edges(spec, names)
    population = spec["population"]
    count = len(names)
    if population * count > SLOTS:
        raise SystemExit(f"{population} candidates of {count} designs exceed {SLOTS} slots")
    state_path, log_path = out / "state.json", out / "search.jsonl"
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    cma = CMA(space.size, population, spec.get("sigma", 0.5), spec.get("seed", 1), state)
    orientation = None if state is None else state.get("orientation")
    device = torch.device("cuda")
    games = spec["games_per_edge"]
    if games % BLOCK_GAMES:
        raise SystemExit(f"games_per_edge in whole blocks of {BLOCK_GAMES}")
    started, deadline = time.time(), time.time() + 3600.0 * args.hours
    first_generation = cma.generation
    with bench_lock():
        arena = DesignArena(spec.get("preset", "real_scale_deathmatch"), device)
        while cma.generation < spec["generations"]:
            # more games once the step has halved: the late candidates differ by less than
            # 64 games resolve
            if cma.sigma < spec.get("sigma", 0.5) / 2:
                games = spec.get("games_late", games)
            samples = cma.ask()
            knobs = [space.knobs(x) for x in samples]
            doctrines = [space.doctrine(x) for x in samples]
            designs = [h for one in knobs for h in space.handles_of(one)]
            bases = [d["base"] for d in space.designs] * population
            card = padded(design_cards(space.rules, bases, designs))
            for candidate, doctrine in enumerate(doctrines):  # each design flies its own
                for index, values in enumerate(doctrine):
                    arena.set_doctrine(candidate * count + index, space.parameters(values))
            matchups = []
            for candidate in range(population):
                block = edge_matchups(count, edges, candidate * count)
                matchups += block * (games // BLOCK_GAMES)
            torch.manual_seed(spec.get("seed", 1) * 100_003 + cma.generation)
            counts = arena.play(card, matchups)
            per = len(edges) * (games // BLOCK_GAMES)
            played = []
            for candidate in range(population):
                mine = counts[candidate * per : (candidate + 1) * per]
                scores = []
                for e in range(len(edges)):
                    total = mine[e :: len(edges)].sum(0)
                    scores.append(float((total[0] + 0.5 * total[1]) / total.sum()))
                played.append(scores)
            if cycle and orientation is None:
                orientation = cycle_orientation(played[0], games, spec)
                print(f"the cycle runs {'forward' if orientation == 1 else 'backward'}", flush=True)
            oriented = edges if orientation != -1 else [(b, a) for a, b in edges]
            losses, rows = [], []
            for candidate in range(population):
                scores = played[candidate]
                if cycle and orientation == -1:
                    scores = [1.0 - p for p in scores]
                row = scored(space, knobs[candidate], oriented, scores, games, doctrines[candidate])
                losses.append(row["loss"])
                rows.append(
                    dict(
                        generation=cma.generation,
                        index=candidate,
                        mean=candidate == 0,
                        x=samples[candidate].tolist(),
                        knobs=knobs[candidate],
                        doctrine=doctrines[candidate],
                        sigma=cma.sigma,
                        games=games,
                        **row,
                    )
                )
            cma.tell(samples, losses)
            with open(log_path, "a") as log:
                for row in rows:
                    log.write(json.dumps(row) + "\n")
            state_path.write_text(json.dumps(cma.state() | dict(orientation=orientation)) + "\n")
            done = cma.generation - first_generation
            per_generation = (time.time() - started) / done
            left = per_generation * (spec["generations"] - cma.generation)
            mean = rows[0]
            print(
                f"generation {cma.generation - 1}: mean loss {mean['loss']:.3f} (edges "
                + ", ".join(f"{s:.2f}" for s in mean["scores"])
                + (
                    "; trimmed " + ", ".join(f"{s:.2f}" for s in mean["trimmed_scores"])
                    if "trimmed_scores" in mean
                    else ""
                )
                + f"; realism {max(mean['distances']):.2f}), best {min(losses):.3f}, "
                f"sigma {cma.sigma:.3f}; {per_generation:.0f} s a generation; "
                f"ETA {clock(left)} (worst {clock(1.3 * left + 120)})",
                flush=True,
            )
            if time.time() > deadline:
                raise SystemExit(f"past the {args.hours} h bound: rerun to resume")


def best_of(log_path: pathlib.Path, last: int = 5) -> dict:
    """The search's best mean row of its last `last` generations (each mean is played
    fresh every generation, so the means are the least selected-on rows)."""
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    final = max(row["generation"] for row in rows)
    means = [row for row in rows if row["mean"] and row["generation"] > final - last]
    return min(means, key=lambda row: row["loss"])


def label_of(row: dict) -> str:
    """A candidate's label: GENERATION:INDEX for a search row, trim:ROUND for a trim's."""
    return row.get("label", f"{row['generation']}:{row['index']}")


def candidate_row(out: pathlib.Path, candidate: str | None) -> dict:
    """A candidate by label: GENERATION:INDEX of the search, "trim" (the trim round of the
    smallest spread, of the last trim's source and card) or trim:ROUND; by default the
    search's best recent mean (`best_of`). Every row has the designs' knobs and doctrine."""
    if candidate and candidate.startswith("trim"):
        path = out / "trim.jsonl"
        if not path.exists():
            raise SystemExit(f"{path}: no trim has played")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        last = rows[-1]
        rows = [r for r in rows if (r["source"], r["card"]) == (last["source"], last["card"])]
        if candidate == "trim":
            return min(rows, key=lambda row: row["spread"])
        for row in rows:
            if row["label"] == candidate:
                return row
        raise SystemExit(f"{candidate}: no such trim round in {path}")
    log_path = out / "search.jsonl"
    if not candidate:
        row = best_of(log_path)
    else:
        generation, index = (int(v) for v in candidate.split(":"))
        for line in log_path.read_text().splitlines():
            row = json.loads(line)
            if row["generation"] == generation and row["index"] == index:
                break
        else:
            raise SystemExit(f"{candidate}: no such candidate in {log_path}")
    return row | dict(doctrine=row.get("doctrine", [{} for _ in row["knobs"]]))


def card_for(space: Space, knobs: list[dict[str, float]], card_set: str | None):
    """(the card, its label) for designs at these knobs: the instant estimate on their bases
    ("estimate"); or with a ship set holding them by name, its solved card when the set's
    handles are the knobs' own ("SET": exact), else that card estimated at the knobs'
    handles ("SET+estimate": exact in the flight, a closed-form change of the rest)."""
    handles = space.handles_of(knobs)
    if not card_set:
        bases = [d["base"] for d in space.designs]
        return padded(design_cards(space.rules, bases, handles)), "estimate"
    from bnb.game import ships as ship_files

    names = [d["name"] for d in space.designs]
    found = {ship.name: ship for ship in ship_files.load(card_set)}
    missing = [name for name in names if name not in found]
    if missing:
        raise SystemExit(f"{card_set} holds no {', '.join(missing)}")
    same = all(
        math.isclose(found[name].handles[key], value, rel_tol=1e-9, abs_tol=1e-12)
        for name, one in zip(names, handles, strict=True)
        for key, value in one.items()
    )
    if same:
        exact = ShipTypes(
            **{
                k: torch.tensor([found[name].card[k] for name in names], dtype=torch.float64)
                for k in FIELDS
            }
        )
        return padded(exact), card_set
    keys = [found[name].key for name in names]
    return padded(design_cards(space.rules, keys, handles, ship_set=card_set)), (
        f"{card_set}+estimate"
    )


def set_doctrines(arena: DesignArena, space: Space, doctrine: list[dict], preset: str) -> None:
    """Slot i flies design i's doctrine in this preset."""
    for index, values in enumerate(doctrine):
        arena.set_doctrine(index, space.parameters(values, preset))


def play_pairs(arena: DesignArena, count: int, games: int, seed: int) -> dict:
    """Every pair of the first `count` slots, `games` games each: {(a, b): (wins, ties,
    losses) of a}."""
    blocks = games // BLOCK_GAMES
    pairs = list(itertools.combinations(range(count), 2))
    torch.manual_seed(seed)
    counts = arena.play(None, [(a, b) for a, b in pairs for _ in range(blocks)])
    return {pair: counts[k * blocks : (k + 1) * blocks].sum(0) for k, pair in enumerate(pairs)}


def score_of(total) -> float:
    """A first player's score from its (wins, ties, losses)."""
    return float((total[0] + 0.5 * total[1]) / total.sum())


def bradley_terry(score: np.ndarray, games: np.ndarray) -> np.ndarray:
    """The Bradley-Terry ratings (Elo, mean 0) of a score matrix (row's score against
    column) and its games, by Newton."""
    rating = np.zeros(score.shape[0])
    for _ in range(200):
        expected = 1.0 / (1.0 + np.exp(rating[None, :] - rating[:, None]))
        gradient = (games * (score - expected)).sum(1)
        hessian = (games * expected * (1 - expected)).sum(1) + 1e-9
        rating += gradient / hessian
        rating -= rating.mean()
    return rating * ELO


def ratings_of(totals: dict, count: int) -> np.ndarray:
    """The Bradley-Terry ratings (Elo) of a round robin's totals (`play_pairs`)."""
    score = np.full((count, count), 0.5)
    games = np.zeros((count, count))
    for (a, b), total in totals.items():
        score[a, b], score[b, a] = score_of(total), 1.0 - score_of(total)
        games[a, b] = games[b, a] = total.sum()
    return bradley_terry(score, games)


def check(args) -> None:
    """The `check` command (the module docstring)."""
    spec = json.loads(pathlib.Path(args.spec).read_text())
    out = pathlib.Path(args.out)
    space = setup(spec)
    names = [design["name"] for design in space.designs]
    row = candidate_row(out, args.candidate)
    card, card_label = card_for(space, row["knobs"], args.card)
    presets = [args.preset or space.preset]
    if not args.preset:
        presets += [p for p in args.presets.split(",") if p and p not in presets]
    device = torch.device("cuda")
    count = len(names)
    edges, cycle = spec_edges(spec, names)
    if cycle and json.loads((out / "state.json").read_text()).get("orientation") == -1:
        edges = [(b, a) for a, b in edges]
    beats = {(names[a], names[b]) for a, b in edges}
    with bench_lock():
        for preset in presets:
            path = out / f"check-{preset}.jsonl"
            arena = DesignArena(preset, device)
            arena.arena.set_card(card)
            set_doctrines(arena, space, row["doctrine"], preset)
            blocks = args.games // BLOCK_GAMES
            pairs = list(itertools.combinations(range(count), 2))
            matchups = [(a, b) for a, b in pairs for _ in range(blocks)]
            mixed = []
            size = arena.game.ships_per_team
            for a, b in pairs:
                for d in range(count):
                    if (names[d], names[a]) in beats and (names[d], names[b]) in beats:
                        lineups = [
                            tuple((a, b)[k % 2] for k in range(size)),
                            tuple((b, a)[k % 2] for k in range(size)),
                        ]
                        for lineup in lineups:
                            # a mixed team is one player: it flies one doctrine (the first
                            # design's), as the arena's doctrine is one set a team
                            first = arena.player(lineup)
                            arena.set_doctrine(first, space.parameters(row["doctrine"][a], preset))
                            mixed.append((a, b, d, lineup))
                            matchups += [(first, d)] * max(blocks // 2, 1)
            torch.manual_seed(args.seed)
            started = time.time()
            counts = arena.play(None, matchups)
            common = dict(candidate=label_of(row), card=card_label, preset=preset)
            with open(path, "a") as log:
                at = 0
                for a, b in pairs:
                    total = counts[at : at + blocks].sum(0)
                    at += blocks
                    score = score_of(total)
                    record = dict(
                        kind="pair",
                        first=names[a],
                        second=names[b],
                        games=int(total.sum()),
                        score=score,
                        stderr=math.sqrt(score * (1 - score) / total.sum()),
                        wins=int(total[0]),
                        ties=int(total[1]),
                        losses=int(total[2]),
                        **common,
                    )
                    log.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
                for a, b, d, lineup in mixed:
                    n = max(blocks // 2, 1)
                    total = counts[at : at + n].sum(0)
                    at += n
                    score = score_of(total)
                    record = dict(
                        kind="mixed",
                        team=[names[k] for k in lineup],
                        against=names[d],
                        games=int(total.sum()),
                        score=score,
                        stderr=math.sqrt(score * (1 - score) / total.sum()),
                        **common,
                    )
                    log.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
            print(f"{preset}: checked in {time.time() - started:.0f} s", flush=True)
            del arena
            torch.cuda.empty_cache()


def trim(args) -> None:
    """The `trim` command (the module docstring)."""
    spec = json.loads(pathlib.Path(args.spec).read_text())
    if "trim" not in spec:
        raise SystemExit("the spec has no trim")
    out = pathlib.Path(args.out)
    space = setup(spec)
    names = [design["name"] for design in space.designs]
    count = len(names)
    knob = spec["trim"]["knob"]
    source = candidate_row(out, args.candidate)
    base_knobs = source.get("untrimmed_knobs", source["knobs"])
    path = out / "trim.jsonl"
    done = []
    if path.exists():
        done = [json.loads(line) for line in path.read_text().splitlines()]
        done = [
            r
            for r in done
            if (r["source"], r["card"].split("+")[0]) == (label_of(source), args.card or "estimate")
        ]
    spec_slope = trim_slope(spec)
    if done:
        factors, slope = done[-1]["next"], done[-1]["next_slope"]
    else:
        factors, slope = source.get("trim", [1.0] * count), spec_slope
    started, deadline = time.time(), time.time() + 3600.0 * args.hours
    first_round = len(done)
    with bench_lock():
        arena = DesignArena(space.preset, torch.device("cuda"))
        set_doctrines(arena, space, source["doctrine"], space.preset)
        for number in range(len(done), args.rounds):
            if done and max(abs(e) for e in done[-1]["elo"]) <= args.target:
                break
            knobs = trimmed_knobs(space, base_knobs, factors)
            factors = [
                after[knob] / before[knob] for before, after in zip(base_knobs, knobs, strict=True)
            ]
            card, card_label = card_for(space, knobs, args.card)
            arena.arena.set_card(card)
            totals = play_pairs(arena, count, args.games, 6_000_000 + number)
            elo = ratings_of(totals, count)
            if done:  # the slope from the rounds' own changes (a secant, pooled)
                previous = done[-1]
                moved = [
                    math.log2(f / g) for f, g in zip(factors, previous["factors"], strict=True)
                ]
                change = [e - p for e, p in zip(elo, previous["elo"], strict=True)]
                spread = sum(m * m for m in moved)
                if spread > 1e-4:
                    measured = sum(m * c for m, c in zip(moved, change, strict=True)) / spread
                    slope = min(max(measured, 0.5 * spec_slope), 2.0 * spec_slope)
            following = [f * 2.0 ** (-e / slope) for f, e in zip(factors, elo, strict=True)]
            row = dict(
                label=f"trim:{number}",
                source=label_of(source),
                card=card_label,
                round=number,
                knob=knob,
                factors=factors,
                knobs=knobs,
                untrimmed_knobs=base_knobs,
                doctrine=source["doctrine"],
                elo=[float(e) for e in elo],
                spread=float(np.abs(elo).max()),
                scores={f"{names[a]} > {names[b]}": score_of(t) for (a, b), t in totals.items()},
                games=args.games,
                slope=slope,
                next=following,
                next_slope=slope,
            )
            with open(path, "a") as log:
                log.write(json.dumps(row) + "\n")
            done.append(row)
            factors = following
            per_round = (time.time() - started) / (number + 1 - first_round)
            print(
                f"trim round {number}: Elo "
                + ", ".join(f"{n} {e:+.0f}" for n, e in zip(names, elo, strict=True))
                + f" (spread {row['spread']:.0f}; {knob} x "
                + ", ".join(f"{f:.3f}" for f in row["factors"])
                + f"; slope {slope:.0f}); next x "
                + ", ".join(f"{f:.3f}" for f in following)
                + f"; ETA {clock(per_round * (args.rounds - number - 1))}",
                flush=True,
            )
            if time.time() > deadline:
                raise SystemExit(f"past the {args.hours} h bound: rerun to resume")
    best = min(done, key=lambda r: r["spread"])
    print(f"the trim: {best['label']} (spread {best['spread']:.0f} Elo)", flush=True)


def _solve_design(arguments):
    """One design's exact card in a worker process (`real_scale.solve_type`)."""
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    from bnb.contracts import GameConfig
    from bnb.game import real_scale
    from bnb.game.rules import rules_of

    key, name, handles, notes, rate = arguments
    rules = rules_of(GameConfig(1, 1, 7800.0, ship_set=real_scale.SHIP_SET))
    return real_scale.solve_type(key, name, handles, notes, rate, rules, log=lambda line: None)


def export(args) -> None:
    """Solve a candidate's designs exactly and write them as a ship set (rules from the
    real-scale set's), one file per design, in the spec's order, with the designs' doctrine
    (the preset's set's values it changes, by design name) in the set's doctrine.json."""
    import concurrent.futures

    from bnb.game import real_scale
    from bnb.game import ships as ship_files
    from bnb.game.ship_design import climb_rate_of

    spec = json.loads(pathlib.Path(args.spec).read_text())
    out = pathlib.Path(args.out)
    space = setup(spec)
    row = candidate_row(out, args.candidate)
    handles = space.handles_of(row["knobs"])
    jobs = []
    for design, one in zip(space.designs, handles, strict=True):
        key = design["name"].lower()
        base = space.handles[design["base"]]
        notes = {
            name: f"designed ({out}, candidate {label_of(row)}): "
            + (
                f"{value:.6g}, the base {design['base']}'s {base[name]:.6g}"
                if not math.isclose(value, base[name], rel_tol=1e-9)
                else f"the base {design['base']}'s"
            )
            for name, value in one.items()
        }
        jobs.append((key, design["name"], one, notes, climb_rate_of(one)))
    ship_files.write_rules(ship_files.rules(real_scale.SHIP_SET), args.ship_set)
    started = time.time()
    solved = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        for job, result in zip(jobs, pool.map(_solve_design, jobs), strict=True):
            solved[job[0]] = result
            print(
                f"{job[1]}: worst flown error {result.worst:.2e}, converged "
                f"{result.converged} ({(time.time() - started) / 60:.1f} min)",
                flush=True,
            )
    ship_files.write([solved[job[0]].ship for job in jobs], args.ship_set)
    doctrine = dict(
        preset=space.preset,
        candidate=label_of(row),
        designs={
            d["name"]: values for d, values in zip(space.designs, row["doctrine"], strict=True)
        },
    )
    directory = ship_files.directory_of(args.ship_set)
    (directory / "doctrine.json").write_text(json.dumps(doctrine, indent=1) + "\n")
    print(f"wrote {directory}", flush=True)


def curl_of(scores: list[float], games: int) -> float:
    """A cycle's curl: the sum of its oriented edges' log-odds."""
    return sum(logit_of(p, games) for p in scores)


ROBUST_SIGMA = 0.1  # the acceptance test's perturbation
ROBUST_CURL = 3 * math.log(0.55 / 0.45)  # its median draw's least curl: edges of 0.55 on average


def robust(args) -> None:
    """Does the cycle hold when the agent changes? For each of `--seeds` draws, every design
    flies its own perturbation of its own doctrine (`bench.scripted_search.perturb` at
    `--sigma`, a different draw per design; its searched dimensions perturbed with the rest,
    the relative tactics it does not search held), and every pair plays `--games` games; one
    row per draw and pair to DIR/robust-<preset>.jsonl (a draw already recorded for this
    candidate, card and sigma is not replayed). The verdict (DIR/robust-summary-<preset>.json):
    per draw the oriented edges and, for a cycle, the curl; the acceptance test (at sigma
    ROBUST_SIGMA) passes when every draw's curl is positive and the median draw's at least
    ROBUST_CURL (for a spec of edges: when four in five of the draws' edges keep their
    direction)."""
    from bench.scripted_search import perturb
    from bnb.controllers.scripted_parameters import RELATIVE_TACTICS

    spec = json.loads(pathlib.Path(args.spec).read_text())
    out = pathlib.Path(args.out)
    space = setup(spec)
    names = [design["name"] for design in space.designs]
    row = candidate_row(out, args.candidate)
    preset = args.preset or space.preset
    card, card_label = card_for(space, row["knobs"], args.card)
    count = len(names)
    path = out / f"robust-{preset}.jsonl"
    recorded = []
    if path.exists():
        recorded = [json.loads(line) for line in path.read_text().splitlines()]
    key = (label_of(row), card_label, args.sigma)

    def mine(r):
        return (r.get("candidate"), r["card"], r["sigma"]) == key

    have = {r["draw"] for r in recorded if mine(r)}
    with bench_lock():
        arena = DesignArena(preset, torch.device("cuda"))
        arena.arena.set_card(card)
        for draw in range(args.seeds):
            if draw in have:
                continue
            generator = torch.Generator().manual_seed(9000 + draw)
            for slot in range(count):
                own = space.parameters(row["doctrine"][slot], preset)
                searched = {d.partition(".")[0] for d in row["doctrine"][slot]}
                frozen = tuple(name for name in RELATIVE_TACTICS if name not in searched)
                arena.set_doctrine(slot, perturb(own, args.sigma, generator, frozen=frozen))
            totals = play_pairs(arena, count, args.games, 7_000_000 + draw)
            with open(path, "a") as log:
                for (a, b), total in totals.items():
                    record = dict(
                        draw=draw,
                        sigma=args.sigma,
                        first=names[a],
                        second=names[b],
                        games=int(total.sum()),
                        score=score_of(total),
                        candidate=label_of(row),
                        card=card_label,
                    )
                    recorded.append(record)
                    log.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
    verdict = robust_verdict(spec, out, [r for r in recorded if mine(r)], names)
    verdict |= dict(candidate=label_of(row), card=card_label, preset=preset)
    (out / f"robust-summary-{preset}.json").write_text(json.dumps(verdict, indent=1) + "\n")
    print(json.dumps(verdict, indent=1))


def robust_verdict(spec: dict, out: pathlib.Path, rows: list[dict], names: list[str]) -> dict:
    """The robustness verdict of one candidate's draws (the `robust` docstring)."""
    edges, cycle = spec_edges(spec, names)
    state = out / "state.json"
    if cycle and state.exists() and json.loads(state.read_text()).get("orientation") == -1:
        edges = [(b, a) for a, b in edges]
    sigma = rows[0]["sigma"] if rows else None
    draws = []
    for draw in sorted({r["draw"] for r in rows}):
        score = {}
        games = 1
        for r in rows:
            if r["draw"] == draw:
                score[(r["first"], r["second"])] = r["score"]
                score[(r["second"], r["first"])] = 1.0 - r["score"]
                games = r["games"]
        oriented = [score[(names[a], names[b])] for a, b in edges]
        one = dict(draw=draw, edges=oriented)
        if cycle:
            one["curl"] = curl_of(oriented, games)
        draws.append(one)
    verdict = dict(sigma=sigma, draws=draws, edges=[f"{names[a]} > {names[b]}" for a, b in edges])
    if not draws:
        return verdict | dict(pass_=None)
    if cycle:
        curls = sorted(d["curl"] for d in draws)
        median = float(np.median(curls))
        verdict |= dict(
            curls_positive=all(c > 0 for c in curls),
            median_curl=median,
            least_median_curl=ROBUST_CURL,
        )
        passed = verdict["curls_positive"] and median >= ROBUST_CURL
    else:
        kept = [p > 0.5 for d in draws for p in d["edges"]]
        verdict["kept_share"] = sum(kept) / len(kept)
        passed = verdict["kept_share"] >= 0.8
    verdict["gate"] = sigma == ROBUST_SIGMA
    verdict["pass"] = bool(passed) if verdict["gate"] else None
    return verdict


PALEY_SIGNS = (1, 1, -1)  # the quadratic character of 1, 2, 3 mod 7


def mixture(weights: list[float], values: list[float], additive: bool) -> float:
    """The weighted mixture of a corner's values: linear for an additive doctrine dimension,
    in logs otherwise."""
    if additive:
        return sum(w * v for w, v in zip(weights, values, strict=True))
    return math.exp(sum(w * math.log(v) for w, v in zip(weights, values, strict=True)))


def compose(args) -> None:
    """The seven-type spec from three cycle searches (one per plane): type i sits at angle
    chi(k) 2 pi k i / 7 in plane k, its knobs and doctrine dimensions of that plane's group
    the mixture of the cycle's three corners at that angle (weights (1 + 2 cos(angle - 2 pi c
    / 3)) / 3 for the corner c, in the order each beats the next; in logs, linear for an
    additive dimension), the fire rate the nearest allowed. Each corner is the plane's trim
    (`--candidate trim`) when it has one, else its best recent mean. Every other knob is the
    common base's (the three planes must share it). Writes the spec (all three groups'
    knobs and doctrine searched, the first plane's trim and axes)."""
    planes = []
    for spec_path, out in zip(args.specs.split(","), args.outs.split(","), strict=True):
        spec = json.loads(pathlib.Path(spec_path).read_text())
        out = pathlib.Path(out)
        names = [design["name"] for design in spec["designs"]]
        edges, _ = spec_edges(spec, names)
        state = json.loads((out / "state.json").read_text())
        order = [a for a, _ in edges]
        if state.get("orientation") == -1:
            order = order[::-1]
        row = candidate_row(out, "trim" if (out / "trim.jsonl").exists() else None)
        corners = [row["knobs"][index] for index in order]
        doctrine = [row["doctrine"][index] for index in order]
        group = spec["designs"][0]["knobs"]
        dimensions = spec["designs"][0].get("doctrine", [])
        planes.append((group, corners, dimensions, doctrine, spec["designs"][0]["base"]))
    bases = {plane[4] for plane in planes}
    if len(bases) != 1:
        raise SystemExit(f"the planes' bases differ ({', '.join(sorted(bases))}): no common base")
    base = planes[0][4]
    designs = []
    names = args.names.split(",")
    for i in range(7):
        start, doctrine_start = {}, {}
        for k, (group, corners, dimensions, doctrine, _) in enumerate(planes, 1):
            angle = PALEY_SIGNS[k - 1] * 2 * math.pi * k * i / 7
            weights = [(1 + 2 * math.cos(angle - 2 * math.pi * c / 3)) / 3 for c in range(3)]
            for name in group:
                value = mixture(weights, [c[name] for c in corners], False)
                if name == "fire_rate":
                    value = min(FIRE_RATES, key=lambda r: abs(math.log(r / value)))
                start[name] = value
            for dimension in dimensions:
                values = [c[dimension] for c in doctrine]
                doctrine_start[dimension] = mixture(weights, values, is_additive(dimension))
        knobs = [name for group, *_ in planes for name in group]
        dimensions = [d for _, _, plane_dimensions, _, _ in planes for d in plane_dimensions]
        designs.append(
            dict(
                name=names[i],
                base=base,
                knobs=knobs,
                start=start,
                doctrine=dimensions,
                doctrine_start=doctrine_start,
            )
        )
    edges = [[names[i], names[(i + d) % 7]] for i in range(7) for d in (1, 2, 4)]
    template = json.loads(pathlib.Path(args.specs.split(",")[0]).read_text())
    spec = {
        key: template[key]
        for key in (
            "preset",
            "band",
            "realism_radius",
            "realism_weight",
            "distinct",
            "distinct_weight",
            "trim",
        )
        if key in template
    }
    spec |= dict(designs=designs, edges=edges, games_per_edge=64, population=5, generations=40)
    spec |= dict(sigma=0.3, seed=707)
    pathlib.Path(args.write).write_text(json.dumps(spec, indent=1) + "\n")
    print(json.dumps(spec, indent=1))


def nash(payoff: np.ndarray, iterations: int = 20000) -> np.ndarray:
    """An approximate Nash equilibrium of the symmetric zero-sum game with this payoff
    (row's score less a half): the average strategy of regret matching in self-play."""
    count = payoff.shape[0]
    regret, total = np.zeros(count), np.zeros(count)
    for _ in range(iterations):
        positive = np.maximum(regret, 0.0)
        strategy = positive / positive.sum() if positive.sum() > 0 else np.full(count, 1 / count)
        total += strategy
        values = payoff @ strategy
        regret += values - strategy @ values
    return total / total.sum()


def verdict_of(spec: dict, out: pathlib.Path, rows: list[dict], preset: str) -> dict:
    """The verdict of one candidate's check rows on one card (the `summary` docstring)."""
    names = [design["name"] for design in spec["designs"]]
    count = len(names)
    score = np.full((count, count), 0.5)
    games = np.zeros((count, count))
    for row in rows:
        if row["kind"] != "pair":
            continue
        a, b = names.index(row["first"]), names.index(row["second"])
        score[a, b], score[b, a] = row["score"], 1.0 - row["score"]
        games[a, b] = games[b, a] = row["games"]
    elo = bradley_terry(score, games)
    payoff = score - 0.5
    np.fill_diagonal(payoff, 0.0)
    equilibrium = nash(payoff)
    edges, cycle = spec_edges(spec, names)
    state = out / "state.json"
    if cycle and state.exists() and json.loads(state.read_text()).get("orientation") == -1:
        edges = [(b, a) for a, b in edges]
    edge_scores = {f"{names[a]} > {names[b]}": float(score[a, b]) for a, b in edges}
    low, high = spec["band"]
    mixed = [row for row in rows if row["kind"] == "mixed"]
    card = rows[0]["card"] if rows else None
    gate = preset == spec.get("preset", "real_scale_deathmatch")
    verdict = dict(
        preset=preset,
        gate=gate,
        candidate=rows[0]["candidate"] if rows else None,
        card=card,
        exact=card is not None and card != "estimate" and not card.endswith("+estimate"),
        elo=dict(zip(names, elo.round(1).tolist(), strict=True)),
        elo_spread=float(np.abs(elo).max()),
        edges=edge_scores,
        directions_kept=all(v > 0.5 for v in edge_scores.values()),
        edges_in_band=all(low - 0.02 <= v <= high + 0.02 for v in edge_scores.values()),
        exploitability=float((payoff.mean(1)).max()),
        nash=dict(zip(names, equilibrium.round(3).tolist(), strict=True)),
        nash_full_support=bool((equilibrium > 0.5 / count).all()),
        mixed=[
            dict(team=row["team"], against=row["against"], against_scores=1.0 - row["score"])
            for row in mixed
        ],
        mixed_pass=all(1.0 - row["score"] >= 0.55 for row in mixed),
    )
    if cycle:
        least = min(r["games"] for r in rows if r["kind"] == "pair") if rows else 1
        verdict["curl"] = curl_of(list(edge_scores.values()), least)
    passed = (
        verdict["elo_spread"] <= 35.0
        and verdict["edges_in_band"]
        and verdict["exploitability"] <= 0.02
        and verdict["nash_full_support"]
        and (verdict["mixed_pass"] or not mixed)
    )
    # only the spec's preset is a gate; the others (Frontline) are reported
    verdict["pass"] = bool(passed) if gate else None
    robust_path = out / f"robust-summary-{preset}.json"
    if gate and robust_path.exists():
        robust_verdict_ = json.loads(robust_path.read_text())
        if (robust_verdict_.get("candidate"), robust_verdict_.get("card")) == (
            verdict["candidate"],
            card,
        ):
            verdict["robust_pass"] = robust_verdict_.get("pass")
    if gate:
        verdict["accepted"] = bool(
            passed and verdict["exact"] and verdict.get("robust_pass") is True
        )
    return verdict


def summary(args) -> None:
    """The check's verdict against the targets (docs/explanation/game/ship-design.md): the
    Bradley-Terry ratings, the edges against the band (and whether each keeps its direction),
    the curl of a cycle, the uniform pick's exploitability, the Nash equilibrium, the
    mixed-team test. Of the rows of `--card` (by default the last check's card) and
    `--candidate` (by default the last check's). `pass` is the targets in the spec's preset
    (None in another: Frontline is an evaluation); `accepted` is a pass on an exact card
    whose robustness test (`robust`, same candidate and card) passed."""
    spec = json.loads(pathlib.Path(args.spec).read_text())
    out = pathlib.Path(args.out)
    rows = [
        json.loads(line) for line in (out / f"check-{args.preset}.jsonl").read_text().splitlines()
    ]
    card = args.card or rows[-1]["card"]
    rows = [row for row in rows if row["card"] == card]
    candidate = args.candidate or (rows[-1].get("candidate") if rows else None)
    rows = [row for row in rows if row.get("candidate") == candidate]
    verdict = verdict_of(spec, out, rows, args.preset)
    (out / f"summary-{args.preset}.json").write_text(json.dumps(verdict, indent=1) + "\n")
    print(json.dumps(verdict, indent=1))


def main() -> None:
    """The commands the module describes."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    candidate_help = "GENERATION:INDEX, trim or trim:ROUND (default: the best recent mean)"
    card_help = "a ship set holding the designs by name (default: the instant estimate)"
    one = sub.add_parser("search")
    one.add_argument("--spec", required=True)
    one.add_argument("--out", required=True)
    one.add_argument("--hours", type=float, default=3.0)
    two = sub.add_parser("check")
    two.add_argument("--spec", required=True)
    two.add_argument("--out", required=True)
    two.add_argument("--candidate", help=candidate_help)
    two.add_argument("--games", type=int, default=1024, help="per pair")
    two.add_argument("--preset", help="only this preset (default: the spec's, then --presets)")
    two.add_argument(
        "--presets", default=FRONTLINE, help="evaluated after the spec's (comma-separated)"
    )
    two.add_argument("--card", help=card_help)
    two.add_argument("--seed", type=int, default=5_000_000)
    seven = sub.add_parser("trim")
    seven.add_argument("--spec", required=True)
    seven.add_argument("--out", required=True)
    seven.add_argument("--candidate", help=candidate_help)
    seven.add_argument("--card", help=card_help)
    seven.add_argument("--games", type=int, default=512, help="per pair and round")
    seven.add_argument("--rounds", type=int, default=3)
    seven.add_argument("--target", type=float, default=20.0, help="Elo from the mean")
    seven.add_argument("--hours", type=float, default=1.0)
    four = sub.add_parser("export")
    four.add_argument("--spec", required=True)
    four.add_argument("--out", required=True)
    four.add_argument("--candidate", help=candidate_help)
    four.add_argument("--ship-set", required=True, help="the set to write, e.g. design-flight")
    six = sub.add_parser("robust")
    six.add_argument("--spec", required=True)
    six.add_argument("--out", required=True)
    six.add_argument("--candidate", help=candidate_help)
    six.add_argument("--card", help=card_help)
    six.add_argument("--preset", help="default: the spec's")
    six.add_argument("--sigma", type=float, default=ROBUST_SIGMA, help="the perturbation")
    six.add_argument("--seeds", type=int, default=6, help="draws")
    six.add_argument("--games", type=int, default=256, help="per pair and draw")
    five = sub.add_parser("compose")
    five.add_argument("--specs", required=True, help="the three cycle specs, comma-separated")
    five.add_argument("--outs", required=True, help="their search directories, in order")
    five.add_argument("--names", required=True, help="seven names, comma-separated")
    five.add_argument("--write", required=True, help="the seven-type spec to write")
    three = sub.add_parser("summary")
    three.add_argument("--spec", required=True)
    three.add_argument("--out", required=True)
    three.add_argument("--preset", default="real_scale_deathmatch")
    three.add_argument("--card", help="the rows of this card (default: the last check's)")
    three.add_argument("--candidate", help="the rows of this candidate (default: the last's)")
    args = parser.parse_args()
    {
        "search": search,
        "check": check,
        "summary": summary,
        "export": export,
        "trim": trim,
        "compose": compose,
        "robust": robust,
    }[args.command](args)


if __name__ == "__main__":
    main()
