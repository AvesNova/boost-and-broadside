"""A map of balanced ship cards: a quality-diversity search (CMA-MAE, `ribs`) over the ship's
flight and combat handles, every candidate card's team played against a fixed reference
card's team on the balance harness (`bench.balance_sweep`, `bench.balance_worker`).

    PYTHONPATH=src:. python -m bench.balance_map search --out OUT --generations 60 --hours 12
    PYTHONPATH=src:. python -m bench.balance_map confirm --out OUT --games 512
    PYTHONPATH=src:. python -m bench.balance_map table --out OUT

The search space (`SPACE`): 13 handles, each a coordinate in [0, 1] of its bounds (log or
linear). Every handle not searched is the reference's (v23 Jujak's). A coordinate vector
decodes (`decode`) to handles deterministically, with one repair: the hit damage is clipped
into the band that keeps every time to kill (`time_to_kill`) inside 0.6-1.4 s for the
candidate on itself and against the reference both ways. The handles are solved into cards
(`api.solve_handles`, batched); a card the solver cannot meet is invalid (objective 0, never
played, counted).

The map's axes (`measures`), from the handles alone (no measurement noise), each in [0, 1]:
agility = 0.5 turn + 0.25 low-speed acceleration + 0.25 nose quickness (each its coordinate;
quickness the reversal time's, reversed); toughness x reach = 0.5 shield + 0.5 gun range.

The objective (`objective_of`): 50 minus the worst searched condition's distance from even,
in win-rate-equivalent points: the kill share k converted by the condition's slope b
(`bench.balance_sweep` conversion fit) as 100 sigmoid(b logit k). 50 is even everywhere.

The run's records (under OUT, durable each generation): evaluations.jsonl (every played card,
each condition's kill share, every evaluation appended), generations.jsonl (one line a
generation), scheduler.pkl (the ribs scheduler after the generation's tell), queue/ (the
workers' job files and outputs). A restart resumes from the last finished generation.
"""

import argparse
import dataclasses
import json
import math
import pathlib
import pickle
import time

import numpy

SHIP_SET = "real-scale"
REFERENCE = "jujak"  # the fixed reference card (v23 Jujak) and the base of every candidate


@dataclasses.dataclass(frozen=True)
class Axis:
    """One searched handle: its bounds and whether its coordinate is logarithmic."""

    handle: str
    low: float
    high: float
    log: bool = True

    def value(self, coordinate: float) -> float:
        """The handle's value at a coordinate in [0, 1]."""
        u = min(max(coordinate, 0.0), 1.0)
        if self.log:
            return math.exp(math.log(self.low) + u * (math.log(self.high) - math.log(self.low)))
        return self.low + u * (self.high - self.low)

    def coordinate(self, value: float) -> float:
        """A value's coordinate in [0, 1] (clipped)."""
        if self.log:
            u = (math.log(value) - math.log(self.low)) / (math.log(self.high) - math.log(self.low))
        else:
            u = (value - self.low) / (self.high - self.low)
        return min(max(u, 0.0), 1.0)


# The reference card's numbers the bounds and the time to kill use (v23 Jujak, c4ede6c8).
REFERENCE_SHIELD = 430.25288750249626
REFERENCE_DAMAGE = 19.1877
REFERENCE_STORE = 497563.2
REFERENCE_SHOT_COST = REFERENCE_STORE / 865.2401087452062  # 575 J
# Time to kill (s) of a sustained tail-on core burst: target shield / (shooter damage x
# fire rate x tail aspect); Jujak on Jujak 0.75 s. Every candidate fires at 30 Hz with the
# reference's armour, so time to kill = TIME_PER_SHIELD_PER_DAMAGE x shield / damage.
TIME_PER_SHIELD_PER_DAMAGE = 0.7495 * REFERENCE_DAMAGE / REFERENCE_SHIELD
TIME_TO_KILL_BAND = (0.6, 1.4)
_SHIELD_BAND = tuple(
    REFERENCE_SHIELD * t / (TIME_PER_SHIELD_PER_DAMAGE * REFERENCE_SHIELD / REFERENCE_DAMAGE)
    for t in TIME_TO_KILL_BAND
)  # the reference shooting the candidate
_DAMAGE_BAND = tuple(
    TIME_PER_SHIELD_PER_DAMAGE * REFERENCE_SHIELD / t for t in reversed(TIME_TO_KILL_BAND)
)  # the candidate shooting the reference

# The 13 searched handles (bounds: the real WWII types' spread from
# src/bnb/reference/real-plane-types, leaning to responsiveness and punch, the author's g
# limit 6-9; the combat knobs inside x0.5-2 of v23 and the time-to-kill band).
SPACE = (
    Axis("sustained_turn_rate", 0.50, 0.85),  # real 0.503 (P-47D) - 0.785 (A6M5)
    Axis("nose_reversal_time", 0.25, 0.55),  # real 0.249 (Fw 190 A-8) - 0.623 (Mosquito)
    Axis("low_speed_acceleration", 8.8, 20.0),  # real 8.87 (P-47D) - 18.97 (Bf 109 K-4)
    Axis("climb_power", 200.0, 500.0),  # climb time (s) at the v23 store; real 290-555
    Axis("cruise_speed", 260.0, 360.0, log=False),  # real 261 (A6M5) - 336 (Tempest V)
    Axis("stall_speed", 75.0, 106.0, log=False),  # real 75 (A6M5) - 105.6 (Fw 190, P-38L)
    Axis("g_limit", 6.0, 9.0, log=False),  # the author's 6-9 (real 6-8)
    Axis("max_shield", *_SHIELD_BAND),  # 344-803: the reference's 0.6-1.4 s on it
    Axis("hit_damage", *_DAMAGE_BAND),  # 10.3-24.0: its 0.6-1.4 s on the reference
    Axis("gun_range", 400.0, 700.0),  # real 400 (Bf 109 K-4) - 600 (Spitfire XIV)
    Axis("store", 0.5 * REFERENCE_STORE, 2.0 * REFERENCE_STORE),  # the ceiling (J), a knob
    Axis("shot_cost", 0.5 * REFERENCE_SHOT_COST, 2.0 * REFERENCE_SHOT_COST),  # J a shot
    Axis("dive_acceleration", 38.0, 50.0),  # boost from cruise; Jujak 40.9, Yongma 46.5
)
NAMES = tuple(axis.handle for axis in SPACE)
INDEX = {name: number for number, name in enumerate(NAMES)}
# The handles a candidate's record keeps (the searched ones as the solver takes them).
STORED = (
    "sustained_turn_rate",
    "nose_reversal_time",
    "low_speed_acceleration",
    "climb_time",
    "cruise_speed",
    "stall_speed",
    "g_limit",
    "max_shield",
    "hit_damage",
    "gun_range",
    "ceiling",
    "shots_per_store",
    "dive_acceleration",
)


def reference_ships() -> dict:
    """The set's ship files by key ({key: Ship}: handles, card)."""
    from bnb.game import ships

    return {ship.key: ship for ship in ships.load(SHIP_SET)}


def time_to_kill(shield: float, damage: float) -> float:
    """Seconds of sustained tail-on core fire to kill: the target's shield, the shooter's hit."""
    return TIME_PER_SHIELD_PER_DAMAGE * shield / damage


def damage_band(shield: float) -> tuple[float, float]:
    """The hit damage that keeps the candidate's own mirror and its fire on the reference
    inside TIME_TO_KILL_BAND, for this shield (never empty inside SPACE's shield bounds)."""
    low_mirror = TIME_PER_SHIELD_PER_DAMAGE * shield / TIME_TO_KILL_BAND[1]
    high_mirror = TIME_PER_SHIELD_PER_DAMAGE * shield / TIME_TO_KILL_BAND[0]
    return max(low_mirror, _DAMAGE_BAND[0]), min(high_mirror, _DAMAGE_BAND[1])


def decode(coordinates, base: dict[str, float]) -> dict[str, float]:
    """The handles of a coordinate vector (len(SPACE),): the base handles (the reference's)
    with the searched ones set, the hit damage repaired into `damage_band`, the climb time
    the climb power scaled by the store, the shots per store the store over the shot cost."""
    values = {axis.handle: axis.value(float(c)) for axis, c in zip(SPACE, coordinates, strict=True)}
    low, high = damage_band(values["max_shield"])
    values["hit_damage"] = min(max(values["hit_damage"], low), high)
    handles = dict(base)
    for name in NAMES:
        if name not in ("climb_power", "store", "shot_cost"):
            handles[name] = values[name]
    handles["ceiling"] = values["store"]
    handles["climb_time"] = values["climb_power"] * values["store"] / REFERENCE_STORE
    handles["shots_per_store"] = values["store"] / values["shot_cost"]
    handles["fire_rate"] = 30.0
    return handles


def encode(handles: dict[str, float]) -> numpy.ndarray:
    """A ship's handles as coordinates (the inverse of `decode`, clipped into the bounds)."""
    values = dict(handles)
    values["store"] = handles["ceiling"]
    values["climb_power"] = handles["climb_time"] * REFERENCE_STORE / handles["ceiling"]
    values["shot_cost"] = handles["ceiling"] / handles["shots_per_store"]
    return numpy.array([axis.coordinate(values[axis.handle]) for axis in SPACE])


def measures(coordinates) -> tuple[float, float]:
    """The map's two axes in [0, 1] from the coordinates (the damage repair moves neither):
    agility (0.5 turn, 0.25 low-speed acceleration, 0.25 nose quickness) and toughness x
    reach (0.5 shield, 0.5 gun range)."""
    c = numpy.clip(numpy.asarray(coordinates, dtype=float), 0.0, 1.0)
    agility = (
        0.5 * c[INDEX["sustained_turn_rate"]]
        + 0.25 * c[INDEX["low_speed_acceleration"]]
        + 0.25 * (1.0 - c[INDEX["nose_reversal_time"]])
    )
    toughness = 0.5 * c[INDEX["max_shield"]] + 0.5 * c[INDEX["gun_range"]]
    return float(agility), float(toughness)


def solve_cards(handle_sets: list[dict[str, float]], processes: int = 8) -> list[dict | str]:
    """The solved cards ({field: value}) of handle sets, or the solver's reason a set fails
    (a string). Solved in batches across processes (`api.solve_handles`, strict=False)."""
    if not handle_sets:
        return []
    import concurrent.futures
    import multiprocessing

    size = max(1, math.ceil(len(handle_sets) / processes))
    parts = [handle_sets[i : i + size] for i in range(0, len(handle_sets), size)]
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(len(parts), mp_context=context) as pool:
        results = list(pool.map(_solve_part, parts))
    return [card for part in results for card in part]


def _solve_part(handle_sets: list[dict[str, float]]) -> list[dict | str]:
    """One process's batch of solves: (B, 1) handles, one card each."""
    import torch

    torch.set_num_threads(1)
    from bnb.game import api
    from bnb.game.handling import NAMES as HANDLE_NAMES
    from bnb.game.handling import Handles
    from bnb.game.handling.errors import Infeasible

    rules = api.rules_of(
        api.PRESETS["real_scale_deathmatch"], types=api.ship_types(ship_set=SHIP_SET)
    )
    handles = Handles(
        **{
            name: torch.tensor([[h[name]] for h in handle_sets], dtype=torch.float64)
            for name in HANDLE_NAMES
        }
    )
    try:
        solution = api.solve_handles(handles, rules, strict=False)
    except Infeasible as error:  # the batch as a whole (a check before any flight)
        if len(handle_sets) == 1:
            return [f"infeasible: {error}"]
        return [card for h in handle_sets for card in _solve_part([h])]
    converged = solution.converged.reshape(len(handle_sets), -1)[:, 0].tolist()
    out = []
    for number, ok in enumerate(converged):
        if not ok:
            reasons = [f for f in solution.failures if f.startswith(f"card {number} ")]
            out.append("no convergence: " + (reasons[0] if reasons else "unknown"))
            continue
        out.append(
            {
                field.name: float(getattr(solution.types, field.name)[number, 0])
                for field in dataclasses.fields(solution.types)
            }
        )
    return out


# --- the evaluation: conditions, the conversion and the objective ----------------------------

SEARCHED = ("deathmatch-2v2", "deathmatch-5v5", "elimination-5v5", "frontline-10v10")
EVERY = (
    "deathmatch-1v1",
    "deathmatch-2v2",
    "deathmatch-5v5",
    "deathmatch-10v10",
    "elimination-5v5",
    "elimination-10v10",
    "frontline-5v5",
    "frontline-10v10",
)
# Kill share -> win-rate slope b by condition (balance-eval's conversion fit,
# docs/rewrite/sections/balance-eval/data/conversion_fit.md: logit(score) = a + b logit(kill
# share)); the intercept a is the Yongma-v-Jujak fit's own offset and is left out. 1v1's fit
# had 3 cards: 1.0. Fit with fog and fields on; the balance settings have fields off.
SLOPES = {
    "deathmatch-1v1": 1.0,
    "deathmatch-2v2": 2.50,
    "deathmatch-5v5": 7.62,
    "deathmatch-10v10": 14.15,
    "elimination-5v5": 2.12,
    "elimination-10v10": 3.00,
    "frontline-5v5": 4.66,
    "frontline-10v10": 6.73,
}
SEARCH_GAMES = 128
RECHECK_GAMES = 384  # a recheck brings a card to 512 games a condition
RECHECK_FLOOR = 38.0  # rechecked: a cell's best with an objective at least this
RECHECKS_PER_GENERATION = 32
BALANCED = 45.0  # inside 45-55 win-rate equivalent in every condition


def win_equivalent(kill_share: float, condition: str) -> float:
    """The kill share as win-rate-equivalent points (0-100)."""
    k = min(max(kill_share, 1e-4), 1.0 - 1e-4)
    return 100.0 / (1.0 + math.exp(-SLOPES[condition] * math.log(k / (1.0 - k))))


def pooled(evaluations: list[dict]) -> dict[str, tuple[float, float, int]]:
    """Each condition's kill share pooled over evaluations (inverse-variance), its standard
    error and its games: {condition: (share, error, games)}."""
    by_condition: dict[str, list[dict]] = {}
    for row in evaluations:
        by_condition.setdefault(row["condition"], []).append(row)
    out = {}
    for condition, rows in by_condition.items():
        weights = [1.0 / max(r["kill_share"][1], 1e-3) ** 2 for r in rows]
        share = sum(w * r["kill_share"][0] for w, r in zip(weights, rows, strict=True))
        out[condition] = (
            share / sum(weights),
            1.0 / math.sqrt(sum(weights)),
            sum(r["games"] for r in rows),
        )
    return out


def objective_of(shares: dict, conditions=SEARCHED) -> dict:
    """The objective (50 minus the worst condition's win-equivalent distance from 50), its
    lower bound (each share moved one standard error away from even), the worst condition
    and the fewest games; None objective when a condition is missing."""
    if any(c not in shares for c in conditions):
        return dict(objective=None, lower=None, worst=None, games=0)
    distance, upper = {}, {}
    for condition in conditions:
        share, error, _ = shares[condition]
        distance[condition] = abs(win_equivalent(share, condition) - 50.0)
        away = share + math.copysign(error, share - 0.5 if share != 0.5 else 1.0)
        upper[condition] = abs(win_equivalent(away, condition) - 50.0)
    worst = max(distance, key=distance.get)
    return dict(
        objective=50.0 - distance[worst],
        lower=50.0 - max(upper.values()),
        worst=worst,
        games=min(shares[c][2] for c in conditions),
        win_equivalent={c: round(win_equivalent(shares[c][0], c), 2) for c in conditions},
    )


def pair_entry(name: str, card: dict, opponent: dict | None = None) -> dict:
    """A pairwise entry (`bench.balance_sweep`): the card's team (A) against the opponent's
    (B; None: the reference), both flying the reference's stored doctrine (one doctrine for
    every card)."""
    a = dict(ship=REFERENCE, factors={k: {"set": v} for k, v in card.items()})
    b = dict(
        ship=REFERENCE,
        factors={} if opponent is None else {k: {"set": v} for k, v in opponent.items()},
    )
    return dict(name=name, a=a, b=b)


# --- the run's records -----------------------------------------------------------------------


class Records:
    """The run's directory: candidates.jsonl (every asked card: coordinates, handles, the
    solved card or the solver's failure, measures), evaluations.jsonl (every condition's
    result for a card), generations.jsonl, scheduler.pkl, queue/<condition>/ (job files for
    the workers) and results/<condition>/ (the workers' outputs)."""

    def __init__(self, root: pathlib.Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.candidates = self._read("candidates.jsonl")
        self.evaluations = self._read("evaluations.jsonl")
        self.by_id = {c["id"]: c for c in self.candidates}
        self.results_of: dict[str, list[dict]] = {}
        for row in self.evaluations:
            self.results_of.setdefault(row["id"], []).append(row)

    def _read(self, name: str) -> list[dict]:
        path = self.root / name
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def append(self, name: str, rows: list[dict]) -> None:
        with (self.root / name).open("a") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
            handle.flush()

    def add_candidates(self, rows: list[dict]) -> None:
        self.append("candidates.jsonl", rows)
        self.candidates.extend(rows)
        self.by_id.update({r["id"]: r for r in rows})

    def add_evaluations(self, rows: list[dict]) -> None:
        self.append("evaluations.jsonl", rows)
        self.evaluations.extend(rows)
        for row in rows:
            self.results_of.setdefault(row["id"], []).append(row)

    def score(self, identifier: str, conditions=SEARCHED) -> dict:
        """A card's objective on everything played so far."""
        return objective_of(pooled(self.results_of.get(identifier, [])), conditions)

    def map(self, dims=(10, 10)) -> dict:
        """The map: each cell's best card by the objective's lower bound ({cell: summary})."""
        cells: dict[str, dict] = {}
        for candidate in self.candidates:
            if "card" not in candidate:
                continue
            score = self.score(candidate["id"])
            if score["objective"] is None:
                continue
            cell = cell_of(candidate["measures"], dims)
            key = f"{cell[0]},{cell[1]}"
            row = dict(id=candidate["id"], measures=candidate["measures"], **score)
            if key not in cells or row["lower"] > cells[key]["lower"]:
                cells[key] = row
        return cells


def cell_of(measures, dims=(10, 10)) -> tuple[int, int]:
    """The grid cell of measures in [0, 1]^2."""
    return tuple(min(int(m * d), d - 1) for m, d in zip(measures, dims, strict=True))


def submit(
    records: Records, label: str, entries: list[dict], games: int, seed: int, conditions=SEARCHED
) -> dict[str, pathlib.Path]:
    """One job per condition into the workers' queues; returns each condition's output."""
    outputs = {}
    for condition in conditions:
        queue = records.root / "queue" / condition
        queue.mkdir(parents=True, exist_ok=True)
        out = records.root / "results" / condition / f"{label}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        job = dict(pairs=entries, out=str(out), games=games, mixed_games=0, seed=seed)
        partial = queue / f"{label}.json.partial"
        partial.write_text(json.dumps(job))
        partial.rename(queue / f"{label}.json")
        outputs[condition] = out
    return outputs


def collect(
    records: Records,
    label: str,
    outputs: dict[str, pathlib.Path],
    count: int,
    log=print,
    poll: float = 5.0,
) -> list[dict]:
    """Wait for every condition's output to hold `count` lines (a failed job raises), then
    the evaluations: one row per card and condition."""
    while True:
        done = all(p.exists() and sum(1 for _ in p.open()) >= count for p in outputs.values())
        if done:
            break
        for condition in outputs:
            failed = records.root / "queue" / condition / "failed" / f"{label}.json.txt"
            if failed.exists():
                raise RuntimeError(f"{condition} job {label} failed:\n{failed.read_text()}")
        time.sleep(poll)
    rows = []
    for condition, path in outputs.items():
        for line in path.read_text().splitlines():
            record = json.loads(line)
            job = record["jobs"]["energy v turn"]
            rows.append(
                dict(
                    id=record["name"],
                    condition=condition,
                    label=label,
                    seed=record["seed"],
                    games=job["games"],
                    kill_share=job["kill_share_a"],
                    point_share=job["point_share_a"],
                    score=job["score"],
                    kills_per_minute=job["kills_per_minute"],
                    deaths_per_minute=job["deaths_per_minute"],
                    match_seconds=job["match_seconds"],
                )
            )
    return rows


# --- the search ------------------------------------------------------------------------------


def seeds() -> list[tuple[str, dict[str, float]]]:
    """The starting cards (name, handles): v23 Jujak and Yongma, and balance-final's options
    E and G (tune/balance-final, version 22's shields; here x5.4 as v23 scaled them)."""
    ships = reference_ships()
    jujak, yongma = dict(ships["jujak"].handles), dict(ships["yongma"].handles)
    out = [("v23-jujak", jujak), ("v23-yongma", yongma)]
    out.append(
        (
            "E-yongma",
            yongma
            | dict(sustained_turn_rate=0.5930 * 1.12, max_shield=yongma["max_shield"] * 0.55),
        )
    )
    out.append(("E-jujak", jujak | dict(sustained_turn_rate=0.7647 * 0.95)))
    out.append(
        (
            "G-yongma",
            yongma
            | dict(
                sustained_turn_rate=0.6600,
                max_shield=yongma["max_shield"] * 0.55,
                ceiling=yongma["ceiling"] * 1.444,
                climb_time=yongma["climb_time"] * 1.444,
                shots_per_store=yongma["shots_per_store"] * 2.137,
            ),
        )
    )
    out.append(
        (
            "G-jujak",
            jujak | dict(sustained_turn_rate=0.7647 * 0.95, max_shield=jujak["max_shield"] * 1.159),
        )
    )
    return out


def make_scheduler(emitters: int, batch: int, sigma: float, learning_rate: float, seed: int):
    """CMA-MAE: a 10 x 10 GridArchive over the two measures with a learning rate under 1,
    `emitters` CMA-ES emitters ranked by improvement, started on the seed cards."""
    from ribs.archives import GridArchive
    from ribs.emitters import EvolutionStrategyEmitter
    from ribs.schedulers import Scheduler

    archive = GridArchive(
        solution_dim=len(SPACE),
        dims=(10, 10),
        ranges=((0.0, 1.0), (0.0, 1.0)),
        learning_rate=learning_rate,
        threshold_min=20.0,
        seed=seed,
    )
    starts = [encode(handles) for _, handles in seeds()]
    order = [1, 4, 2, 0, 5, 3]  # Yongma, G Yongma, E Yongma, Jujak, ...: the far ones first
    bounds = [(0.0, 1.0)] * len(SPACE)
    emitter_list = [
        EvolutionStrategyEmitter(
            archive,
            x0=starts[order[n % len(order)]],
            sigma0=sigma,
            ranker="imp",
            bounds=bounds,
            batch_size=batch,
            seed=seed + 1 + n,
        )
        for n in range(emitters)
    ]
    return Scheduler(archive, emitter_list)


def prepare(
    records: Records, label: str, solutions, base: dict, processes: int, emitter_of=None
) -> list[dict]:
    """Candidates of coordinate vectors: decoded, solved, measured, recorded."""
    handle_sets = [decode(x, base) for x in solutions]
    cards = solve_cards(handle_sets, processes)
    rows = []
    for number, (x, handles, card) in enumerate(zip(solutions, handle_sets, cards, strict=True)):
        row = dict(
            id=f"{label}-{number:03d}",
            generation=label,
            emitter=None if emitter_of is None else emitter_of(number),
            coordinates=[round(float(v), 6) for v in x],
            handles={k: handles[k] for k in STORED},
            measures=list(measures(x)),
        )
        if isinstance(card, str):
            row["failure"] = card[:300]
        else:
            row["card"] = card
        rows.append(row)
    records.add_candidates(rows)
    return rows


def rechecks(records: Records, limit: int) -> list[str]:
    """Cards to bring to 512 games: each cell's best by the objective at or above
    RECHECK_FLOOR with fewer games, best first, up to `limit`."""
    chosen = []
    by_cell: dict[tuple, list] = {}
    for candidate in records.candidates:
        if "card" not in candidate:
            continue
        score = records.score(candidate["id"])
        if score["objective"] is None or score["objective"] < RECHECK_FLOOR:
            continue
        by_cell.setdefault(cell_of(candidate["measures"]), []).append((score, candidate["id"]))
    for rows in by_cell.values():
        score, identifier = max(rows, key=lambda r: r[0]["objective"])
        if score["games"] < SEARCH_GAMES + RECHECK_GAMES:
            chosen.append((score["objective"], identifier))
    chosen.sort(reverse=True)
    return [identifier for _, identifier in chosen[:limit]]


def clock(seconds: float) -> str:
    """A clock time (local) this many seconds from now."""
    return time.strftime("%H:%M", time.localtime(time.time() + seconds))


def search(args) -> None:
    """The CMA-MAE run: each generation asks, solves, plays (SEARCH_GAMES a condition, seed
    0) together with the rechecks (RECHECK_GAMES, seed 1), tells, and writes its records
    before the next; a restart resumes after the last finished generation."""
    records = Records(args.out)
    state = args.out / "scheduler.pkl"
    base = dict(reference_ships()[REFERENCE].handles)
    if state.exists():
        with state.open("rb") as handle:
            saved = pickle.load(handle)
        scheduler, generation = saved["scheduler"], saved["generation"] + 1
        print(f"resumed at generation {generation}", flush=True)
    else:
        scheduler = make_scheduler(
            args.emitters, args.batch, args.sigma, args.learning_rate, args.seed
        )
        generation = 0
    started = time.time()
    deadline = started + args.hours * 3600.0
    if generation == 0 and not any(c["generation"] == "seed" for c in records.candidates):
        named = seeds()
        rows = prepare(records, "seed", [encode(h) for _, h in named], base, args.processes)
        for row, (name, _) in zip(rows, named, strict=True):
            row["name"] = name
        valid = [r for r in rows if "card" in r]
        outputs = submit(
            records,
            "seed",
            [pair_entry(r["id"], r["card"]) for r in valid],
            SEARCH_GAMES + RECHECK_GAMES,
            1,
        )
        records.add_evaluations(collect(records, "seed", outputs, len(valid)))
        for row in valid:
            score = records.score(row["id"])
            scheduler.archive.add(
                numpy.array([row["coordinates"]]),
                numpy.array([score["objective"]]),
                numpy.array([row["measures"]]),
            )
            print(f"seed {row['name']}: {json.dumps(score)}", flush=True)
    per_generation = []
    while generation < args.generations and time.time() < deadline:
        begun = time.time()
        label = f"g{generation:04d}"
        solutions = scheduler.ask()
        batch = args.batch
        rows = prepare(records, label, solutions, base, args.processes, lambda n: n // batch)
        valid = [r for r in rows if "card" in r]
        recheck = rechecks(records, RECHECKS_PER_GENERATION)
        outputs = submit(
            records, label, [pair_entry(r["id"], r["card"]) for r in valid], SEARCH_GAMES, 0
        )
        recheck_outputs = {}
        if recheck:
            recheck_outputs = submit(
                records,
                f"{label}-recheck",
                [pair_entry(i, records.by_id[i]["card"]) for i in recheck],
                RECHECK_GAMES,
                1,
            )
        records.add_evaluations(collect(records, label, outputs, len(valid)))
        if recheck:
            records.add_evaluations(
                collect(records, f"{label}-recheck", recheck_outputs, len(recheck))
            )
        objectives = []
        for row in rows:
            score = records.score(row["id"]) if "card" in row else dict(objective=None)
            objectives.append(0.0 if score["objective"] is None else score["objective"])
        scheduler.tell(numpy.array(objectives), numpy.array([r["measures"] for r in rows]))
        partial = state.with_suffix(".partial")
        with partial.open("wb") as handle:
            pickle.dump(dict(scheduler=scheduler, generation=generation), handle)
        partial.replace(state)
        cells = records.map()
        (args.out / "map.json").write_text(json.dumps(cells, indent=1))
        took = time.time() - begun
        per_generation.append(took)
        left = min(
            args.generations - generation - 1, int((deadline - time.time()) / max(took, 1.0))
        )
        mean = sum(per_generation[-5:]) / len(per_generation[-5:])
        line = dict(
            generation=generation,
            time=time.strftime("%H:%M:%S"),
            seconds=round(took),
            asked=len(rows),
            solved=len(valid),
            infeasible=len(rows) - len(valid),
            rechecked=len(recheck),
            best=max(objectives),
            cells=len(cells),
            balanced_cells=sum(c["objective"] >= BALANCED for c in cells.values()),
            confident_cells=sum(
                c["lower"] >= BALANCED and c["games"] >= 512 for c in cells.values()
            ),
            archive_cells=len(scheduler.archive),
        )
        records.append("generations.jsonl", [line])
        print(
            f"{label} {line['time']}: {took / 60:.1f} min; solved {len(valid)}/{len(rows)}, "
            f"rechecked {len(recheck)}; best {max(objectives):.1f}; map {len(cells)} cells, "
            f"{line['balanced_cells']} balanced (objective >= {BALANCED}), "
            f"{line['confident_cells']} at 512 games with the lower bound >= {BALANCED}; "
            f"bound: {left} more generations, done about {clock(left * mean)} KST",
            flush=True,
        )
        generation += 1
    print("search done", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("search")
    one.add_argument("--out", type=pathlib.Path, required=True)
    one.add_argument("--generations", type=int, default=60)
    one.add_argument("--hours", type=float, default=12.0)
    one.add_argument("--emitters", type=int, default=4)
    one.add_argument("--batch", type=int, default=32, help="candidates an emitter asks")
    one.add_argument("--sigma", type=float, default=0.15)
    one.add_argument("--learning-rate", type=float, default=0.05)
    one.add_argument("--seed", type=int, default=0)
    one.add_argument("--processes", type=int, default=8, help="solver processes")
    args = parser.parse_args()
    if args.command == "search":
        search(args)


if __name__ == "__main__":
    main()
