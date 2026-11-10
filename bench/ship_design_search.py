"""Design real-scale ship types whose head-to-head results form cycles
(docs/explanation/game/ship-design.md): a CMA-ES search over the designs' knobs
(`bnb.game.ship_design`), each generation's candidates played on one fixed arena whose card
is written in place, so nothing recompiles.

    uv run --no-sync python -m bench.ship_design_search search --spec SPEC.json --out DIR
    uv run --no-sync python -m bench.ship_design_search check --spec SPEC.json --out DIR \\
        [--candidate GENERATION:INDEX] [--games 1024] [--preset real_scale_frontline]

SPEC.json (see `docs/explanation/game/ship-design/`): the designs (name, base type, the
knobs the search moves, optional start values), the edges (each [winner, loser]) or a
cycle (names in order; the search picks its direction at the first generation), the band
an edge's score should lie in, and the search's sizes. Each design's knobs start at its
base's values; a knob's search coordinate is its log in envelope deviations from there
(the fire rate: one unit a step between FIRE_RATES), held within the envelope's bounds.

The loss of a candidate (a value for every design's searched knobs) is

    sum over edges of (the edge's log-odds' distance outside the band)^2
    + realism_weight x sum over designs of max(0, Mahalanobis distance - realism_radius)^2
    + distinct_weight x sum over design pairs of max(0, distinct - their z distance)^2,

the edges played `games_per_edge` games each (half with each design on each side), every
ship of a team one design, both teams the scripted agent (its real-scale set for the
preset, T = 1). Each generation plays the CMA-ES mean and `population - 1` samples; its
rows (every candidate's knobs, scores and loss) are appended to DIR/search.jsonl and the
optimiser's state to DIR/state.json as it ends, so a stopped search resumes. The search
stops at `generations` or `--hours`; each generation prints its ETA.

`check` plays a candidate (by default the best mean of the search) at `--games` games per
edge on every pair of designs (a round robin), and the mixed-team test: for every pair
{A, B} and every design D that the spec says beats both, a team of A and B (alternating,
3 + 2 and 2 + 3 in half the games each) against a team of D. Rows to DIR/check-<preset>.jsonl.

`summary` turns a check into the verdict against the targets: the Bradley-Terry ratings, the
edges against the band, the uniform pick's exploitability, the Nash equilibrium (regret
matching) and its support, and the mixed-team test (DIR/summary-<preset>.json).
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
from bnb.game.ship_types import ShipTypes
from bnb.league.matches import Arena, ArenaConfig, Player
from bnb.runtime.opponents import Rung

SLOTS = 36  # the arena's card: a fixed number of type slots, so a new card never recompiles
BLOCKS = 36  # matchups a batch
BLOCK_GAMES = 64


def clock(seconds: float) -> str:
    return (datetime.datetime.now() + datetime.timedelta(seconds=seconds)).strftime("%H:%M")


# ---------------------------------------------------------------- the designs


@dataclasses.dataclass
class Space:
    """The search's coordinates: per design, its base's handles and knobs, and the knobs it
    moves; `x` is every design's moved knobs' coordinates, in the spec's order."""

    spec: dict
    handles: dict
    envelope: Envelope
    rules: object

    def __post_init__(self) -> None:
        self.designs = self.spec["designs"]
        self.columns = [
            (index, knob) for index, design in enumerate(self.designs) for knob in design["knobs"]
        ]

    @property
    def size(self) -> int:
        return len(self.columns)

    def base(self, index: int) -> dict:
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
        for value, (index, name) in zip(x, self.columns, strict=True):
            if name == "fire_rate":
                start = FIRE_RATES.index(out[index]["fire_rate"])
                step = min(max(start + int(round(float(value))), 0), len(FIRE_RATES) - 1)
                out[index]["fire_rate"] = FIRE_RATES[step]
            else:
                deviation = float(self.envelope.deviation[CONTINUOUS.index(name)])
                out[index][name] = out[index][name] * math.exp(float(value) * deviation)
        return [self.envelope.clamp(knobs) for knobs in out]

    def handles_of(self, knobs: list[dict[str, float]]) -> list[dict[str, float]]:
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


def band_loss(score: float, games: int, band) -> float:
    """An edge's squared log-odds distance outside the band (the score held off 0 and 1
    by half a game)."""
    p = min(max(score, 0.5 / games), 1.0 - 0.5 / games)
    logit = math.log(p / (1.0 - p))
    low, high = (math.log(b / (1.0 - b)) for b in band)
    return max(0.0, low - logit, logit - high) ** 2


# ---------------------------------------------------------------- the arena


class DesignArena:
    """A fixed arena of SLOTS type slots and BLOCKS matchups a batch: one player per slot
    (every ship that slot's type), and mixed teams added as needed. `play(card, matchups)`
    writes the card in place and plays the matchups, BLOCKS at a time."""

    def __init__(self, preset: str, device: torch.device, block_games: int = BLOCK_GAMES):
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
            ArenaConfig(blocks=BLOCKS, block_environments=block_games),
            card=card,
        )
        self.device = device

    def player(self, lineup: tuple[int, ...]) -> int:
        """The index of a player whose team flies this lineup (added the first time)."""
        for index, one in enumerate(self.arena.players):
            if one.ship_type == lineup:
                return index
        one = Player(f"lineup {lineup}", rung=Rung(1.0), ship_type=lineup)
        self.arena.players.append(one)
        self.arena.team_types.append(lineup)
        return len(self.arena.players) - 1

    def play(self, card: ShipTypes, matchups: list[tuple[int, int]]) -> np.ndarray:
        """(len(matchups), 3) each matchup's first player's wins, ties and losses."""
        self.arena.set_card(card)
        out = []
        for start in range(0, len(matchups), BLOCKS):
            chunk = list(matchups[start : start + BLOCKS])
            real = len(chunk)
            chunk += [(0, 1)] * (BLOCKS - real)  # padding, not read
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


def search(args) -> None:
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
            samples = cma.ask()
            knobs = [space.knobs(x) for x in samples]
            designs = [h for one in knobs for h in space.handles_of(one)]
            bases = [d["base"] for d in space.designs] * population
            card = padded(design_cards(space.rules, bases, designs))
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
            if cycle and orientation is None:  # the way round the mean is nearer the band
                forward = sum(band_loss(p, games, spec["band"]) for p in played[0])
                backward = sum(band_loss(1 - p, games, spec["band"]) for p in played[0])
                orientation = 1 if forward <= backward else -1
                print(f"the cycle runs {'forward' if orientation == 1 else 'backward'}", flush=True)
            losses, rows = [], []
            for candidate in range(population):
                scores = played[candidate]
                if cycle and orientation == -1:
                    scores = [1.0 - p for p in scores]
                edge_loss = sum(band_loss(s, games, spec["band"]) for s in scores)
                penalty = space.penalty(knobs[candidate])
                loss = edge_loss + penalty["realism"] + penalty["distinct"]
                losses.append(loss)
                rows.append(
                    dict(
                        generation=cma.generation,
                        index=candidate,
                        mean=candidate == 0,
                        x=samples[candidate].tolist(),
                        knobs=knobs[candidate],
                        scores=scores,
                        edge_loss=edge_loss,
                        loss=loss,
                        sigma=cma.sigma,
                        **penalty,
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


def check(args) -> None:
    spec = json.loads(pathlib.Path(args.spec).read_text())
    out = pathlib.Path(args.out)
    space = setup(spec)
    names = [design["name"] for design in space.designs]
    log_path = out / "search.jsonl"
    if args.candidate:
        generation, index = (int(v) for v in args.candidate.split(":"))
        row = next(
            json.loads(line)
            for line in log_path.read_text().splitlines()
            if json.loads(line)["generation"] == generation and json.loads(line)["index"] == index
        )
    else:
        row = best_of(log_path)
    knobs = row["knobs"]
    preset = args.preset or spec.get("preset", "real_scale_deathmatch")
    card = padded(
        design_cards(space.rules, [d["base"] for d in space.designs], space.handles_of(knobs))
    )
    if args.card:  # an exact card from ship files
        from bnb.game import ships as ship_files

        found = {ship.name: ship for ship in ship_files.load(args.card)}
        exact = ShipTypes(
            **{
                k: torch.tensor([found[name].card[k] for name in names], dtype=torch.float64)
                for k in vars(card)
            }
        )
        card = padded(exact)
    device = torch.device("cuda")
    count = len(names)
    edges, cycle = spec_edges(spec, names)
    if cycle and json.loads((out / "state.json").read_text()).get("orientation") == -1:
        edges = [(b, a) for a, b in edges]
    beats = {(names[a], names[b]) for a, b in edges}
    path = out / f"check-{preset}.jsonl"
    with bench_lock():
        arena = DesignArena(preset, device)
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
                        first = arena.player(lineup)
                        mixed.append((a, b, d, lineup))
                        matchups += [(first, d)] * max(blocks // 2, 1)
        torch.manual_seed(args.seed)
        started = time.time()
        counts = arena.play(card, matchups)
        with open(path, "a") as log:
            at = 0
            for a, b in pairs:
                total = counts[at : at + blocks].sum(0)
                at += blocks
                score = float((total[0] + 0.5 * total[1]) / total.sum())
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
                    candidate=f"{row['generation']}:{row['index']}",
                    card=args.card or "estimate",
                )
                log.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            for a, b, d, lineup in mixed:
                n = max(blocks // 2, 1)
                total = counts[at : at + n].sum(0)
                at += n
                score = float((total[0] + 0.5 * total[1]) / total.sum())
                record = dict(
                    kind="mixed",
                    team=[names[k] for k in lineup],
                    against=names[d],
                    games=int(total.sum()),
                    score=score,
                    stderr=math.sqrt(score * (1 - score) / total.sum()),
                    candidate=f"{row['generation']}:{row['index']}",
                    card=args.card or "estimate",
                )
                log.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
        print(f"checked in {time.time() - started:.0f} s", flush=True)


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


def summary(args) -> None:
    """The check's verdict against the targets (docs/explanation/game/ship-design.md)."""
    spec = json.loads(pathlib.Path(args.spec).read_text())
    out = pathlib.Path(args.out)
    names = [design["name"] for design in spec["designs"]]
    rows = [
        json.loads(line) for line in (out / f"check-{args.preset}.jsonl").read_text().splitlines()
    ]
    if args.card:
        rows = [row for row in rows if row["card"] == args.card]
    count = len(names)
    score = np.full((count, count), 0.5)
    games = np.zeros((count, count))
    for row in rows:
        if row["kind"] != "pair":
            continue
        a, b = names.index(row["first"]), names.index(row["second"])
        score[a, b], score[b, a] = row["score"], 1.0 - row["score"]
        games[a, b] = games[b, a] = row["games"]
    # Bradley-Terry ratings (Elo, mean 0) by Newton on the pair scores
    rating = np.zeros(count)
    for _ in range(200):
        expected = 1.0 / (1.0 + np.exp(rating[None, :] - rating[:, None]))
        gradient = (games * (score - expected)).sum(1)
        hessian = (games * expected * (1 - expected)).sum(1) + 1e-9
        rating += gradient / hessian
        rating -= rating.mean()
    elo = rating * 400.0 / math.log(10.0)
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
    verdict = dict(
        preset=args.preset,
        elo=dict(zip(names, elo.round(1).tolist(), strict=True)),
        elo_spread=float(np.abs(elo).max()),
        edges=edge_scores,
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
    verdict["pass"] = (
        verdict["elo_spread"] <= 35.0
        and verdict["edges_in_band"]
        and verdict["exploitability"] <= 0.02
        and verdict["nash_full_support"]
        and (verdict["mixed_pass"] or not mixed)
    )
    (out / f"summary-{args.preset}.json").write_text(json.dumps(verdict, indent=1) + "\n")
    print(json.dumps(verdict, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("search")
    one.add_argument("--spec", required=True)
    one.add_argument("--out", required=True)
    one.add_argument("--hours", type=float, default=3.0)
    two = sub.add_parser("check")
    two.add_argument("--spec", required=True)
    two.add_argument("--out", required=True)
    two.add_argument("--candidate", help="GENERATION:INDEX (default: the best recent mean)")
    two.add_argument("--games", type=int, default=1024, help="per pair")
    two.add_argument("--preset", help="default: the spec's")
    two.add_argument("--card", help="a ship set holding the designs' exact cards, by name")
    two.add_argument("--seed", type=int, default=5_000_000)
    three = sub.add_parser("summary")
    three.add_argument("--spec", required=True)
    three.add_argument("--out", required=True)
    three.add_argument("--preset", default="real_scale_deathmatch")
    three.add_argument("--card", help="only the rows of this card (a ship set, or 'estimate')")
    args = parser.parse_args()
    {"search": search, "check": check, "summary": summary}[args.command](args)


if __name__ == "__main__":
    main()
