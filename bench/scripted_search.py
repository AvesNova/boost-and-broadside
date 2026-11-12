"""The scripted agent's parameter search (scripted spec A3): a population round robin, kept so
it can be rerun.

    uv run --no-sync python -m bench.scripted_search --preset frontline --log frontline.jsonl
    uv run --no-sync python -m bench.scripted_search --preset frontline --log frontline.jsonl \
        --generations 6  # resume

The agent's defaults are one parameter set per game mode (`MODE_DEFAULTS`), each from a
search of its own preset (`--preset`: the candidates play only that preset); without
`--preset` one set is searched over every standard preset. A preset plays its ship set's
card with the game's lineup, or every ship of the type `--ship-type` (an index into the
set's card); the first generation is the default `ScriptedParameters()` and perturbations
of it, or with `--from-defaults` the preset's own set in `MODE_DEFAULTS` and perturbations
of that (the real-scale variant's light retune:

    uv run --no-sync python -m bench.scripted_search --preset real_scale_deathmatch \
        --ship-type 0 --from-defaults --log real_scale_deathmatch.jsonl
).

`--mixed` plays mixed lineups instead: each game draws every seat's type from the
preset's set (`--types`, all by default; less `--hold-out`, kept back to check that a set
generalises) and both teams fly the same draw, seat for seat, so neither side is favoured
by its ships (`mixed_lineups`).

`--champion` plays against a champion instead of the standard references: the preset's set
in `MODE_DEFAULTS` (`--champion defaults`) or a search log's candidate (`LOG:GENERATION:
INDEX`), anchored at 0. Ratings anchored on a reference that every candidate beats are held
by the prior, so the plateau rule stalls; so when the median candidate's score against the
champion reaches `--promote-at`, the generation's top candidate becomes the champion (from
the next generation on) and the plateau history starts again. Each row logs its champion
(`champion`: "defaults", the spec given, or "GENERATION:INDEX" of this log), so a resumed
search keeps it. The real-scale balancing's searches (docs/explanation/game/real-scale-balance.md):

    uv run --no-sync python -m bench.scripted_search --preset real_scale_deathmatch \
        --mixed --hold-out "Spitfire XIV,Ki-84,Zerstörer,Lantern" --champion defaults \
        --from-defaults --plateau-on median --generations 40 --log dm.jsonl

Each generation plays a population of `--population` candidates side by side, one
parameter-table row per view, so one compiled rollout per preset: every candidate plays
`--games` fresh 5v5 games per preset, fog on, the seat drawn per game, at each preset searched.
A share `--reference-share` of those games is split evenly over the reference
players (`REFERENCES`: the scripted agent at its defaults for the mode, and the chaser); the
rest are
against other candidates drawn at random. Each candidate seat plays at a temperature drawn per
game from `TEMPERATURES`, so the ranking rewards strength across temperatures; the references
play as they are.

The ranking is a Bradley-Terry fit over every game of the generation, the first reference
(the defaults) anchored at 0 (`bnb.league.ratings.fit_ratings`). The next generation keeps
the top `--keep` and fills the rest with children of them: a uniform crossover of two parents
(half the children; the other half one parent), then a log-normal perturbation of every
searched number with a step `sigma`, which shrinks by `--decay` per generation. Each
logistic's scale stays at or above its floor (`SCALE_FLOORS`), and each switch's at or above
its own (`SWITCH_SCALE_FLOORS`), its rate within `SWITCH_RATES`; the memory's fade centre
stays within `FADE_CENTRE`. The kept candidates replay with fresh games, so a lucky
generation does not carry its luck. The search stops after `--generations` generations, or
earlier when the best rating (`--plateau-on`: the top, or the median) of the last
`--patience` generations is not `--min-gain` points above the best before them.

Every candidate's parameters, rating, its error and its score against each reference per
preset are appended to --log as a JSON line; a run resumes from its log's last generation.

Seeds: the children draw from a generator seeded with (--seed, generation); generation g's
games seed the default generators with `game_seed(seed, g)`, below `SEED_LIMIT` for every
seed and generation this tool accepts. The acceptance tests
(`tests/sections/test_scripted_acceptance.py`) seed their games at or above it.

Holds the benchmark lock. The section notes record the runs that chose the defaults; the last
generation of each is kept as `scripted_populations.json` (`scripted_defaults`).
"""

import argparse
import datetime
import json
import math
import os
import pathlib
import time
from collections.abc import Callable

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import torch  # noqa: E402

from bench.lock import bench_lock  # noqa: E402
from bnb.contracts import Action, Obs, Outputs, Pool  # noqa: E402
from bnb.controllers.scripted import Chaser, Scripted  # noqa: E402
from bnb.controllers.scripted_defaults import MODE_DEFAULTS  # noqa: E402
from bnb.controllers.scripted_parameters import (  # noqa: E402
    ADDITIVE_BOUNDS,
    MODES,
    NAMES,
    OWN_TIME,
    SCALE_FLOORS,
    STANDARD_MODES,
    SWITCH_SCALE_FLOORS,
    Logistic,
    Ramp,
    ScriptedParameters,
    Switch,
    from_row,
    pack,
    table,
    unpack,
)
from bnb.game import api  # noqa: E402
from bnb.game.handling.characteristics import Characteristics, characteristics  # noqa: E402
from bnb.game.modes.rules import PRESETS  # noqa: E402
from bnb.league.ratings import fit_ratings  # noqa: E402
from bnb.runtime.evaluate import first_results  # noqa: E402

PRESET_NAMES = MODES  # the presets with a parameter set each, one search each (`--preset`)
TEMPERATURES = (1.0, 1.5, 2.0, 3.0)  # each candidate seat's, drawn per game
SCALE_CEILING = 10.0  # a logistic's or switch's largest scale, in multiples of its floor
SWITCH_RATES = (1.0, 30.0)  # a switch's fastest rate, per second: its bounds
FADE_CENTRE = (0.5, 2.0)  # seconds: the bounds of the memory fade logistic's centre
PRIOR_GAMES = 1.0  # each player's virtual games against the anchor, half won (fit_ratings)
MAXIMUM_SEED, MAXIMUM_GENERATIONS = 999, 1000
SEED_LIMIT = MAXIMUM_GENERATIONS * (MAXIMUM_SEED + 1)  # every search game seed is below this

# The reference players by name: a controller from the agent's characteristics and the ships
# per team. The first is the anchor of every generation's fit.
Reference = Callable[[Characteristics, int], object]
REFERENCES: dict[str, Reference] = {
    "scripted": lambda card, ships: Scripted(card, ships),
    "chaser": lambda card, ships: Chaser(card, ships),
}
CHAMPION = "champion"  # the reference's name with `--champion`


def champion_references(parameters: ScriptedParameters) -> dict[str, Reference]:
    """The references of a search against a champion: the champion alone."""
    return {CHAMPION: lambda card, ships: Scripted(card, ships, parameters)}


def mixed_lineups(
    views: int, ships_per_team: int, pool: tuple[int, ...], generator=None
) -> torch.Tensor:
    """(views, 2 x ships_per_team) int32: each view's seat types drawn uniformly from `pool`
    (type indices, with replacement), the second team flying the first's, seat for seat."""
    choice = torch.randint(len(pool), (views, ships_per_team), generator=generator)
    team = torch.tensor(pool, dtype=torch.int32)[choice]
    return torch.cat([team, team], dim=1)


def type_pool(ship_set: str, types: str | None, hold_out: str | None) -> tuple[int, ...]:
    """The type indices of a set named in `types` (comma-separated names; all when None),
    less those in `hold_out`."""
    names = list(api.type_names(ship_set))
    chosen = names if not types else [name.strip() for name in types.split(",")]
    held = [] if not hold_out else [name.strip() for name in hold_out.split(",")]
    unknown = [name for name in chosen + held if name not in names]
    if unknown:
        raise SystemExit(f"not types of {ship_set}: {unknown}")
    return tuple(names.index(name) for name in chosen if name not in held)


def game_seed(seed: int, generation: int) -> int:
    """The default generators' seed for a generation's games, below SEED_LIMIT for every seed
    and generation the tool accepts."""
    return MAXIMUM_GENERATIONS * seed + generation


class Mixed:
    """Several controllers over the same views, each row playing one of them: `choice` (R,)
    int64, the index into `controllers` of each row's player. Every part runs on every row,
    with a carry of its own (a tuple, one per part)."""

    def __init__(self, controllers: list, choice: torch.Tensor) -> None:
        self.controllers, self.choice = controllers, choice
        self.reads = tuple(dict.fromkeys(name for c in controllers for name in c.reads))

    def init_carry(self, views: int, device: torch.device) -> tuple:
        """Every part's carry."""
        return tuple(controller.init_carry(views, device) for controller in self.controllers)

    def step(self, carry: tuple, obs: Obs, reset: torch.Tensor):
        """Every part's action on every row, each row keeping its own player's."""
        action, carries = None, []
        for index, (controller, part_carry) in enumerate(zip(self.controllers, carry, strict=True)):
            part, part_carry, _ = controller.step(part_carry, obs, reset)
            part = part.packed()
            carries.append(part_carry)
            mine = (self.choice == index)[:, None, None]
            action = part if action is None else torch.where(mine, part, action)
        return Action.unpack(action), tuple(carries), Outputs()


def clamp(parameters: ScriptedParameters) -> ScriptedParameters:
    """The parameters inside the search's bounds: each logistic's and switch's scale between
    its floor and SCALE_CEILING floors, a switch's rate within SWITCH_RATES, the fade's
    centre within FADE_CENTRE, the recovery shield fraction at most 0.9, the combat radius
    beyond the shoot logistic's centre, each additive number (`ADDITIVE_BOUNDS`: the own
    time's) within its bounds."""

    def bounded(value: float, low: float, high: float) -> float:
        return min(max(value, low), high)

    values = {}
    for name in NAMES:
        value = getattr(parameters, name)
        if isinstance(value, Logistic):
            floor = SCALE_FLOORS[name]
            scale = bounded(value.scale, floor, SCALE_CEILING * floor)
            centre = (
                bounded(value.centre, *FADE_CENTRE)
                if name == "memory_fade_in_seconds"
                else value.centre
            )
            value = Logistic(centre, scale, value.rising)
        elif isinstance(value, Switch):
            floor = SWITCH_SCALE_FLOORS[name]
            value = Switch(
                bounded(value.rate, *SWITCH_RATES),
                value.centre,
                bounded(value.scale, floor, SCALE_CEILING * floor),
            )
        elif name in ADDITIVE_BOUNDS:
            value = bounded(value, *ADDITIVE_BOUNDS[name])
        values[name] = value
    values["recovery_shield_fraction"] = min(values["recovery_shield_fraction"], 0.9)
    values["combat_radius_in_gun_range"] = max(
        values["combat_radius_in_gun_range"], 1.1 * values["shoot_in_gun_range"].centre
    )
    return ScriptedParameters(**values)


def perturb(
    parameters: ScriptedParameters,
    sigma: float,
    generator: torch.Generator,
    frozen: tuple[str, ...] = OWN_TIME,
) -> ScriptedParameters:
    """Every searched number times exp(sigma N(0, 1)) (a ramp's ends sorted; its values and a
    logistic's direction kept; a switch's rate, centre and scale each), aggression plus
    2 sigma N(0, 1), an additive number (`ADDITIVE_BOUNDS`) plus sigma N(0, 1) times its
    bounds' width; then `clamp`. The `frozen` fields are kept and draw nothing (by default
    the own time's, so a search without it draws as before it existed)."""

    def factor() -> float:
        return math.exp(sigma * float(torch.randn(1, generator=generator)))

    values = {}
    for name in NAMES:
        value = getattr(parameters, name)
        if name in frozen:
            values[name] = value
        elif name in ADDITIVE_BOUNDS:
            low, high = ADDITIVE_BOUNDS[name]
            step = sigma * (high - low) * float(torch.randn(1, generator=generator))
            values[name] = value + step
        elif isinstance(value, Ramp):
            low, high = sorted(end * factor() for end in (value.low, value.high))
            values[name] = Ramp(low, max(high, low + 1e-3), value.at_low, value.at_high)
        elif isinstance(value, Logistic):
            values[name] = Logistic(value.centre * factor(), value.scale * factor(), value.rising)
        elif isinstance(value, Switch):
            values[name] = Switch(*(number * factor() for number in value))
        elif name == "aggression":
            values[name] = value + 2.0 * sigma * float(torch.randn(1, generator=generator))
        else:
            values[name] = value * factor()
    return clamp(ScriptedParameters(**values))


def crossover(
    first: ScriptedParameters, second: ScriptedParameters, generator: torch.Generator
) -> ScriptedParameters:
    """Each field (a whole logistic or ramp) from one parent or the other, evenly."""
    pick = torch.rand(len(NAMES), generator=generator) < 0.5
    chosen = {
        name: getattr(first if bool(take) else second, name)
        for name, take in zip(NAMES, pick, strict=True)
    }
    return clamp(ScriptedParameters(**chosen))


def next_generation(
    ranked: list[ScriptedParameters],
    size: int,
    keep: int,
    sigma: float,
    generator,
    frozen: tuple[str, ...] = OWN_TIME,
) -> list[ScriptedParameters]:
    """The top `keep` of `ranked` (best first), and children of them to `size` (`perturb`,
    with its `frozen` fields kept)."""
    parents, children = ranked[:keep], []
    while len(parents) + len(children) < size:
        first = parents[int(torch.randint(keep, (1,), generator=generator))]
        if len(children) % 2 == 0:
            second = parents[int(torch.randint(keep, (1,), generator=generator))]
            first = crossover(first, second, generator)
        children.append(perturb(first, sigma, generator, frozen))
    return parents + children


def play(
    candidates: list[ScriptedParameters],
    games: int,
    reference_share: float,
    device: torch.device,
    presets: tuple[str, ...] = STANDARD_MODES,
    ship_type: int | None = None,
    pool: tuple[int, ...] | None = None,
    reference_players: dict[str, Reference] = REFERENCES,
) -> tuple[torch.Tensor, dict]:
    """One generation's games. Returns the win matrix (K, K), K = the candidates then the
    references, a draw half a win each, and per reference each candidate's score against it
    per preset. ship_type: every ship flies this type of the preset's card (None: the
    game's lineup); pool: mixed lineups of these types instead (`mixed_lineups`)."""
    count, references = len(candidates), list(reference_players)
    per_reference = round(reference_share * games / len(references))
    against_population = games - per_reference * len(references)
    wins = torch.zeros(count + len(references), count + len(references), dtype=torch.float64)
    scores = {name: [[0.0] * len(presets) for _ in range(count)] for name in references}
    rows = table(candidates, device)
    for preset_index, preset in enumerate(presets):
        game = PRESETS[preset]
        types = api.ship_types(device, ship_set=game.ship_set)
        card = characteristics(types, api.rules_of(game))
        # each candidate's games: against the population, then each reference in turn
        player = torch.arange(count).repeat_interleave(games)
        opponent = torch.empty_like(player)
        for index in range(count):
            start = index * games
            others = torch.randint(count - 1, (against_population,))
            opponent[start : start + against_population] = others + (others >= index).long()
            for reference_index in range(len(references)):
                first = start + against_population + reference_index * per_reference
                opponent[first : first + per_reference] = count + reference_index
        views = player.shape[0]
        temperature = torch.tensor(TEMPERATURES)[torch.randint(len(TEMPERATURES), (2, views))]
        temperature = temperature.to(device)
        choice = torch.where(opponent < count, 0, opponent - count + 1).to(device)
        population = Scripted(
            card,
            game.ships_per_team,
            unpack(rows[opponent.clamp_max(count - 1).to(device)]),
            temperature[1],
        )
        rival = Mixed(
            [population]
            + [reference_players[name](card, game.ships_per_team) for name in references],
            choice,
        )
        candidate = Scripted(
            card, game.ships_per_team, unpack(rows[player.to(device)]), temperature[0]
        )
        torch._dynamo.reset()  # a fresh rollout per preset; the compile cache is not reused
        lineup = None
        if ship_type is not None:
            lineup = torch.full((views, game.num_ships), ship_type, dtype=torch.int32)
        if pool is not None:
            lineup = mixed_lineups(views, game.ships_per_team, pool)
        results = first_results(
            game,
            types,
            (Pool("search", 1.0, ("player", "opponent")),),
            {"player": candidate, "opponent": rival},
            "player",
            views,
            device,
            ship_type=lineup,
        ).cpu()
        points = (results > 0).double() + 0.5 * (results == 0).double()
        wins.index_put_((player, opponent), points, accumulate=True)
        wins.index_put_((opponent, player), 1.0 - points, accumulate=True)
        for reference_index, name in enumerate(references):
            for index in range(count):
                mine = (player == index) & (opponent == count + reference_index)
                scores[name][index][preset_index] = float(points[mine].mean())
    return wins, scores


def plateaued(tops: list[float], patience: int, min_gain: float) -> bool:
    """Whether the best of the last `patience` top ratings is not `min_gain` above the best
    before them."""
    if len(tops) <= patience:
        return False
    return max(tops[-patience:]) < max(tops[:-patience]) + min_gain


def statistic(ratings: list[float], kind: str) -> float:
    """A generation's rating for the plateau rule: its top, or its median (steadier: a
    lucky top does not hold the rule off its plateau)."""
    ordered = sorted(ratings, reverse=True)
    return ordered[0] if kind == "top" else ordered[len(ordered) // 2]


def promoted(rows: list[dict], promote_at: float | None) -> str | None:
    """The candidate a generation's rows promote to champion ("GENERATION:INDEX", its top
    rated) when the median candidate's mean score against the champion reaches `promote_at`;
    else None."""
    if promote_at is None or not rows or CHAMPION not in rows[0]["scores"]:
        return None
    scores = sorted(sum(row["scores"][CHAMPION]) / len(row["scores"][CHAMPION]) for row in rows)
    if scores[len(scores) // 2] < promote_at:
        return None
    top = max(rows, key=lambda row: row["rating"])
    return f"{top['generation']}:{top['index']}"


def resume(
    path: pathlib.Path, kind: str = "top", promote_at: float | None = None
) -> tuple[int, list[ScriptedParameters], list[float], str | None]:
    """From a search log: the next generation's index, the last generation's candidates (best
    first), the rating for the plateau rule (`statistic`) of every generation against the
    next generation's champion, and that champion's label (None without one)."""
    rows = [json.loads(line) for line in open(path)]
    last = max(row["generation"] for row in rows)
    by_generation = [[row for row in rows if row["generation"] == g] for g in range(last + 1)]
    champion = by_generation[-1][0].get("champion")
    tops = [
        statistic([row["rating"] for row in one], kind)
        for one in by_generation
        if one[0].get("champion") == champion
    ]
    if (label := promoted(by_generation[-1], promote_at)) is not None:
        champion, tops = label, []
    ranked = sorted(by_generation[-1], key=lambda r: -r["rating"])
    return last + 1, [from_row(row["parameters"]) for row in ranked], tops, champion


def champion_of(label: str, preset: str, log_path: pathlib.Path) -> ScriptedParameters:
    """A champion's parameters: "defaults" (the preset's set in MODE_DEFAULTS),
    "GENERATION:INDEX" (a candidate of this search's log) or "LOG:GENERATION:INDEX"."""
    if label == "defaults":
        return MODE_DEFAULTS.sets[preset]
    path, generation, index = (str(log_path), *label.split(":"))
    if label.count(":") == 2:
        path, generation, index = label.rsplit(":", 2)
    for line in open(path):
        row = json.loads(line)
        if row["generation"] == int(generation) and row["index"] == int(index):
            return from_row(row["parameters"])
    raise SystemExit(f"champion {label}: no such candidate")


def main() -> None:
    """Run the search the module describes, appending every candidate to the log."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--generations", type=int, default=12, help="this call's, at most")
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--keep", type=int, default=8)
    parser.add_argument("--games", type=int, default=128, help="per candidate and preset")
    parser.add_argument("--reference-share", type=float, default=0.25)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--decay", type=float, default=0.85, help="sigma's factor per generation")
    parser.add_argument("--minimum-sigma", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-gain", type=float, default=10.0, help="rating points")
    parser.add_argument(
        "--plateau-on", choices=("top", "median"), default="top", help="the plateau rule's rating"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log", default="scripted_search.jsonl")
    parser.add_argument(
        "--preset",
        choices=PRESET_NAMES,
        help="search this preset's parameter set alone (default: one set over every standard "
        "preset)",
    )
    parser.add_argument(
        "--ship-type", type=int, help="every ship flies this type (default: the game's lineup)"
    )
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="mixed lineups: each seat a type drawn from --types, both teams the same draw",
    )
    parser.add_argument("--types", help="comma-separated type names (default: all)")
    parser.add_argument("--hold-out", help="comma-separated type names left out of the draws")
    parser.add_argument(
        "--champion",
        help="play against this champion alone: 'defaults' or LOG:GENERATION:INDEX (needs "
        "--preset)",
    )
    parser.add_argument(
        "--promote-at",
        type=float,
        default=0.75,
        help="with --champion: the median score against it that promotes the top candidate",
    )
    parser.add_argument(
        "--own-time",
        action="store_true",
        help="search the own time's exponent too (default: kept)",
    )
    parser.add_argument(
        "--from-defaults",
        action="store_true",
        help="start from the preset's set in MODE_DEFAULTS (needs --preset)",
    )
    args = parser.parse_args()
    presets = (args.preset,) if args.preset else STANDARD_MODES
    if (args.from_defaults or args.champion or args.mixed) and not args.preset:
        raise SystemExit("--from-defaults, --champion and --mixed need --preset")
    pool = None
    if args.mixed:
        pool = type_pool(PRESETS[args.preset].ship_set, args.types, args.hold_out)
    frozen = () if args.own_time else OWN_TIME
    if not 0 <= args.seed <= MAXIMUM_SEED:
        raise SystemExit(f"seed in [0, {MAXIMUM_SEED}]")
    device = torch.device("cuda")
    log_path = pathlib.Path(args.log)

    def sigma_at(generation: int) -> float:
        return max(args.sigma * args.decay**generation, args.minimum_sigma)

    def generator_at(generation: int) -> torch.Generator:
        return torch.Generator().manual_seed(args.seed * MAXIMUM_GENERATIONS + generation)

    promote_at = args.promote_at if args.champion else None
    champion = args.champion
    if log_path.exists() and log_path.stat().st_size:
        start, ranked, tops, champion = resume(log_path, args.plateau_on, promote_at)
        population = next_generation(
            ranked, args.population, args.keep, sigma_at(start), generator_at(start), frozen
        )
    else:
        start, tops, generator = 0, [], generator_at(0)
        first = clamp(
            MODE_DEFAULTS.sets[args.preset] if args.from_defaults else ScriptedParameters()
        )
        population = [first] + [
            perturb(first, args.sigma, generator, frozen) for _ in range(args.population - 1)
        ]
    clock = time.perf_counter()
    with bench_lock(), open(log_path, "a") as log:
        for generation in range(start, min(start + args.generations, MAXIMUM_GENERATIONS)):
            torch.manual_seed(game_seed(args.seed, generation))
            players = REFERENCES
            if champion is not None:
                players = champion_references(champion_of(champion, args.preset, log_path))
            wins, scores = play(
                population,
                args.games,
                args.reference_share,
                device,
                presets,
                args.ship_type,
                pool,
                players,
            )
            anchor = len(population)  # the first reference
            ratings, errors = fit_ratings(wins, anchor, PRIOR_GAMES)
            references = {name: float(ratings[anchor + k]) for k, name in enumerate(players)}
            rows = []
            for index, parameters in enumerate(population):
                row = dict(
                    generation=generation,
                    index=index,
                    rating=float(ratings[index]),
                    stderr=float(errors[index]),
                    scores={name: scores[name][index] for name in players},
                    presets=presets,
                    champion=champion,
                    references=references,
                    parameters=pack(parameters),
                )
                log.write(json.dumps(row) + "\n")
                rows.append(row)
            log.flush()
            order = sorted(range(len(population)), key=lambda index: -float(ratings[index]))
            spread = [float(ratings[index]) for index in order]
            top = order[0]
            summary = ", ".join(
                f"{name} {sum(scores[name][top]) / len(presets):.3f}" for name in players
            )
            elapsed = time.perf_counter() - clock
            left = min(start + args.generations, MAXIMUM_GENERATIONS) - generation - 1

            def eta(margin: float) -> str:
                per = elapsed / (generation - start + 1)
                later = datetime.datetime.now() + datetime.timedelta(seconds=margin * per * left)
                return later.strftime("%H:%M")

            middle = sorted(sum(score) / len(presets) for score in scores[list(players)[0]])
            print(
                f"generation {generation}: top {spread[0]:.0f} +- {float(errors[top]):.0f} "
                f"(candidate {top}), median {spread[len(spread) // 2]:.0f}, last "
                f"{spread[-1]:.0f}; the top's scores {summary} "
                f"(the median's {middle[len(middle) // 2]:.3f}); "
                f"{elapsed:.0f} s; the bound ETA {eta(1.0)} (worst {eta(1.3)})",
                flush=True,
            )
            tops.append(statistic(spread, args.plateau_on))
            ranked = [population[index] for index in order]
            if (label := promoted(rows, promote_at)) is not None:
                print(f"champion {champion} -> {label} (the median candidate beat it)", flush=True)
                champion, tops = label, []
            if plateaued(tops, args.patience, args.min_gain):
                print(f"the {args.plateau_on} rating has plateaued", flush=True)
                break
            population = next_generation(
                ranked,
                args.population,
                args.keep,
                sigma_at(generation + 1),
                generator_at(generation + 1),
                frozen,
            )
    print("top", repr(ranked[0]))


if __name__ == "__main__":
    main()
