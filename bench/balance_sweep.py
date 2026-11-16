"""Ship-balance measurement on the game-loop harness (`bench.game_loop.Loop`): card
candidates and card pairs played by the scripted agent on both sides, every candidate's games
in one compiled graph, scored by kill share.

    PYTHONPATH=src:. python -m bench.balance_sweep sweep --condition deathmatch-5v5 \
        --candidates candidates.json --out out/deathmatch-5v5.jsonl --games 512
    PYTHONPATH=src:. python -m bench.balance_sweep pairwise --condition deathmatch-5v5 \
        --pairs pairs.json --out out/pairs.jsonl --games 512
    PYTHONPATH=src:. python -m bench.balance_sweep check --device cpu   # the switches' proofs
    PYTHONPATH=src:. python -m bench.balance_sweep speed --condition deathmatch-5v5 --fog on

The measurement's settings (`BALANCE`, the author's decisions of 2026-11-16; the game's and
training's own defaults are unchanged):

- no fog and no fields (the scripted agent ignores fields; fog slows the games): `Settings`
  `fog` and `fields`, off here only;
- seeded spawns: game i of every job starts on the spawn layout drawn from (seed, i)
  (`bench.game_loop.seeded_uniforms`), so candidates are compared on the same spawns (common
  random numbers); the ticks' own randomness (respawns, spread, decisions) is from the
  default generator, seeded per chunk;
- the whole episode: the preset's own clock, no opening skipped; deathmatch has no early win;
  elimination and Frontline play their first match to its end (a wipe, the front's
  threshold or the clock), a batch stopping once every first match has ended;
- vision equal for every ship (the cards' own vision; no condition changes it).

The conditions (`CONDITIONS`): deathmatch 1v1, 2v2, 5v5, 10v10 on the density rule's map
(`api.arena_radius_for`: the preset's 5v5 map scaled by sqrt(ships / 5), then the mode's
floor of 3000 m, which the 1v1 and 2v2 maps sit on); elimination and Frontline 5v5 and
10v10 on the preset's own map (as the game plays them since 2026-11-15).

The primary metric is the kill share (`balance_summary`): side A's share of the kills by
hits over every game's whole first match (pooled: A's kills over all kills, the standard
error by game); with it the kills and the deaths (of any cause) a minute for each side, and
the point share (each side's points the other side's deaths of any cause, the game's own
score). The game score (wins and half the draws) is kept as the secondary metric.

Entries. A sweep candidate (`--candidates`, the balance-final sweep's form): {"name",
"card": {"energy": {...}, "turn": {...}}, "energy": {...}, "turn": {...}}, the card factors
on the base Yongma and Jujak rows (`bench.game_loop.variant_types`: a field -> factor, or
{"set": value}) and optional doctrine fields replacing the stored doctrines
(`imugi_doctrine.named`'s form); its jobs: "energy v turn" (`--games`: a Yongma team against
a Jujak team, A the Yongma team) and, with `--mixed-games`, "mixed" and "mixed reversed"
against the stored doctrines on the same card (a doctrine change's strength). A pair
(`--pairs`): {"name", "a": card, "b": card}, a card {"ship": a key of the set (default
yongma for a, jujak for b), "factors": {...}, "doctrine": {...}}: a team of card A against a
team of card B, each card flying its ship's stored doctrine (with the card's doctrine fields),
A the card-A team. Any card against any card, so sampled cards can play a reference.

The agent: `CONTROLLER`, Imugi with the mode's parameter set on both sides. Another
controller is a factory (the loop) -> agent (`bench.game_loop.Loop`'s `controller`); to
switch, change that one line.

Batches: a chunk of `--variants` entries shares one card table (`Loop.set_table`, data: no
recompile) and plays as many batches as its games fill; each entry's line is appended to
`--out` as its chunk ends (an entry already there is skipped, so a stopped sweep resumes).
`worker.py`'s persistent form: `bench.balance_worker`.
"""

import argparse
import concurrent.futures
import dataclasses
import json
import math
import multiprocessing
import os
import pathlib
import time

import torch

from bench.game_loop import (
    FILLER,
    SHIP_SET,
    Group,
    Job,
    Loop,
    Player,
    batches,
    clock,
    imugi_controller,
    job_summary,
    run,
)
from bnb.controllers.imugi_defaults import MODE_DEFAULTS
from bnb.controllers.imugi_doctrine import from_named, named, stored
from bnb.controllers.imugi_parameters import pack
from bnb.game import api, ships
from bnb.game.handling import characteristics as characteristics_module
from bnb.game.ship_types import ShipTypes

# The agent both sides fly: a factory (the loop) -> agent. The one line to change to switch
# the controller (feat/lethality-sticky's card-derived agent, once it lands).
CONTROLLER = imugi_controller


@dataclasses.dataclass(frozen=True)
class Condition:
    """A balance condition: the mode, the team size and the map."""

    mode: str
    ships: int
    density_map: bool  # the density rule's map (`api.arena_radius_for`), else the preset's


CONDITIONS = {
    "deathmatch-1v1": Condition("deathmatch", 1, True),
    "deathmatch-2v2": Condition("deathmatch", 2, True),
    "deathmatch-5v5": Condition("deathmatch", 5, True),
    "deathmatch-10v10": Condition("deathmatch", 10, True),
    "elimination-5v5": Condition("elimination", 5, False),
    "elimination-10v10": Condition("elimination", 10, False),
    "frontline-5v5": Condition("frontline", 5, False),
    "frontline-10v10": Condition("frontline", 10, False),
}


@dataclasses.dataclass(frozen=True)
class Settings:
    """A measurement's switches (`BALANCE` the balance measurement's)."""

    fog: bool = False
    fields: bool = False
    spawn_seed: int | None = 0  # None: unseeded spawns
    seconds: float | None = None  # None: the preset's whole clock (a check may shorten it)


BALANCE = Settings()
KEPT = (  # what a job's line keeps of `bench.game_loop.job_summary` and `balance_summary`
    "games",
    "kill_share_a",
    "point_share_a",
    "kills_per_minute",
    "deaths_per_minute",
    "match_seconds",
    "score",
    "kill_score",
    "wins",
    "draws",
    "kills_per_game",
    "kill_share",
    "kill_share_by_pair",
    "passes_per_kill",
    "no_merge_kills",
    "trade_fraction",
)


def preset_of(condition: Condition) -> str:
    """The condition's preset (`bench.game_loop.MODES`)."""
    return f"real_scale_{condition.mode}"


def mode_parameters(condition: Condition) -> tuple[float, ...]:
    """Imugi's parameter set for the condition's mode, packed."""
    return tuple(pack(MODE_DEFAULTS.sets[preset_of(condition)]))


def balance_loop(
    condition: Condition,
    environments: int,
    variants: int,
    settings: Settings = BALANCE,
    device="cuda",
    compiled: bool = True,
    controller=None,
) -> Loop:
    """The compiled loop of one condition under the settings."""
    seconds = settings.seconds or api.PRESETS[preset_of(condition)].episode_seconds
    return Loop(
        environments,
        variants,
        seconds,
        device=device,
        compiled=compiled,
        ships_per_team=condition.ships,
        fog=settings.fog,
        mode=condition.mode,
        fields=settings.fields,
        fixed_map=not condition.density_map,
        controller=controller or CONTROLLER,
        spawn_seed=settings.spawn_seed,
        stop_when_finished=True,
    )


def doctrines_of(key: str) -> dict:
    """A ship's stored doctrine (`imugi_doctrines.json`) in `named` form."""
    return named(stored()[SHIP_SET][key], figures=6)


def card(entry: dict, default_ship: str) -> dict:
    """A card spec with its defaults: {"ship", "factors", "doctrine"}."""
    return dict(
        ship=entry.get("ship", default_ship),
        factors=entry.get("factors", {}),
        doctrine=entry.get("doctrine", {}),
    )


def as_pair(entry: dict) -> dict:
    """A sweep candidate or a pair, as a pair: {"name", "a", "b", "mixed"} (mixed: the entry
    is a sweep candidate, whose mixed jobs play its doctrines on the same card)."""
    if "a" in entry or "b" in entry:
        return dict(
            name=entry["name"],
            a=card(entry.get("a", {}), "yongma"),
            b=card(entry.get("b", {}), "jujak"),
            mixed=False,
        )
    factors = entry.get("card", {})
    return dict(
        name=entry["name"],
        a=dict(ship="yongma", factors=factors.get("energy", {}), doctrine=entry.get("energy", {})),
        b=dict(ship="jujak", factors=factors.get("turn", {}), doctrine=entry.get("turn", {})),
        mixed=True,
    )


def card_table(a_cards: list[dict], b_cards: list[dict], device="cpu") -> ShipTypes:
    """The card table (2K,): the K A cards' rows, then the K B cards' (each its ship's row of
    the set, its factors multiplied or set)."""
    if len(a_cards) != len(b_cards):
        raise ValueError("as many A cards as B cards")
    keys = [ship.key for ship in ships.load(SHIP_SET)]
    full = api.ship_types(ship_set=SHIP_SET)
    rows = []
    for spec in [*a_cards, *b_cards]:
        index = keys.index(spec["ship"])
        row = {f.name: getattr(full, f.name)[index].clone() for f in dataclasses.fields(full)}
        for name, factor in spec["factors"].items():
            if isinstance(factor, dict):  # {"set": value}: an absolute value
                row[name] = torch.full_like(row[name], factor["set"])
            else:
                row[name] = row[name] * factor
        rows.append(row)
    return ShipTypes(
        **{
            f.name: torch.stack([row[f.name] for row in rows]).to(device)
            for f in dataclasses.fields(full)
        }
    )


def player_of(pair: dict, parameters: tuple[float, ...]) -> Player:
    """The pair's player: card A's ship's stored doctrine on the kind-0 rows, card B's on the
    kind-1 rows, each with its card's doctrine fields."""
    return Player(
        pair["name"],
        from_named(doctrines_of(pair["a"]["ship"]) | pair["a"]["doctrine"]),
        from_named(doctrines_of(pair["b"]["ship"]) | pair["b"]["doctrine"]),
        parameters=parameters,
    )


def reference_player(pair: dict, parameters: tuple[float, ...]) -> Player:
    """The stored doctrines on the pair's ships (a mixed job's other side)."""
    return player_of(dict(pair, a=dict(pair["a"], doctrine={}), b=dict(pair["b"], doctrine={}),
                          name="stored"), parameters)  # fmt: skip


def chunks(pairs: list[dict], variants: int):
    """The pairs cut into chunks of `variants` (one card table each): (first index, the
    chunk's pairs, its A cards, its B cards), the table padded with the base cards."""
    for first in range(0, len(pairs), variants):
        part = pairs[first : first + variants]
        padding = [dict(ship=s, factors={}, doctrine={}) for s in ("yongma", "jujak")]
        a_cards = [p["a"] for p in part] + [padding[0]] * (variants - len(part))
        b_cards = [p["b"] for p in part] + [padding[1]] * (variants - len(part))
        yield first, part, a_cards, b_cards


def jobs_of(part: list[dict], games: int, mixed_games: int, parameters) -> list[Job]:
    """A chunk's jobs, in order: each pair's "energy v turn" (A the card-A team), then for
    a sweep candidate with `mixed_games`, its "mixed" and "mixed reversed" against the stored
    doctrines."""
    jobs = []
    for variant, pair in enumerate(part):
        player = player_of(pair, parameters)
        jobs.append(Job(variant, "energy v turn", player, player, games))
        if pair["mixed"] and mixed_games > 0:
            reference = reference_player(pair, parameters)
            jobs.append(Job(variant, "mixed", player, reference, mixed_games // 2))
            jobs.append(Job(variant, "mixed reversed", player, reference, mixed_games // 2))
    return jobs


def ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> tuple[float, float]:
    """A pooled ratio over games (sum over sum) and its standard error by game (the ratio
    estimator's: the residuals' spread)."""
    total = float(denominator.sum())
    if total <= 0:
        return 0.5, 0.5
    value = float(numerator.sum()) / total
    residual = numerator - value * denominator
    n = numerator.shape[0]
    error = math.sqrt(float((residual * residual).sum()) * n / max(n - 1, 1)) / total
    return value, error


def balance_summary(out: dict, tick_length: float) -> dict:
    """The balance numbers of a job's games over their whole first matches: A's kill share
    (kills by hits) and point share (the other side's deaths of any cause), pooled with the
    standard error by game; kills and deaths a minute per side (A, B: a team's, per game
    minute); the mean match length (s)."""
    kills, deaths = out["match_kills"], out["match_deaths"]
    minutes = float(out["match_ticks"].sum()) * tick_length / 60.0
    points = deaths.flip(-1)
    return dict(
        kill_share_a=ratio(kills[:, 0], kills.sum(1)),
        point_share_a=ratio(points[:, 0], points.sum(1)),
        kills_per_minute=(kills.sum(0) / max(minutes, 1e-9)).tolist(),
        deaths_per_minute=(deaths.sum(0) / max(minutes, 1e-9)).tolist(),
        match_seconds=float(out["match_ticks"].float().mean()) * tick_length,
    )


def _prewarm_one(table: ShipTypes, rules, device: str) -> None:
    """One chunk's card characteristics into the disk cache under the loop's device key (the
    entry `Loop.set_table` then loads), measured on this process's CPU."""
    torch.set_num_threads(1)
    key = list(characteristics_module._key(table, rules))
    key[1] = device
    path = characteristics_module._disk_path(tuple(key))
    if path.exists():
        return
    measured = characteristics_module.measure_characteristics(table, rules)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.stem}.{os.getpid()}.partial")
    torch.save({f.name: getattr(measured, f.name) for f in dataclasses.fields(measured)}, partial)
    partial.replace(path)


def prewarm(loop: Loop, pairs: list[dict], variants: int, processes: int = 12) -> None:
    """Measure every chunk's card characteristics before the GPU is taken: `set_table`
    measures them on the CPU (minutes for a large store on a slow CPU), so a job would
    otherwise hold the GPU lock with the GPU idle."""
    device = str(loop.types.max_energy.device)
    tables = [card_table(a, b) for _, _, a, b in chunks(pairs, variants)]
    if not tables:
        return
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        min(processes, len(tables)), mp_context=context
    ) as pool:
        for future in [pool.submit(_prewarm_one, t, loop.rules, device) for t in tables]:
            future.result()


def warm(loop: Loop, parameters: tuple[float, ...], variants: int, ticks: int = 3) -> None:
    """Compile the tick a job plays, outside any lock: one tick of a filler game, then a few
    ticks of a job-like batch (a non-base card table, the three lineups with a candidate and
    the stored doctrines, seeded spawns), then the base table back."""
    loop._tick(*loop.start([Group(0, "mixed", FILLER, FILLER, loop.B)]))
    pair = as_pair({"name": "warm", "card": {"energy": {"max_shield": 1.0}}})
    entries = [dict(pair, name=f"warm{v}") for v in range(variants)]
    _, part, a_cards, b_cards = next(chunks(entries, variants))
    loop.set_table(card_table(a_cards, b_cards, loop.device))
    share = max(2, loop.B // (3 * variants)) // 2 * 2
    jobs = jobs_of(part, share, share, parameters)
    groups = [group for _, group in batches(jobs, loop.B)[0]]
    arguments = loop.start(groups)
    for _ in range(ticks):
        arguments = loop._tick(*arguments)
    torch.cuda.synchronize(loop.device) if loop.device.type == "cuda" else None
    base = as_pair({"name": "base"})
    _, _, a_cards, b_cards = next(chunks([base] * variants, variants))
    loop.set_table(card_table(a_cards, b_cards, loop.device))


def sweep(
    loop: Loop,
    entries: list[dict],
    out: pathlib.Path,
    games: int,
    mixed_games: int,
    seed: int,
    variants: int,
    condition_name: str,
    parameters: tuple[float, ...],
    log=print,
) -> None:
    """Every entry's jobs on a built loop, a chunk of `variants` entries per card table;
    each entry's line is appended to `out` as its chunk ends (`bench.balance_worker` calls
    this on its one compiled loop). seed: the spawn seed (`Loop.spawn_seed`; game i of every
    job on the same spawns) and the ticks' randomness per chunk."""
    pairs = [as_pair(entry) for entry in entries]
    by_name = {entry["name"]: entry for entry in entries}
    seconds = loop.ticks * loop.rules.tick_length
    loop.spawn_seed = seed
    started = time.time()
    for first, part, a_cards, b_cards in chunks(pairs, variants):
        loop.set_table(card_table(a_cards, b_cards, loop.device))
        jobs = jobs_of(part, games, mixed_games, parameters)
        torch.manual_seed(seed + first)
        outs = run(loop, jobs, log=log)
        rows = [
            job_summary(job, o, seconds) | balance_summary(o, loop.rules.tick_length)
            for job, o in zip(jobs, outs, strict=True)
        ]
        with out.open("a") as handle:
            for number, pair in enumerate(part):
                mine = [r for r in rows if r["variant"] == number]
                line = dict(
                    name=pair["name"],
                    condition=condition_name,
                    mode=loop.mode,
                    ships=loop.S,
                    seconds=seconds,
                    fog=loop.fog,
                    fields=loop.game.num_fields > 0,
                    seed=seed,
                    entry=by_name[pair["name"]],
                    jobs={r["lineup"]: {k: r[k] for k in KEPT if k in r} for r in mine},
                )
                handle.write(json.dumps(line) + "\n")
                head = mine[0]
                share, error = head["kill_share_a"]
                log(
                    f"{pair['name']}: A kill share {share:.3f} ± {error:.3f}, kills/min A "
                    f"{head['kills_per_minute'][0]:.2f} B {head['kills_per_minute'][1]:.2f}; "
                    f"score {head['score'][0]:.3f} ± {head['score'][1]:.3f}"
                )
        elapsed = time.time() - started
        finished = first + len(part)
        left = elapsed / finished * (len(pairs) - finished)
        log(f"{finished}/{len(pairs)} entries; done about {clock(left)} KST")


def check(args) -> None:
    """The switches' proofs on a small loop: fog and fields off in the state and the
    observation, and the seeded spawns repeating (the same seed and game index: the same
    layout; another seed: another)."""
    condition = CONDITIONS[args.condition]
    settings = dataclasses.replace(BALANCE, seconds=args.seconds)
    loop = balance_loop(condition, args.environments, 1, settings, args.device, compiled=False)
    groups = [Group(0, "energy v turn", FILLER, FILLER, loop.B)]
    state, observation, _, _ = loop.start(groups)
    shown = api.display(state)
    from bnb.contracts.obs import GLOBAL, SHIP

    alive = observation.ships[..., SHIP["alive"]] > 0.5
    seen = observation.ships[..., SHIP["visible"]] > 0.5
    print(
        f"{args.condition}: fogged environments {int(state.match.fogged.sum())}/{loop.B}, "
        f"fogged global column max {float(observation.globals[..., GLOBAL['fogged']].max())}, "
        f"fields in the rules {loop.game.num_fields}, present fields "
        f"{int(state.fields.present.sum())}, living rows seen "
        f"{int((alive & seen).sum())}/{int(alive.sum())}"
    )
    first = torch.stack([shown["ships.x"], shown["ships.y"]], -1)
    again = torch.stack([api.display(loop.start(groups)[0])[k] for k in ("ships.x", "ships.y")], -1)
    loop.spawn_seed = settings.spawn_seed + 1
    other = torch.stack([api.display(loop.start(groups)[0])[k] for k in ("ships.x", "ships.y")], -1)
    shifted = [Group(0, "energy v turn", FILLER, FILLER, loop.B, first_game=2)]
    loop.spawn_seed = settings.spawn_seed
    moved = torch.stack(
        [api.display(loop.start(shifted)[0])[k] for k in ("ships.x", "ships.y")], -1
    )
    print(
        f"spawns, seed {settings.spawn_seed}: repeat exactly {bool(torch.equal(first, again))}; "
        f"seed {settings.spawn_seed + 1} differs {not torch.allclose(first, other)}; "
        f"game i+2 at position i matches game i+2: {bool(torch.equal(moved[:-2], first[2:]))}"
    )


def speed(args) -> None:
    """Games per second of one condition with fog and fields as given: compile and warm
    outside the lock, then time `--batches` batches of `--seconds` under it."""
    condition = CONDITIONS[args.condition]
    settings = Settings(fog=args.fog == "on", fields=args.fields == "on", seconds=args.seconds)
    begun = time.time()
    loop = balance_loop(condition, args.environments, args.variants, settings, args.device)
    parameters = mode_parameters(condition)
    warm(loop, parameters, args.variants)
    print(f"compiled and warm in {time.time() - begun:.0f} s", flush=True)
    pair = as_pair({"name": "base"})
    jobs = jobs_of([pair], loop.B, 0, parameters)
    from bench.balance_worker import held

    with held(args.lock):
        timings = []
        for _ in range(args.batches):
            started = time.time()
            out = loop.play([group for _, group in batches(jobs, loop.B)[0]])
            timings.append(time.time() - started)
    best = min(timings)
    kills = float(out["match_kills"].sum())
    print(
        f"{args.condition} fog {args.fog} fields {args.fields}: B={loop.B}, "
        f"{loop.ticks} ticks ({loop.ticks * loop.rules.tick_length:.0f} s): batches "
        f"{', '.join(f'{t:.1f}' for t in timings)} s; best {loop.B / best:.2f} games/s, "
        f"{loop.B * loop.ticks / best / 1e3:.0f}k env-ticks/s; kills {kills / loop.B:.1f} a "
        f"game; peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
        flush=True,
    )


def main() -> None:
    """The command line: a one-shot sweep or pairwise run, the checks and the speed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("sweep", "pairwise"):
        one = sub.add_parser(name)
        one.add_argument("--condition", choices=CONDITIONS, required=True)
        one.add_argument("--candidates" if name == "sweep" else "--pairs", type=pathlib.Path,
                         required=True, dest="entries")  # fmt: skip
        one.add_argument("--out", type=pathlib.Path, required=True)
        one.add_argument("--games", type=int, default=512)
        one.add_argument("--mixed-games", type=int, default=0)
        one.add_argument("--seed", type=int, default=0)
        one.add_argument("--environments", type=int, default=4096)
        one.add_argument("--variants", type=int, default=32, help="entries a card table holds")
        one.add_argument("--device", default="cuda")
        one.add_argument("--eager", action="store_true", help="no torch.compile (a CPU check)")
        one.add_argument("--seconds", type=float, default=None, help="a check's short clock")
    one = sub.add_parser("check")
    one.add_argument("--condition", choices=CONDITIONS, default="deathmatch-5v5")
    one.add_argument("--environments", type=int, default=8)
    one.add_argument("--seconds", type=float, default=2.0)
    one.add_argument("--device", default="cpu")
    one = sub.add_parser("speed")
    one.add_argument("--condition", choices=CONDITIONS, default="deathmatch-5v5")
    one.add_argument("--fog", choices=("on", "off"), default="off")
    one.add_argument("--fields", choices=("on", "off"), default="off")
    one.add_argument("--environments", type=int, default=2048)
    one.add_argument("--variants", type=int, default=32)
    one.add_argument("--seconds", type=float, default=60.0)
    one.add_argument("--batches", type=int, default=2)
    one.add_argument("--device", default="cuda")
    one.add_argument("--lock", type=pathlib.Path, default=None)
    args = parser.parse_args()
    if args.command == "check":
        return check(args)
    if args.command == "speed":
        return speed(args)
    condition = CONDITIONS[args.condition]
    entries = json.loads(args.entries.read_text())
    done = set()
    if args.out.exists():
        done = {json.loads(line)["name"] for line in args.out.read_text().splitlines()}
    entries = [e for e in entries if e["name"] not in done]
    if not entries:
        print("every entry is done", flush=True)
        return
    settings = dataclasses.replace(BALANCE, seconds=args.seconds)
    loop = balance_loop(
        condition, args.environments, args.variants, settings, args.device, not args.eager
    )
    parameters = mode_parameters(condition)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sweep(
        loop, entries, args.out, args.games, args.mixed_games, args.seed, args.variants,
        args.condition, parameters,
    )  # fmt: skip


if __name__ == "__main__":
    main()
