"""The final freeze check's statistics (docs/rewrite/sections/freeze-final/results.md): each
condition's measures with their standard errors, from `bench.mode_dynamics` records (one
JSON line per batch of each environment's first game).

Definitions (a "game" is one environment's first game; ticks are 1/30 s):

- **deathmatch** (recorded with tickets off, so every game plays its 600 s clock): each
  game is replayed with the frozen tickets T(N) (`phase.tickets`, 4 N^(4/3) to the nearest
  5): it ends at the tick a team's points reach T, else on the clock, won by the kill lead
  (a draw when level). A point is a death of the other team, whatever its cause. Kills a
  minute: both teams' points up to the game's end over its length in minutes, per game,
  then averaged.
- **elimination**: the game's end tick and whether it ended before the clock (a wipe);
  boundary deaths, the pooled share of deaths with no shooter (a ratio of sums, its SE
  clustered by game); the last death, the death(s) on the game's last death tick, counted
  as the boundary's when any of them had no shooter (games ended by a wipe only).
- **frontline**: ended by capture, the front reached its threshold before the clock;
  P(reversal) as `bench.front_reversal` (pooled over move pairs, SE clustered by game); the
  first capturer's share, among games with a capture and a winner (by the threshold or by
  the front's sign at the clock), of those its first capturer won.
- **energy against turn** (team 0 and team 1 of different types, `--lineup
  "yongma|jujak;jujak|yongma"`): the Yongma team's score, a win 1, a draw 1/2, by the
  mode's result (deathmatch: the replayed ticket result above).
- **kills by aspect** (`bnb.contracts.ASPECTS`, the killing hit's angle off the victim's
  nose: under 10 degrees a mutual head-on, 10-45 offset head-on, 45-135 flank, beyond the
  tail): pooled over the batches' first games, kills with a shooter only; SE binomial
  (kills within a game are not independent, so these are a little optimistic).

A share's SE is binomial, sqrt(p (1 - p) / n), over games; a mean's is sd / sqrt(n); a
median's is from a game bootstrap (400 resamples, fixed seed).
"""

import json
import math
import pathlib
import random
import sys

from bnb.contracts import ASPECTS
from bnb.game import api
from bnb.game.modes import phase

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parents[3]))
from bench.front_reversal import counts  # noqa: E402

YONGMA = 0  # the real-scale set's type index of Yongma (Jujak is 1)


def load(path: pathlib.Path, label: str) -> tuple[list[dict], list[dict]]:
    """(games, batch entries) of one label."""
    entries = []
    if path.exists():
        for line in path.read_text().splitlines():
            entry = json.loads(line)
            if entry["label"] == label:
                entries.append(entry)
    return [game for entry in entries for game in entry["games"]], entries


def share(hits: float, total: int) -> tuple[float, float]:
    """A share and its binomial SE."""
    if total == 0:
        return (math.nan, math.nan)
    p = hits / total
    return (p, math.sqrt(max(p * (1 - p), 0.0) / total))


def mean(values: list[float]) -> tuple[float, float]:
    """A mean and its SE."""
    n = len(values)
    if n < 2:
        return (values[0] if values else math.nan, math.nan)
    m = sum(values) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    return (m, sd / math.sqrt(n))


def quantile(values: list[float], q: float) -> float:
    """The q-quantile (linear interpolation)."""
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def quantile_se(values: list[float], q: float, resamples: int = 400) -> tuple[float, float]:
    """A quantile and its bootstrap SE."""
    draws = random.Random(7)
    estimate = quantile(values, q)
    boots = [
        quantile([values[draws.randrange(len(values))] for _ in values], q)
        for _ in range(resamples)
    ]
    return (estimate, math.sqrt(sum((b - estimate) ** 2 for b in boots) / resamples))


def ratio(numerators: list[float], denominators: list[float]) -> tuple[float, float]:
    """A pooled ratio sum(a) / sum(b) and its game-clustered SE (linearised)."""
    total = sum(denominators)
    n = len(numerators)
    if total == 0 or n < 2:
        return (math.nan, math.nan)
    r = sum(numerators) / total
    residual = sum((a - r * b) ** 2 for a, b in zip(numerators, denominators, strict=True))
    return (r, math.sqrt(residual * n / (n - 1)) / total)


def tickets_of(ships: int) -> int:
    """The frozen deathmatch tickets for `ships` enemy ships."""
    rules = api.rules_of(api.PRESETS["real_scale_deathmatch"])
    return int(phase.tickets(rules.pools[0].mode, ships))


def deathmatch_game(game: dict, ships: int, episode_ticks: int) -> dict:
    """A tickets-off game replayed with the frozen tickets: its end tick, whether tickets
    ended it, team 0's result and both teams' points by its end."""
    limit = tickets_of(ships)
    points = [0, 0]
    for tick, team in game["events"]:
        points[team] += 1
        if points[team] >= limit:
            return dict(
                end=tick + 1, by_tickets=True, result=1 if team == 0 else -1, points=sum(points)
            )
    lead = points[0] - points[1]
    return dict(
        end=episode_ticks, by_tickets=False, result=(lead > 0) - (lead < 0), points=sum(points)
    )


def last_death_by_boundary(game: dict) -> bool | None:
    """Whether the game's last death tick has a boundary death (None: no death)."""
    ticks = [t for t in game["death_tick"] if t >= 0]
    if not ticks:
        return None
    last = max(ticks)
    return any(
        t == last and boundary
        for t, boundary in zip(game["death_tick"], game["death_boundary"], strict=True)
    )


def yongma_score(game: dict, result: int) -> float | None:
    """The Yongma team's score in an energy-against-turn game (None: not one)."""
    ships = len(game["ship_type"]) // 2
    team0, team1 = set(game["ship_type"][:ships]), set(game["ship_type"][ships:])
    if len(team0) != 1 or len(team1) != 1 or team0 == team1:
        return None
    yongma_team0 = team0 == {YONGMA}
    score = 0.5 if result == 0 else float((result > 0) == yongma_team0)
    return score


def aspects(entries: list[dict]) -> dict:
    """Kills by aspect class (shares and SEs) and by killer and victim type, pooled."""
    totals = [sum(entry["kill_aspect"][k] for entry in entries) for k in range(len(ASPECTS))]
    kills = sum(totals)
    by_type = [[sum(e["kills_by_type"][i][j] for e in entries) for j in range(2)] for i in range(2)]
    return dict(
        kills=kills,
        **{name: share(count, kills) for name, count in zip(ASPECTS, totals, strict=True)},
        kills_by_type=by_type,
    )


def measures(kind: str, games: list[dict], entries: list[dict]) -> dict:
    """The condition's measures (value, SE) and its game count."""
    if not games:
        return dict(games=0)
    tick = entries[0]["tick_length"]
    episode = entries[0]["episode_ticks"]
    ships = entries[0]["ships"]
    out = dict(games=len(games), ships=ships, radius=entries[0]["radius"])
    scores = []
    if kind == "deathmatch":
        replayed = [deathmatch_game(game, ships, episode) for game in games]
        lengths = [r["end"] * tick for r in replayed]
        out["tickets"] = tickets_of(ships)
        out["by_tickets"] = share(sum(r["by_tickets"] for r in replayed), len(games))
        out["median_s"] = quantile_se(lengths, 0.5)
        out["p80_s"] = quantile_se(lengths, 0.8)
        out["kills_per_minute"] = mean([r["points"] / (r["end"] * tick / 60.0) for r in replayed])
        scores = [yongma_score(g, r["result"]) for g, r in zip(games, replayed, strict=True)]
    elif kind == "elimination":
        lengths = [game["end_tick"] * tick for game in games]
        out["median_s"] = quantile_se(lengths, 0.5)
        out["p80_s"] = quantile_se(lengths, 0.8)
        out["p90_s"] = quantile_se(lengths, 0.9)
        out["on_at_clock"] = share(sum(not g["by_rule"] for g in games), len(games))
        out["boundary_deaths"] = ratio(
            [g["boundary_deaths"] for g in games], [g["deaths"] for g in games]
        )
        wiped = [last_death_by_boundary(g) for g in games if g["by_rule"]]
        wiped = [w for w in wiped if w is not None]
        out["last_death_boundary"] = share(sum(wiped), len(wiped))
        scores = [yongma_score(g, g["result"]) for g in games]
    elif kind == "frontline":
        lengths = [game["end_tick"] * tick for game in games]
        out["by_capture"] = share(sum(g["by_rule"] for g in games), len(games))
        out["median_s"] = quantile_se(lengths, 0.5)
        pairs = [counts(g) for g in games]
        out["p_reversal"] = ratio([r for r, _ in pairs], [n for _, n in pairs])
        decided = [g for g in games if g["events"] and g["result"] != 0]
        first_won = [(g["events"][0][1] == 0) == (g["result"] > 0) for g in decided]
        out["first_capturer_wins"] = share(sum(first_won), len(first_won))
        out["moves_per_game"] = mean([len(g["events"]) for g in games])
        scores = [yongma_score(g, g["result"]) for g in games]
    scores = [s for s in scores if s is not None]
    if scores:
        out["yongma_score"] = mean(scores)
    out["aspects"] = aspects(entries)
    return out


# The adaptive stop: every listed measure's SE at or under its target (a look per batch).
TARGETS = {
    "deathmatch": {"by_tickets": 0.03},
    "elimination": {"last_death_boundary": 0.03, "boundary_deaths": 0.03, "on_at_clock": 0.03},
    "frontline": {"by_capture": 0.03, "p_reversal": 0.03, "first_capturer_wins": 0.03},
}
ENERGY_TURN_TARGET = 0.02  # the Yongma team's score


def done(kind: str, result: dict, energy_turn: bool) -> bool:
    """Whether a condition's measures meet their SE targets."""
    targets = {"yongma_score": ENERGY_TURN_TARGET} if energy_turn else TARGETS[kind]
    for name, target in targets.items():
        value = result.get(name)
        if value is None or not math.isfinite(value[1]) or value[1] > target:
            return False
    return True


def _f(value: tuple[float, float], scale: float = 1.0, digits: int = 0) -> str:
    """'value ± SE' at a scale (100 for percentages)."""
    v, s = value
    return f"{v * scale:.{digits}f} ± {s * scale:.{digits}f}"


def _clock(value: tuple[float, float]) -> str:
    """A time in s as m:ss ± s."""
    v, s = value
    return f"{int(v // 60)}:{int(round(v % 60)):02d} ± {s:.0f} s"


def main() -> None:
    """Print the results' tables (Markdown) from data/modes.jsonl."""
    path = HERE / "data" / "modes.jsonl"
    labels = []
    for line in path.read_text().splitlines():
        label = json.loads(line)["label"]
        if label not in labels:
            labels.append(label)
    kind_of = {"dm": "deathmatch", "elim": "elimination", "front": "frontline"}
    rows = {label: measures(kind_of[label.split("-")[0]], *load(path, label)) for label in labels}
    print("| deathmatch | games | tickets | ended by tickets (%) | median | p80 | kills/min |")
    print("|---|---|---|---|---|---|---|")
    for label, r in rows.items():
        if label.startswith("dm-") and "-et-" not in label:
            print(
                f"| {label[3:]} | {r['games']} | {r['tickets']} | {_f(r['by_tickets'], 100, 1)} "
                f"| {_clock(r['median_s'])} | {_clock(r['p80_s'])} "
                f"| {_f(r['kills_per_minute'], 1, 2)} |"
            )
    print()
    print(
        "| elimination | games | median | p80 | p90 | on at the clock (%) "
        "| deaths by the boundary (%) | last death by the boundary (%) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for label, r in rows.items():
        if label.startswith("elim-") and "-et-" not in label:
            print(
                f"| {label[5:]} | {r['games']} | {_clock(r['median_s'])} | {_clock(r['p80_s'])} "
                f"| {_clock(r['p90_s'])} | {_f(r['on_at_clock'], 100, 1)} "
                f"| {_f(r['boundary_deaths'], 100, 1)} | {_f(r['last_death_boundary'], 100, 1)} |"
            )
    print()
    print(
        "| frontline | games | ended by capture (%) | median | P(reversal) "
        "| first capturer wins (%) | moves a game |"
    )
    print("|---|---|---|---|---|---|---|")
    for label, r in rows.items():
        if label.startswith("front-") and "-et-" not in label:
            print(
                f"| {label[6:]} | {r['games']} | {_f(r['by_capture'], 100, 1)} "
                f"| {_clock(r['median_s'])} | {_f(r['p_reversal'], 1, 3)} "
                f"| {_f(r['first_capturer_wins'], 100, 1)} | {_f(r['moves_per_game'], 1, 2)} |"
            )
    print()
    print("| energy (Yongma) team against turn (Jujak) team | games | Yongma team's score (%) |")
    print("|---|---|---|")
    for label, r in rows.items():
        if "-et-" in label:
            print(f"| {label} | {r['games']} | {_f(r['yongma_score'], 100, 1)} |")
    print()
    print(
        "| kills by aspect (mixed lineups) | kills | mutual head-on | offset head-on | flank "
        "| tail | Y on Y / Y on J / J on Y / J on J |"
    )
    print("|---|---|---|---|---|---|---|")
    for label, r in rows.items():
        if "-et-" not in label:
            a = r["aspects"]
            by = a["kills_by_type"]
            print(
                f"| {label} | {a['kills']} | "
                + " | ".join(_f(a[name], 100, 1) for name in ASPECTS)
                + f" | {by[0][0]} / {by[0][1]} / {by[1][0]} / {by[1][1]} |"
            )


if __name__ == "__main__":
    main()
