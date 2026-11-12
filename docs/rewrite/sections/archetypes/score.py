"""The archetype search's table: every goal of a pair in one row, from `sizes.py` results.

    python docs/rewrite/sections/archetypes/score.py round1/sizes.json [more.json ...] [--sort]

Ship A is the first slot of the card table (the Yongma row, lineup "energy"), ship B the
second (the Jujak row, "turn"). Per candidate:

- A v B: the score of A's side at each size played, and the worst |score - 0.5|.
- AA and BB (the 1v1 mirrors): the mutual head-on kill share, life (s), merges a kill
  (mean / median / p90) and the post-merge choice entropy (bits over same sense, opposite
  sense, extend, turn v extender: 0 is one choice always, 2 is all four equally).
- A v B 1v1: kills m / o / f / t, merges a kill and the choice entropy.
- Pace over every lineup played: life (s) and kills a minute a ship, min-max; the largest
  trade fraction.
- The score against the versus player (mixed / reversed, 5v5): the scripted agent's
  strength on the pair.
- `badness`: the sum of the misses against the goals, 0 when every goal is met (the sort key).
"""

import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
CHOICES = ("same sense", "opposite sense", "extend", "turn v extender")
MIRRORS = {"AA": "energy mirror", "BB": "turn mirror"}


def entropy(row: dict) -> float | None:
    """Bits of the post-merge choice mix, pooled over both types."""
    counts = [0.0] * len(CHOICES)
    for choices in row.get("post_merge_choices", {}).values():
        for i, c in enumerate(CHOICES):
            counts[i] += choices[c]["count"]
    total = sum(counts)
    if total <= 0:
        return None
    return -sum(c / total * math.log2(c / total) for c in counts if c > 0)


def merges(row: dict) -> tuple[float, float | None, float | None]:
    e = row["engagement"]["two turn diameters"]
    return e["merges_per_kill_mean"], e["merges_per_kill_median"], e["merges_per_kill_p90"]


def outside(value: float, low: float, high: float) -> float:
    """How far value is outside [low, high], relative to the band's edge."""
    if value < low:
        return (low - value) / low
    if value > high:
        return (value - high) / high
    return 0.0


def evaluate(result: dict, name: str) -> dict:
    def row(size, lineup):
        return next(
            (
                r
                for r in result["sizes"].get(str(size), [])
                if r["candidate"] == name and r["lineup"] == lineup
            ),
            None,
        )

    sizes = sorted((int(s) for s in result["sizes"]), key=int)
    evt = {s: row(s, "energy v turn") for s in sizes}
    evt = {s: r for s, r in evt.items() if r}
    out = dict(name=name, evt={s: r["score"][0] for s, r in evt.items()})
    out["worst"] = max((abs(v - 0.5) for v in out["evt"].values()), default=math.nan)
    played = list(evt.values())
    for key, lineup in MIRRORS.items():
        r = row(1, lineup)
        if r:
            played.append(r)
            out[key] = dict(
                mutual=r["kill_share"][0],
                life=r["life_seconds"],
                merges=merges(r),
                entropy=entropy(r),
                kills=r["kill_share"],
            )
    one = evt.get(1)
    if one:
        out["AB1"] = dict(
            kills=one["kill_share"],
            merges=merges(one),
            entropy=entropy(one),
            life=one["life_seconds"],
        )
    five = evt.get(5)
    if five:
        out["AB5_kills"] = five["kill_share"]
    out["life"] = [r["life_seconds"] for r in played]
    out["kpm"] = [r["kills_per_minute_per_ship"] for r in played]
    out["trades"] = max((r["trade_fraction"] for r in played), default=math.nan)
    versus = [
        r["score"][0]
        for lineup in ("mixed", "mixed reversed")
        for r in result.get("versus", [])
        if r["candidate"] == name and r["lineup"] == lineup
    ]
    out["versus"] = versus
    # The misses: balance beyond 0.05, mirror mutual head-on beyond 0.3, life outside
    # 45-300 s, kills a minute outside 0.2-1.4, mirror merges a kill outside 2-4.5, choice
    # entropy under 1 bit, trades over 0.1, strength off 0.5 by more than 0.05.
    bad = max(0.0, out["worst"] - 0.05) * 10 if not math.isnan(out["worst"]) else 0.0
    for key in MIRRORS:
        if key in out:
            m = out[key]
            bad += max(0.0, m["mutual"] - 0.3) * 2
            bad += outside(m["merges"][0], 2.0, 4.5)
            if m["entropy"] is not None:
                bad += max(0.0, 1.0 - m["entropy"])
    bad += sum(outside(v, 45.0, 300.0) for v in out["life"])
    bad += sum(outside(v, 0.2, 1.4) for v in out["kpm"])
    bad += max(0.0, out["trades"] - 0.1) * 5 if not math.isnan(out["trades"]) else 0.0
    bad += sum(max(0.0, abs(v - 0.5) - 0.05) * 5 for v in versus)
    out["badness"] = bad
    return out


def fmt_mirror(m: dict | None) -> str:
    if not m:
        return ""
    mean, median, p90 = m["merges"]
    ent = "" if m["entropy"] is None else f"; H {m['entropy']:.2f}"
    return (
        f"{m['mutual']:.2f}, {m['life']:.0f} s; {mean:.1f} / "
        f"{'-' if median is None else round(median)} / {'-' if p90 is None else round(p90)}{ent}"
    )


def table(rows: list[dict]) -> str:
    sizes = sorted({s for r in rows for s in r["evt"]})
    lines = [
        "| Candidate | A v B "
        + " / ".join(f"{s}v{s}" for s in sizes)
        + " | Worst | AA 1v1 mutual, life; merges m / med / p90; H | BB 1v1 | A v B 1v1 kills "
        "m / o / f / t; merges; H | 5v5 kills m / o / f / t | Life s | Kills/min/ship "
        "| Trades | v r9 | Badness |",
        "|---" * 12 + "|",
    ]
    for r in rows:
        ab = r.get("AB1")
        ab_text = ""
        if ab:
            mean, median, _ = ab["merges"]
            ent = "" if ab["entropy"] is None else f"; H {ab['entropy']:.2f}"
            ab_text = (
                " / ".join(f"{v:.2f}" for v in ab["kills"])
                + f"; {mean:.1f} / {'-' if median is None else round(median)}{ent}"
            )
        five = " / ".join(f"{v:.2f}" for v in r["AB5_kills"]) if r.get("AB5_kills") else ""
        lines.append(
            f"| {r['name']} | "
            + " / ".join(f"{r['evt'][s]:.2f}" if s in r["evt"] else "-" for s in sizes)
            + f" | {r['worst']:.2f} | {fmt_mirror(r.get('AA'))} | {fmt_mirror(r.get('BB'))} | "
            f"{ab_text} | {five} | {min(r['life']):.0f}-{max(r['life']):.0f} | "
            f"{min(r['kpm']):.2f}-{max(r['kpm']):.2f} | {r['trades']:.2f} | "
            + " / ".join(f"{v:.2f}" for v in r["versus"])
            + f" | {r['badness']:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    rows = []
    for path in paths:
        result = json.loads((HERE / path).read_text())
        for c in result["candidates"]:
            if any(c["name"] == r["candidate"] for rs in result["sizes"].values() for r in rs):
                rows.append(evaluate(result, c["name"]))
    if "--sort" in sys.argv:
        rows.sort(key=lambda r: r["badness"])
    print(table(rows))


if __name__ == "__main__":
    main()
