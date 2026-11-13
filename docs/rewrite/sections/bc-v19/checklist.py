"""The run checklist: read a run's metrics.jsonl (and the live Elo table, when given) and
print one line per check, `ok` or `FLAG`, in a fixed order, so a watcher (a script or an
agent) can tell at a glance whether anything looks wrong. Thresholds are deliberately
loose: a FLAG asks for a look, not a stop.

    uv run --no-sync python checklist.py METRICS.jsonl [LATEST_ELO.txt]
"""

import json
import math
import re
import statistics
import sys


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def series(rows: list[dict], key: str) -> list[float]:
    return [row[key] for row in rows if finite(row.get(key))]


def keys(rows: list[dict], pattern: str) -> list[str]:
    seen = set()
    for row in rows[-5:]:
        seen.update(key for key in row if re.search(pattern, key))
    return sorted(seen)


def main() -> None:
    rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
    out: list[tuple[str, str, str]] = []

    def check(name: str, bad: bool, detail: str) -> None:
        out.append(("FLAG" if bad else "ok", name, detail))

    if not rows:
        print("FLAG metrics: no rows yet")
        return
    last, updates = rows[-1], len(rows)
    out.append(("info", "progress", f"{updates} updates, steps {last.get('steps', '?')}"))

    # 1. alerts and warnings the run raises itself
    raised = [key.removeprefix("alerts/") for key in keys(rows, r"^alerts/") if last.get(key)]
    check("alerts", bool(raised), ", ".join(raised) or "none")

    # 2. non-finite values anywhere in the last row
    nonfinite = [key for key, value in last.items() if isinstance(value, float) and not math.isfinite(value)]
    nonfinite_counts = [key for key in keys(rows, r"nonfinite") if finite(last.get(key)) and last[key] > 0]
    check("nonfinite", bool(nonfinite_counts), ", ".join(nonfinite_counts) or "none")
    out.append(("info", "nan series (often a quantity with no data, e.g. a cosine of a zero gradient)", str(len(nonfinite))))

    # 3. input envelope and code support
    envelope = {k.split("/")[-1]: last[k] for k in keys(rows, r"^input/envelope_violation_share/") if finite(last.get(k))}
    bad = {k: v for k, v in envelope.items() if v > 0.01}
    check("input envelope (share > 1%)", bool(bad), str({k: round(v, 4) for k, v in bad.items()}) or "none")
    clips = {k: last[k] for k in keys(rows, r"support_clip") if finite(last.get(k)) and last[k] > 0.01}
    check("code support clipping (> 1%)", bool(clips), str({k: round(v, 4) for k, v in clips.items()}) or "none")

    # 4. the training objective: BC KL (falling, not rising) or PPO's KL and entropy
    for key in ("bc/kl", "policy/kl", "kl", "bc/entropy/student", "policy/entropy"):
        values = series(rows, key)
        if len(values) >= 6:
            recent, earlier = statistics.mean(values[-3:]), min(values[: len(values) - 3])
            rising = key.endswith("kl") and "bc" in key and recent > 1.15 * earlier and updates > 8
            check(f"{key}", rising, f"last {values[-1]:.4g}, mean of last 3 {recent:.4g}, best earlier {earlier:.4g}")

    # 5. every head's loss: flag one more than 25% above its best after update 10
    for key in keys(rows, r"(^|/)loss$|cross_entropy$|^bc/kl$"):
        values = series(rows, key)
        if len(values) >= 10:
            best, recent = min(values[3:]), statistics.mean(values[-3:])
            check(f"head {key}", recent > 1.25 * best and recent - best > 0.02, f"recent {recent:.4g}, best {best:.4g}")

    # 6. the critic
    for key in keys(rows, r"^critic/explained_variance/"):
        values = series(rows, key)
        if values and updates > 10:
            check(f"{key}", values[-1] < 0.5, f"{values[-1]:.3f}")

    # 7. gradients: norms against their own median, the actor's share
    for key in keys(rows, r"^gradient/.*norm"):
        values = series(rows, key)
        if len(values) >= 6:
            median = statistics.median(values)
            check(f"{key}", values[-1] > 5 * median, f"last {values[-1]:.4g}, median {median:.4g}")

    # 8. advantages (PPO): no level owning or vanishing from the summed advantage
    for key in keys(rows, r"^advantage/share/[a-z_]+$"):
        values = series(rows, key)
        if values:
            check(f"{key}", values[-1] > 0.7 or values[-1] < 0.02, f"{values[-1]:.3f}")
    for key in keys(rows, r"^gradient/policy_level_cosine/[a-z_]+/policy$"):
        values = series(rows, key)
        if values:
            check(f"{key}", values[-1] < 0.0, f"{values[-1]:.3f}")

    # 9. combat in the training games: hits and deaths per game holding up
    for key in keys(rows, r"^episode/event/(hits|combat_deaths)/[a-z_]+/(self_play|scripted|league)$"):
        values = series(rows, key)
        if len(values) >= 10:
            early, recent = statistics.mean(values[3:8]), statistics.mean(values[-3:])
            check(f"{key}", early > 0 and recent < 0.5 * early, f"recent {recent:.4g}, early {early:.4g}")

    # 10. speed
    times = [row.get("time/update_seconds") or row.get("update_seconds") for row in rows]
    times = [t for t in times if finite(t)]
    if len(times) >= 4:
        median = statistics.median(times[2:])
        check("update time", times[-1] > 2 * median, f"last {times[-1]:.1f} s, median {median:.1f} s")

    # 11. the live Elo (the laptop's table): the averaged policy's latest ratings by mode
    if len(sys.argv) > 2:
        try:
            lines = open(sys.argv[2]).read().splitlines()
        except OSError:
            lines = []
        latest: dict[str, list[tuple[int, float, float]]] = {}
        for line in lines:
            match = re.match(r"(real_scale_\w+)\s+\S*step_(\d+)\S*\s+(-?[\d.]+)\s+\+-\s+([\d.]+)", line)
            if match:
                mode, step, rating, error = match.groups()
                latest.setdefault(mode, []).append((int(step), float(rating), float(error)))
        for mode, points in latest.items():
            points = sorted(set(points))
            text = ", ".join(f"{s // 1_000_000}M {r:.0f}±{e:.0f}" for s, r, e in points[-4:])
            falling = len(points) >= 3 and points[-1][1] < points[-3][1] - 2 * points[-1][2]
            check(f"elo {mode}", falling, text)

    for status, name, detail in out:
        print(f"{status:4} {name}: {detail}")


if __name__ == "__main__":
    main()
