"""Draw a run's training-time live Elo, with its post-hoc calibrated endpoints marked.

The lines are the live Elo the trainer kept while it ran (checkpoints/<run>/elo_history.jsonl):
sequential updates against a drifting pool, read as a shape only, not as a rating. The
points are the post-hoc calibrated ratings of the final live checkpoint and the best
averaged checkpoint, from the run's newest elo-calibration artifact.

    uv run --no-sync python scripts/plot_elo_history.py --run wild-universe-731 \
        --out docs/results/v1-731/elo_history_731.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run_dir = ROOT / "checkpoints" / args.run
    rows = [json.loads(line) for line in (run_dir / "elo_history.jsonl").read_text().splitlines()]
    steps = [row["global_step"] / 1e6 for row in rows]
    calibration = sorted((run_dir / "artifacts" / "elo-calibration").glob("*/result.json"))[-1]
    players = {p["label"]: p for p in json.loads(calibration.read_text())["players"]}
    final = players[f"ckpt_{max(p['global_step'] or 0 for p in players.values())}"]
    averaged = players["best_avg"]

    blue, orange, ink, muted, grid = "#2160b8", "#d9622b", "#1f2328", "#6e7781", "#e6e8eb"
    figure, axes = plt.subplots(figsize=(8, 4.6), dpi=110)
    axes.plot(
        steps,
        [row["live"] for row in rows],
        color=blue,
        lw=1.6,
        label="live policy, training-time Elo (uncalibrated)",
    )
    # The averaged policy is unrated (0) until it first enters the pool.
    first = next(i for i, row in enumerate(rows) if row["avg"])
    axes.plot(
        steps[first:],
        [row["avg"] for row in rows[first:]],
        color=blue,
        lw=1.2,
        alpha=0.45,
        label="averaged policy, training-time Elo (uncalibrated)",
    )
    for player, marker, label in (
        (final, "o", "final live checkpoint, calibrated (±1 SE)"),
        (averaged, "D", "best averaged checkpoint, calibrated (±1 SE)"),
    ):
        axes.errorbar(
            [player["global_step"] / 1e6],
            [player["calibrated_elo"]],
            yerr=[player["stderr"]],
            fmt=marker,
            color=orange,
            ms=6,
            capsize=3,
            label=label,
            zorder=5,
        )
    axes.axhline(1000, color=muted, lw=1, ls="--")
    axes.annotate(
        "scripted controller 1000",
        (steps[-1], 1000),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=muted,
    )
    axes.set_title(
        f"Live Elo during training, {args.run}",
        loc="left",
        fontsize=12,
        fontweight="bold",
        color=ink,
    )
    axes.set_xlabel("environment steps (millions)", color=ink)
    axes.set_ylabel("Elo (scripted = 1000)", color=ink)
    axes.grid(True, color=grid, lw=0.8)
    axes.set_axisbelow(True)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.tick_params(colors=muted)
    axes.legend(loc="lower right", frameon=False, fontsize=8)
    figure.tight_layout()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
