"""Charts for bench/energy_duel.py: paths, speed and store over time, damage per condition.

Usage: uv run --no-sync python bench/energy_duel_charts.py DIR (reads DIR/results.json and
DIR/traces.pt, writes DIR/paths.png, DIR/speed.png, DIR/store.png, DIR/damage.png).
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

INK, MUTED, GRID, BLUE, RED = "#1f2328", "#6e7781", "#d0d7de", "#2f6fbf", "#c2412d"
PLANS = ("one-circle", "two-circle")
plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
    }
)


def row_label(name: str) -> str:
    return name.split(",")[0]


def grid(rows):
    fig, axes = plt.subplots(
        len(rows), 2, figsize=(10.8, 3.1 * len(rows) + 1.2), dpi=100, squeeze=False
    )
    return fig, axes


def main(folder: str):
    results = json.load(open(f"{folder}/results.json"))
    data = torch.load(f"{folder}/traces.pt")
    summary = results["summary"]
    names = data["names"]
    traces = data["traces"]
    h = data["tick_length"]
    rows = []
    for row in summary:
        label = row_label(row["name"])
        if label not in rows:
            rows.append(label)
    index = {(row_label(r["name"]), r["plan"]): i for i, r in enumerate(summary)}
    end = data["end_tick"]

    # (a) paths.
    fig, axes = grid(rows)
    for r, label in enumerate(rows):
        for col, plan in enumerate(PLANS):
            ax = axes[r, col]
            i = index[(label, plan)]
            stop = int(end[i])
            for slot, colour in ((0, BLUE), (1, RED)):
                xs = traces["x"][:stop, i, slot] / 1000
                ys = traces["y"][:stop, i, slot] / 1000
                ax.plot(xs, ys, color=colour, linewidth=1.4)
                ax.plot(xs[0], ys[0], "s", color=colour, markersize=7)
            for hit in data["hits"]:
                if hit[0] == i:
                    _, tick, attacker, damage, x, y = hit
                    # Mark the victim's position, in the attacker's colour.
                    ax.plot(
                        x / 1000,
                        y / 1000,
                        "o",
                        color=BLUE if attacker == 0 else RED,
                        markersize=6,
                        markeredgecolor="white",
                        markeredgewidth=0.6,
                    )
            s = summary[i]
            ax.set_title(
                f"{label}, {plan}\nBlue dealt {s['damage_by_blue']:.0f}, "
                f"Red dealt {s['damage_by_red']:.0f}",
                loc="left",
            )
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(color=GRID, linewidth=0.6)
            ax.tick_params(colors=MUTED)
    angle = results["setup"].get("exchange_angle_deg")
    tag = f" (exchange {angle:g} deg)" if angle is not None else ""
    fig.suptitle(
        f"Paths, one seed per condition (km). Blue = mode{tag}.\n"
        "Red = coasts. Squares: start. Dots: damaging hits (at the victim, shooter's colour).",
        x=0.02,
        ha="left",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(f"{folder}/paths.png")
    plt.close(fig)

    # (b) speed and store over time.
    for key, ylabel, scale, out in (
        ("speed", "speed, m/s", 1.0, "speed"),
        ("energy", "store, % of full", None, "store"),
    ):
        fig, axes = grid(rows)
        full = results["setup"]["max_store"]
        for r, label in enumerate(rows):
            for col, plan in enumerate(PLANS):
                ax = axes[r, col]
                i = index[(label, plan)]
                stop = int(end[i])
                time = torch.arange(stop) * h
                for slot, colour, who in ((0, BLUE, "Blue"), (1, RED, "Red")):
                    values = traces[key][:stop, i, slot]
                    values = values * (100.0 / full) if scale is None else values
                    ax.plot(time, values, color=colour, linewidth=1.8, label=who)
                ax.set_title(f"{label}, {plan}", loc="left")
                ax.set_xlim(0, 60)
                if scale is None:
                    ax.set_ylim(-3, 103)
                ax.grid(color=GRID, linewidth=0.6)
                ax.tick_params(colors=MUTED)
                if r == len(rows) - 1:
                    ax.set_xlabel("seconds")
                if col == 0:
                    ax.set_ylabel(ylabel)
        axes[0, 1].legend(loc="upper right", frameon=False)
        title = "Speed" if key == "speed" else "Store"
        fig.suptitle(
            f"{title} over time, one seed per condition (a line stops at a kill)\n{tag.strip()}",
            x=0.02,
            ha="left",
            fontsize=16,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.975 if tag else 0.985))
        fig.savefig(f"{folder}/{out}.png")
        plt.close(fig)

    # (c) damage each way per condition.
    labels = [r["name"] for r in summary]
    fig, ax = plt.subplots(figsize=(10.8, 0.62 * len(labels) + 2.0), dpi=100)
    y = torch.arange(len(labels)).float()
    blue = [r["damage_by_blue"] for r in summary]
    red = [r["damage_by_red"] for r in summary]
    ax.barh(y - 0.2, blue, height=0.38, color=BLUE, label="dealt by Blue")
    ax.barh(y + 0.2, red, height=0.38, color=RED, label="dealt by Red")
    for k, r in enumerate(summary):
        outcome = (
            f"Red kills {r['red_kill']:.0%}"
            if r["red_kill"] > 0
            else f"Blue kills {r['blue_kill']:.0%}"
            if r["blue_kill"] > 0
            else "clock"
        )
        ax.text(max(blue[k], red[k]) + 2, k, outcome, va="center", fontsize=12, color=MUTED)
    ax.axvline(results["setup"]["max_shield"], color=MUTED, linewidth=1, linestyle="--")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 175)
    ax.set_xlabel("mean damage dealt in 60 s (shield 100; a full hit 21)")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title(f"Damage each way, 64 seeds per condition\n{tag.strip()}", loc="left", fontsize=16)
    fig.tight_layout()
    fig.savefig(f"{folder}/damage.png")
    plt.close(fig)
    del names


if __name__ == "__main__":
    main(sys.argv[1])
