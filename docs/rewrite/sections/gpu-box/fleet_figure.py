"""fleet_figure.py OUT.png: zero-shot fleet-size generalisation (training is 5v5, 2600 m arena) of the
champion (Yemong) and the nearest MLP, against the scripted agent, in the training arena.
Left two panels: Elo (scripted = 1000) by fleet size per mode, from the fleet ratings (5v5 and 10v10
in fields with the null's calibration rungs; 20v20 and 40v40 with only the three players: a hollow
marker is a saturated rating, one side winning every game, so a bound, not a value). Right: the
deathmatch win rate at fixed odds (n network ships against n x 1, 1.2, 1.5, 2 scripted), 95% Wilson
intervals."""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

out = sys.argv[1]
SIZES = (5, 10, 20, 40)
# (rating, stderr, saturated) from the fleet ratings' leaderboards (gpu-box-log 2026-11-10)
ELO = {
    "frontline": {
        "champion": [(1997, 29, False), (1875, 27, False), (1134, 21, False), (526, 28, False)],
        "nearest": [(1303, 18, False), (1240, 17, False), (1102, 25, False), (1041, 19, False)],
    },
    "deathmatch": {
        "champion": [(2404, 32, False), (2778, 32, False), (2551, 142, True), (2445, 246, True)],
        "nearest": [(1386, 15, False), (1345, 12, False), (859, 19, False), (-440, 246, True)],
    },
}
COLOURS = {"champion": "#2a78d6", "nearest": "#1baf7a"}
LABELS = {"champion": "Champion (Yemong, RL 180M)", "nearest": "Nearest MLP (RL 40.9M)"}
ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

odds = {}
for path in pathlib.Path("runs/crossover/fixed-odds").glob("*.json"):
    d = json.loads(path.read_text())
    agent = "champion" if "deadreckon" in d["agent"] else "nearest"
    odds[(agent, d["own"])] = d["rows"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=150, facecolor=surface)
for ax, mode in zip(axes[:2], ("frontline", "deathmatch")):
    ax.set_facecolor(surface)
    for agent in ("champion", "nearest"):
        values = ELO[mode][agent]
        ys = [v[0] for v in values]
        ax.plot(SIZES, ys, color=COLOURS[agent], linewidth=2, zorder=2, label=LABELS[agent])
        for x, (y, e, saturated) in zip(SIZES, values):
            ax.errorbar(x, y, yerr=None if saturated else e, fmt="o", markersize=7,
                        color=COLOURS[agent], markerfacecolor=surface if saturated else COLOURS[agent],
                        markeredgewidth=2, capsize=3, zorder=3)
            if saturated:
                ax.annotate("≥" if y > 1000 else "≤", (x, y), xytext=(7, -2),
                            textcoords="offset points", fontsize=10, color=muted)
    ax.axhline(1000, color=muted, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(5.2, 1000, " scripted = 1000", va="bottom", fontsize=8.5, color=muted)
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES, [f"{n}v{n}" for n in SIZES])
    ax.set_title(f"{mode.capitalize()}: Elo by fleet size", color=ink, fontsize=11.5, loc="left")
    ax.set_xlabel("ships per team (zero-shot beyond 5v5; 2600 m training arena)", color=muted,
                  fontsize=9)
    ax.set_ylabel("Elo (scripted = 1000)", color=muted, fontsize=9)

ax = axes[2]
ax.set_facecolor(surface)
markers = {10: "o", 20: "s", 40: "D"}
for agent in ("champion", "nearest"):
    for n in (10, 20, 40):
        rows = odds.get((agent, n))
        if not rows:
            continue
        xs = [r["ratio"] for r in rows]
        ys = [r["win_rate"] for r in rows]
        lo = [r["win_rate"] - r["win_low"] for r in rows]
        hi = [r["win_high"] - r["win_rate"] for r in rows]
        ax.errorbar(xs, ys, yerr=[lo, hi], color=COLOURS[agent], marker=markers[n], markersize=6,
                    linewidth=1.8, capsize=3, alpha=1.0 if agent == "champion" else 0.85,
                    linestyle="-" if agent == "champion" else (0, (4, 2)),
                    label=f"{'Champion' if agent == 'champion' else 'Nearest MLP'}, {n} ships")
ax.axhline(0.5, color=muted, linewidth=1, linestyle=(0, (4, 3)))
ax.set_xticks((1.0, 1.2, 1.5, 2.0), ["1:1", "1.2:1", "1.5:1", "2:1"])
ax.set_ylim(-0.03, 1.03)
ax.set_title("Deathmatch: win rate when outnumbered", color=ink, fontsize=11.5, loc="left")
ax.set_xlabel("scripted ships : network ships", color=muted, fontsize=9)
ax.set_ylabel("network's win rate vs scripted (95% CI)", color=muted, fontsize=9)
ax.legend(frameon=False, fontsize=7.5, labelcolor=ink, loc="upper right")

for ax in axes:
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8.5)
axes[0].legend(frameon=False, fontsize=8.5, labelcolor=ink, loc="lower left")
fig.suptitle("Zero-shot fleet-size generalisation (trained 5v5 only)", color=ink, fontsize=13,
             x=0.01, ha="left")
fig.text(0.01, 0.005, "Hollow markers: saturated ratings (one side won every game), shown as bounds. "
         "5v5 and 10v10 Elo from fields with calibration rungs; 20v20 and 40v40 from champion, "
         "nearest MLP and scripted only. Win rates: 128 games per point.", fontsize=7.5, color=muted)
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.savefig(out, facecolor=surface)
print(out)
