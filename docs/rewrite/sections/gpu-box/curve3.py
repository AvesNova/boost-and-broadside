"""The whole BC -> RL Elo curve of the three agent types, from tournament 10 (one field, so every
Elo is on one scale): the champion (Yemong, dead reckoning + hidden physics), the null (Yemong,
absolute head, C-fixed) and the nearest model (a 2-layer MLP over the nearest ships, fields and
zones). x: total environment steps (BC, then RL); y: Elo (scripted = 1000), the mean of frontline
and deathmatch, uniform averages, +-1 standard error; the nearest model's live weights dashed
(its uniform average is far below them). Two panels below: each mode alone.
Usage: curve3.py RATINGS_DIR OUT.png"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ratings_dir, out = pathlib.Path(sys.argv[1]), sys.argv[2]
modes = {}
for mode in ("frontline", "deathmatch"):
    if not (ratings_dir / f"{mode}.batches.jsonl").exists():
        continue  # a preview while the tournament runs: the modes rated so far
    last = json.loads((ratings_dir / f"{mode}.batches.jsonl").read_text().strip().splitlines()[-1])
    modes[mode] = {p["player"]: (p["rating"] + 1000.0, p["stderr"]) for p in last["ratings"]}

BC_END = 120586240
YEMONG_BC = [10485760, 20971520, 30408704, 40894464, 50331648, 60817408, 70254592, 80740352,
             90177536, 100663296, 110100480, 120586240]
YEMONG_RL = [10485760, 20971520, 30408704, 40894464, 50331648, 60817408, 70254592, 80740352,
             90177536, 100663296, 110100480, 121634816, 141557760, 160432128, 180355072]
NEAREST_BC = [20971520, 30408704, 40894464, 60817408, 80740352, 100663296, 120586240]
NEAREST_RL = [20971520, 40894464, 60817408, 80740352, 100663296, 121634816, 141557760,
              160432128, 180355072]
# (label, colour, [(x steps, player path)], dashed)
SERIES = []
for run_date, line, label, colour in (
    ("2026-11-09", "deadreckon-split-hiddenphysics", "Champion (Yemong: dead reckoning + hidden physics)", "#2a78d6"),
    ("2026-11-09", "absolute", "Null (Yemong: absolute head, C-fixed)", "#eb6834"),
):
    pts = [(s, f"runs/{run_date}-bc-{line}/policies/step_{s}/uniform_average") for s in YEMONG_BC]
    pts += [(BC_END + s, f"runs/{run_date}-rl-{line}/policies/step_{s}/uniform_average") for s in YEMONG_RL]
    SERIES.append((label, colour, pts, False))
NEAREST = "#1baf7a"
pts = [(s, f"runs/2026-11-10-bc-nearest/policies/step_{s}/uniform_average") for s in NEAREST_BC]
pts += [(BC_END + s, f"runs/2026-11-10-rl-nearest/policies/step_{s}/uniform_average") for s in NEAREST_RL]
SERIES.append(("Nearest MLP (uniform average)", NEAREST, pts, False))
live = [(s, f"runs/2026-11-10-bc-nearest/policies/step_{s}/live") for s in (30408704, 60817408, 120586240)]
live += [(BC_END + s, f"runs/2026-11-10-rl-nearest/policies/step_{s}/live") for s in (60817408, 121634816, 180355072)]
SERIES.append(("Nearest MLP (live weights)", NEAREST, live, True))


AVAILABLE = tuple(modes)


def value(player, which):
    if which == "mean" and len(AVAILABLE) == 1:
        return modes[AVAILABLE[0]][player]
    if which == "mean":
        (f, fe), (d, de) = modes["frontline"][player], modes["deathmatch"][player]
        return (f + d) / 2.0, ((fe**2 + de**2) ** 0.5) / 2.0
    return modes[which][player]


ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
if len(AVAILABLE) == 2:
    fig = plt.figure(figsize=(12, 10), dpi=150, facecolor=surface)
    spec = fig.add_gridspec(2, 2, height_ratios=(1.35, 1), hspace=0.32, wspace=0.12)
    panels = [(fig.add_subplot(spec[0, :]), "mean", "Mean of frontline and deathmatch"),
              (fig.add_subplot(spec[1, 0]), "frontline", "Frontline"),
              (fig.add_subplot(spec[1, 1]), "deathmatch", "Deathmatch")]
else:
    fig = plt.figure(figsize=(12, 6.2), dpi=150, facecolor=surface)
    panels = [(fig.add_subplot(1, 1, 1), "mean",
               f"{AVAILABLE[0].capitalize()} only (preview: the other mode is still being rated)")]
missing = set()
for ax, which, title in panels:
    ax.set_facecolor(surface)
    for label, colour, points, dashed in SERIES:
        missing |= {p for _, p in points if any(p not in modes[m] for m in AVAILABLE)}
        points = [(x, p) for x, p in points if all(p in modes[m] for m in AVAILABLE)]
        xs = [x / 1e6 for x, _ in points]
        vals = [value(p, which) for _, p in points]
        ys, es = [v[0] for v in vals], [v[1] for v in vals]
        if not dashed:
            ax.fill_between(xs, [y - e for y, e in zip(ys, es)], [y + e for y, e in zip(ys, es)],
                            color=colour, alpha=0.13, linewidth=0)
        ax.plot(xs, ys, color=colour, linewidth=2, linestyle=(0, (5, 3)) if dashed else "-",
                marker="D" if dashed else "o", markersize=4.5, markeredgecolor=surface,
                markeredgewidth=1.2, label=label, zorder=3)
        if which == "mean":
            ax.annotate(f"{ys[-1]:.0f}", (xs[-1], ys[-1]), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=10, color=ink)
    ax.axhline(1000, color=muted, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.axvline(BC_END / 1e6, color=muted, linewidth=1, zorder=1)
    ax.set_xlim(0, (BC_END + 180355072) / 1e6 + (16 if which == "mean" else 6))
    ax.set_title(title, color=ink, fontsize=11 if which != "mean" else 12, loc="left")
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=muted, labelsize=9)
    if which == "mean":
        lo, hi = ax.get_ylim()
        ax.text(BC_END / 1e6 - 3, lo + (hi - lo) * 0.03, "BC", ha="right", va="bottom", fontsize=11, color=muted)
        ax.text(BC_END / 1e6 + 3, lo + (hi - lo) * 0.03, "RL", ha="left", va="bottom", fontsize=11, color=muted)
        ax.text(3, 1000 + (hi - lo) * 0.012, "scripted teacher = 1000", fontsize=9, color=muted)
        ax.set_ylabel("Elo (scripted = 1000)", color=muted, fontsize=10)
        ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=ink)
    if which != "mean" or len(AVAILABLE) == 1:
        ax.set_xlabel("environment steps (millions): BC, then RL", color=muted, fontsize=9)
fig.suptitle("Boost and Broadside: BC → RL Elo, champion vs null vs nearest MLP (tournament 10)",
             color=ink, fontsize=13, x=0.01, ha="left", y=0.995)
fig.text(0.01, 0.005, "Uniform average of each run's weights at each snapshot (nearest: also its live "
         "weights, dashed); band ±1 standard error. Yemong RL from its BC 120.6M uniform average, "
         "nearest RL from its BC 120.6M exponential average. One field, 8 × 256 games per batch.",
         fontsize=7.5, color=muted)
fig.savefig(out, facecolor=surface, bbox_inches="tight")
print(out, "missing:", sorted(missing) or "none")
