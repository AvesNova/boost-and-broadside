"""Three derived charts from the fan data (no new games), light + dark PNG and data JSON each.

  crossover_by_size     50% opponent ratio against trained team size, final live and
                        averaged, 95% CI from a parametric bootstrap of the five grid
                        points (binomial resampling).
  game_length_by_ratio  mean game length (seconds of game time, ticks / 60) against opponent ratio,
                        one line per size, final averaged checkpoint.
  averaged_vs_final     win rate of averaged and final live weights at each ratio, one
                        panel per size.

Usage: extras.py [FAN_DIR]   (default: the current directory; uses whichever
fan_731_n{N}.jsonl exist there). Game counts in the chart notes are read from the data.
"""

import json
import math
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) > 1:
    os.chdir(sys.argv[1])
TICK_SECONDS = 1 / 60  # ShipConfig.dt of run 731; step_count counts physics ticks
FINAL, AVG = "/step_000671846400.pt", "/avg_step_000671846400.pt"
SIZES = [n for n in (4, 8, 16, 32) if os.path.exists(f"fan_731_n{n}.jsonl")]
data = {
    n: [json.loads(line) for line in open(f"fan_731_n{n}.jsonl") if line.strip()] for n in SIZES
}
rng = np.random.default_rng(0)

THEMES = {
    "": dict(
        bg="#ffffff",
        ink="#1f2328",
        muted="#6e7781",
        grid="#e6e8eb",
        live="#2160b8",
        avg="#c2571a",
        ramp=["#9ec5f4", "#5b9be8", "#2160b8", "#123f80"],
    ),
    "_dark": dict(
        bg="#0f1115",
        ink="#e6e8eb",
        muted="#9198a1",
        grid="#262a31",
        live="#6aa5ec",
        avg="#f0a060",
        ramp=["#1f4f8f", "#3b7fd6", "#80b2f2", "#c8defb"],
    ),
}


def pts(n, name):
    return sorted((r for r in data[n] if r["ckpt"].endswith(name)), key=lambda r: r["scripted"])


def games_note(name):
    """'4,096 games per point (3,072 at 32 ships)': the most common count, then exceptions."""
    per_size = {n: sorted({r["games"] for r in pts(n, name)}) for n in SIZES}
    assert all(len(g) == 1 for g in per_size.values()), per_size
    counts = [g[0] for g in per_size.values()]
    main = max(set(counts), key=counts.count)
    rest = [f"{g[0]:,} at {n} ships" for n, g in per_size.items() if g[0] != main]
    return f"{main:,} games per point" + (f" ({', '.join(rest)})" if rest else "")


def crossing(xs, ys):
    for i in range(len(xs) - 1):
        if ys[i] >= 0.5 > ys[i + 1]:
            return xs[i] + (ys[i] - 0.5) / (ys[i] - ys[i + 1]) * (xs[i + 1] - xs[i])
    return None


def wilson(w, n, z=1.96):
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def style(ax, t):
    ax.set_facecolor(t["bg"])
    ax.grid(True, color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=t["muted"])


def legend(ax, t, **kw):
    leg = ax.legend(frameon=False, fontsize=9, **kw)
    for txt in leg.get_texts():
        txt.set_color(t["ink"])


# ---- crossover_by_size
cross = {}
for n in SIZES:
    cross[n] = {}
    for key, name in (("final_live", FINAL), ("final_averaged", AVG)):
        p = pts(n, name)
        if len(p) < 5:
            continue
        ratios = [r["scripted"] / n for r in p]
        x = crossing(ratios, [r["win_rate"] for r in p])
        boot = []
        for _ in range(4000):
            ys = [rng.binomial(r["games"], r["win_rate"]) / r["games"] for r in p]
            boot.append(crossing(ratios, ys))
        ok = [b for b in boot if b is not None]
        cross[n][key] = dict(
            ratio=x,
            opponents=None if x is None else x * n,
            ratio_ci95=[float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))]
            if ok
            else None,
            bootstrap_no_crossing_share=1 - len(ok) / len(boot),
        )
json.dump(
    dict(
        run="wild-universe-731",
        method="linear interpolation of win rate (ties as non-wins); "
        "95% CI = 2.5/97.5 percentiles of 4000 parametric bootstrap resamples of the five "
        "grid points",
        sizes={str(n): v for n, v in cross.items()},
    ),
    open("crossover_by_size.json", "w"),
    indent=1,
)
for s, t in THEMES.items():
    fig, ax = plt.subplots(figsize=(7, 4.4), dpi=150)
    fig.patch.set_facecolor(t["bg"])
    style(ax, t)
    for key, lab, c, mk, dx in (
        ("final_live", "final live weights", t["live"], "o", 0.97),
        ("final_averaged", "final averaged weights", t["avg"], "D", 1.03),
    ):
        ns = [n for n in SIZES if cross[n].get(key) and cross[n][key]["ratio"] is not None]
        if not ns:
            continue
        xs = [n * dx for n in ns]
        ys = [cross[n][key]["ratio"] for n in ns]
        lo = [ys[i] - cross[n][key]["ratio_ci95"][0] for i, n in enumerate(ns)]
        hi = [cross[n][key]["ratio_ci95"][1] - ys[i] for i, n in enumerate(ns)]
        ax.errorbar(xs, ys, yerr=[lo, hi], color=c, lw=2, marker=mk, ms=6, capsize=3, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(n) for n in SIZES])
    ax.set_xlim(SIZES[0] / 1.4, SIZES[-1] * 1.4)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}x"))
    ax.set_xlabel("trained ships", color=t["ink"])
    ax.set_ylabel("scripted ships per trained ship at 50% wins", color=t["ink"])
    ax.set_title(
        "How many scripted ships it takes to hold them even",
        color=t["ink"],
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    legend(ax, t, loc="best")
    fig.text(
        0.01,
        0.01,
        "Run 731. Crossover interpolated between grid ratios 1.0-2.0; bars: 95% bootstrap CI; "
        "ties count as losses.",
        fontsize=8,
        color=t["muted"],
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(f"crossover_by_size{s}.png", facecolor=t["bg"])
    plt.close(fig)

# ---- game_length_by_ratio
glen = {}
for n in SIZES:
    p = pts(n, AVG)
    glen[n] = [
        dict(
            scripted=r["scripted"],
            ratio=r["scripted"] / n,
            mean_ticks=r["mean_steps"],
            mean_seconds=r["mean_steps"] * TICK_SECONDS,
            ties=r["ties"],
            capped=r.get("capped"),
            games=r["games"],
        )
        for r in p
    ]
json.dump(
    dict(
        run="wild-universe-731",
        checkpoint="avg_step_000671846400.pt",
        tick_seconds=TICK_SECONDS,
        episode_cap_ticks=1024,
        note="mean over all games incl. capped ones (capped games count as 1024 ticks)",
        sizes={str(n): v for n, v in glen.items()},
    ),
    open("game_length_by_ratio.json", "w"),
    indent=1,
)
for s, t in THEMES.items():
    fig, ax = plt.subplots(figsize=(7, 4.4), dpi=150)
    fig.patch.set_facecolor(t["bg"])
    style(ax, t)
    for i, n in enumerate(SIZES):
        g = glen[n]
        ax.plot(
            [x["ratio"] for x in g],
            [x["mean_seconds"] for x in g],
            color=t["ramp"][i],
            lw=2,
            marker="o",
            ms=4.5,
        )
    # direct labels at the line ends, nudged apart so they never overlap
    ends = sorted((glen[n][-1]["mean_seconds"], n) for n in SIZES)
    placed = []
    for y, n in ends:
        y = max(y, placed[-1] + 0.8) if placed else y
        placed.append(y)
        ax.annotate(
            f"{n} ships",
            (glen[n][-1]["ratio"], y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=t["ink"],
        )
    ax.axhline(1024 * TICK_SECONDS, color=t["muted"], lw=1, ls="--")
    ax.annotate(
        "episode cap",
        (1.0, 1024 * TICK_SECONDS),
        xytext=(0, -12),
        textcoords="offset points",
        fontsize=8,
        color=t["muted"],
    )
    ax.set_xlim(0.95, 2.2)
    ax.set_ylim(0, 1024 * TICK_SECONDS * 1.05)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.set_xlabel("scripted ships per trained ship", color=t["ink"])
    ax.set_ylabel("mean game length (seconds of game time)", color=t["ink"])
    ax.set_title(
        "Game length against the odds", color=t["ink"], loc="left", fontsize=12, fontweight="bold"
    )
    fig.text(
        0.01,
        0.01,
        f"Run 731, averaged weights at the end; {games_note(AVG)}; game time = ticks / 60.",
        fontsize=8,
        color=t["muted"],
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(f"game_length_by_ratio{s}.png", facecolor=t["bg"])
    plt.close(fig)

# ---- averaged_vs_final
avf = {}
for n in SIZES:
    f, a = {r["scripted"]: r for r in pts(n, FINAL)}, {r["scripted"]: r for r in pts(n, AVG)}
    avf[n] = []
    for sc in sorted(set(f) & set(a)):
        d = a[sc]["win_rate"] - f[sc]["win_rate"]
        se = math.sqrt(a[sc]["se"] ** 2 + f[sc]["se"] ** 2)
        avf[n].append(
            dict(
                scripted=sc,
                ratio=sc / n,
                final_live=f[sc]["win_rate"],
                averaged=a[sc]["win_rate"],
                difference=d,
                difference_ci95=1.96 * se,
                games=f[sc]["games"],
            )
        )
json.dump(
    dict(
        run="wild-universe-731",
        note="independent samples; CI of difference = 1.96*sqrt(se_a^2+se_f^2)",
        sizes={str(n): v for n, v in avf.items()},
    ),
    open("averaged_vs_final.json", "w"),
    indent=1,
)
for s, t in THEMES.items():
    fig, axes = plt.subplots(
        1, len(SIZES), figsize=(2.3 * len(SIZES) + 1, 4.2), dpi=150, sharey=True, squeeze=False
    )
    fig.patch.set_facecolor(t["bg"])
    for ax, n in zip(axes[0], SIZES):
        style(ax, t)
        for x in avf[n]:
            ax.plot(
                [x["ratio"]] * 2,
                [x["final_live"], x["averaged"]],
                color=t["muted"],
                lw=1.5,
                zorder=1,
            )
        for key, c, mk, lab in (
            ("final_live", t["live"], "o", "final live"),
            ("averaged", t["avg"], "D", "averaged"),
        ):
            ax.scatter(
                [x["ratio"] for x in avf[n]],
                [x[key] for x in avf[n]],
                color=c,
                marker=mk,
                s=28,
                zorder=2,
                label=lab,
            )
        ax.set_title(f"{n} trained ships", color=t["ink"], fontsize=10)
        ax.axhline(0.5, color=t["muted"], lw=1, ls="--")
        ax.set_xticks([1, 1.5, 2])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
        ax.set_xlim(0.85, 2.15)
    axes[0][0].set_ylim(0, 1.02)
    axes[0][0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0][0].set_ylabel("win rate", color=t["ink"])
    legend(axes[0][0], t, loc="lower left")
    fig.supxlabel("scripted ships per trained ship", color=t["ink"], fontsize=10, y=0.07)
    fig.suptitle(
        "Averaged against final live weights",
        color=t["ink"],
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.01,
        f"Run 731 at the end of training; {games_note(FINAL)}; ties count as losses.",
        fontsize=8,
        color=t["muted"],
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(f"averaged_vs_final{s}.png", facecolor=t["bg"])
    plt.close(fig)
print("sizes", SIZES)
for n in SIZES:
    print(
        n,
        {
            k: (
                round(v["ratio"], 3) if v["ratio"] else None,
                [round(c, 3) for c in v["ratio_ci95"]] if v["ratio_ci95"] else None,
            )
            for k, v in cross[n].items()
        },
    )
