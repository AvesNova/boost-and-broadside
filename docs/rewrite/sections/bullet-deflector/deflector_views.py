"""Two views of the deflector (the egg, proposal 2026-11-27), both from the game's own step
(lean.tool.hitbench's synthetic bullets; no copy of a game number):

1. apparent size: the hit width (the band of aim offsets that do damage) in metres and as the
   half-angle seen from the shooter (mrad), by range from astern and from the side, for the
   deflector off and the egg at largest turns 30, 45 and 60 deg; and by aspect at 300 m;
2. bullet paths: a row of parallel bullets past the target in the target's frame, per largest
   turn and aspect, drawn THROUGH the ship (the paths are flown with a hit radius of 1 cm, so no
   bullet stops; the colour says whether it hit with the real hit radius), as in the design
   sketches. The turn falls mid-tick at the egg's entry: drawn at the crossing of the incoming
   and outgoing tick-to-tick lines.

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/bullet-deflector/deflector_views.py
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from lean.game import card, core
from lean.tool import hitbench

OUT = Path(__file__).parent / "workbench"
COLOURS = hitbench.COLOURS + ("#7d56c7", "#5e5d59")
RANGES = torch.tensor([50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 650.0])
TURNS_DEG = (30.0, 45.0, 60.0)
CONFIG = dataclasses.replace(hitbench.CONFIG, deflector="egg")


def variants() -> list[tuple[str, dict]]:
    out = [("off", dict(card.DEFLECTOR_PRESETS["off"]))]
    for turn in TURNS_DEG:
        egg = {**card.DEFLECTOR_PRESETS["egg"], "deflector_turn": math.radians(turn)}
        out.append((f"egg, {turn:g} deg", egg))
    return out


def widths(inputs: dict, aspects_deg, ranges: torch.Tensor) -> torch.Tensor:
    """(A, R) hit width, metres, from the step."""
    probe = core.reset(CONFIG, 1, torch.tensor([[1, 1]]), None, "cpu")
    radius = hitbench._value(core.with_hit_model(probe.constants, **inputs), "hit_radius")
    step = hitbench.OFFSET_STEP
    offsets = torch.arange(-radius - 1.0, radius + 1.0 + 1e-6, step)
    aspects = torch.tensor([math.radians(a) for a in aspects_deg])
    damage, _ = hitbench._damage(CONFIG, inputs, aspects, ranges, offsets)
    return (damage > 0.0).float().sum(-1) * step


def apparent_figure(path: Path) -> list[str]:
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    lines = [
        "| deflector | hit width astern, m | side (90 deg), m | head-on, m | half-angle astern "
        "at 300 m, mrad | game table (still ship) astern / side / head-on, m |",
        "|---|---|---|---|---|---|",
    ]
    i300 = int((RANGES == 300.0).nonzero())
    aspects = list(range(0, 181, 10))
    probe = core.reset(CONFIG, 1, torch.tensor([[1, 1]]), None, "cpu")
    radius = hitbench._value(probe.constants, "hit_radius")
    for k, (name, inputs) in enumerate(variants()):
        colour = COLOURS[k % len(COLOURS)]
        by_range = widths(inputs, (180, 90), RANGES)
        half = 1e3 * 0.5 * by_range / RANGES
        a.plot(RANGES, half[0], color=colour, lw=2, marker="o", ms=4, label=f"{name}, astern")
        a.plot(RANGES, half[1], color=colour, lw=2, ls="--", label=f"{name}, 90 deg")
        by_aspect = widths(inputs, aspects, RANGES[i300 : i300 + 1])[:, 0]
        b.plot(aspects, by_aspect, color=colour, lw=2, marker="o", ms=3, label=name)
        table = core.hit_widths(torch.tensor([radius]), inputs)[0]
        game = [
            float(core.hit_width(table, torch.tensor(math.cos(math.radians(d)))))
            for d in (180, 90, 0)
        ]
        lines.append(
            f"| {name} | {float(by_range[0, i300]):.1f} | {float(by_range[1, i300]):.1f} | "
            f"{float(by_aspect[0]):.1f} | {float(half[0, i300]):.1f} | "
            + " / ".join(f"{g:.1f}" for g in game)
            + " |"
        )
    a.set(
        xlabel="range, m",
        ylabel="half-angle of the hit width, mrad",
        title="Apparent size: half the hit width over the range",
    )
    a.set_yscale("log")
    b.set(
        xlabel="aspect: bullet path off the target's nose, deg (0 head-on)",
        ylabel="hit width, m",
        title="Hit width by aspect at 300 m (the target at cruise)",
    )
    b.set_xticks([0, 30, 45, 90, 135, 180])
    for axis in (a, b):
        axis.grid(True, color="#e4e3df", lw=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=7, frameon=False)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return lines


def paths(inputs: dict, aspect_deg: float, offsets: torch.Tensor, ticks: int = 5):
    """Tick positions relative to the target (ticks + 1, O, 2) flown through the ship (hit
    radius 1 cm), the real hit flags (O,), and the target's nose direction."""
    aspect = torch.tensor([math.radians(aspect_deg)])
    ranges = torch.tensor([300.0])
    real, _ = hitbench._damage(CONFIG, inputs, aspect, ranges, offsets)
    through = {**inputs, "hit_radius": 0.01, "core_radius": 0.0}
    state = hitbench._place(CONFIG, through, aspect, ranges, offsets)
    envs = state.x.shape[0]
    idle = torch.zeros(envs, 2, 3, dtype=torch.long)
    state = state.replace(pending=idle)
    noise = {"spread": torch.zeros(envs, 2, 2)}
    s = hitbench.SLOT

    def relative(st):
        return torch.stack(
            [
                st.bullets[:, core.BULLET_X, 0, s] - st.x[:, 1],
                st.bullets[:, core.BULLET_Y, 0, s] - st.y[:, 1],
            ],
            -1,
        )

    nose = torch.stack([state.heading_cos[0, 1], state.heading_sin[0, 1]])
    points = [relative(state)]
    for _ in range(ticks):
        state, _ = core.step(CONFIG, state, idle, noise)
        points.append(relative(state))
    return torch.stack(points), real.flatten() > 0.0, nose


def _kinked(points: torch.Tensor) -> torch.Tensor:
    """One bullet's polyline (T, 2) with a mid-tick turn put at the crossing of the incoming
    and outgoing tick-to-tick lines (where the direction changed by more than 0.01 rad; the
    drag's slow drift is not drawn as a kink)."""
    out = [points[0]]
    for k in range(1, len(points) - 1):
        before, after = points[k] - points[k - 1], points[k + 1] - points[k]
        cross = float(before[0] * after[1] - before[1] * after[0])
        if abs(cross) > 1e-2 * float(before.norm() * after.norm()):
            gap = points[k] - points[k - 1]  # p_{k-1} + u before = p_k + v after
            u = float(gap[0] * after[1] - gap[1] * after[0]) / cross
            out.append(points[k - 1] + u * before)
        out.append(points[k])
    out.append(points[-1])
    return torch.stack(out)


def egg_outline(inputs: dict, nose: torch.Tensor) -> torch.Tensor:
    theta = torch.linspace(0.0, 2.0 * math.pi, 200)
    along = torch.where(torch.cos(theta) >= 0, inputs["egg_front"], inputs["egg_rear"])
    u, v = along * torch.cos(theta), inputs["egg_half_width"] * torch.sin(theta)
    return torch.stack([u * nose[0] - v * nose[1], u * nose[1] + v * nose[0]], -1)


def path_figure(path: Path) -> None:
    aspects = (0, 45, 90, 135, 180)
    rows = variants()[1:]
    offsets = torch.arange(-30.0, 30.01, 2.0)
    fig, axes = plt.subplots(
        len(rows),
        len(aspects),
        figsize=(4 * len(aspects), 4.1 * len(rows)),
        constrained_layout=True,
    )
    probe = core.reset(CONFIG, 1, torch.tensor([[1, 1]]), None, "cpu")
    radius = hitbench._value(probe.constants, "hit_radius")
    for row, (name, inputs) in enumerate(rows):
        for col, aspect in enumerate(aspects):
            ax = axes[row, col]
            points, hit, nose = paths(inputs, aspect, offsets)
            for i in range(len(offsets)):
                line = _kinked(points[:, i])
                colour = COLOURS[1] if hit[i] else COLOURS[0]
                ax.plot(line[:, 0], line[:, 1], color=colour, lw=0.9)
            outline = egg_outline(inputs, nose)
            ax.plot(outline[:, 0], outline[:, 1], color="#5e5d59", lw=1, ls="--")
            theta = torch.linspace(0, 2 * math.pi, 100)
            circle_x, circle_y = radius * torch.cos(theta), radius * torch.sin(theta)
            ax.plot(circle_x, circle_y, color="#5e5d59", lw=1, alpha=0.7)
            ax.annotate(
                "",
                xy=(2.2 * radius * nose[0], 2.2 * radius * nose[1]),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#5e5d59", lw=1.2),
            )
            ax.set_aspect("equal")
            ax.set_xlim(-75, 75)
            ax.set_ylim(-75, 75)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{name}, {aspect} deg off the nose", fontsize=9)
    fig.suptitle(
        "Bullet paths through a target at cruise (target's frame, 300 m shot, rays 2 m apart), "
        "from the game's step, drawn through the ship. Dashed: the egg; circle: the hit radius "
        f"{radius:g} m; arrow: nose.\nOrange: the bullet hits (real hit radius); blue: misses.",
        fontsize=10,
    )
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    text = apparent_figure(OUT / "apparent_size.png")
    print("\n".join(text))
    path_figure(OUT / "bullet_paths.png")
    print("wrote", OUT / "apparent_size.png", OUT / "bullet_paths.png")
