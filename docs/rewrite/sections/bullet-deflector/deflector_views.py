"""Two views of the deflector, both from the game's own step (lean.tool.hitbench's synthetic
bullets; no copy of a game number):

1. apparent size: the target's edge (the widest pass offset that still does damage) and its
   damage-weighted effective radius, in metres and as the half-angle seen from the shooter
   (mrad), by range, for the deflector off and at several strengths D (same D at every aspect,
   nose cone off so only D acts), and by aspect for the presets;
2. streamlines: a row of parallel bullet paths past the target in the target's frame, per
   preset and aspect, from the step's tick-end positions (the bend happens at the closest point:
   drawn as the incoming line to that point, the push, then the bent outgoing line).

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
STRENGTHS = (0.0, 1.0, 2.0, 3.0, 6.0, 9.0)
RANGES = torch.tensor([50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 650.0, 800.0])


def edges(name: str, overrides: dict, aspects_deg, ranges: torch.Tensor):
    """(A, R) edge radius and effective radius (m), from one step."""
    inputs = {**card.DEFLECTOR_PRESETS[name], **overrides}
    config = dataclasses.replace(hitbench.CONFIG, deflector="symmetric")
    radius = float(card.DEFLECTOR_PRESETS[name].get("hit_radius", 0.0)) or None
    probe = core.reset(config, 1, torch.tensor([[1, 1]]), None, "cpu")
    reach = core.with_hit_model(probe.constants, **inputs)
    radius = hitbench._value(reach, "hit_radius")
    step = hitbench.OFFSET_STEP
    offsets = torch.arange(-radius - 1.0, radius + 1.0 + 1e-6, step)
    aspects = torch.tensor([math.radians(a) for a in aspects_deg])
    damage, _ = hitbench._damage(config, inputs, aspects, ranges, offsets)
    plain = {**inputs, "deflector_front": 0.0, "deflector_side": 0.0, "nose_cone": 0.0}
    full, _ = hitbench._damage(config, plain, torch.tensor([math.pi]), ranges, torch.zeros(1))
    hit = damage > 0.0
    edge = torch.where(hit, offsets.abs(), 0.0).amax(-1)  # widest damaging offset
    edge = torch.where(hit.any(-1), edge + 0.5 * step, 0.0)
    effective = damage.sum(-1) * step / (2.0 * full[0, :, 0])
    return edge, effective, radius


def apparent_figure(path: Path) -> list[str]:
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    lines = [
        "| D, m | edge radius, m | effective radius, m | edge half-angle at 300 m, mrad "
        "| looks like the undeflected ship at |",
        "|---|---|---|---|---|",
    ]
    base = None
    for k, d in enumerate(STRENGTHS):
        over = dict(deflector_front=d, deflector_side=d, nose_cone=0.0)
        edge, effective, radius = edges("symmetric", over, (180,), RANGES)
        edge, effective = edge[0], effective[0]
        if base is None:
            base = edge
        colour = COLOURS[k % len(COLOURS)]
        label = "no deflector" if d == 0 else f"D = {d:g} m"
        a.plot(RANGES, 1e3 * edge / RANGES, color=colour, lw=2, marker="o", ms=4, label=label)
        i300 = int((RANGES == 300.0).nonzero())
        e = float(edge[i300])
        lines.append(
            f"| {d:g} | {e:.1f} | {float(effective[i300]):.2f} | {1e3 * e / 300:.1f} | "
            + (f"{300 * float(base[i300]) / e:.0f} m |" if e > 0 else "never hit |")
        )
    a.set(
        xlabel="range, m",
        ylabel="half-angle of the target's edge, mrad",
        title="Apparent size from astern: the edge's half-angle\n(same D all round, nose cone off)",
    )
    a.set_yscale("log")
    # metres against D, astern and beam
    ds = torch.linspace(0.0, 12.0, 25)
    for aspect, style in ((180, "-"), (90, "--")):
        e_list, f_list = [], []
        for d in ds:
            over = dict(deflector_front=float(d), deflector_side=float(d), nose_cone=0.0)
            e, f, radius = edges("symmetric", over, (aspect,), RANGES[3:4])
            e_list.append(float(e[0, 0]))
            f_list.append(float(f[0, 0]))
        b.plot(ds, e_list, color=COLOURS[0], lw=2, ls=style, label=f"edge, {aspect} deg")
        b.plot(ds, f_list, color=COLOURS[1], lw=2, ls=style, label=f"effective, {aspect} deg")
    b.axhline(radius, color="#5e5d59", lw=1, ls=":", label=f"hit radius {radius:g} m")
    b.set(
        xlabel="deflector D, m",
        ylabel="radius, m",
        title="Edge and effective radius against D (300 m)",
    )
    # presets by aspect
    aspects = list(range(0, 181, 5))
    for k, name in enumerate(card.DEFLECTOR_PRESETS):
        e, f, _ = edges(name, {}, aspects, RANGES[3:4])
        c.plot(aspects, e[:, 0], color=COLOURS[k], lw=2, label=f"{name}: edge")
        c.plot(aspects, f[:, 0], color=COLOURS[k], lw=2, ls="--", label=f"{name}: effective")
    c.set(
        xlabel="aspect: bullet path off the target's nose, deg (0 head-on)",
        ylabel="radius, m",
        title="Presets by aspect (300 m)",
    )
    c.set_xticks([0, 30, 45, 90, 135, 180])
    for axis in (a, b, c):
        axis.grid(True, color="#e4e3df", lw=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=7, frameon=False)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return lines


def streams(name: str, aspect_deg: float, offsets: torch.Tensor, ticks: int = 4):
    """Tick positions relative to the target, (ticks + 1, O, 2), hit flags (O,), the bend tick's
    closest point (O, 2) and the target's nose direction."""
    inputs = dict(card.DEFLECTOR_PRESETS[name])
    config = dataclasses.replace(hitbench.CONFIG, deflector=name)
    state = hitbench._place(
        config, inputs, torch.tensor([math.radians(aspect_deg)]), torch.tensor([300.0]), offsets
    )
    counts = state.bullet_counts.clone()
    counts[:, core.MOVES_LEFT, 0, hitbench.SLOT] = ticks + 2
    state = state.replace(bullet_counts=counts)
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
    hit = torch.zeros(envs, dtype=torch.bool)
    alive = torch.ones(envs, dtype=torch.bool)
    for _ in range(ticks):
        state, out = core.step(config, state, idle, noise)
        hit |= out.hit_damage[:, 0, 1] > 0
        alive &= state.bullet_counts[:, core.MOVES_LEFT, 0, s] > 0
        p = relative(state)
        points.append(torch.where(alive[:, None], p, torch.nan))
    return torch.stack(points), hit, nose


def stream_figure(path: Path) -> None:
    aspects = (0, 30, 45, 90, 180)
    names = ("symmetric", "frontal")
    offsets = torch.arange(-24.0, 24.01, 2.0)
    fig, axes = plt.subplots(
        len(names), len(aspects), figsize=(4 * len(aspects), 8.4), constrained_layout=True
    )
    probe = core.reset(hitbench.CONFIG, 1, torch.tensor([[1, 1]]), None, "cpu")
    radius = hitbench._value(probe.constants, "hit_radius")
    for row, name in enumerate(names):
        for col, aspect in enumerate(aspects):
            ax = axes[row, col]
            pts, hit, nose = streams(name, aspect, offsets)
            first = pts[0]
            # hitbench puts every pass's closest point mid-tick at offset x across, so the
            # centred bullet starts half a straight move before the centre: move = -2 x start.
            centre = int((offsets.abs() < 1e-6).nonzero())
            move = (-2.0 * first[centre]).expand_as(first)
            for i in range(len(offsets)):
                ref = torch.stack([first[i] - 3 * move[i], first[i] + 4 * move[i]])
                ax.plot(ref[:, 0], ref[:, 1], color="#c9c8c3", lw=0.6, zorder=0)
            closest = first + 0.5 * move  # hitbench places the closest approach mid-tick
            for i in range(len(offsets)):
                incoming = torch.stack([first[i] - 3 * move[i], closest[i]])
                colour = COLOURS[1] if hit[i] else COLOURS[0]
                ax.plot(incoming[:, 0], incoming[:, 1], color=colour, lw=1.2)
                if hit[i] or torch.isnan(pts[2, i, 0]):
                    ax.plot(*closest[i], marker="x", color=COLOURS[1], ms=5)
                    continue
                after = pts[1:, i]
                step_after = after[1] - after[0]
                joined = after[0] - 0.5 * step_after  # the bent line back to the closest point
                bent = torch.cat([joined[None], after])
                ax.plot(
                    [closest[i, 0], joined[0]],
                    [closest[i, 1], joined[1]],
                    color=colour,
                    lw=1.2,
                    ls=":",
                )
                ax.plot(bent[:, 0], bent[:, 1], color=colour, lw=1.2)
            theta = torch.linspace(0, 2 * math.pi, 100)
            ax.plot(radius * torch.cos(theta), radius * torch.sin(theta), color="#5e5d59", lw=1.5)
            ax.annotate(
                "",
                xy=(1.8 * radius * nose[0], 1.8 * radius * nose[1]),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#5e5d59", lw=1.5),
            )
            ax.set_aspect("equal")
            ax.set_xlim(-60, 60)
            ax.set_ylim(-60, 60)
            ax.set_title(f"{name}, {aspect} deg off the nose", fontsize=10)
            ax.grid(True, color="#e4e3df", lw=0.6)
    fig.suptitle(
        "Bullet paths past a target (target's frame, 300 m shot, rows 2 m apart). Circle: hit "
        f"radius {radius:g} m; arrow: nose. x: the bullet stopped on the ship (drawn at the closest point); orange: it did damage, blue with x: no damage (nose cone). "
        "Dotted: the push at the closest point. Grey: the straight paths with no deflector.",
        fontsize=10,
    )
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    text = apparent_figure(OUT / "apparent_size.png")
    print("\n".join(text))
    stream_figure(OUT / "streamlines.png")
    print("wrote", OUT / "apparent_size.png", OUT / "streamlines.png")
