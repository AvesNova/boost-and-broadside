"""A still of the deflector's bend: bullets drawn tick to tick (as the sandbox draws them, each
tick-end position joined to the next), in the target's frame, for the three presets.

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/bullet-deflector/bend_still.py OUT.png

The game's step on synthetic bullets (lean.tool.hitbench's placement: the target flying at
cruise, the bullet at the launch speed slowed over 300 m), a fan of offsets from -14 to 14 m,
head-on (0 deg) and from the side (90 deg); 4 ticks around the pass. Positions are drawn
relative to the target's position at the same tick, its nose up.
"""

import dataclasses
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from lean.game import card, core  # noqa: E402
from lean.tool import hitbench  # noqa: E402

OFFSETS = torch.linspace(-14.0, 14.0, 15)


def paths(preset: str, aspect: float):
    config = dataclasses.replace(hitbench.CONFIG, deflector="symmetric")
    inputs = card.DEFLECTOR_PRESETS[preset]
    state = hitbench._place(config, inputs, torch.tensor([aspect]), torch.tensor([300.0]), OFFSETS)
    # Start two ticks before the pass: step the bullet back along its path.
    speed = state.bullets[:, core.BULLET_SPEED, 0, hitbench.SLOT]
    bullets = state.bullets.clone()
    bullets[:, core.BULLET_X, 0, hitbench.SLOT] -= 2.0 * speed * config.tick_length
    counts = state.bullet_counts.clone()
    counts[:, core.MOVES_LEFT, 0, hitbench.SLOT] = 6
    state = state.replace(bullets=bullets, bullet_counts=counts)
    # The target's path two ticks earlier is close enough to its straight line: move it back.
    state = state.replace(
        x=state.x - 2.0 * config.tick_length * state.velocity_x,
        y=state.y - 2.0 * config.tick_length * state.velocity_y,
    )
    envs = state.x.shape[0]
    idle = torch.zeros(envs, 2, 3, dtype=torch.long)
    points, alive = [], []
    for _ in range(5):
        heading = math.atan2(float(state.heading_sin[0, 1]), float(state.heading_cos[0, 1]))
        turn = math.pi / 2 - heading  # nose up
        dx = state.bullets[:, core.BULLET_X, 0, hitbench.SLOT] - state.x[:, 1]
        dy = state.bullets[:, core.BULLET_Y, 0, hitbench.SLOT] - state.y[:, 1]
        points.append(
            torch.stack(
                [
                    dx * math.cos(turn) - dy * math.sin(turn),
                    dx * math.sin(turn) + dy * math.cos(turn),
                ],
                -1,
            )
        )
        alive.append(state.bullet_moves_left[:, 0, hitbench.SLOT] > 0)
        state, _ = core.step(
            config, state.replace(pending=idle), idle, {"spread": torch.zeros(envs, 2, 2)}
        )
    radius = float(torch.as_tensor(state.constants["hit_radius"]).flatten()[0])
    return torch.stack(points, 1), torch.stack(alive, 1), radius


def main():
    out = sys.argv[1]
    presets = list(card.DEFLECTOR_PRESETS)
    fig, axes = plt.subplots(
        2, len(presets), figsize=(4 * len(presets), 8), constrained_layout=True
    )
    for column, preset in enumerate(presets):
        for row, aspect in enumerate((0.0, 90.0)):
            axis = axes[row, column]
            xy, alive, radius = paths(preset, math.radians(aspect))
            for i in range(len(OFFSETS)):
                live = alive[i].tolist()
                last = max(k for k, a in enumerate(live) if a) if any(live) else 0
                # a bullet that hit ends at its contact tick: drawn to that tick's end
                end = min(last + 2, xy.shape[1])
                axis.plot(xy[i, :end, 0], xy[i, :end, 1], color="#2a78d6", lw=1.2, marker=".", ms=4)
            axis.add_patch(plt.Circle((0, 0), radius, fill=False, color="#eb6834", lw=1.2))
            axis.annotate(
                "", xy=(0, 8), xytext=(0, -8), arrowprops=dict(arrowstyle="->", color="#0b0b0b")
            )
            axis.set(
                xlim=(-60, 60),
                ylim=(-60, 60),
                aspect="equal",
                title=f"{preset}, bullets from {aspect:.0f} deg off the nose",
            )
            axis.grid(True, color="#e4e3df", lw=0.6)
    fig.suptitle(
        "Bullets drawn tick to tick in the target's frame (nose up; circle: hit radius). "
        "A hit ends the line.",
        fontsize=10,
    )
    fig.savefig(out, dpi=100)


if __name__ == "__main__":
    main()
