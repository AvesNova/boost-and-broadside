"""The hit-model workbench (proposal 2026-11-27): what a hit model does to one bullet, measured
with the game's own step on synthetic bullets, on the CPU, in seconds.

    PYTHONPATH=. uv run --no-sync python -m lean.tool.hitbench [--presets circle off]
        [--set hit_radius=14 deflector_turn_deg=30 bullet_spread=10 ...] [--sigma 2] [--out DIR]

For each preset (card.DEFLECTOR_PRESETS, with the `--set` inputs of core.HIT_MODEL on top), a few
ticks of core.step on E synthetic environments: a target at the origin flying straight at its
cruise speed, and one of the shooter's bullets placed outside the target's deflector circle on a
straight relative line that would pass the target's centre at a given offset, at a given aspect (the
bullet's path off the target's nose, 0 head-on) and range (the bullet's speed is the launch
speed, cruise plus muzzle, slowed by the game's drag over the range). The damage is the steps'
summed `hit_damage` (the deflector turns the bullet at the circle as in play; the nose armour
holds a hit within its inner angle to no damage). From it:

- the hit width: the band of offsets that do damage; the effective hit radius: the integral of
  damage over the offset over twice the full damage at that range (a centred hit from astern
  with the deflector off: hit damage x kinetic), and the target's apparent angular size, 2 x
  effective radius / range, in milliradians;
- per bullet, against a Gaussian aim error sigma (milliradians, the shooter's) plus the game's
  spread (spread_speed per axis over the launch speed): the hit chance (damage above 0) and the
  expected damage; damage per second while tracking (x the fire rate); the time to kill from a
  full shield (the shield over the expected damage, plus one more hit, over the fire rate; no
  recharge, since a hit restarts the wait).

Prints a table per preset and writes `hitbench.png` (and `hitbench.md`) in --out. Nothing here
is a copy of a game number: every value is read from the constants of the state it steps.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import time
from pathlib import Path

import torch

from lean.contract import REAL
from lean.game import card, core

ASPECTS_DEG = (0, 10, 20, 25, 30, 35, 40, 45, 50, 60, 90, 120, 150, 180)
RANGES = (100.0, 200.0, 300.0, 400.0, 500.0, 650.0)
OFFSET_STEP = 0.1  # m
SIGMA_PANEL = 40.0  # mrad: the aim-error panel runs 0 to this
CONFIG = core.GameConfig(field_count=0, fog=False)
SLOT = 1  # the test bullet's slot in the shooter's ring
TICKS = 5  # steps per measurement: the bullet's pass falls inside them at every range


@dataclasses.dataclass
class Bench:
    """One preset's measurement: damage (aspect, range, offset) and the constants it used."""

    name: str
    inputs: dict
    offsets: torch.Tensor  # (O,)
    damage: torch.Tensor  # (A, R, O)
    full: torch.Tensor  # (R,): a centred hit from astern, deflector off
    launch_speed: float
    fire_rate: float
    max_shield: float


def _value(constants, name: str) -> float:
    return float(torch.as_tensor(constants[name]).flatten()[0])


def _place(config, inputs, aspects, ranges, offsets):
    """The synthetic state: E = A x R x O environments (aspect, range, offset)."""
    a, r, o = torch.meshgrid(aspects, ranges, offsets, indexing="ij")
    a, r, o = a.flatten(), r.flatten(), o.flatten()
    envs = len(a)
    state = core.reset(config, envs, torch.tensor([[1, 1]]).expand(envs, 2), None, "cpu")
    state = state.replace(constants=core.with_hit_model(state.constants, **inputs))
    c = state.constants
    cruise, launch = _value(c, "cruise_speed"), _value(c, "bullet_launch_speed")
    drag = _value(c, "bullet_drag")
    bullet_speed = launch * torch.exp(-drag * r)
    # The bullet flies +x; the target's nose is the aspect off the reversed path.
    nose = math.pi - a
    nose_x, nose_y = torch.cos(nose), torch.sin(nose)
    relative_x = bullet_speed - cruise * nose_x
    relative_y = -cruise * nose_y
    length = torch.sqrt(relative_x**2 + relative_y**2)
    across_x, across_y = -relative_y / length, relative_x / length
    # Back along the relative line from the pass, beyond the deflector circle.
    back = float(inputs["deflector_radius"]) + 20.0
    start_x = o * across_x - back * relative_x / length
    start_y = o * across_y - back * relative_y / length
    zeros, ones = torch.zeros(envs), torch.ones(envs)
    state = core.with_velocity(
        state.replace(
            x=torch.stack([zeros - 5000.0, zeros], 1),
            y=torch.stack([zeros, zeros], 1),
            heading_cos=torch.stack([ones, nose_x], 1),
            heading_sin=torch.stack([zeros, nose_y], 1),
            speed=torch.stack([ones * cruise, ones * cruise], 1),
            angle=torch.zeros(envs, 2),
            angle_cos=torch.ones(envs, 2),
            angle_sin=torch.zeros(envs, 2),
            nose_rate=torch.zeros(envs, 2),
        )
    )
    bullets = state.bullets.clone()
    bullets[:, :, 0, SLOT] = 0.0
    bullets[:, core.BULLET_X, 0, SLOT] = start_x
    bullets[:, core.BULLET_Y, 0, SLOT] = start_y
    bullets[:, core.BULLET_DIR_X, 0, SLOT] = 1.0
    bullets[:, core.BULLET_SPEED, 0, SLOT] = bullet_speed
    bullet_state = state.bullet_state.clone()
    bullet_state[:, 0, SLOT] = REAL
    # The tick after the slot's: the test bullet flies BULLET_SLOTS - 1 more ticks (> TICKS).
    return state.replace(
        bullets=bullets,
        bullet_state=bullet_state,
        tick=torch.full((envs,), SLOT + 1, dtype=torch.long),
    )


def _damage(config, inputs, aspects, ranges, offsets) -> tuple[torch.Tensor, object]:
    state = _place(config, inputs, aspects, ranges, offsets)
    constants = state.constants
    envs = state.x.shape[0]
    idle = torch.zeros(envs, 2, 3, dtype=torch.long)
    damage = torch.zeros(envs)
    for _ in range(TICKS):
        state, out = core.step(
            config, state.replace(pending=idle), idle, {"spread": torch.zeros(envs, 2, 2)}
        )
        damage = damage + out.hit_damage[:, 0, 1]
    shape = (len(aspects), len(ranges), len(offsets))
    return damage.reshape(shape), constants


def measure(name: str, overrides: dict | None = None) -> Bench:
    """One preset (with `overrides`, core.HIT_MODEL names) measured on the step."""
    inputs = {**card.DEFLECTOR_PRESETS[name], **(overrides or {})}
    # The deflector's code on for every preset ("off": no turn, as the skipped code).
    config = dataclasses.replace(CONFIG, deflector="circle")
    probe = core.reset(config, 1, torch.tensor([[1, 1]]), None, "cpu")
    reach = core.with_hit_model(probe.constants, **inputs)
    radius = _value(reach, "hit_radius")
    offsets = torch.arange(-radius - 1.0, radius + 1.0 + 1e-6, OFFSET_STEP)
    aspects = torch.tensor([math.radians(a) for a in ASPECTS_DEG])
    ranges = torch.tensor(RANGES)
    damage, constants = _damage(config, inputs, aspects, ranges, offsets)
    # Full damage: a centred hit from astern, the deflector and nose cone off.
    plain = {**inputs, "deflector_turn": 0.0, "nose_cone": 0.0}
    full, _ = _damage(config, plain, torch.tensor([math.pi]), ranges, torch.zeros(1))
    interval = _value(constants, "fire_interval")
    return Bench(
        name,
        core.hit_model_of(constants),
        offsets,
        damage,
        full[0, :, 0],
        _value(constants, "bullet_launch_speed"),
        1.0 / (interval * config.tick_length),
        _value(constants, "max_shield"),
    )


def hit_width(bench: Bench) -> torch.Tensor:
    """(A, R) metres: the band of offsets that do damage."""
    return (bench.damage > 0.0).float().sum(-1) * OFFSET_STEP


def effective_radius(bench: Bench) -> torch.Tensor:
    """(A, R) metres: the damage's integral over the offset over twice the centred hit's at
    that aspect and range (never turned, inside the core: the full damage; the aspect sets the
    bullet's slowing on the way, the target flying on)."""
    centre = bench.damage[..., int(bench.offsets.abs().argmin())]
    return bench.damage.sum(-1) * OFFSET_STEP / (2.0 * centre.clamp_min(1e-9))


def per_bullet(bench: Bench, sigma_mrad: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(A, R) hit chance and expected damage per bullet at aim error sigma (mrad) plus the
    game's spread: the lateral miss at range r is Gaussian with standard deviation r x
    sqrt(sigma^2 + (spread / launch speed)^2)."""
    spread = bench.inputs["bullet_spread"] / bench.launch_speed
    angle = math.hypot(sigma_mrad * 1e-3, spread)
    ranges = torch.tensor(RANGES)
    width = (ranges * angle).clamp_min(1e-6)[:, None]  # (R, 1)
    density = torch.exp(-0.5 * (bench.offsets[None] / width) ** 2) / (
        width * math.sqrt(2 * math.pi)
    )
    weight = density * OFFSET_STEP  # (R, O); offsets beyond the grid do nothing
    chance = ((bench.damage > 0.0).float() * weight).sum(-1)
    expected = (bench.damage * weight).sum(-1)
    return chance, expected


def time_to_kill(bench: Bench, sigma_mrad: float) -> torch.Tensor:
    """(A, R) seconds from a full shield: shield / expected damage + 1 / hit chance shots."""
    chance, expected = per_bullet(bench, sigma_mrad)
    shots = bench.max_shield / expected.clamp_min(1e-9) + 1.0 / chance.clamp_min(1e-9)
    return torch.where(expected > 0.0, shots / bench.fire_rate, torch.inf)


def table(bench: Bench, sigma: float, range_index: int = 2) -> str:
    """Markdown: by aspect at RANGES[range_index]: effective radius, mrad, hit chance, damage per
    bullet, damage per second, time to kill."""
    radius = effective_radius(bench)
    chance, expected = per_bullet(bench, sigma)
    kill = time_to_kill(bench, sigma)
    r = RANGES[range_index]
    lines = [
        f"**{bench.name}** at {r:.0f} m, aim error {sigma:g} mrad + spread "
        f"{1e3 * bench.inputs['bullet_spread'] / bench.launch_speed:.2f} mrad; full hit "
        f"{float(bench.full[range_index]):.2f}",
        "",
        "| aspect (deg off nose) | hit width, m | effective radius, m | apparent size, mrad "
        "| hit chance | damage / bullet | damage / s | time to kill, s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    width = hit_width(bench)
    for i, aspect in enumerate(ASPECTS_DEG):
        effective = float(radius[i, range_index])
        ttk = float(kill[i, range_index])
        lines.append(
            f"| {aspect} | {float(width[i, range_index]):.1f} | {effective:.2f} | "
            f"{2e3 * effective / r:.1f} | "
            f"{float(chance[i, range_index]):.3f} | {float(expected[i, range_index]):.3f} | "
            f"{float(expected[i, range_index]) * bench.fire_rate:.1f} | "
            f"{'never' if math.isinf(ttk) else f'{ttk:.2f}'} |"
        )
    return "\n".join(lines)


COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")  # dataviz categorical order


def figure(benches: list[Bench], sigma: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    range_index = 2
    r = RANGES[range_index]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    (a, b), (c, d) = axes
    for k, bench in enumerate(benches):
        colour = COLOURS[k % len(COLOURS)]
        radius = effective_radius(bench)[:, range_index]
        a.plot(ASPECTS_DEG, radius, color=colour, lw=2, marker="o", ms=4, label=bench.name)
        for aspect, style in ((90, "-"), (0, ":")):
            i = ASPECTS_DEG.index(aspect)
            b.plot(
                bench.offsets,
                bench.damage[i, range_index],
                color=colour,
                lw=2,
                ls=style,
                label=f"{bench.name}, {aspect} deg",
            )
        sigmas = torch.linspace(0.0, SIGMA_PANEL, 41)
        for aspect, style in ((180, "-"), (90, "--")):
            i = ASPECTS_DEG.index(aspect)
            chances = [float(per_bullet(bench, float(s))[0][i, range_index]) for s in sigmas]
            c.plot(
                sigmas, chances, color=colour, lw=2, ls=style, label=f"{bench.name}, {aspect} deg"
            )
        kill = time_to_kill(bench, sigma)
        for aspect, style in ((180, "-"), (90, "--")):
            i = ASPECTS_DEG.index(aspect)
            d.plot(
                RANGES,
                kill[i],
                color=colour,
                lw=2,
                ls=style,
                marker="o",
                ms=4,
                label=f"{bench.name}, {aspect} deg",
            )
    a.set(
        xlabel="aspect: bullet path off the target's nose, deg",
        ylabel="effective hit radius, m",
        title=f"Effective hit radius at {r:.0f} m",
    )
    b.set(
        xlabel="offset of the pass from the target's centre, m",
        ylabel="damage per bullet",
        title=f"Damage against the aim offset at {r:.0f} m (solid 90 deg, dotted head-on)",
    )
    c.set(
        xlabel="aim error sigma, mrad (plus the game's spread)",
        ylabel="hit chance per bullet",
        title=f"Hit chance at {r:.0f} m (solid astern, dashed 90 deg)",
    )
    d.set(
        xlabel="range, m",
        ylabel="time to kill from a full shield, s",
        title=f"Time to kill, aim error {sigma:g} mrad (solid astern, dashed 90 deg)",
    )
    for axis in axes.flat:
        axis.grid(True, color="#e4e3df", lw=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=7, frameon=False)
    a.set_xticks([0, 30, 45, 90, 135, 180])
    fig.savefig(path, dpi=110)
    plt.close(fig)


def parse_set(items: list[str]) -> dict:
    out = {}
    for item in items:
        name, _, value = item.partition("=")
        if name.endswith("_deg"):
            name, value = name[: -len("_deg")], math.radians(float(value))
        out[name] = float(value)
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--presets", nargs="+", default=list(card.DEFLECTOR_PRESETS))
    parser.add_argument(
        "--set", nargs="*", default=[], help="hit-model inputs, name=value (angles: name_deg=)"
    )
    parser.add_argument("--sigma", type=float, default=10.0, help="aim error, mrad")
    parser.add_argument(
        "--out", type=Path, default=Path("docs/rewrite/sections/bullet-deflector/workbench")
    )
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    started = time.perf_counter()
    overrides = parse_set(args.set)
    benches = [measure(name, overrides) for name in args.presets]
    measured = time.perf_counter() - started
    args.out.mkdir(parents=True, exist_ok=True)
    text = "\n\n".join(table(bench, args.sigma) for bench in benches)
    if overrides:
        text = f"Inputs set: {overrides}\n\n" + text
    print(text)
    (args.out / "hitbench.md").write_text(text + "\n")
    figure(benches, args.sigma, args.out / "hitbench.png")
    print(
        f"\nmeasured in {measured:.1f} s, all in {time.perf_counter() - started:.1f} s; "
        f"wrote {args.out / 'hitbench.png'} and hitbench.md"
    )


if __name__ == "__main__":
    main()
