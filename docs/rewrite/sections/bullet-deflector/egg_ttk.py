"""Time to kill under the egg by spread, range and aspect, from the workbench (lean.tool.hitbench:
the game's step on synthetic bullets), for a few hit damages; and which (hit damage, spread)
pairs land in the author's band (2026-11-27: about 0.4 s the floor with perfect aim, every
shot hitting close; about 1.0 s the upper limit of a well-aimed burst where only dispersion and
distance cost hits).

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/bullet-deflector/egg_ttk.py OUT.md
"""

import sys

import torch

from lean.game import core
from lean.tool import hitbench

SPREADS_MRAD = (None, 5.0, 10.0, 20.0)  # None: today's
DAMAGES = (7.0, 8.5, 10.5, 14.0)
SIGMAS_MRAD = (0.0, 3.0)  # the shooter's own aim error on top of the spread
ASPECTS = (180, 90, 0)


def main():
    torch.set_num_threads(8)
    launch = float(
        torch.as_tensor(core.ship_card(core.GameConfig())["bullet_launch_speed"]).flatten()[0]
    )
    lines = [
        "Time to kill from a full shield, seconds (workbench: shield / expected damage per "
        "bullet + 1 / hit chance, at the fire rate; no recharge). Rows: hit damage, spread, "
        "aim error; columns: range x aspect (astern / 90 deg / head-on).",
        "",
    ]
    header = (
        "| deflector | hit damage | spread, mrad (m/s) | aim error, mrad | "
        + " | ".join(f"{r:.0f} m {a}" for r in hitbench.RANGES for a in ASPECTS)
        + " |"
    )
    lines += [header, "|---" * (4 + len(hitbench.RANGES) * len(ASPECTS)) + "|"]
    band = []
    for preset in ("egg", "off"):
        for damage in DAMAGES:
            for mrad in SPREADS_MRAD:
                spread = None if mrad is None else mrad * 1e-3 * launch
                overrides = {"hit_damage": damage}
                if spread is not None:
                    overrides["bullet_spread"] = spread
                bench = hitbench.measure(preset, overrides)
                shown = 1e3 * bench.inputs["bullet_spread"] / bench.launch_speed
                for sigma in SIGMAS_MRAD:
                    kill = hitbench.time_to_kill(bench, sigma)
                    cells = []
                    for r in range(len(hitbench.RANGES)):
                        for a in ASPECTS:
                            value = float(kill[hitbench.ASPECTS_DEG.index(a), r])
                            cells.append("never" if value == float("inf") else f"{value:.2f}")
                    lines.append(
                        f"| {preset} | {damage:g} | {shown:.1f} "
                        f"({bench.inputs['bullet_spread']:.1f}) "
                        f"| {sigma:g} | " + " | ".join(cells) + " |"
                    )
                    if preset == "egg" and sigma == 0.0:
                        astern = kill[hitbench.ASPECTS_DEG.index(180)]
                        close, far = float(astern[0]), float(astern[-2])  # 100 m and 500 m
                        band.append((damage, shown, close, far))
                print(lines[-1], flush=True)
    lines += [
        "",
        "Band check (egg, perfect aim, astern): the 100 m time near the 0.4 s floor (0.35 to "
        "0.5 s) and the 500 m time at most about 1.0 s (1.05 s).",
        "",
        "| hit damage | spread, mrad | TTK 100 m, s | TTK 500 m, s | in band |",
        "|---|---|---|---|---|",
    ]
    for damage, shown, close, far in band:
        ok = 0.35 <= close <= 0.5 and far <= 1.05
        lines.append(
            f"| {damage:g} | {shown:.1f} | {close:.2f} | {far:.2f} | {'yes' if ok else 'no'} |"
        )
    text = "\n".join(lines) + "\n"
    print(text)
    open(sys.argv[1], "w").write(text)


if __name__ == "__main__":
    main()
