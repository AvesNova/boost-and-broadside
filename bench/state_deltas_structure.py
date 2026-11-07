"""The strong structure of the one-tick changes (docs/rewrite/sections/state-deltas.md,
section 2), from the subsamples bench/state_deltas_measure.py saved: point masses (exact
zero, discrete steps), the clusters by power and stick, and the rare large events and how
often they occur. Prints markdown tables."""

import json
import pathlib

import numpy as np

SCRATCH = pathlib.Path("/tmp/bnb-scratch-deltas")
PLAYERS = ("scripted", "calm-fern")
PRESETS = ("frontline", "deathmatch", "deathmatch_wide")
EVENTS = ("fired", "shot", "boundary", "field", "zone", "respawned", "boost", "reverse")


def share(mask) -> str:
    """A boolean mask's share as a percentage."""
    return f"{100 * mask.mean():.2f}%"


def main() -> None:
    """Print the structure tables and write structure.json."""
    files = {}
    for player in PLAYERS:
        for preset in PRESETS:
            path = SCRATCH / "samples" / player / f"{preset}.npz"
            if path.exists():
                files[player, preset] = np.load(path)
    out = {}
    print("| player | preset | " + " | ".join(EVENTS) + " |")
    print("|---|---|" + "---|" * len(EVENTS))
    for (player, preset), f in files.items():
        cells = [share(f[f"_situation.{e}"]) for e in EVENTS]
        print(f"| {player} | {preset} | " + " | ".join(cells) + " |")
    print()
    # point masses and discrete steps
    rows = []
    for (player, preset), f in files.items():
        entry = {}
        cooldown = np.round(f["gun_cooldown.countdown.1.value"]).astype(int)
        values, counts = np.unique(cooldown, return_counts=True)
        entry["gun_cooldown residual values (ticks: share)"] = {
            int(v): round(float(c / cooldown.size), 5) for v, c in zip(values, counts)
        }
        wait = f["shield_recharge_wait.countdown.1.value"]
        entry["recharge wait residual: |r| < 1e-5"] = float((np.abs(wait) < 1e-5).mean())
        entry["recharge wait residual: reset to the delay (r > 1 s)"] = float((wait > 1).mean())
        shield = f["shield.recharge.1.value"]
        entry["shield residual: |r| < 1e-6"] = float((np.abs(shield) < 1e-6).mean())
        entry["shield residual: loss > 0.05 (a hit)"] = float((shield < -0.05).mean())
        entry["shield residual: small loss (boundary), 0 > r > -0.05"] = float(
            ((shield < -1e-6) & (shield > -0.05)).mean()
        )
        energy = f["energy.persistence.1.value"]
        fired = f["_situation.fired"]
        entry["energy change when fired: median"] = float(np.median(energy[fired]))
        entry["energy change when not fired: p0.1 / p99.9"] = [
            float(np.percentile(energy[~fired], 0.1)),
            float(np.percentile(energy[~fired], 99.9)),
        ]
        index = f["log_refractive_index.persistence.1.value"]
        entry["ln n change exactly 0"] = float((index == 0).mean())
        entry["ln n change |d| > 0.05 (a field edge crossed fast)"] = float(
            (np.abs(index) > 0.05).mean()
        )
        nose = f["nose_rate.persistence.1.value"]
        field = f["_situation.field"]
        entry["nose rate |d| > 2 rad/s: share, of which in a field"] = [
            float((np.abs(nose) > 2).mean()),
            float(field[np.abs(nose) > 2].mean()) if (np.abs(nose) > 2).any() else 0.0,
        ]
        velocity = f["velocity.rotated.1.norm"]
        big = velocity > 3.0
        entry["velocity residual > 3 m/s: share, of which in a field"] = [
            float(big.mean()),
            float(field[big].mean()) if big.any() else 0.0,
        ]
        out[f"{player} {preset}"] = entry
        rows.append((player, preset, entry))
    print("```")
    print(json.dumps(out, indent=1))
    print("```")
    # clusters: the body-frame velocity change by power and stick sign
    print()
    print("| player | preset | power | velocity along nose: mean / p5 / p95 (m/s) | energy: mean |")
    print("|---|---|---|---|---|")
    for (player, preset), f in files.items():
        along = f["velocity.rotated.1.nose_along"]
        energy = f["energy.persistence.1.value"]
        fired = f["_situation.fired"]
        for power in ("coast", "boost", "reverse"):
            m = f[f"_situation.{power}"] & ~fired
            if m.any():
                p5, p95 = np.percentile(along[m], (5, 95))
                print(
                    f"| {player} | {preset} | {power} | {along[m].mean():.3f} / {p5:.3f} / "
                    f"{p95:.3f} | {energy[m].mean():.2e} |"
                )
    (SCRATCH / "structure.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
