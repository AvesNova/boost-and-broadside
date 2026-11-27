"""Round 2 (revised brief): markdown tables from the surface (surface-r2.json, top-r2.json,
report-r2.json) and the proposed persona box (CPU only).

    summarise2.py > tables-2-surface.md      # also writes proposal-2.json beside this file
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse2  # noqa: E402

from lean.scripted import PARAMETERS  # noqa: E402
from lean.scripted.parameters import SAMPLED  # noqa: E402

HERE = Path(__file__).resolve().parent
INERT = ("boundary_weight", "seek_weight", "field_weight")


def main():
    top = json.loads((HERE / "top-r2.json").read_text())
    report = json.loads((HERE / "report-r2.json").read_text())
    fit = json.loads((HERE / "surface-r2.json").read_text())
    names = top["names"]
    print("## The surface: fit\n")
    rows = [
        [
            k,
            f"{v[0]:.5f}",
            f"{v[1]:.5f}",
            f"{(fit['coin'] - v[0]) / (fit['coin'] - fit['saturated']):.1%}",
            f"{(fit['coin'] - v[1]) / (fit['coin'] - fit['saturated']):.1%}",
        ]
        for k, v in fit["cv"].items()
    ]
    print(
        analyse2.table(
            ["lambda", "per size", "shared", "per size explained", "shared explained"], rows
        )
    )
    print(
        f"\nCoin {fit['coin']:.5f}, saturated {fit['saturated']:.5f} per game; agents "
        f"{fit['agents']}, games {fit['games']}.\n"
    )
    print("## Peak, LCB optimum, robust centre (relative Elo per size: offsets arbitrary)\n")
    rows = []
    for key in ("peak", "lcb", "centre"):
        e = top[key]
        rows.append(
            [key, f"{e['g']:.0f}", " / ".join(f"{v:.0f}" for v in e["per_size"]), f"{e['sd']:.0f}"]
        )
    print(analyse2.table(["point", "mean over sizes", "1v1 / 4v4 / 16v16", "SD"], rows) + "\n")
    print("## Rows: old yardstick, robust centre, peak, the 50-Elo region (5-95%)\n")
    rows = []
    for i, name in enumerate(names):
        c, p = top["centre"]["theta"][i], top["peak"]["theta"][i]
        lo, hi = top["region"]["low"][i], top["region"]["high"][i]
        if name in PARAMETERS and name in SAMPLED:
            row = PARAMETERS[name]

            def value(u, row=row):
                if row.law == "log":
                    return row.default * math.exp(row.width * (2 * u - 1))
                return row.default + row.width * (2 * u - 1)

            rows.append(
                [
                    name,
                    f"{row.default:g}",
                    f"{value(c):.3g} (u {c:.2f})",
                    f"{value(p):.3g} (u {p:.2f})",
                    f"{value(lo):.3g} - {value(hi):.3g} (u {lo:.2f} - {hi:.2f})",
                ]
            )
        else:
            rows.append([name, "-", f"v {c:+.2f}", f"v {p:+.2f}", f"v {lo:+.2f} - {hi:+.2f}"])
    print(
        analyse2.table(["row", "old yardstick", "robust centre", "peak", "50-Elo region"], rows)
        + "\n"
    )
    print(
        "Crowd rows (v): the crowd value is the duel value x e^(2 width v); "
        "store_crowd k = e^(2 v).\n"
    )
    # Flat directions: the Hessian's eigenvectors with the smallest |curvature|.
    values = np.array(top["hessian_eigenvalues"])
    vectors = np.array(top["hessian_eigenvectors"])
    order = np.argsort(np.abs(values))
    print("## Directions at the centre: curvature (Elo per box^2) and the largest loadings\n")
    rows = []
    for k in list(order[:5]) + list(np.argsort(values)[:3]):
        v = vectors[k]
        load = np.argsort(-np.abs(v))[:4]
        rows.append([f"{values[k]:+.1f}", ", ".join(f"{names[j]} {v[j]:+.2f}" for j in load)])
    print(analyse2.table(["curvature", "loadings"], rows) + "\n")
    for size in ("1", "4", "16"):
        print(f"## {size}v{size}: largest interactions (Elo per box^2), Sobol on the surface\n")
        rows = [
            [f"{a} x {b}", f"{c:+.0f} +- {e:.0f}"] for a, b, c, e in report["products"][size][:6]
        ]
        print(analyse2.table(["product", "coefficient"], rows) + "\n")
        s = report["sobol"][size]
        order_s = np.argsort(-np.array(s["total"]))[:10]
        rows = [
            [
                SAMPLED[i],
                f"{s['first'][i]:.3f}",
                f"{s['total'][i]:.3f}",
                f"{s['total'][i] - s['first'][i]:+.3f}",
            ]
            for i in order_s
        ]
        print(analyse2.table(["row", "S1", "ST", "ST - S1"], rows))
        print(f"\nModel Elo sd over the box: {s['sd']:.0f}.\n")
    print(
        "## Working ranges: u where f is within 50 / 100 Elo of the row's best "
        "(slice through the centre)\n"
    )
    rows = []
    for name in SAMPLED:
        cells = [name]
        for size in ("1", "4", "16"):
            r = report["ranges"][size][name]
            cells.append(
                f"best {r['best_u']:.2f}; {r['u50'][0]:.2f}-{r['u50'][1]:.2f} / "
                f"{r['u100'][0]:.2f}-{r['u100'][1]:.2f}"
            )
        rows.append(cells)
    print(analyse2.table(["row", "1v1", "4v4", "16v16"], rows) + "\n")
    # The proposed persona box: centre = the robust centre; per kept row a half-width (u units)
    # = half the narrowest 50-Elo slice range over the sizes, clipped to [0.1, 0.5].
    proposal = {"centre": top["centre"]["config"], "box": {}}
    rows = []
    for name in SAMPLED:
        if name in INERT:
            continue
        spans = [
            report["ranges"][s][name]["u50"][1] - report["ranges"][s][name]["u50"][0]
            for s in ("1", "4", "16")
        ]
        half = float(np.clip(min(spans) / 2.0, 0.1, 0.5))
        row = PARAMETERS[name]
        width = 2.0 * half * row.width  # in the row's law units (log or linear)
        proposal["box"][name] = {"law": row.law, "width": width, "half_u": half}
        rows.append([name, row.law, f"{row.width:g}", f"{width:.3g}", f"{half:.2f}"])
    print("## Proposed persona box (around the robust centre; inert rows held)\n")
    print(
        analyse2.table(
            ["row", "law", "round-1 width", "proposed width", "half-width (box u)"], rows
        )
        + "\n"
    )
    print("League personas (surface predictions, relative Elo):\n")
    rows = [
        [k, f"{v['g']:.0f}", " / ".join(f"{x:.0f}" for x in v["per_size"]), f"{v['sd']:.0f}"]
        for k, v in report["league"].items()
    ]
    print(analyse2.table(["persona", "mean", "1v1 / 4v4 / 16v16", "SD"], rows) + "\n")
    (HERE / "proposal-2.json").write_text(json.dumps(proposal, indent=1))


if __name__ == "__main__":
    main()
