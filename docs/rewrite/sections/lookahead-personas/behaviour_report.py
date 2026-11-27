"""Round 2 (revised brief): behaviour tables around the robust centre (CPU only).

    behaviour_report.py DATA > tables-2-behaviour.md

Sobol indices (SALib, first and total order) of the mean JS from the centre over the spread-1
box around it (blur2.py saltelli, base 32: 736 configurations per size); the label blur at
spreads 0.25 / 0.5 / 1 (blur2.py blur, inert rows held); the league candidates' pairwise JS
(league_js.py).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse2  # noqa: E402

from lean.scripted.parameters import SAMPLED  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    from SALib.analyze import sobol

    data = Path(sys.argv[1])
    problem = {"num_vars": len(SAMPLED), "names": list(SAMPLED), "bounds": [[0, 1]] * len(SAMPLED)}
    out = {"sobol": {}, "blur": {}}
    tables = {}
    for size in (1, 4, 16):
        path = data / f"saltelli_{size}v{size}.npz"
        if not path.exists():
            continue
        y = np.load(path)["js"]
        si = sobol.analyze(problem, y, calc_second_order=False, seed=1)
        out["sobol"][size] = {
            "mean_js": float(y.mean()),
            "S1": si["S1"].tolist(),
            "ST": si["ST"].tolist(),
            "S1_conf": si["S1_conf"].tolist(),
            "ST_conf": si["ST_conf"].tolist(),
        }
        tables[size] = si
    if tables:
        order = np.argsort(-np.mean([t["ST"] for t in tables.values()], 0))
        rows = []
        for i in order:
            cells = [SAMPLED[i]]
            for t in tables.values():
                cells.append(f"{t['S1'][i]:.3f} / {t['ST'][i]:.3f} +- {t['ST_conf'][i]:.3f}")
            rows.append(cells)
        header = ["row"] + [f"{s}v{s} S1 / ST" for s in tables]
        print("## Behaviour: Sobol indices of the JS from the robust centre (spread-1 box)\n")
        print(analyse2.table(header, rows))
        means = ", ".join(f"{s}v{s} {out['sobol'][s]['mean_js']:.4f}" for s in tables)
        print(f"\nMean JS from the centre over the sample: {means}.\n")
    rows = []
    for path in sorted(data.glob("blur-box_*.json")):
        r = json.loads(path.read_text())
        size, spread = path.stem.split("_")[1], path.stem.split("_")[2]
        out["blur"][f"{size} {spread}"] = r
        rows.append(
            [
                size,
                spread,
                r["ship_states"],
                f"{r['centre_entropy']:.3f}",
                f"{r['member_entropy']:.3f}",
                f"{r['blur']:.4f} ({r['blur'] / r['member_entropy']:.1%})",
                f"{r['from_centre_mean']:.4f}",
                f"{r['from_centre_max']:.4f}",
                f"{r['mixture_from_centre']:.5f}",
            ]
        )
    if rows:
        print("## Label blur around the robust centre (64 personas; boundary, seek, field held)\n")
        print(
            analyse2.table(
                [
                    "size",
                    "spread",
                    "ship-states",
                    "centre H",
                    "members' H",
                    "blur (share)",
                    "JS to centre mean",
                    "max",
                    "mixture to centre",
                ],
                rows,
            )
            + "\n"
        )
    path = HERE / "league-js-2.json"
    if path.exists():
        league = json.loads(path.read_text())
        out["league"] = league
        for size, r in league.items():
            names = r["names"]
            m = np.array(r["js"])
            print(f"## Pairwise JS between the league candidates, {size}\n")
            print(
                analyse2.table(
                    [""] + names,
                    [
                        [a] + [f"{m[i, j]:.4f}" for j in range(len(names))]
                        for i, a in enumerate(names)
                    ],
                )
                + "\n"
            )
    (HERE / "behaviour-2.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
