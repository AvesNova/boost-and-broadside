"""Round 3: the revised sampler (CPU): as surface3.py sampler, but each direction's sd capped
at 0.35 box units (about the box sampler's per-row sd) instead of 0.75, since the surface does
not resolve the curvature of the flat directions (local3.py: r 0.04-0.14 on the sampler's
measured drops). c for a predicted drop of 50; 24 personas (group sampler2).

    sampler2.py OUT.json
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round3  # noqa: E402
import surface  # noqa: E402
import surface3  # noqa: E402

HERE = Path(__file__).resolve().parent
surface3.SD_CAP = 0.35


def main():
    space = surface.Space2(surface.Surface(HERE / "surface-r3.json"))
    centre = np.array(json.loads((HERE / "top-r3.json").read_text())["centre"]["theta"])
    sampler = surface3.Sampler(space, centre, surface3.hessian_at(space, centre))
    out = {"sd_cap": surface3.SD_CAP, "floor": surface3.FLOOR, "c": {}}
    for target in (25, 50, 100):
        c = surface3.solve(
            lambda c: surface3.mean_drop(
                space, centre, sampler.draw(c, 4000, np.random.default_rng(1))
            ),
            target,
        )
        thetas = sampler.draw(c, 4000, np.random.default_rng(2))
        drops = space.g(centre)[0][0] - space.g(thetas)[0]
        out["c"][str(target)] = {
            "c": c,
            "capped": int(np.sum(sampler.sds(c) >= surface3.SD_CAP - 1e-9)),
            "predicted_drop_mean": float(drops.mean()),
            "predicted_drop_quantiles": np.quantile(drops, [0.1, 0.5, 0.9]).tolist(),
            "rejected_share": surface3.rejected_share(space, centre, sampler, c),
            "covariance": sampler.covariance(c).tolist(),
            "row_sd": np.sqrt(np.diag(sampler.covariance(c))).tolist(),
        }
        print(
            target,
            round(c, 2),
            out["c"][str(target)]["capped"],
            np.round(out["c"][str(target)]["predicted_drop_quantiles"]),
            file=sys.stderr,
        )
    c = out["c"]["50"]["c"]
    group = {
        f"gauss50b-{k:02d}": space.config(t)
        for k, t in enumerate(sampler.draw(c, 24, np.random.default_rng(11)))
    }
    round3.write_group("sampler2", group)
    out["names"], out["free"], out["centre"] = surface3.NAMES, sampler.free, centre.tolist()
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
