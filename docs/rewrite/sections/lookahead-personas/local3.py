"""Round 3: a local surface (round-3 games only: the sweep, the design and the sampler games,
all around the summit) and its check on the sampler personas (CPU only).

    local3.py OUT.json

1. Fit surface.py's per-size quadratic on the round-3 store without the sampler agents; predict
   each sampler persona's drop from the round-3 centre (mean over sizes) and compare with the
   measured drop (sampler-games-3.json): r and RMSE.
2. The same check for the global surface (surface-r3.json, all rounds).
3. Fit on every round-3 game; the Hessian of its mean surface at the round-3 centre
   (eigenvalues), and the sampler rebuilt on it (c for predicted drops 25 / 50 / 100).
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
SIZES = surface.SIZES


def fit(exclude):
    pairings, agents = surface.load((str(round3.DATA / "round3.sqlite"),))
    agents = {k: v for k, v in agents.items() if surface3.positive(v) and k not in exclude}
    data = surface.Data(pairings, agents)
    coin, saturated = surface.reference_deviances(data)
    cv = surface.cross_validate(data, 100.0)
    out = {
        "models": {},
        "support": {},
        "cv_explained": (coin - cv[0]) / (coin - saturated),
        "agents": len(data.ids),
    }
    for size in SIZES:
        rows = [r for r in data.rows if r[0] == size]
        beta, covariance = surface.fit_matrix(
            surface.design_per_size(data, size, rows),
            np.array([r[3] for r in rows], float),
            np.array([r[4] for r in rows], float),
            surface.penalty_vector(100.0, surface.PER_SIZE_BLOCKS),
        )
        out["models"][str(size)] = {"beta": beta.tolist(), "covariance": covariance.tolist()}
        xs = np.array([data.x[(size, i)] for i in data.ids])
        out["support"][str(size)] = [
            np.quantile(xs, 0.02, 0).tolist(),
            np.quantile(xs, 0.98, 0).tolist(),
        ]
    return out


def check(surface_path, games, centre):
    space = surface.Space2(surface.Surface(surface_path))
    measured, predicted = [], []
    for label, rows in games["rows"].items():
        if label == "reference":
            continue
        theta = surface.theta_of(rows)
        measured.append(
            -np.mean([games["setups"][s][label]["elo"] - 1000 for s in ("1v1", "4v4", "16v16")])
        )
        predicted.append(space.g(centre)[0][0] - space.g(theta)[0][0])
    measured, predicted = np.array(measured), np.array(predicted)
    return {
        "r": float(np.corrcoef(measured, predicted)[0, 1]),
        "rmse": float(np.sqrt(np.mean((measured - predicted) ** 2))),
        "measured_mean": float(measured.mean()),
        "predicted_mean": float(predicted.mean()),
    }


def main():
    out_path = Path(sys.argv[1])
    games = json.loads((HERE / "sampler-games-3.json").read_text())
    centre = np.array(json.loads((HERE / "top-r3.json").read_text())["centre"]["theta"])
    sampler_ids = {
        round3.agent_of(rows).id for label, rows in games["rows"].items() if label != "reference"
    }
    scratch = round3.DATA / "local"
    scratch.mkdir(exist_ok=True)
    result = {}
    held = fit(sampler_ids)
    (scratch / "local-without-sampler.json").write_text(json.dumps(held))
    result["local_without_sampler"] = {
        "cv_explained": held["cv_explained"],
        **check(scratch / "local-without-sampler.json", games, centre),
    }
    result["global_r3"] = check(HERE / "surface-r3.json", games, centre)
    full = fit(set())
    (scratch / "local-all.json").write_text(json.dumps(full))
    space = surface.Space2(surface.Surface(scratch / "local-all.json"))
    sampler = surface3.Sampler(space, centre, surface3.hessian_at(space, centre))
    result["local_all"] = {
        "cv_explained": full["cv_explained"],
        "eigenvalues": sampler.values.tolist(),
    }
    for target in (25, 50, 100):
        c = surface3.solve(
            lambda c: surface3.mean_drop(
                space, centre, sampler.draw(c, 3000, np.random.default_rng(1))
            ),
            target,
        )
        result["local_all"][f"c{target}"] = {
            "c": c,
            "capped": int(np.sum(sampler.sds(c) >= surface3.SD_CAP - 1e-9)),
        }
    out_path.write_text(json.dumps(result, indent=1))
    print(
        json.dumps(
            {k: {q: v for q, v in d.items() if q != "eigenvalues"} for k, d in result.items()},
            indent=1,
        ),
        file=sys.stderr,
    )
    print("local eigenvalues of -H:", np.round(sampler.values), file=sys.stderr)


if __name__ == "__main__":
    main()
