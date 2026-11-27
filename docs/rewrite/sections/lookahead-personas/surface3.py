"""Round 3: the skill surface refitted on round-2 + round-3 games, the summit, and the
curvature-shaped persona sampler (CPU only).

    surface3.py oat REPORT.json                   # step 1: per-row curves, degenerate flags
    surface3.py design OAT.json --count 64        # step 1: Sobol design over the enlarged top
    surface3.py fit OUT.json                      # step 2: surface.py's model on all games
    surface3.py top SURFACE.json OUT.json         # step 2: peak, centre, Hessian, far sides
    surface3.py sampler SURFACE.json TOP.json     # step 3: c for drops 25/50/100, the groups

Model and coordinates as surface.py (Space2: 21 unit rows on the default boxes, 6 crowd
values, store_crowd; Elo = the mean of the per-size quadratics).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round3  # noqa: E402
import surface  # noqa: E402

from lean.scripted.parameters import PARAMETERS, SAMPLED  # noqa: E402

HERE = Path(__file__).resolve().parent
STORES = surface.STORES + (str(round3.DATA / "round3.sqlite"),)
INERT = ("boundary_weight", "seek_weight", "field_weight")
NAMES = list(SAMPLED) + [f"{n}_crowd" for n in surface.CROWD_ROWS] + ["store_crowd"]
FLAGS = {"shots": 0.5, "speed": 0.8}  # under this x the reference's: flagged
STYLE_KEYS = ("store", "speed", "climb", "dive", "turning", "two_circle", "shots", "hits",
              "hit_range")


def mean_elo(entry_by_size):
    values = [entry_by_size[s]["elo"] for s in ("1v1", "4v4", "16v16")]
    return None if None in values else float(np.mean(values))


# --------------------------------------------------------------------------- step 1


def oat_command(a):
    r = json.loads(Path(a.report).read_text())
    setups = r["setups"]
    labels = [k for k in r["rows"] if k != "reference"]
    out = {"rows": {}, "flags": {}}
    reference_style = {s: setups[s]["reference"]["style"] for s in setups}
    for label in labels:
        per = {s: setups[s][label] for s in setups}
        flags = []
        for s, e in per.items():
            st, ref = e.get("style"), reference_style[s]
            if e.get("games"):
                draws = e["draws"] / e["games"]
                if draws > 0.2:
                    flags.append(f"{s} draws {draws:.2f}")
            if st and ref:
                for key, factor in FLAGS.items():
                    if st[key] is not None and st[key] < factor * ref[key]:
                        flags.append(f"{s} {key} {st[key]:.2f} vs {ref[key]:.2f}")
                if st["hit_range"] is not None and st["hit_range"] < 0.5 * ref["hit_range"]:
                    flags.append(f"{s} hit_range {st['hit_range']:.2f} vs {ref['hit_range']:.2f}")
            if e["elo"] is not None and e["elo"] < 700:
                flags.append(f"{s} elo {e['elo']:.0f}")
        if flags:
            out["flags"][label] = flags
        name = label.split("@")[0]
        delta = float(label.split("@")[1]) if "@" in label else None
        if delta is not None:
            out["rows"].setdefault(name, {})[delta] = {
                s: [per[s]["elo"], per[s]["elo_error"]] for s in per
            }
    Path(a.out or "/dev/null").write_text(json.dumps(out, indent=1))
    for name, curve in out["rows"].items():
        line = []
        for delta in sorted(curve):
            values = [curve[delta][s][0] for s in ("1v1", "4v4", "16v16")]
            line.append(
                f"{delta:+.2f}: "
                + "/".join("-" if v is None else f"{v - 1000:+.0f}" for v in values)
            )
        print(f"{name:26s} " + "  ".join(line))
    for label in labels:
        if "@" not in label:
            per = [setups[s][label]["elo"] for s in ("1v1", "4v4", "16v16")]
            print(f"{label:26s} " + "/".join(f"{v - 1000:+.0f}" for v in per))
    print("flags:")
    for label, flags in out["flags"].items():
        print(f"  {label}: {'; '.join(flags)}")


def design_command(a):
    """Sobol points over the enlarged top: per row, the centre's offsets [low, high] (box
    units) from a.ranges (JSON {row: [low, high]}); crowd values and store_crowd +-0.5 box."""
    ranges = json.loads(Path(a.ranges).read_text())
    centre = round3.rows_of(round3.CENTRE)
    names = list(ranges)
    sampler = qmc.Sobol(len(names), scramble=True, seed=20261128)
    unit = sampler.random(a.count)
    group = {}
    for k, point in enumerate(unit):
        rows = dict(centre)
        for name, u in zip(names, point, strict=True):
            low, high = ranges[name]
            rows = round3.moved(rows, name, low + (high - low) * float(u))
        group[f"design{k:02d}"] = rows
    round3.write_group("design", group)


# --------------------------------------------------------------------------- step 2


def positive(values):
    """Off the log scale (collision weight 0, step 1's hard end): left out of the fit."""
    return all(values[n] > 0 for n in SAMPLED if PARAMETERS[n].law == "log")


def fit_command(a):
    surface.load_default = surface.load
    pairings, agents = surface.load(STORES)
    agents = {k: v for k, v in agents.items() if positive(v)}
    surface.load = lambda stores=STORES: surface.load_default(stores)  # top's starts too
    data = surface.Data(pairings, agents)
    games = {s: sum(r[3] for r in data.rows if r[0] == s) for s in surface.SIZES}
    print(f"agents {len(data.ids)}, pairings {len(data.rows)}, games {games}", file=sys.stderr)
    coin, saturated = surface.reference_deviances(data)
    cv = surface.cross_validate(data, 100.0)
    explained = (coin - cv[0]) / (coin - saturated)
    print(f"lambda 100 held-out: per size {explained:.1%}", file=sys.stderr)
    out = {"lambda": 100.0, "cv_explained": explained, "models": {}, "support": {},
           "agents": len(data.ids), "games": {str(s): int(g) for s, g in games.items()}}
    for size in surface.SIZES:
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
    Path(a.out).write_text(json.dumps(out))


HELD = None  # the inert rows' theta (the round-2 centre's), held in the top search


class HeldSpace(surface.Space2):
    """Space2 with the inert rows (boundary, seek, field) held at the round-2 centre: a
    deviation counts as outside the support."""

    def outside(self, theta):
        global HELD
        if HELD is None:
            HELD = surface.theta_of(round3.rows_of(round3.CENTRE))
        theta = np.atleast_2d(theta)
        index = [NAMES.index(n) for n in INERT]
        return super().outside(theta) + ((theta[:, index] - HELD[index]) ** 2).sum(-1)


def hessian_at(space, theta, h=0.05):
    n = space.dimension
    out = np.zeros((n, n))
    eye = np.eye(n) * h
    for i in range(n):
        for j in range(i, n):
            value = (
                space.g(theta + eye[i] + eye[j])[0][0]
                - space.g(theta + eye[i] - eye[j])[0][0]
                - space.g(theta - eye[i] + eye[j])[0][0]
                + space.g(theta - eye[i] - eye[j])[0][0]
            ) / (4 * h * h)
            out[i, j] = out[j, i] = value
    return out


def top_command(a):
    """surface.py's top (peak, the 50-Elo region by Metropolis, its centroid), then the
    summit's shape: per row and size the peak's distance to the data support, the Hessian
    of the mean surface at the peak (eigenvalues, vectors), slices through the peak along
    each row out to 3 box units either side (where it falls 50 / 100 / 200)."""
    surface.load_default = surface.load
    def load(stores=STORES):
        pairings, agents = surface.load_default(stores)
        return pairings, {k: v for k, v in agents.items() if positive(v)}

    surface.load = load
    surface.Space2 = HeldSpace
    surface.top_command(argparse.Namespace(surface=a.surface, out=a.out, band=50.0, kappa=1.0,
                                           steps=a.steps))
    top = json.loads(Path(a.out).read_text())
    s = surface.Surface(a.surface)
    space = surface.Space2(s)
    peak = np.array(top["peak"]["theta"])
    centre = np.array(top["centre"]["theta"])
    edge = {}
    for size in surface.SIZES:
        low, high = s.support[size]
        x = space.x_at(peak, size)[0]
        edge[size] = {
            n: [float(x[i] - low[i]), float(high[i] - x[i])]
            for i, n in enumerate(SAMPLED)
            if min(x[i] - low[i], high[i] - x[i]) < 0.03
        }
    hessian = hessian_at(space, peak)
    free = [i for i, n in enumerate(NAMES) if n not in INERT]
    values, small = np.linalg.eigh(hessian[np.ix_(free, free)])
    vectors = np.zeros((len(NAMES), len(free)))
    vectors[free] = small
    slices = {}
    grid = np.linspace(-3, 3, 241)
    for i, n in enumerate(NAMES):
        thetas = np.repeat(peak[None], len(grid), 0)
        thetas[:, i] += grid
        f = space.g(thetas)[0]
        f0 = space.g(peak)[0][0]
        entry = {}
        for drop in (50, 100, 200):
            inside = grid[f >= f0 - drop]
            entry[str(drop)] = [float(inside.min()), float(inside.max())]
        entry["best_offset"] = float(grid[np.argmax(f)])
        slices[n] = entry
    top["summit"] = {
        "peak_at_support_edge": {str(k): v for k, v in edge.items()},
        "hessian_peak": hessian.tolist(),
        "eigenvalues_peak": values.tolist(),
        "eigenvectors_peak": vectors.T.tolist(),
        "slices_peak": slices,
        "centre_minus_peak": (centre - peak).tolist(),
    }
    Path(a.out).write_text(json.dumps(top, indent=1))
    print("peak at a support edge:", {k: list(v) for k, v in edge.items()}, file=sys.stderr)
    order = np.argsort(values)
    for k in list(order[:4]) + list(order[-4:]):
        v = vectors[:, k]
        main = np.argsort(-np.abs(v))[:3]
        print(f"eig {values[k]:+.0f}: " + ", ".join(f"{NAMES[m]} {v[m]:+.2f}" for m in main),
              file=sys.stderr)


# --------------------------------------------------------------------------- step 3

STICKINESS_MAX = 0.15
HORIZON_MIN = 0.35
LEAGUE_STEP_MAX = 1.0  # box units along a direction (the data's reach)
FLOOR = 10.0  # Elo per box^2: flatter (or upward) directions are given this curvature
SD_CAP = 0.75  # box units: no direction's sd beyond this (the data's reach)


class Sampler:
    """N(centre, c Sigma), Sigma = V diag(1 / max(lambda, FLOOR)) V^T over the non-inert
    coordinates (lambda the eigenvalues of -H of the mean surface at the centre), each
    direction's sd capped at SD_CAP / sqrt(c) x sqrt(c) (so at every c), inert rows held."""

    def __init__(self, space, centre, hessian):
        self.space, self.centre = space, centre
        self.free = [i for i, n in enumerate(NAMES) if n not in INERT]
        a = -hessian[np.ix_(self.free, self.free)]
        self.values, self.vectors = np.linalg.eigh(a)

    def sds(self, c):
        return np.minimum(np.sqrt(c / np.maximum(self.values, FLOOR)), SD_CAP)

    def covariance(self, c):
        sd = self.sds(c)
        return (self.vectors * sd**2) @ self.vectors.T

    def draw(self, c, count, rng):
        z = rng.standard_normal((count * 4, len(self.free))) * self.sds(c)
        thetas = np.repeat(self.centre[None], len(z), 0)
        thetas[:, self.free] += z @ self.vectors.T
        keep = [t for t in thetas if not rejected(self.space, t)]
        return np.array(keep[:count])


def rejected(space, theta) -> bool:
    """Degenerate play (round 2 + step 1): team stalls from stickiness at 0.17 and above
    (round 2: Elo near the floor at 4v4 / 16v16 from 0.17; fine at 0.106 with a ratio to the
    temperature of 8.6; step 1: fine at a ratio of 44), and a horizon under 0.35 s (step 1:
    -230 to -300 at 0.24 s, the look-ahead's first point is 0.25 s; at 0.1 s shots fall to a
    tenth). The round-2 ratio rule (stickiness > 5 x temperature) is kept as a reported
    count only."""
    rows = space.config(theta)
    return rows["stickiness"] > STICKINESS_MAX or rows["horizon"] < HORIZON_MIN


def rejected_share(space, centre, sampler, c):
    z = np.random.default_rng(5).standard_normal((4000, len(sampler.free))) * sampler.sds(c)
    thetas = np.repeat(centre[None], len(z), 0)
    thetas[:, sampler.free] += z @ sampler.vectors.T
    return float(np.mean([rejected(space, t) for t in thetas]))


def box_draw(space, centre, scale, count, rng):
    """Round 2's proposal box (proposal-2.json half_u per row, uniform, crowd rows fixed)
    around the centre, its half-widths x scale."""
    box = json.loads((HERE / "proposal-2.json").read_text())["box"]
    out = []
    while len(out) < count:
        theta = centre.copy()
        for name, b in box.items():
            theta[NAMES.index(name)] += scale * b["half_u"] * rng.uniform(-1, 1)
        if not rejected(space, theta):
            out.append(theta)
    return np.array(out)


def mean_drop(space, centre, thetas):
    return float(space.g(centre)[0][0] - space.g(thetas)[0].mean())


def solve(function, target, low=1e-2, high=1e5):
    for _ in range(50):
        middle = math.sqrt(low * high)
        if function(middle) < target:
            low = middle
        else:
            high = middle
    return math.sqrt(low * high)


def sampler_command(a):
    s = surface.Surface(a.surface)
    space = surface.Space2(s)
    top = json.loads(Path(a.top).read_text())
    centre = np.array(top["centre"]["theta"])
    hessian = hessian_at(space, centre)
    sampler = Sampler(space, centre, hessian)
    rng = np.random.default_rng(7)
    out = {"centre": centre.tolist(), "names": NAMES, "free": sampler.free,
           "eigenvalues": sampler.values.tolist(), "floor": FLOOR, "sd_cap": SD_CAP, "c": {}}
    group = {}
    for target, count in ((25, 16), (50, 32), (100, 16)):
        c = solve(lambda c: mean_drop(space, centre, sampler.draw(c, 4000,
                                                                  np.random.default_rng(1))),
                  target)
        thetas = sampler.draw(c, 4000, np.random.default_rng(2))
        drops = space.g(centre)[0][0] - space.g(thetas)[0]
        per_size = [float(np.mean(space.g(centre)[2][k][0] - space.g(thetas)[2][k]))
                    for k in range(3)]
        out["c"][str(target)] = {
            "c": c,
            "sd_per_direction": sampler.sds(c).tolist(),
            "capped": int(np.sum(sampler.sds(c) >= SD_CAP - 1e-9)),
            "rejected_share": rejected_share(space, centre, sampler, c),
            "predicted_drop_mean": float(drops.mean()),
            "predicted_drop_quantiles": np.quantile(drops, [0.1, 0.5, 0.9]).tolist(),
            "predicted_drop_per_size": per_size,
            "covariance": sampler.covariance(c).tolist(),
            "row_sd": np.sqrt(np.diag(sampler.covariance(c))).tolist(),
        }
        for k, theta in enumerate(sampler.draw(c, count, rng)):
            group[f"gauss{target}-{k:02d}"] = space.config(theta)
        print(f"drop {target}: c {c:.1f}, capped {out['c'][str(target)]['capped']}, "
              f"predicted per size {np.round(per_size)}", file=sys.stderr)
    scale = solve(lambda k: mean_drop(space, centre, box_draw(space, centre, k, 2000,
                                                              np.random.default_rng(4))), 50,
                  1e-3, 1e2)
    out["box_scale_50"] = scale
    for k, theta in enumerate(box_draw(space, centre, scale, 32, rng)):
        group[f"box50-{k:02d}"] = space.config(theta)
    print(f"box scale for a 50 drop: {scale:.2f} of round 2's half-widths", file=sys.stderr)
    # League candidates: along the flattest directions that move behaviour, both signs, at a
    # predicted drop of 30.
    share = np.array([surface.BEHAVIOUR_SHARE.get(NAMES[i], 0.05) for i in sampler.free])
    behaviour = (sampler.vectors**2 * share[:, None]).sum(0)
    score = behaviour / np.maximum(sampler.values, FLOOR)
    chosen = np.argsort(-score)[: a.directions]
    out["league_directions"] = []
    for k in chosen:
        v = np.zeros(len(NAMES))
        v[sampler.free] = sampler.vectors[:, k]
        for sign in (-1, 1):
            t = solve(lambda t: space.g(centre)[0][0] - space.g(centre + sign * t * v)[0][0]
                      if not rejected(space, centre + sign * t * v) else 1e9, 30, 1e-3, LEAGUE_STEP_MAX)
            theta = centre + sign * t * v
            label = f"flat{k:02d}{'+' if sign > 0 else '-'}"
            group[label] = space.config(theta)
            out["league_directions"].append(
                {"label": label, "eigenvalue": float(sampler.values[k]), "step": t,
                 "main": [[NAMES[sampler.free[m]], float(sampler.vectors[m, k])]
                          for m in np.argsort(-np.abs(sampler.vectors[:, k]))[:4]]})
    round3.write_group("sampler", group)
    Path(a.out).write_text(json.dumps(out, indent=1))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("oat")
    p.add_argument("report")
    p.add_argument("--out")
    p = sub.add_parser("design")
    p.add_argument("ranges")
    p.add_argument("--count", type=int, default=64)
    p = sub.add_parser("fit")
    p.add_argument("out")
    p = sub.add_parser("top")
    p.add_argument("surface")
    p.add_argument("out")
    p.add_argument("--steps", type=int, default=40000)
    p = sub.add_parser("sampler")
    p.add_argument("surface")
    p.add_argument("top")
    p.add_argument("out")
    p.add_argument("--directions", type=int, default=6)
    a = parser.parse_args()
    {"oat": oat_command, "design": design_command, "fit": fit_command, "top": top_command,
     "sampler": sampler_command}[a.command](a)


if __name__ == "__main__":
    main()
