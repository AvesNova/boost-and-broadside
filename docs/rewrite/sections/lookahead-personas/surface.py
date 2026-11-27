"""Round 2 (revised brief, 2026-11-28): one skill surface over the parameter space, fitted to
every round-2 game's outcome (CPU only).

    surface.py fit OUT.json [--stores ...] [--lambda 3]     # per-size and shared models, CV
    surface.py top SURFACE.json OUT.json                     # peak, robust centre, flat directions

Data: every game in the round-2 stores (skill2, screen, CMA, around, active), the
current game version, fields and fog, the study clock of its size, equal sizes 1, 4 and 16.
Round 1's games are left out: other conditions (1v1 without fog or fields, the 60 s clock) and
an older game version; its 64 personas are round 2's population anyway.

Coordinates: each agent's effective position per size on the 21 sampled rows, x = u - 1/2 with
u the row's unit coordinate on its spread-1 box around the table default (log law: u = 1/2 +
ln(value / default) / (2 width)). A crowded row's value is the duel value x ratio^w_s, w_s the
mean crowd weight of the agent's measure and half at that size (crowd-2.json histograms); the
store weight is scaled by store_crowd's mean factor at that size against today's k = 1 (the
near-enemies histogram). So crowd rows are positions, not features of their own.

Model: Bradley-Terry on the pairings' outcomes (a draw half a win), P(A beats B) = sigmoid(f(A)
- f(B)), f a quadratic in x: linear, squared and every pairwise product, ridge-penalised
(lambda on the squared and product terms, lambda / 10 on the linear ones). "per size": one f
per size; "shared": f(x, s) = a shared quadratic + a per-size linear and squared part (the
products shared). Compared by 5-fold cross-validation over agents (each fold's agents' games
held out: the held-out deviance per game). Uncertainty: the Laplace covariance.
"""

import argparse
import itertools
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import skill2  # noqa: E402

from lean.eval.rating import ELO_PER_NAT  # noqa: E402
from lean.scripted.parameters import CROWDED, PARAMETERS, SAMPLED  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = Path("/home/vizia/avesnova/boost-and-broadside/data/lookahead-personas-2")
STORES = (
    "skill2.sqlite",
    "screen/verify.sqlite",
    "cma/tune.sqlite",
    "around.sqlite",
    "active/active.sqlite",
)  # the validation ladder's stores (ladder/) are kept out of the fit
SIZES = (1, 4, 16)
D = len(SAMPLED)
PAIRS = list(itertools.combinations(range(D), 2))
MEASURES = ("near", "known", "own")


def crowd_weights():
    crowd = json.loads((HERE / "crowd-2.json").read_text())

    def weight(size, measure, half):
        centres, counts = np.array(
            crowd[f"{size}v{size}"][MEASURES[int(round(measure))]]["histogram"]
        ).T
        excess = np.clip(centres - 1.0, 0.0, None)
        return float((counts * excess / (excess + half)).sum() / counts.sum())

    def store_factor(size, k):
        centres, counts = np.array(crowd[f"{size}v{size}"]["near"]["histogram"]).T
        m = np.maximum(centres, 1.0)
        factor = ((1 + k) / (1 + k * m) * counts).sum() / counts.sum()
        today = (2.0 / (1.0 + m) * counts).sum() / counts.sum()
        return float(factor / today)

    return weight, store_factor


WEIGHT, STORE_FACTOR = crowd_weights()


def effective_units(values: dict, size: int) -> np.ndarray:
    """The agent's unit coordinates (21) at a size, crowd rows folded in."""
    w = WEIGHT(size, values.get("crowd_measure", 0.0), values.get("crowd_half", 1.0))
    out = []
    for name in SAMPLED:
        row = PARAMETERS[name]
        value = values[name]
        if name in CROWDED:
            value = value * values.get(f"{name}_crowd", 1.0) ** w
        if name == "store_weight":
            value = value * STORE_FACTOR(size, values.get("store_crowd", 1.0))
        if row.law == "log":
            out.append(0.5 + math.log(value / row.default) / (2.0 * row.width))
        else:
            out.append(0.5 + (value - row.default) / (2.0 * row.width))
    return np.array(out)


def features(x: np.ndarray) -> np.ndarray:
    """(n, 21) centred coordinates -> (n, 21 + 21 + 210) linear, squared, products."""
    x = np.atleast_2d(x)
    products = np.stack([x[:, i] * x[:, j] for i, j in PAIRS], -1)
    return np.concatenate([x, x * x - 1.0 / 12.0, products], -1)


def load(stores=STORES):
    """{size: {pair (id_a, id_b): [games, score_a]}}, {id: values}."""
    pairings = {size: {} for size in SIZES}
    agents = {}
    for name in stores:
        path = DATA / name
        if not path.exists():
            continue
        connection = sqlite3.connect(path)
        for identity, kind, description in connection.execute(
            "SELECT id, kind, description FROM agents"
        ):
            if kind == "scripted":
                agents[identity] = json.loads(description)
        conditions = {}
        for identity, description in connection.execute("SELECT id, description FROM conditions"):
            d = json.loads(description)
            sizes, game = d["sizes"], d["game"]
            if sizes[0] != sizes[1] or sizes[0] not in SIZES:
                continue
            if game["clock_ticks"] != skill2.CLOCKS[sizes[0]] or not game["fog"]:
                continue
            if game["field_count"] != 10:
                continue
            conditions[identity] = sizes[0]
        query = (
            "SELECT agent_a, agent_b, conditions, COUNT(*), SUM(result = 0) + 0.5 * SUM(result = 2)"
            " FROM games GROUP BY agent_a, agent_b, conditions"
        )
        for a, b, condition, games, score in connection.execute(query):
            if condition not in conditions or a == b:
                continue
            key = (a, b) if a < b else (b, a)
            score_a = score if a < b else games - score
            entry = pairings[conditions[condition]].setdefault(key, [0, 0.0])
            entry[0] += games
            entry[1] += score_a
    return pairings, agents


def objective(beta, diff, games, score, penalty):
    logit = diff @ beta
    # -log-likelihood of binomial scores: score log s(l) + (games - score) log s(-l)
    nll = np.sum(games * np.logaddexp(0.0, logit) - score * logit)
    prob = 1.0 / (1.0 + np.exp(-logit))
    gradient = diff.T @ (games * prob - score) + penalty * beta
    return nll + 0.5 * np.sum(penalty * beta * beta), gradient


def fit_matrix(diff, games, score, penalty):
    result = minimize(
        objective,
        np.zeros(diff.shape[1]),
        args=(diff, games, score, penalty),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 3000},
    )
    beta = result.x
    logit = diff @ beta
    prob = 1.0 / (1.0 + np.exp(-logit))
    hessian = (diff * (games * prob * (1 - prob))[:, None]).T @ diff + np.diag(penalty)
    return beta, np.linalg.inv(hessian)


def deviance(beta, diff, games, score):
    logit = diff @ beta
    return float(np.sum(games * np.logaddexp(0.0, logit) - score * logit)) / max(games.sum(), 1)


class Data:
    def __init__(self, pairings, agents):
        self.agents = agents
        self.rows = []  # (size, a, b, games, score_a)
        for size, entries in pairings.items():
            for (a, b), (games, score) in entries.items():
                if a in agents and b in agents:
                    self.rows.append((size, a, b, games, score))
        self.ids = sorted({r[1] for r in self.rows} | {r[2] for r in self.rows})
        self.x = {
            (size, identity): effective_units(agents[identity], size) - 0.5
            for size in SIZES
            for identity in self.ids
        }


def design_per_size(data, size, rows):
    f = {i: features(data.x[(size, i)])[0] for i in {r[1] for r in rows} | {r[2] for r in rows}}
    diff = np.array([f[r[1]] - f[r[2]] for r in rows])
    return diff


def penalty_vector(lam, blocks):
    """blocks: list of ('linear' | 'square' | 'product', width)."""
    out = []
    for kind, width in blocks:
        value = lam / 10.0 if kind == "linear" else lam
        if kind == "product" and ADDITIVE:
            value = 1e9
        out += [value] * width
    return np.array(out)


PER_SIZE_BLOCKS = [("linear", D), ("square", D), ("product", len(PAIRS))]
ADDITIVE = False  # --additive: the products held at 0 (penalty 1e9)


def shared_design(data, rows):
    """Shared quadratic + per-size linear and squared parts (3 sizes)."""
    out = []
    for size, a, b, _, _ in rows:
        fa, fb = features(data.x[(size, a)])[0], features(data.x[(size, b)])[0]
        d = fa - fb
        per = np.zeros(len(SIZES) * 2 * D)
        k = SIZES.index(size)
        per[k * 2 * D : (k + 1) * 2 * D] = d[: 2 * D]
        out.append(np.concatenate([d, per]))
    return np.array(out)


SHARED_BLOCKS = PER_SIZE_BLOCKS + [("square", len(SIZES) * 2 * D)]


def cross_validate(data, lam, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    fold_of = {
        i: int(f) for i, f in zip(data.ids, rng.integers(0, folds, len(data.ids)), strict=True)
    }
    # The references (agents in 20 or more pairings: the yardsticks the games were played
    # against) are never held out: a fold without them would drop whole studies from training.
    count = {}
    for r in data.rows:
        count[r[1]] = count.get(r[1], 0) + 1
        count[r[2]] = count.get(r[2], 0) + 1
    for identity, n in count.items():
        if n >= 20:
            fold_of[identity] = -1
    per, shared, total_games = 0.0, 0.0, 0.0
    for fold in range(folds):
        test = [r for r in data.rows if fold_of[r[1]] == fold or fold_of[r[2]] == fold]
        train = [r for r in data.rows if not (fold_of[r[1]] == fold or fold_of[r[2]] == fold)]
        if not test:
            continue
        for size in SIZES:
            tr = [r for r in train if r[0] == size]
            te = [r for r in test if r[0] == size]
            if not tr or not te:
                continue
            beta, _ = fit_matrix(
                design_per_size(data, size, tr),
                np.array([r[3] for r in tr], float),
                np.array([r[4] for r in tr], float),
                penalty_vector(lam, PER_SIZE_BLOCKS),
            )
            g = np.array([r[3] for r in te], float)
            per += (
                deviance(beta, design_per_size(data, size, te), g, np.array([r[4] for r in te]))
                * g.sum()
            )
        beta, _ = fit_matrix(
            shared_design(data, train),
            np.array([r[3] for r in train], float),
            np.array([r[4] for r in train], float),
            penalty_vector(lam, SHARED_BLOCKS),
        )
        g = np.array([r[3] for r in test], float)
        shared += (
            deviance(beta, shared_design(data, test), g, np.array([r[4] for r in test])) * g.sum()
        )
        total_games += g.sum()
    return per / total_games, shared / total_games


def reference_deviances(data):
    """Per game: the coin (every game at 1/2) and the saturated (each pairing at its own
    observed rate) deviances, the scale for the held-out deviance."""
    games = np.array([r[3] for r in data.rows], float)
    score = np.array([r[4] for r in data.rows], float)
    rate = np.clip(score / games, 1e-9, 1 - 1e-9)
    saturated = -np.sum(score * np.log(rate) + (games - score) * np.log(1 - rate)) / games.sum()
    return math.log(2.0), float(saturated)


def fit_command(arguments):
    pairings, agents = load()
    data = Data(pairings, agents)
    print(
        f"agents {len(data.ids)}, pairings {len(data.rows)}, games "
        f"{sum(r[3] for r in data.rows)} ("
        + ", ".join(f"{s}v{s} {sum(r[3] for r in data.rows if r[0] == s)}" for s in SIZES)
        + ")",
        file=sys.stderr,
    )
    coin, saturated = reference_deviances(data)
    print(f"deviance per game: coin {coin:.5f}, saturated {saturated:.5f}", file=sys.stderr)
    cv = {}
    for lam in arguments.lambdas:
        cv[lam] = cross_validate(data, lam)
        explained = [(coin - v) / (coin - saturated) for v in cv[lam]]
        print(
            f"lambda {lam}: held-out deviance per game: per size {cv[lam][0]:.5f} "
            f"({explained[0]:.1%} of coin-to-saturated), shared {cv[lam][1]:.5f} "
            f"({explained[1]:.1%})",
            file=sys.stderr,
        )
    best_lam = min(cv, key=lambda k: cv[k][0])
    models = {}
    for size in SIZES:
        rows = [r for r in data.rows if r[0] == size]
        beta, covariance = fit_matrix(
            design_per_size(data, size, rows),
            np.array([r[3] for r in rows], float),
            np.array([r[4] for r in rows], float),
            penalty_vector(best_lam, PER_SIZE_BLOCKS),
        )
        # In-sample: the agents' fitted Elo against a plain per-agent BT is not computed here.
        models[size] = {"beta": beta.tolist(), "covariance": covariance.tolist()}
    support = {}
    for size in SIZES:
        xs = np.array([data.x[(size, i)] for i in data.ids])
        support[size] = [np.quantile(xs, 0.02, 0).tolist(), np.quantile(xs, 0.98, 0).tolist()]
    out = {
        "lambda": best_lam,
        "cv": {str(k): v for k, v in cv.items()},
        "coin": coin,
        "saturated": saturated,
        "models": {str(k): v for k, v in models.items()},
        "support": {str(k): v for k, v in support.items()},
        "agents": len(data.ids),
        "games": {str(s): int(sum(r[3] for r in data.rows if r[0] == s)) for s in SIZES},
    }
    Path(arguments.out).write_text(json.dumps(out))
    print(f"chosen lambda {best_lam}; wrote {arguments.out}", file=sys.stderr)


# --------------------------------------------------------------------------- the top


class Surface:
    def __init__(self, path):
        s = json.loads(Path(path).read_text())
        self.beta = {int(k): np.array(v["beta"]) for k, v in s["models"].items()}
        self.covariance = {int(k): np.array(v["covariance"]) for k, v in s["models"].items()}
        self.support = {int(k): np.array(v) for k, v in s["support"].items()}

    def elo(self, size, x):
        """Predicted Elo (relative: an arbitrary offset) and its Laplace SD at centred x."""
        phi = features(x)
        mean = phi @ self.beta[size] * ELO_PER_NAT
        sd = np.sqrt(np.einsum("ij,jk,ik->i", phi, self.covariance[size], phi)) * ELO_PER_NAT
        return mean, sd


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("out")
    fit.add_argument("--lambdas", type=float, nargs="+", default=[3.0, 10.0, 30.0, 100.0])
    fit.add_argument("--additive", action="store_true")
    top = sub.add_parser("top")
    top.add_argument("surface")
    top.add_argument("out")
    top.add_argument("--band", type=float, default=50.0)
    top.add_argument("--kappa", type=float, default=1.0)
    top.add_argument("--steps", type=int, default=40000)
    report = sub.add_parser("report")
    report.add_argument("surface")
    report.add_argument("top")
    report.add_argument("out")
    report.add_argument("--league", type=int, default=6)
    arguments = parser.parse_args()
    global ADDITIVE
    ADDITIVE = getattr(arguments, "additive", False)
    if arguments.command == "fit":
        fit_command(arguments)
    elif arguments.command == "top":
        top_command(arguments)
    elif arguments.command == "report":
        report_command(arguments)


# --------------------------------------------------------------------------- top (appended)

CROWD_ROWS = (
    "angles_weight",
    "energy_weight",
    "aggression",
    "speed_weight",
    "preferred_speed",
    "reply_temperature",
)
KNOWN = 1.0


class Space2:
    """theta = 21 unit coordinates (u, on the default boxes), a crowd value v per CROWD_ROWS
    row (box units: the crowd value is the duel value x e^(2 width v)) and the store_crowd
    coordinate s (k = e^(2 s)); the crowd measure is the known enemies, half 1."""

    def __init__(self, surface):
        self.surface = surface
        self.w = {size: WEIGHT(size, KNOWN, 1.0) for size in SIZES}
        self.store_shift = {}
        width = PARAMETERS["store_weight"].width
        self.dimension = D + len(CROWD_ROWS) + 1

    def x_at(self, theta, size):
        theta = np.atleast_2d(theta)
        x = theta[:, :D] - 0.5
        x = x.copy()
        for k, name in enumerate(CROWD_ROWS):
            x[:, SAMPLED.index(name)] += theta[:, D + k] * self.w[size]
        width = PARAMETERS["store_weight"].width
        factors = np.array([STORE_FACTOR(size, math.exp(2.0 * s)) for s in theta[:, -1]])
        x[:, SAMPLED.index("store_weight")] += np.log(factors) / (2.0 * width)
        return x

    def outside(self, theta):
        """Squared distance outside each size's data support (0 inside)."""
        total = 0.0
        for size in SIZES:
            low, high = self.surface.support[size]
            x = self.x_at(theta, size)
            total = total + (np.clip(low - x, 0, None) ** 2 + np.clip(x - high, 0, None) ** 2).sum(
                -1
            )
        return total

    def g(self, theta):
        means, variances = [], []
        for size in SIZES:
            m, sd = self.surface.elo(size, self.x_at(theta, size))
            means.append(m)
            variances.append(sd**2)
        return np.mean(means, 0), np.sqrt(np.sum(variances, 0)) / len(SIZES), means

    def config(self, theta) -> dict:
        """The rows over the defaults (a JSON object) of a point."""
        values = {}
        for name, u in zip(SAMPLED, theta[:D], strict=True):
            row = PARAMETERS[name]
            if row.law == "log":
                values[name] = row.default * math.exp(row.width * (2.0 * u - 1.0))
            else:
                values[name] = row.default + row.width * (2.0 * u - 1.0)
        for k, name in enumerate(CROWD_ROWS):
            values[f"{name}_crowd"] = math.exp(2.0 * PARAMETERS[name].width * theta[D + k])
        values["store_crowd"] = math.exp(2.0 * theta[-1])
        values["crowd_measure"] = KNOWN
        values["crowd_half"] = 1.0
        return values


def theta_of(values: dict) -> np.ndarray:
    """An agent's point in Space2 (its crowd measure ignored: rows only)."""
    theta = []
    for name in SAMPLED:
        row = PARAMETERS[name]
        value = values[name]
        if row.law == "log":
            theta.append(0.5 + math.log(value / row.default) / (2.0 * row.width))
        else:
            theta.append(0.5 + (value - row.default) / (2.0 * row.width))
    for name in CROWD_ROWS:
        theta.append(math.log(values.get(f"{name}_crowd", 1.0)) / (2.0 * PARAMETERS[name].width))
    theta.append(math.log(values.get("store_crowd", 1.0)) / 2.0)
    return np.array(theta)


def top_command(arguments):
    surface = Surface(arguments.surface)
    space = Space2(surface)
    penalty = 1e5
    _, agents = load()

    def negative(theta):
        mean, _, _ = space.g(theta)
        return float(-mean[0] + penalty * space.outside(theta)[0])

    def negative_lcb(theta):
        mean, sd, _ = space.g(theta)
        return float(-(mean[0] - arguments.kappa * sd[0]) + penalty * space.outside(theta)[0])

    starts = [theta_of(v) for v in agents.values() if "crowd_measure" in v]
    starts = sorted(starts, key=lambda t: -space.g(t)[0][0])[:4]
    results = {}
    for label, function in (("peak", negative), ("lcb", negative_lcb)):
        best = None
        for start in starts:
            r = minimize(function, start, method="L-BFGS-B")
            if best is None or r.fun < best.fun:
                best = r
        results[label] = best.x
    peak = results["peak"]
    if space.g(results["lcb"])[0][0] > space.g(peak)[0][0]:  # the better optimum found
        peak = results["lcb"]
    top_value = space.g(peak)[0][0]
    # The region within `band` Elo of the peak, inside the support: a uniform sample by
    # random-walk Metropolis (hard constraints), its centroid the robust centre.
    rng = np.random.default_rng(1)
    theta = peak.copy()
    samples = []
    step = 0.03
    accepted = 0
    for t in range(arguments.steps):
        proposal = theta + step * rng.standard_normal(theta.shape)
        if (
            space.outside(proposal)[0] <= 0
            and space.g(proposal)[0][0] >= top_value - arguments.band
        ):
            theta = proposal
            accepted += 1
        if t % 20 == 0:
            samples.append(theta.copy())
    samples = np.array(samples[len(samples) // 5 :])
    centre = samples.mean(0)
    hessian = np.zeros((space.dimension, space.dimension))
    h = 0.05
    base = space.g(centre)[0][0]
    for i in range(space.dimension):
        for j in range(i, space.dimension):
            e_i = np.eye(space.dimension)[i] * h
            e_j = np.eye(space.dimension)[j] * h
            value = (
                space.g(centre + e_i + e_j)[0][0]
                - space.g(centre + e_i - e_j)[0][0]
                - space.g(centre - e_i + e_j)[0][0]
                + space.g(centre - e_i - e_j)[0][0]
            ) / (4 * h * h)
            hessian[i, j] = hessian[j, i] = value
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    names = list(SAMPLED) + [f"{n}_crowd" for n in CROWD_ROWS] + ["store_crowd"]
    out = {
        "peak": {
            "theta": peak.tolist(),
            "config": space.config(peak),
            "g": space.g(peak)[0][0],
            "per_size": [float(m[0]) for m in space.g(peak)[2]],
            "sd": space.g(peak)[1][0],
        },
        "lcb": {
            "theta": results["lcb"].tolist(),
            "config": space.config(results["lcb"]),
            "g": space.g(results["lcb"])[0][0],
            "per_size": [float(m[0]) for m in space.g(results["lcb"])[2]],
            "sd": space.g(results["lcb"])[1][0],
        },
        "centre": {
            "theta": centre.tolist(),
            "config": space.config(centre),
            "g": base,
            "per_size": [float(m[0]) for m in space.g(centre)[2]],
            "sd": space.g(centre)[1][0],
        },
        "region": {
            "band": arguments.band,
            "samples": len(samples),
            "acceptance": accepted / arguments.steps,
            "sd": samples.std(0).tolist(),
            "low": np.quantile(samples, 0.05, 0).tolist(),
            "high": np.quantile(samples, 0.95, 0).tolist(),
        },
        "names": names,
        "hessian_eigenvalues": eigenvalues.tolist(),
        "hessian_eigenvectors": eigenvectors.T.tolist(),
        "region_samples": samples[:: max(1, len(samples) // 400)].tolist(),
    }
    Path(arguments.out).write_text(json.dumps(out, indent=1))
    print(
        f"peak g {out['peak']['g']:.0f} per size {np.round(out['peak']['per_size'])} sd {out['peak']['sd']:.0f}",
        file=sys.stderr,
    )
    print(
        f"lcb g {out['lcb']['g']:.0f} per size {np.round(out['lcb']['per_size'])} sd {out['lcb']['sd']:.0f}",
        file=sys.stderr,
    )
    print(
        f"centre g {base:.0f} per size {np.round(out['centre']['per_size'])} sd {out['centre']['sd']:.0f}; region {len(samples)} samples, acceptance {accepted / arguments.steps:.2f}",
        file=sys.stderr,
    )
    flat = np.argsort(np.abs(eigenvalues))[:5]
    print(
        "flattest directions (Elo per box^2 curvature):",
        np.round(eigenvalues[flat], 1),
        file=sys.stderr,
    )
    print(
        "steepest:",
        np.round(eigenvalues[:3], 0),
        "most positive:",
        np.round(eigenvalues[-3:], 0),
        file=sys.stderr,
    )


# --------------------------------------------------------------------------- report (appended)

# Round 1's behaviour share per row (the redraw JS share of all rows, 4v4, results.md): the
# metric for spreading league personas over rows that change behaviour.
BEHAVIOUR_SHARE = {
    "temperature": 0.18,
    "power_stickiness": 0.17,
    "speed_weight": 0.11,
    "horizon": 0.09,
    "store_weight": 0.08,
    "stickiness": 0.08,
    "angles_weight": 0.07,
    "reply_temperature": 0.07,
    "preferred_speed": 0.06,
    "energy_weight": 0.06,
    "caution": 0.05,
    "power_temperature": 0.05,
    "power_horizon": 0.04,
    "aggression": 0.04,
    "threat_weight": 0.03,
    "collision_weight": 0.03,
    "fire_tolerance": 0.02,
    "field_weight": 0.01,
    "disengaged_energy_weight": 0.01,
    "boundary_weight": 0.0,
    "seek_weight": 0.0,
}


def report_command(arguments):
    surface = Surface(arguments.surface)
    top = json.loads(Path(arguments.top).read_text())
    space = Space2(surface)
    centre = np.array(top["centre"]["theta"])
    out = {"products": {}, "sobol": {}, "ranges": {}, "league": {}}
    rng = np.random.default_rng(3)
    for size in SIZES:
        beta = surface.beta[size] * ELO_PER_NAT
        error = np.sqrt(np.diag(surface.covariance[size])) * ELO_PER_NAT
        products = beta[2 * D :]
        order = np.argsort(-np.abs(products))[:8]
        out["products"][size] = [
            [
                SAMPLED[PAIRS[k][0]],
                SAMPLED[PAIRS[k][1]],
                float(products[k]),
                float(error[2 * D + k]),
            ]
            for k in order
        ]
        # Sobol of f over the spread-1 box (uniform u, no crowd), first and total order.
        a = rng.random((40000, D)) - 0.5
        b = rng.random((40000, D)) - 0.5
        fa, fb = features(a) @ beta, features(b) @ beta
        variance = np.var(np.concatenate([fa, fb]))
        first, total = [], []
        for i in range(D):
            ab = a.copy()
            ab[:, i] = b[:, i]
            fab = features(ab) @ beta
            first.append(float(np.mean(fb * (fab - fa)) / variance))
            total.append(float(0.5 * np.mean((fa - fab) ** 2) / variance))
        out["sobol"][size] = {"first": first, "total": total, "sd": float(np.sqrt(variance))}
        # Working range: the slice of each row through the centre's effective point at this
        # size; the u where f is within 50 and 100 Elo of the slice's best, inside the support.
        x0 = space.x_at(centre, size)[0]
        low, high = surface.support[size]
        ranges = {}
        for i, name in enumerate(SAMPLED):
            grid = np.linspace(low[i], high[i], 81)
            xs = np.repeat(x0[None], len(grid), 0)
            xs[:, i] = grid
            f = surface.elo(size, xs)[0]
            best = f.max()
            within50 = grid[f >= best - 50]
            within100 = grid[f >= best - 100]
            ranges[name] = {
                "best_u": float(grid[np.argmax(f)] + 0.5),
                "centre_u": float(x0[i] + 0.5),
                "u50": [float(within50.min() + 0.5), float(within50.max() + 0.5)],
                "u100": [float(within100.min() + 0.5), float(within100.max() + 0.5)],
                "support": [float(low[i] + 0.5), float(high[i] + 0.5)],
                "drop_at_support_edges": [float(best - f[0]), float(best - f[-1])],
            }
        out["ranges"][size] = ranges
    # League personas: farthest-point picks among the top region's samples, distance weighted
    # by each row's behaviour share (round 1), starting from the sample farthest from the centre.
    samples = np.array(top["region_samples"])
    weights = np.array([BEHAVIOUR_SHARE[n] for n in SAMPLED] + [0.05] * (len(CROWD_ROWS) + 1))

    def distance(p, q):
        return np.sqrt((weights * (p - q) ** 2).sum(-1))

    chosen = [int(np.argmax(distance(samples, centre)))]
    while len(chosen) < arguments.league:
        d = np.min([distance(samples, samples[j]) for j in chosen], 0)
        chosen.append(int(np.argmax(d)))
    for k, j in enumerate(chosen):
        g, sd, per = space.g(samples[j])
        out["league"][f"league{k + 1}"] = {
            "theta": samples[j].tolist(),
            "config": space.config(samples[j]),
            "g": float(g[0]),
            "per_size": [float(m[0]) for m in per],
            "sd": float(sd[0]),
        }
    Path(arguments.out).write_text(json.dumps(out, indent=1))
    for size in SIZES:
        print(
            f"{size}v{size} largest products:",
            [(a[:8], b[:8], round(c)) for a, b, c, _ in out["products"][size][:4]],
            file=sys.stderr,
        )
    for k, v in out["league"].items():
        print(k, round(v["g"]), np.round(v["per_size"]), file=sys.stderr)


if __name__ == "__main__":
    main()
