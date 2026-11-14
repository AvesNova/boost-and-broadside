"""The group-sequential test of `bnb evaluate` (`bnb.league.evaluation`): a fixed number of
planned looks at an estimate D of a difference θ (the overall score, A's win probability
less one half), each look stopping with A wins, B wins or a tie, or continuing.

**The statistic.** At a look, Z = D / SE(D), and the information is I = 1 / SE(D)². With
I_max the information planned for the last look, the look's information fraction is
t = I / I_max, and Z t^(1/2) I_max^(1/2) behaves as a Brownian motion in t with drift
θ I_max^(1/2) (the canonical joint distribution of a sequence of looks; Jennison and
Turnbull, *Group Sequential Methods*, 2000, chapter 3).

**Boundaries.** At look k, A wins when Z ≥ u_k, B wins when Z ≤ -u_k, and the test stops
with a tie when |Z| ≤ l_k (the inner wedge); otherwise it continues. Both are set by
error spending (Lan and DeMets 1983), O'Brien-Fleming type, at the looks' actual
fractions, so a look may come at any information:

- the probability of a winner when θ = 0 (two-sided α, α / 2 a side) is spent as
  α(t) = 4 (1 - Φ(z_{1-α/4} / t^(1/2))) in all, on the two sides alike;
- the probability of a tie when θ = ±δ (β) is spent as β(t) = 2 (1 - Φ(z_{1-β/2} / t^(1/2))).

u_k spends α's increment under θ = 0, l_k spends β's under θ = δ, each given the paths still
running after look k - 1 (the inner wedge binds: a tie stops the test). Where l_k would
reach u_k the look stops surely (l_k = u_k). The last look always decides: l_K = u_K.

**I_max** (`plan`) is the information at which a test with the planned looks, at equal
fractions k / K, has l_K = u_K exactly, so both error rates are spent at once: a drift
η_δ = δ I_max^(1/2) found by bisection, I_max = (η_δ / δ)².

**Computation.** The density of the Brownian motion's value among the paths still running
is held at Simpson's-rule nodes over each look's continuation region (one or two
intervals) and carried from look to look by the Gaussian increment's kernel (the
recursive integration of Armitage, McPherson and Rowe 1969); a boundary's tail
probability is the kernel's tail integrated against the previous look's density, and is
solved for by bisection.
(`operating_characteristics` gives each outcome's probability and the expected
information fraction at the stop for any drift.)
"""

import dataclasses
import math
from statistics import NormalDist

import torch

NORMAL = NormalDist()
INTERVAL_POINTS = 401  # quadrature nodes per interval of a look's continuation region
GRID_WIDTH = 9.0  # the region's reach, in standard deviations, about the drift's path


@dataclasses.dataclass(frozen=True)
class SequentialDesign:
    """looks: the planned looks K, at information fractions k / K; alpha: the two-sided
    chance of a winner when the true difference is 0; beta: the chance of a tie when it is
    ±delta (the indifference zone's edge, in the score's units)."""

    looks: int = 4
    alpha: float = 0.05
    beta: float = 0.2
    delta: float = 0.02

    def __post_init__(self) -> None:
        if self.looks < 1:
            raise ValueError(f"{self.looks} looks: at least one")
        if not 0.0 < self.alpha < 1.0 or not 0.0 < self.beta < 1.0:
            raise ValueError(f"alpha {self.alpha} and beta {self.beta}: each in (0, 1)")
        if not self.delta > 0.0:
            raise ValueError(f"an indifference zone of {self.delta}: above 0")


def alpha_spent(fraction: float, alpha: float) -> float:
    """The two-sided α spent by information fraction t (O'Brien-Fleming type)."""
    if fraction <= 0.0:
        return 0.0
    return 4.0 * (1.0 - NORMAL.cdf(NORMAL.inv_cdf(1.0 - alpha / 4.0) / math.sqrt(fraction)))


def beta_spent(fraction: float, beta: float) -> float:
    """The β spent by information fraction t (O'Brien-Fleming type)."""
    if fraction <= 0.0:
        return 0.0
    return 2.0 * (1.0 - NORMAL.cdf(NORMAL.inv_cdf(1.0 - beta / 2.0) / math.sqrt(fraction)))


@dataclasses.dataclass(frozen=True)
class Boundaries:
    """Per look: fractions (t_k), upper (u_k: A wins at Z ≥ u_k, B at Z ≤ -u_k) and inner
    (l_k: a tie at |Z| ≤ l_k; 0 where a look cannot stop with a tie)."""

    fractions: tuple[float, ...]
    upper: tuple[float, ...]
    inner: tuple[float, ...]

    def decision(self, look: int, z: float) -> str:
        """The outcome at look `look` (from 0) for statistic z: "A", "B", "tie" or
        "continue"."""
        if z >= self.upper[look]:
            return "A"
        if z <= -self.upper[look]:
            return "B"
        if abs(z) <= self.inner[look]:
            return "tie"
        return "continue"


def _simpson(low: float, high: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Simpson's rule on [low, high]: (nodes, weights), `INTERVAL_POINTS` of each."""
    nodes = torch.linspace(low, high, INTERVAL_POINTS, dtype=torch.float64)
    step = (high - low) / (INTERVAL_POINTS - 1)
    weights = torch.full((INTERVAL_POINTS,), 2.0, dtype=torch.float64)
    weights[1::2] = 4.0
    weights[0] = weights[-1] = 1.0
    return nodes, weights * step / 3.0


class _Paths:
    """The paths still running, as a density at quadrature nodes over the continuation
    region of the Brownian motion's value s = Z t^(1/2), under one drift (per unit of t)."""

    def __init__(self, drift: float) -> None:
        self.drift = drift
        self.nodes: torch.Tensor | None = None  # None before the first look
        self.weights: torch.Tensor | None = None
        self.density: torch.Tensor | None = None
        self.time = 0.0

    def _tail_below(self, fraction: float, threshold: torch.Tensor) -> torch.Tensor:
        """P(running until now, and s at `fraction` >= threshold), thresholds of any shape."""
        spread = fraction - self.time
        mean = self.drift * spread
        if self.density is None:
            return torch.special.ndtr(-(threshold - mean) / math.sqrt(spread))
        z = (threshold[..., None] - self.nodes - mean) / math.sqrt(spread)
        return (torch.special.ndtr(-z) * (self.density * self.weights)).sum(-1)

    def tail(self, fraction: float, threshold: float) -> float:
        """P(running until now, and s at `fraction` >= threshold)."""
        return float(self._tail_below(fraction, torch.tensor(threshold, dtype=torch.float64)))

    def running(self) -> float:
        """The probability still running."""
        if self.density is None:
            return 1.0
        return float((self.density * self.weights).sum())

    def inside(self, fraction: float, threshold: float) -> float:
        """P(running until now, and |s| at `fraction` <= threshold)."""
        if threshold <= 0.0:
            return 0.0
        return self.tail(fraction, -threshold) - self.tail(fraction, threshold)

    def advance(self, fraction: float, upper: float, inner: float) -> None:
        """Move to `fraction`, keeping the paths with inner < |s| < upper (s-scale)."""
        spread = fraction - self.time
        mean = self.drift * spread
        width = math.sqrt(spread)
        # the region's reach: beyond GRID_WIDTH standard deviations of every path, nothing
        reach = GRID_WIDTH * math.sqrt(fraction) + abs(self.drift) * fraction
        upper = min(upper, reach)
        if inner >= upper:
            intervals = []
        elif inner > 0.0:
            intervals = [(-upper, -inner), (inner, upper)]
        else:
            intervals = [(-upper, upper)]
        if not intervals:
            nodes = weights = torch.zeros(1, dtype=torch.float64)
        else:
            parts = [_simpson(low, high) for low, high in intervals]
            nodes = torch.cat([part[0] for part in parts])
            weights = torch.cat([part[1] for part in parts])
        if self.density is None:
            new = torch.exp(-0.5 * ((nodes - mean) / width) ** 2) / (width * math.sqrt(2 * math.pi))
        else:
            difference = (nodes[:, None] - self.nodes[None, :] - mean) / width
            kernel = torch.exp(-0.5 * difference**2) / (width * math.sqrt(2 * math.pi))
            new = kernel @ (self.density * self.weights)
        self.nodes, self.weights, self.density = nodes, weights, new
        self.time = fraction


def _solve(function, target: float, low: float, high: float) -> float:
    """x in [low, high] with function(x) = target, for a function falling in x (bisection);
    an end where the target lies beyond it."""
    if function(low) <= target:
        return low
    if function(high) >= target:
        return high
    for _ in range(80):
        middle = 0.5 * (low + high)
        if function(middle) > target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _build(
    design: SequentialDesign, fractions: list[float], drift_delta: float, final: bool = True
) -> tuple[Boundaries, float]:
    """The boundaries at these fractions (see `boundaries`), and the last look's inner
    boundary by β spending alone, before it is set to the upper (for `plan`)."""
    if not fractions or any(b <= a for a, b in zip(fractions, fractions[1:], strict=False)):
        raise ValueError(f"fractions {fractions}: rising, at least one")
    null, alternative = _Paths(0.0), _Paths(drift_delta)
    upper, inner = [], []
    spent_alpha, spent_beta = 0.0, 0.0
    unforced = 0.0
    for look, fraction in enumerate(fractions):
        root = math.sqrt(fraction)
        alpha_target = alpha_spent(min(fraction, 1.0), design.alpha) - spent_alpha
        # each side spends half: P0(s >= u t^(1/2), running) = alpha_target / 2
        u = _solve(lambda z: null.tail(fraction, z * root), alpha_target / 2.0, 0.0, 40.0)
        beta_target = beta_spent(min(fraction, 1.0), design.beta) - spent_beta
        # P_delta(|s| <= l t^(1/2), running) = beta_target: rising in l, so solved on its negative
        unforced = _solve(
            lambda z: -alternative.inside(fraction, z * root), -beta_target, 0.0, 40.0
        )
        found = u if (final and look == len(fractions) - 1) or unforced >= u else unforced
        upper.append(u)
        inner.append(found)
        spent_alpha += 2.0 * null.tail(fraction, u * root)
        spent_beta += alternative.inside(fraction, found * root)
        null.advance(fraction, u * root, found * root)
        alternative.advance(fraction, u * root, found * root)
    return Boundaries(tuple(fractions), tuple(upper), tuple(inner)), unforced


def boundaries(
    design: SequentialDesign, fractions: list[float], drift_delta: float, final: bool = True
) -> Boundaries:
    """The boundaries at looks at these information fractions (rising) for the planned drift
    at delta, eta_delta = delta I_max^(1/2) (see the module docstring); final: the last of
    them is the test's last look, which always decides (l = u)."""
    return _build(design, fractions, drift_delta, final)[0]


def planned_fractions(design: SequentialDesign) -> list[float]:
    """The planned looks' information fractions, k / K."""
    return [(look + 1) / design.looks for look in range(design.looks)]


def plan(design: SequentialDesign) -> tuple[float, Boundaries]:
    """(eta_delta, the planned boundaries): the drift at delta at which the planned looks'
    last boundaries meet (l_K = u_K before it is forced), so the test has its alpha and
    beta; I_max is (eta_delta / delta)^2."""
    fractions = planned_fractions(design)
    low, high = 0.5, 20.0
    for _ in range(50):
        middle = 0.5 * (low + high)
        found, unforced = _build(design, fractions, middle)
        # the last look's beta-spending inner boundary less its upper rises in the drift
        if unforced < found.upper[-1]:
            low = middle
        else:
            high = middle
    drift = 0.5 * (low + high)
    return drift, boundaries(design, fractions, drift)


@dataclasses.dataclass(frozen=True)
class Operating:
    """At one drift: the chances of each outcome, and the expected information fraction at
    the stop (the expected cost is this times I_max's cost)."""

    a_wins: float
    b_wins: float
    tie: float
    expected_fraction: float


def operating_characteristics(found: Boundaries, drift: float) -> Operating:
    """Each outcome's probability and the expected fraction at the stop, for paths with
    `drift` (θ I_max^(1/2))."""
    paths = _Paths(drift)
    a_wins = b_wins = tie = expected = 0.0
    for fraction, u, l_value in zip(found.fractions, found.upper, found.inner, strict=True):
        root = math.sqrt(fraction)
        upper_mass = paths.tail(fraction, u * root)
        lower_mass = paths.running() - paths.tail(fraction, -u * root)
        inner_mass = paths.inside(fraction, l_value * root)
        a_wins += upper_mass
        b_wins += lower_mass
        tie += inner_mass
        expected += fraction * (upper_mass + lower_mass + inner_mass)
        paths.advance(fraction, u * root, l_value * root)
    return Operating(a_wins, b_wins, tie, expected)


def simulate_paths(
    found: Boundaries, drift: float, paths: int, generator: torch.Generator
) -> Operating:
    """`operating_characteristics` by Monte Carlo: Brownian paths with `drift` stopped by
    the boundaries (a check of the grid computation)."""
    fractions = torch.tensor(found.fractions, dtype=torch.float64)
    spread = torch.diff(fractions, prepend=torch.zeros(1, dtype=torch.float64))
    steps = torch.randn(paths, len(fractions), generator=generator, dtype=torch.float64)
    values = torch.cumsum(steps * spread.sqrt() + drift * spread, 1)
    z = values / fractions.sqrt()
    running = torch.ones(paths, dtype=torch.bool)
    counts = dict(A=0.0, B=0.0, tie=0.0)
    expected = 0.0
    for look in range(len(fractions)):
        u, l_value = found.upper[look], found.inner[look]
        a = running & (z[:, look] >= u)
        b = running & (z[:, look] <= -u)
        t = running & (z[:, look].abs() <= l_value) & ~a & ~b
        counts["A"] += float(a.sum())
        counts["B"] += float(b.sum())
        counts["tie"] += float(t.sum())
        expected += float(fractions[look]) * float((a | b | t).sum())
        running &= ~(a | b | t)
    return Operating(
        counts["A"] / paths, counts["B"] / paths, counts["tie"] / paths, expected / paths
    )
