"""The rating tournament's statistics (league spec 6): the Bradley-Terry posterior every
tournament fits, the targets it plays to, and the plan of each batch's blocks.

**The posterior** (`fit_posterior`). Ratings on the Elo scale (`SCALE` points per natural
log-odds unit), the anchor pinned at 0, every other player given a Gaussian prior centred
on the anchor with standard deviation `prior_stderr` Elo. The maximum a posteriori is found
by Newton's method (the log-posterior is strictly concave), and the covariance is the
inverse of the log-posterior's negative Hessian there: the games' Fisher information, the
graph Laplacian with edge weights n_ij p_ij (1 - p_ij), plus the prior's precision. A draw
counts as half a win and a fractional outcome (a kill share) as its fraction.

The prior is a precision, not virtual games. A virtual game's information is p (1 - p) at
the estimate, which vanishes as a clean sweep drives the estimate out: a player who wins
every game against the anchor then has a Fisher information that stays about one virtual
game's however many games it plays (the estimate grows as the log of the games, and n p
(1 - p) stays constant), so its standard error never falls. A Gaussian prior keeps every
variance below prior_stderr squared, and every estimate finite, whatever the results.

**The targets** (`Targets`). Each player may have a standard-error target (a loose floor
for a ladder of checkpoints, a tight one for the checkpoints a decision rests on), and a
named contrast, the rating difference between two players, may have its own. A target is
met when its standard error is at most the target.

**The plan** (`plan_batch`). A batch is `blocks` blocks of `block_games` games, one pair
per block. The plan minimises the weighted A-optimal criterion

    sum over unmet targets t of  c_t' Sigma c_t / target_t^2,

Sigma the covariance after the batch: the posterior precision (every game so far, and the
prior) plus the batch's information, so the games already played count (diminishing
returns). A game between i and j is planned at the information p (1 - p) it gives at the
far end of a one-standard-error interval of their rating difference, the interval at most
`MARGIN_CAP` wide (`planning_information`): the less informative end, so a pair whose result
is near certain, or may be, is discounted, while a pair no game has narrowed is not taken
for hopeless.

1. Continuous design weights over the candidate pairs (the allowed pairs whose planned
   information is not negligible), found by the multiplicative algorithm for A-optimality
   (`design_weights`): a distribution of the batch's games over pairs, not one pair.
2. Rounding to whole blocks, one block at a time: the block goes to the heaviest pair of
   the design, and the design is solved again for the games left, the blocks planned so
   far counted in the precision, over the pairs that still fit the batch: at most
   `players` distinct players (the engine loads each player a batch holds) and each
   network group's lanes (`bnb.league.matches.default_lanes`).

Optionally a share of the blocks goes first to the least-played pairs among the decision
players, so every triangle among them is played (`triangle_share`; the loop check needs
direct games).

**Out of reach.** A target the remaining batches (`reach`) cannot bring in is left out of
the criterion while others can be met: its batches needed are counted at the better of the
plan's rate for it and its best single pair's (the whole batch on that pair), its
precision rising by that gain each batch. When every unmet target is out of reach the
batch plays for them all, unless each needs more than `HOPELESS` times the batches left:
then the plan has criterion 0, which stops the tournament. The usual hopeless case is a
player who beats the anchor every game and has no closer partner: more games against the
anchor move its estimate out and hardly narrow it. The plan names the targets it found out
of reach (`Plan.out_of_reach`).

Offline statistics in float64 on the CPU (code style, Dtypes).
"""

import dataclasses
import math

import torch

from .ratings import SCALE

DTYPE = torch.float64
PRIOR_STDERR = 1000.0  # Elo: the prior's standard deviation about the anchor
INFORMATION_MARGIN = 1.0  # standard errors of the difference a game is planned at
MARGIN_CAP = 200.0  # Elo: the planning margin at most (an unplayed pair's error is the prior's)
HOPELESS = 10.0  # a target needing this many times the batches left is not played for
NEWTON_STEPS = 200
DECREMENT_TOLERANCE = 1e-13  # on the Newton decrement, natural units
DESIGN_STEPS = 300  # iterations of the multiplicative algorithm
NEGLIGIBLE_INFORMATION = 1e-4  # a pair's planned information a game below which it is left out
NEGLIGIBLE_WEIGHT = 1e-6  # design weights below this share are dropped


@dataclasses.dataclass(frozen=True)
class Posterior:
    """The fitted field: ratings (K,) in Elo with the anchor at 0; covariance (K, K) in Elo
    squared, the anchor's row and column 0; precision (K, K) in natural units, the anchor's
    row and column 0 (the free players' block is the inverse of the covariance's); games
    (K, K) each pair's games."""

    ratings: torch.Tensor
    covariance: torch.Tensor
    precision: torch.Tensor
    games: torch.Tensor
    anchor: int

    @property
    def stderr(self) -> torch.Tensor:
        """(K,) each rating's standard error in Elo (the anchor's 0)."""
        return self.covariance.diagonal().clamp_min(0.0).sqrt()

    def restrict(self, players: list[int]) -> Posterior:
        """The posterior's marginal over `players` (indices, the anchor among them): their
        ratings, covariance and games, and the precision of that marginal (the inverse of
        its covariance), so a plan over their pairs sees every other player's games."""
        index = torch.tensor(players)
        anchor = players.index(self.anchor)
        covariance = self.covariance[index[:, None], index[None, :]]
        free = torch.arange(len(players)) != anchor
        precision = torch.zeros_like(covariance)
        inside = free.nonzero()[:, 0]
        precision[inside[:, None], inside[None, :]] = torch.linalg.inv(
            covariance[inside][:, inside] / SCALE**2
        )
        games = self.games[index[:, None], index[None, :]]
        return Posterior(self.ratings[index], covariance, precision, games, anchor)


def _as_float(values) -> torch.Tensor:
    return torch.as_tensor(values, dtype=DTYPE).cpu()


def _laplacian(count: int, pairs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """(K, K) the graph Laplacian of `pairs` (P, 2) with edge weights (P,)."""
    laplacian = torch.zeros((count, count), dtype=DTYPE)
    first, second = pairs[:, 0], pairs[:, 1]
    laplacian.index_put_((first, second), -weight, accumulate=True)
    laplacian.index_put_((second, first), -weight, accumulate=True)
    laplacian.index_put_((first, first), weight, accumulate=True)
    laplacian.index_put_((second, second), weight, accumulate=True)
    return laplacian


def _log_posterior(theta: torch.Tensor, wins: torch.Tensor, prior_precision: float):
    difference = theta[:, None] - theta[None, :]
    return (wins * torch.nn.functional.logsigmoid(difference)).sum() - 0.5 * prior_precision * (
        theta.square().sum()
    )


def fit_posterior(wins, anchor: int, prior_stderr: float = PRIOR_STDERR) -> Posterior:
    """The Bradley-Terry posterior of a field (module docstring): wins (K, K) the row's wins
    over the column, a draw half a win each (fractional outcomes allowed)."""
    wins = _as_float(wins)
    count = wins.shape[0]
    games = wins + wins.T
    free = torch.arange(count) != anchor
    prior_precision = (SCALE / prior_stderr) ** 2  # natural units
    theta = torch.zeros(count, dtype=DTYPE)
    for _ in range(NEWTON_STEPS):
        probability = torch.sigmoid(theta[:, None] - theta[None, :])
        gradient = ((wins - games * probability).sum(1) - prior_precision * theta)[free]
        weight = games * probability * (1.0 - probability)
        hessian = (torch.diag(weight.sum(1)) - weight)[free][:, free]
        hessian = hessian + prior_precision * torch.eye(hessian.shape[0], dtype=DTYPE)
        step = torch.linalg.solve(hessian, gradient)
        if float(gradient @ step) < DECREMENT_TOLERANCE:
            break
        current = _log_posterior(theta, wins, prior_precision)
        length = 1.0
        while length > 1e-12:
            trial = theta.clone()
            trial[free] += length * step
            if _log_posterior(trial, wins, prior_precision) >= current - 1e-15 * abs(
                float(current)
            ):
                theta = trial
                break
            length *= 0.5
        else:
            break
    probability = torch.sigmoid(theta[:, None] - theta[None, :])
    weight = games * probability * (1.0 - probability)
    precision = torch.diag(weight.sum(1)) - weight
    precision[free, free] += prior_precision
    precision[anchor, :] = 0.0
    precision[:, anchor] = 0.0
    covariance = torch.zeros((count, count), dtype=DTYPE)
    block = torch.linalg.inv(precision[free][:, free])
    covariance[free.nonzero()[:, 0][:, None], free.nonzero()[:, 0][None, :]] = block
    return Posterior(theta * SCALE, covariance * SCALE**2, precision, games, anchor)


@dataclasses.dataclass(frozen=True)
class Targets:
    """Standard-error targets in Elo: stderr (K,) each player's (inf: none; the anchor's is
    ignored), contrasts ((i, j, target), ...) the difference r_i - r_j's."""

    stderr: torch.Tensor
    contrasts: tuple[tuple[int, int, float], ...] = ()

    def rows(self, anchor: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(rows (T, K), targets (T,)): each target's direction over the players and its
        standard error in Elo."""
        count = self.stderr.shape[0]
        directions, targets = [], []
        for index in range(count):
            if index != anchor and math.isfinite(float(self.stderr[index])):
                row = torch.zeros(count, dtype=DTYPE)
                row[index] = 1.0
                directions.append(row)
                targets.append(float(self.stderr[index]))
        for first, second, target in self.contrasts:
            row = torch.zeros(count, dtype=DTYPE)
            row[first] += 1.0
            row[second] -= 1.0
            directions.append(row)
            targets.append(float(target))
        if not directions:
            return torch.zeros((0, count), dtype=DTYPE), torch.zeros(0, dtype=DTYPE)
        return torch.stack(directions), torch.tensor(targets, dtype=DTYPE)

    def labels(self, names: list[str], anchor: int) -> list[str]:
        """Each target's name, in the order of `rows`: a player's name, or "A - B" for a
        contrast."""
        players = [
            name
            for index, name in enumerate(names)
            if index != anchor and math.isfinite(float(self.stderr[index]))
        ]
        return players + [
            f"{names[first]} - {names[second]}" for first, second, _ in self.contrasts
        ]

    def standard_errors(self, posterior: Posterior) -> tuple[torch.Tensor, torch.Tensor]:
        """(each target's standard error now (T,), its target (T,)), in Elo."""
        rows, targets = self.rows(posterior.anchor)
        variance = ((rows @ posterior.covariance) * rows).sum(1)
        return variance.clamp_min(0.0).sqrt(), targets

    def met(self, posterior: Posterior) -> bool:
        """Whether every target is met."""
        errors, targets = self.standard_errors(posterior)
        return bool((errors <= targets).all())


def target_vector(names: list[str], default: float, tight: dict[str, float]) -> torch.Tensor:
    """(K,) each player's target: `tight`'s where it names the player, else `default`."""
    return torch.tensor([tight.get(name, default) for name in names], dtype=DTYPE)


def planning_information(difference: torch.Tensor, variance: torch.Tensor, margin: float):
    """A game's information about a pair's rating difference as the plan counts it:
    p (1 - p) at |difference| + margin standard errors, the margin at most `MARGIN_CAP` Elo
    (natural units): the less informative end of the interval, but no wider than that, or a
    pair the games have not narrowed (its error the prior's) would look hopeless."""
    spread = (margin * variance.clamp_min(0.0).sqrt()).clamp_max(MARGIN_CAP / SCALE)
    far = difference.abs() + spread
    probability = torch.sigmoid(far)
    return probability * (1.0 - probability)


def _pair_quadratic(matrix: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """(P,) v' A v for each pair's direction v = e_i - e_j."""
    first, second = pairs[:, 0], pairs[:, 1]
    return matrix[first, first] + matrix[second, second] - 2.0 * matrix[first, second]


def _inverse(precision: torch.Tensor, free: torch.Tensor) -> torch.Tensor:
    """The covariance (K, K) of a precision over the free players, padded with zeros."""
    count = precision.shape[0]
    covariance = torch.zeros((count, count), dtype=DTYPE)
    index = free.nonzero()[:, 0]
    covariance[index[:, None], index[None, :]] = torch.linalg.inv(precision[index][:, index])
    return covariance


def design_weights(
    precision: torch.Tensor,
    free: torch.Tensor,
    pairs: torch.Tensor,
    information: torch.Tensor,
    games: float,
    weight_matrix: torch.Tensor,
    steps: int = DESIGN_STEPS,
    start: torch.Tensor | None = None,
) -> torch.Tensor:
    """The continuous design (P,) over `pairs` (P, 2), summing to 1, that minimises
    tr(W (precision + games sum_p xi_p information_p v_p v_p')^-1) (natural units; W =
    `weight_matrix` (K, K)), by the multiplicative algorithm: xi_p <- xi_p (d_p / xi'd)^1/2,
    d_p = information_p v_p' Sigma W Sigma v_p the criterion's decrease per unit of p.
    start: the weights to start from (uniform by default; each kept above a floor, so a
    pair the start left out can come back)."""
    count = precision.shape[0]
    weights = torch.full((pairs.shape[0],), 1.0 / pairs.shape[0], dtype=DTYPE)
    if start is not None:
        weights = start.clamp_min(1e-3 / pairs.shape[0])
        weights = weights / weights.sum()
    for _ in range(steps):
        total = precision + games * _laplacian(count, pairs, weights * information)
        covariance = _inverse(total, free)
        gain = information * _pair_quadratic(covariance @ weight_matrix @ covariance, pairs)
        mean = float(weights @ gain)
        if mean <= 0.0:
            break
        weights = weights * (gain / mean).clamp_min(0.0).sqrt()
        weights = weights / weights.sum()
    return weights


@dataclasses.dataclass(frozen=True)
class Layout:
    """A batch's shape as the plan sees it: blocks of block_games games; players: distinct
    players a batch may hold at most (None: no limit beyond the blocks' slots); groups and
    lanes: the engine's network groups (each player's, None for a scripted one) and each
    group's lanes, the most of its slots a batch may hold (None: no lanes)."""

    blocks: int
    block_games: int
    players: int | None = None
    groups: tuple[int | None, ...] | None = None
    lanes: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.players is not None and self.players < 2:
            raise ValueError(f"{self.players} players per batch: a block needs two")


@dataclasses.dataclass(frozen=True)
class Plan:
    """A batch's blocks (one pair (i, j) each, i < j); the design weights it rounded
    ({pair: weight}); the weighted criterion over the targets it played for, now and after
    the batch; and the unmet targets left out as out of reach (indices into
    `Targets.rows`), with each one's batches needed at its planned rate."""

    blocks: list[tuple[int, int]]
    weights: dict[tuple[int, int], float]
    criterion: float
    after: float
    out_of_reach: dict[int, float] = dataclasses.field(default_factory=dict)

    @property
    def progress(self) -> float:
        """The share of the weighted criterion the batch is planned to remove."""
        return 0.0 if self.criterion <= 0.0 else 1.0 - self.after / self.criterion


class _Lanes:
    """The lanes a batch has left by network group (a scripted player needs none)."""

    def __init__(self, layout: Layout) -> None:
        self.groups, self.left = layout.groups, None
        if layout.lanes is not None:
            self.left = list(layout.lanes)

    def fits(self, pair: tuple[int, int]) -> bool:
        if self.left is None:
            return True
        needs = [self.groups[index] for index in pair if self.groups[index] is not None]
        return all(needs.count(group) <= self.left[group] for group in set(needs))

    def take(self, pair: tuple[int, int]) -> None:
        if self.left is None:
            return
        for index in pair:
            if self.groups[index] is not None:
                self.left[self.groups[index]] -= 1


class _Batch:
    """One batch being planned: its blocks so far, the players and lanes they hold, and the
    precision with their information added."""

    def __init__(self, precision, upper, information, layout: Layout) -> None:
        self.precision, self.upper, self.information = precision.clone(), upper, information
        self.layout, self.lanes = layout, _Lanes(layout)
        self.seats = layout.players if layout.players is not None else 2 * layout.blocks
        self.blocks: list[tuple[int, int]] = []
        self.held: set[int] = set()

    @property
    def left(self) -> int:
        return self.layout.blocks - len(self.blocks)

    def fits(self, pair: tuple[int, int], kept: set[int] | None = None) -> bool:
        seated = len(self.held | set(pair)) <= self.seats
        return seated and self.lanes.fits(pair) and (kept is None or set(pair) <= kept)

    def add(self, index: int) -> None:
        pair = tuple(self.upper[index].tolist())
        self.blocks.append(pair)
        self.held.update(pair)
        self.lanes.take(pair)
        one = slice(index, index + 1)
        self.precision = self.precision + self.layout.block_games * _laplacian(
            self.precision.shape[0], self.upper[one], self.information[one]
        )


def _plan(
    batch: _Batch,
    free: torch.Tensor,
    weight_matrix: torch.Tensor,
    games: torch.Tensor,
) -> dict[tuple[int, int], float]:
    """Fill `batch`'s blocks left for the criterion tr(W Sigma) (module docstring, steps 1 and
    2): before each block the design over the pairs that still fit (the players and lanes
    the blocks already planned hold), for the batch's games left, and the block to its
    heaviest pair. Returns the first design's weights. With W zero, or no pair worth a
    game, each block goes to the least-played pair that fits. Raises ValueError when no
    allowed pair fits."""
    upper, information, layout = batch.upper, batch.information, batch.layout
    useful = information >= NEGLIGIBLE_INFORMATION
    active = bool(weight_matrix.abs().sum() > 0.0)
    weights: dict[tuple[int, int], float] = {}
    previous: torch.Tensor | None = None  # the last design, over every pair
    games = games.clone()
    while batch.left > 0:
        fits = torch.tensor([batch.fits(tuple(pair)) for pair in upper.tolist()])
        if not bool(fits.any()):
            raise ValueError(f"no allowed pair fits the engine's lanes {layout.lanes}")
        among = (fits & useful).nonzero()[:, 0]
        if active and among.shape[0] > 0:
            design = design_weights(
                batch.precision,
                free,
                upper[among],
                information[among],
                batch.left * layout.block_games,
                weight_matrix,
                DESIGN_STEPS if previous is None else DESIGN_STEPS // 3,
                None if previous is None else previous[among],
            )
            if previous is None:
                weights = {
                    tuple(upper[index].tolist()): float(value)
                    for index, value in zip(among.tolist(), design.tolist(), strict=True)
                    if value >= NEGLIGIBLE_WEIGHT
                }
            previous = torch.zeros(upper.shape[0], dtype=DTYPE)
            previous[among] = design
            index = int(among[int(torch.argmax(design))])
        else:
            options = fits.nonzero()[:, 0]
            played = games[upper[options, 0], upper[options, 1]]
            index = int(options[int(torch.argmin(played))])
        batch.add(index)
        first, second = upper[index].tolist()
        games[first, second] += layout.block_games
        games[second, first] += layout.block_games
    return weights


def pair_weights(
    posterior: Posterior,
    targets: Targets,
    allowed,
    games: float,
    margin: float = INFORMATION_MARGIN,
) -> dict[tuple[int, int], float]:
    """The continuous design ({(i, j): weight}, summing to 1) over the allowed pairs for
    `games` more games, for every unmet target (`design_weights`, step 1 of the plan, not
    rounded); empty when every target is met or no pair is worth a game."""
    allowed = torch.as_tensor(allowed, dtype=torch.bool).cpu()
    count = allowed.shape[0]
    free = torch.arange(count) != posterior.anchor
    rows, goals = targets.rows(posterior.anchor)
    errors, _ = targets.standard_errors(posterior)
    unmet = errors > goals
    natural = rows[unmet] * SCALE / goals[unmet][:, None]
    covariance = posterior.covariance / SCALE**2
    theta = posterior.ratings / SCALE
    upper = torch.triu_indices(count, count, 1).T
    upper = upper[allowed[upper[:, 0], upper[:, 1]]]
    information = planning_information(
        theta[upper[:, 0]] - theta[upper[:, 1]], _pair_quadratic(covariance, upper), margin
    )
    useful = information >= NEGLIGIBLE_INFORMATION
    if not bool(unmet.any()) or not bool(useful.any()):
        return {}
    weights = design_weights(
        posterior.precision, free, upper[useful], information[useful], games, natural.T @ natural
    )
    return {
        tuple(pair): float(weight)
        for pair, weight in zip(upper[useful].tolist(), weights.tolist(), strict=True)
        if weight >= NEGLIGIBLE_WEIGHT
    }


def plan_batch(
    posterior: Posterior,
    targets: Targets,
    allowed,
    layout: Layout,
    reach: float = math.inf,
    margin: float = INFORMATION_MARGIN,
    triangle_share: float = 0.0,
    decision: tuple[int, ...] = (),
    hopeless: float = HOPELESS,
) -> Plan:
    """The batch's blocks (module docstring). allowed (K, K) bool: the pairs that may play.
    reach: the batches the tournament may still play; an unmet target that would need more
    at the rate this batch is planned to lower its variance (its precision rising by the
    batch's gain each batch) is left out as out of reach, and the batch is planned again
    for the others; hopeless: the factor on the batches left past which targets all out of
    reach stop the tournament (module docstring; inf: never). decision: the decision
    players, whose least-played pairs take `triangle_share` of the blocks first. Raises
    ValueError when no allowed pair fits the lanes."""
    allowed = torch.as_tensor(allowed, dtype=torch.bool).cpu()
    count, anchor = allowed.shape[0], posterior.anchor
    free = torch.arange(count) != anchor
    rows, goals = targets.rows(anchor)
    errors, _ = targets.standard_errors(posterior)
    natural = rows * SCALE / goals[:, None]  # each row over its target, natural units
    covariance = posterior.covariance / SCALE**2
    theta = posterior.ratings / SCALE
    upper = torch.triu_indices(count, count, 1).T
    upper = upper[allowed[upper[:, 0], upper[:, 1]]]
    if upper.shape[0] == 0:
        raise ValueError("no pair is allowed to play")
    variance = _pair_quadratic(covariance, upper)
    information = planning_information(theta[upper[:, 0]] - theta[upper[:, 1]], variance, margin)
    playing = (errors > goals).nonzero()[:, 0].tolist()
    # each unmet target's batches to its target at its best single pair's rate (the whole
    # batch on that pair): a floor on what one batch can do for it
    batch_games = layout.blocks * layout.block_games
    projected = natural @ covariance  # (T, K): Sigma c per target
    along = projected[:, upper[:, 0]] - projected[:, upper[:, 1]]  # (T, P): v' Sigma c
    single_before = (projected * natural).sum(1)
    single_drop = (batch_games * information * along.square()) / (
        1.0 + batch_games * information * variance
    )
    single_after = (single_before - single_drop.max(1).values).clamp_min(1e-300)
    single = _batches_needed(single_before, single_after)
    out_of_reach: dict[int, float] = {}
    for _ in range(3):  # plan, leave out what is out of reach, plan again for the others
        weight_matrix = natural[playing].T @ natural[playing]
        batch = _Batch(posterior.precision, upper, information, layout)
        position = {tuple(pair): number for number, pair in enumerate(upper.tolist())}
        games = posterior.games.clone()
        reserved = round(triangle_share * layout.blocks) if len(decision) >= 3 else 0
        for _ in range(reserved):  # the decision players' least-played pairs first
            options = [
                (float(games[i, j]), (i, j))
                for i in decision
                for j in decision
                if i < j and (i, j) in position and batch.fits((i, j))
            ]
            if not options:
                break
            pair = min(options)[1]
            batch.add(position[pair])
            games[pair[0], pair[1]] += layout.block_games
            games[pair[1], pair[0]] += layout.block_games
        weights = _plan(batch, free, weight_matrix, games)
        after_covariance = _inverse(batch.precision, free)
        before = ((natural[playing] @ covariance) * natural[playing]).sum(1)
        after = ((natural[playing] @ after_covariance) * natural[playing]).sum(1)
        needed = torch.minimum(_batches_needed(before, after), single[playing])
        far = {t: batches for t, batches in zip(playing, needed.tolist(), strict=True)}
        far = {t: batches for t, batches in far.items() if batches > reach}
        out_of_reach.update(far)
        if not far:
            break
        if len(far) == len(playing):  # every target left is out of reach
            if min(far.values()) > hopeless * reach:
                return Plan(batch.blocks, weights, 0.0, 0.0, out_of_reach)
            break  # but not hopeless: the batch plays for them all
        playing = [t for t in playing if t not in far]
    return Plan(batch.blocks, weights, float(before.sum()), float(after.sum()), out_of_reach)


def _batches_needed(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    """Batches to a target at one batch's gain in precision, the variances `before` and
    `after` it in units of the target's squared: the precision has to rise from 1 / before
    to 1, by 1 / after - 1 / before a batch."""
    gain = (1.0 / after - 1.0 / before).clamp_min(1e-300)
    return ((1.0 - 1.0 / before) / gain).clamp_min(0.0)
