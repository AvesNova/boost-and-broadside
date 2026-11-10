"""Ratings with a cyclic part: the multidimensional Elo of Balduzzi et al. (2018,
"Re-evaluating Evaluation", mElo-2k), for a field whose results one number does not fit
(rock, paper, scissors).

The log-odds that player i beats player j are

    z_ij = r_i - r_j + c_i' W c_j,

r the transitive ratings (Bradley-Terry's) and c_i a 2k-vector per player, W the 2k x 2k
block-diagonal of k rotations [[0, 1], [-1, 0]], so z is antisymmetric and the second term
is a sum of k cyclic games (each a pair of coordinates per player: i beats j by the area
their two vectors span). k = 0 is Bradley-Terry. The cyclic vectors carry an L2 penalty
(`penalty`, per player and coordinate), chosen by held-out likelihood.

Results come as counts (K, K, 3): the row's wins, draws and losses against the column, a
draw half a win. `cross_validate` splits every pair's games into folds (each game an
outcome of 1, 0.5 or 0), fits each model on all folds but one and scores the held-out
games' log-likelihood: a model that fits better out of sample than Bradley-Terry is
evidence of a cyclic structure that more games would not remove. `residuals` gives each
pair's standardised difference between its observed score and a fit's prediction.

Before a cycle, the link: the deviations from Bradley-Terry may be in how much one player
beats another, not in which way, and a different link from score to probability fits
those without any cycle. So every model also comes with each `LINKS` link: "logit"
(Bradley-Terry's p = sigmoid(z)), "probit" (Thurstone's p = Phi(z): lighter tails) and
"power" (p = sigmoid(z (z^2 + 0.01)^((gamma - 1) / 2)), about sign(z) |z|^gamma with
gamma fitted: a margin-dependent link that stretches or squeezes the large margins).
`majority_cycles` counts the triples whose
majority results go round (i beats j, j beats k, k beats i, each by more than half).

On the Elo scale (`SCALE` points a natural log-odds unit) the transitive part is
SCALE x r, and the cyclic term of a pair SCALE x c_i' W c_j (in the logit link's units)."""

import dataclasses
import math

import torch

from .ratings import SCALE

DTYPE = torch.float64
RATING_PENALTY = 1e-6  # a tiny L2 on the ratings: their constant is otherwise free
STEPS = 200  # L-BFGS iterations
START_SCALE = 0.3  # the cyclic vectors' random start (zero is a saddle point)
LINKS = ("logit", "probit", "power")
GAMMA_BOUND = 1.5  # the power link's |ln gamma| at most: gamma within 0.22 to 4.5
POWER_FLOOR = 0.1  # the power link's smoothing: z (z^2 + floor^2)^((gamma - 1) / 2)


def link_logs(z: torch.Tensor, link: str, log_gamma: torch.Tensor | None = None):
    """(log p, log (1 - p)) of the row winning at score z under `link`."""
    if link == "logit":
        return torch.nn.functional.logsigmoid(z), torch.nn.functional.logsigmoid(-z)
    if link == "probit":
        return torch.special.log_ndtr(z), torch.special.log_ndtr(-z)
    if link == "power":
        gamma = log_gamma.clamp(-GAMMA_BOUND, GAMMA_BOUND).exp()
        bent = z * (z.square() + POWER_FLOOR**2) ** (0.5 * (gamma - 1.0))
        return torch.nn.functional.logsigmoid(bent), torch.nn.functional.logsigmoid(-bent)
    raise ValueError(f"link {link!r} is not one of {LINKS}")


def rotation(k: int) -> torch.Tensor:
    """(2k, 2k) the block-diagonal of k rotations [[0, 1], [-1, 0]]."""
    block = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=DTYPE)
    return torch.block_diag(*([block] * k)) if k else torch.zeros((0, 0), dtype=DTYPE)


def log_odds(ratings: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """(K, K) z_ij = r_i - r_j + c_i' W c_j."""
    transitive = ratings[:, None] - ratings[None, :]
    if vectors.shape[1] == 0:
        return transitive
    return transitive + vectors @ rotation(vectors.shape[1] // 2) @ vectors.T


@dataclasses.dataclass(frozen=True)
class Fit:
    """A fitted model: the ratings (K,) and the cyclic vectors (K, 2k), in natural log-odds
    units, and the penalised negative log-likelihood it reached."""

    ratings: torch.Tensor
    vectors: torch.Tensor
    loss: float
    link: str = "logit"
    gamma: float = 1.0

    def log_odds(self) -> torch.Tensor:
        """(K, K) the fit's score z of the row against the column (the logit link's
        log-odds)."""
        return log_odds(self.ratings, self.vectors)

    def probability(self) -> torch.Tensor:
        """(K, K) the fit's probability of the row beating the column."""
        gamma = torch.tensor(math.log(self.gamma), dtype=DTYPE)
        return link_logs(self.log_odds(), self.link, gamma)[0].exp()


def fit(
    wins: torch.Tensor,
    games: torch.Tensor,
    k: int,
    penalty: float = 0.0,
    seed: int = 0,
    link: str = "logit",
) -> Fit:
    """mElo-2k by maximum penalised likelihood under `link` (`LINKS`; the power link's
    exponent fitted with it). wins (K, K): the row's wins over the column (a draw half
    each); games (K, K): the pair's games, symmetric."""
    wins, games = wins.to(DTYPE), games.to(DTYPE)
    count = wins.shape[0]
    upper = torch.triu(torch.ones((count, count), dtype=torch.bool), 1) & (games > 0)
    won, played = wins[upper], games[upper]
    generator = torch.Generator().manual_seed(seed)
    ratings = torch.zeros(count, dtype=DTYPE, requires_grad=True)
    vectors = (
        START_SCALE * torch.randn((count, 2 * k), generator=generator, dtype=DTYPE)
    ).requires_grad_()
    if link == "power":  # from the logit link's fit: gamma 1 is its exact start
        start = fit(wins, games, k, penalty, seed, "logit")
        with torch.no_grad():
            ratings.copy_(start.ratings)
            vectors.copy_(start.vectors)
    log_gamma = torch.zeros((), dtype=DTYPE, requires_grad=True)
    parameters = [ratings, vectors] if k else [ratings]
    if link == "power":
        parameters.append(log_gamma)
    optimiser = torch.optim.LBFGS(
        parameters,
        max_iter=STEPS,
        tolerance_grad=1e-10,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def loss() -> torch.Tensor:
        z = log_odds(ratings, vectors)[upper]
        log_win, log_loss = link_logs(z, link, log_gamma)
        value = -(won * log_win).sum() - ((played - won) * log_loss).sum()
        value = value + RATING_PENALTY * ratings.square().sum()
        return value + penalty * vectors.square().sum()

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        value = loss()
        value.backward()
        return value

    for _ in range(5):  # L-BFGS restarts its memory; a few passes settle it
        optimiser.step(closure)
    with torch.no_grad():
        final = float(loss())
        shift = ratings.mean()
        return Fit(
            (ratings - shift).detach().clone(),
            vectors.detach().clone(),
            final,
            link,
            float(log_gamma.clamp(-GAMMA_BOUND, GAMMA_BOUND).exp()),
        )


def outcomes(counts: torch.Tensor, folds: int, seed: int = 0):
    """Every game of counts (K, K, 3) as (first, second, outcome, fold) tensors: each pair
    i < j's wins, draws and losses as outcomes 1, 0.5 and 0 of i, shuffled and dealt to the
    folds in turn."""
    generator = torch.Generator().manual_seed(seed)
    count = counts.shape[0]
    firsts, seconds, results, fold_of = [], [], [], []
    for i in range(count):
        for j in range(i + 1, count):
            won, drawn, lost = (int(value) for value in counts[i, j].tolist())
            total = won + drawn + lost
            if total == 0:
                continue
            values = torch.tensor([1.0] * won + [0.5] * drawn + [0.0] * lost, dtype=DTYPE)
            values = values[torch.randperm(total, generator=generator)]
            start = int(torch.randint(folds, (1,), generator=generator))
            firsts.append(torch.full((total,), i))
            seconds.append(torch.full((total,), j))
            results.append(values)
            fold_of.append((torch.arange(total) + start) % folds)
    return torch.cat(firsts), torch.cat(seconds), torch.cat(results), torch.cat(fold_of)


def _tables(first, second, result, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The wins and games (K, K) of a set of games."""
    wins = torch.zeros((count, count), dtype=DTYPE)
    games = torch.zeros((count, count), dtype=DTYPE)
    wins.index_put_((first, second), result, accumulate=True)
    wins.index_put_((second, first), 1.0 - result, accumulate=True)
    ones = torch.ones_like(result)
    games.index_put_((first, second), ones, accumulate=True)
    games.index_put_((second, first), ones, accumulate=True)
    return wins, games


def game_log_likelihood(model: Fit, first, second, result) -> torch.Tensor:
    """Each game's log-likelihood under a fit, a draw half a win."""
    gamma = torch.tensor(math.log(model.gamma), dtype=DTYPE)
    log_win, log_loss = link_logs(model.log_odds()[first, second], model.link, gamma)
    return result * log_win + (1.0 - result) * log_loss


def cross_validate(
    counts: torch.Tensor,
    dimensions: tuple[int, ...] = (0, 1, 2, 3),
    penalties: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0),
    folds: int = 5,
    seed: int = 0,
    links: tuple[str, ...] = LINKS,
) -> dict:
    """Each model's held-out log-likelihood a game over `folds` folds: Bradley-Terry (k = 0)
    and mElo-2k for each k in `dimensions` and each penalty, each under every link of
    `links`. Returns per model ("link=L, k=K, penalty=P") the mean held-out log-likelihood
    a game, and its difference from Bradley-Terry's (the logit link, k = 0) with the
    difference's standard error over the games (a paired comparison); "best": the best
    model's name; "best_transitive": the best with k = 0, and each model's gain over it
    (with its standard error)."""
    count = counts.shape[0]
    first, second, result, fold = outcomes(counts, folds, seed)
    per_game = {}
    for link, k in ((link, k) for link in links for k in dimensions):
        for penalty in penalties if k else (0.0,):
            name = f"link={link}, k={k}, penalty={penalty:g}"
            held = torch.empty_like(result)
            for f in range(folds):
                train = fold != f
                wins, games = _tables(first[train], second[train], result[train], count)
                model = fit(wins, games, k, penalty, seed, link)
                test = ~train
                held[test] = game_log_likelihood(model, first[test], second[test], result[test])
            per_game[name] = held
    base = per_game[f"link=logit, k=0, penalty={0.0:g}"]
    models = {}
    for name, held in per_game.items():
        difference = held - base
        models[name] = dict(
            log_likelihood=float(held.mean()),
            gain_over_bradley_terry=float(difference.mean()),
            gain_stderr=float(difference.std() / math.sqrt(len(difference))),
        )
    best = max(models, key=lambda name: models[name]["log_likelihood"])
    transitive = max(
        (name for name in models if ", k=0," in name),
        key=lambda name: models[name]["log_likelihood"],
    )
    for name, held in per_game.items():  # against the best link without a cycle
        difference = held - per_game[transitive]
        models[name]["gain_over_best_transitive"] = float(difference.mean())
        models[name]["gain_over_best_transitive_stderr"] = float(
            difference.std() / math.sqrt(len(difference))
        )
    return dict(
        games=len(result), folds=folds, models=models, best=best, best_transitive=transitive
    )


def parse_model(name: str) -> tuple[str, int, float]:
    """(link, k, penalty) of a `cross_validate` model name."""
    parts = dict(part.split("=") for part in name.split(", "))
    return parts["link"], int(parts["k"]), float(parts["penalty"])


def majority_cycles(counts: torch.Tensor) -> list[tuple[int, int, int, float]]:
    """The triples (i, j, k) whose majority results go round, i over j over k over i (each
    pair's score above one half, every pair played), each once with its weakest edge's
    score: the cycles by who wins, not by how much."""
    games = counts.sum(-1)
    score = (counts[..., 0] + 0.5 * counts[..., 1]) / games.clamp_min(1.0)
    beats = (score > 0.5) & (games > 0)
    count = counts.shape[0]
    found = []
    for i in range(count):
        for j in range(i + 1, count):
            for k in range(j + 1, count):
                for a, b, c in ((i, j, k), (i, k, j)):
                    if beats[a, b] and beats[b, c] and beats[c, a]:
                        weakest = min(float(score[a, b]), float(score[b, c]), float(score[c, a]))
                        found.append((a, b, c, weakest))
    return found


def residuals(counts: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """(K, K) each pair's observed score minus the predicted p (K, K), over the
    prediction's binomial standard error at its games (0 where no games)."""
    games = counts.sum(-1).to(DTYPE)
    score = (counts[..., 0] + 0.5 * counts[..., 1]).to(DTYPE) / games.clamp_min(1.0)
    stderr = torch.sqrt(p * (1.0 - p) / games.clamp_min(1.0))
    return torch.where(games > 0, (score - p) / stderr.clamp_min(1e-12), torch.zeros_like(p))


def decomposition(model: Fit) -> dict:
    """The transitive and cyclic parts of a fit over every pair (i < j), on the Elo scale:
    each part's root-mean-square log-odds term, and the cyclic share of the summed
    squares."""
    count = model.ratings.shape[0]
    upper = torch.triu(torch.ones((count, count), dtype=torch.bool), 1)
    transitive = (model.ratings[:, None] - model.ratings[None, :])[upper]
    cyclic = (model.log_odds()[upper]) - transitive
    t, c = transitive.square().sum(), cyclic.square().sum()
    return dict(
        transitive_rms_elo=float(SCALE * transitive.square().mean().sqrt()),
        cyclic_rms_elo=float(SCALE * cyclic.square().mean().sqrt()),
        cyclic_share=float(c / (t + c)),
    )
