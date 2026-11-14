"""The rating tournament's statistics (league spec 6, `bnb.league.design`): the posterior
with its Gaussian prior, the targets, the continuous A-optimal design and the batch plan
with its players-per-batch and lane limits, the out-of-reach rule, and the saturated
anchor pair that the earlier greedy allocation kept playing."""

import collections
import math

import pytest
import torch

from bnb.league import design, tournament
from bnb.league.matches import ArenaConfig
from bnb.league.ratings import SCALE, fit_ratings, win_probability

DTYPE = torch.float64


def _wins(results: dict[tuple[int, int], tuple[float, float]], count: int) -> torch.Tensor:
    """(K, K) wins from {(i, j): (i's wins, j's wins)}."""
    wins = torch.zeros((count, count), dtype=DTYPE)
    for (first, second), (won, lost) in results.items():
        wins[first, second] += won
        wins[second, first] += lost
    return wins


def _play(truth: torch.Tensor, block_games: int, generator):
    """A batch's results drawn from the rating model (no draws)."""

    def play(matchups):
        rows = []
        for first, second in matchups:
            probability = win_probability(truth[first], truth[second])
            games = torch.tensor(float(block_games), dtype=DTYPE)
            won = int(torch.binomial(games, probability, generator))
            rows.append((won, 0, block_games - won))
        return torch.tensor(rows)

    return play


# ---- the posterior


def test_a_wide_prior_gives_the_bradley_terry_fit():
    wins = _wins({(0, 1): (30, 50), (1, 2): (40, 40), (0, 2): (20, 60), (2, 3): (70, 10)}, 4)
    posterior = design.fit_posterior(wins, 0, prior_stderr=1e7)
    ratings, stderr = fit_ratings(wins, 0, prior_games=1e-9)
    assert posterior.ratings.tolist() == pytest.approx(ratings.tolist(), abs=1e-4)
    assert posterior.stderr.tolist() == pytest.approx(stderr.tolist(), rel=1e-6)
    assert float(posterior.ratings[0]) == 0.0 and float(posterior.stderr[0]) == 0.0


def test_the_posterior_is_the_maximum_and_its_covariance_the_inverse_hessian():
    wins = _wins({(0, 1): (12, 4), (1, 2): (9, 9), (0, 2): (3, 13)}, 3)
    posterior = design.fit_posterior(wins, 0, prior_stderr=300.0)
    precision = (SCALE / 300.0) ** 2

    def log_posterior(free: torch.Tensor) -> torch.Tensor:
        theta = torch.cat([torch.zeros(1, dtype=DTYPE), free])
        difference = theta[:, None] - theta[None, :]
        likelihood = (wins * torch.nn.functional.logsigmoid(difference)).sum()
        return likelihood - 0.5 * precision * free.square().sum()

    at = (posterior.ratings[1:] / SCALE).clone().requires_grad_()
    (gradient,) = torch.autograd.grad(log_posterior(at), at)
    assert gradient.abs().max() < 1e-7
    hessian = torch.autograd.functional.hessian(log_posterior, at.detach())
    covariance = torch.linalg.inv(-hessian) * SCALE**2
    assert posterior.covariance[1:, 1:].flatten().tolist() == pytest.approx(
        covariance.flatten().tolist(), rel=1e-8
    )


def test_a_clean_sweep_stays_finite_and_within_the_prior():
    """576 straight wins against the anchor: the rating is finite and its error at most the
    prior's, where the old virtual-game prior's error never fell with more games."""
    for games in (96, 576, 5760):
        posterior = design.fit_posterior(_wins({(0, 1): (0, games)}, 2), 0)
        assert math.isfinite(float(posterior.ratings[1])) and float(posterior.ratings[1]) > 500
        assert 0.0 < float(posterior.stderr[1]) < design.PRIOR_STDERR
    _, few = fit_ratings(_wins({(0, 1): (0, 96)}, 2), 0, 1.0)
    _, many = fit_ratings(_wins({(0, 1): (0, 5760)}, 2), 0, 1.0)
    assert float(many[1]) > 0.9 * float(few[1])  # the old fit's error: flat in the games


# ---- the targets


def test_targets_list_players_then_contrasts_and_say_when_met():
    targets = design.Targets(torch.tensor([math.inf, 40.0, 20.0], dtype=DTYPE), ((1, 2, 30.0),))
    rows, goals = targets.rows(0)
    assert rows.tolist() == [[0, 1, 0], [0, 0, 1], [0, 1, -1]] and goals.tolist() == [40, 20, 30]
    assert targets.labels(["scripted", "a", "b"], 0) == ["a", "b", "a - b"]
    wins = _wins({(0, 1): (50, 50), (1, 2): (50, 50), (0, 2): (50, 50)}, 3)
    assert design.Targets(torch.full((3,), 40.0, dtype=DTYPE)).met(design.fit_posterior(wins, 0))
    assert not targets.met(design.fit_posterior(wins, 0))


def test_the_config_gives_decision_players_and_contrasts_their_targets():
    config = tournament.TournamentConfig(
        target_stderr=45.0, decision=("b",), decision_stderr=15.0, contrasts=(("a", "b"),)
    )
    targets = config.targets(["scripted", "a", "b"])
    assert targets.stderr.tolist() == [45.0, 45.0, 15.0]
    assert targets.contrasts == ((1, 2, config.contrast_stderr),)
    with pytest.raises(ValueError, match="not in the field"):
        config.targets(["scripted", "a"])
    with pytest.raises(ValueError, match="a block needs two"):
        tournament.TournamentConfig(players_per_batch=1)


# ---- the design


def test_the_design_meets_the_equivalence_theorem():
    """At the A-optimal design every pair's directional gain is at most the design's mean,
    with equality on its support (Kiefer-Wolfowitz for the weighted A criterion)."""
    count = 5
    free = torch.arange(count) != 0
    precision = torch.eye(count, dtype=DTYPE) * 0.03
    precision[0, 0] = 0.0
    pairs = torch.triu_indices(count, count, 1).T
    information = torch.full((pairs.shape[0],), 0.2, dtype=DTYPE)
    weight_matrix = torch.diag(free.to(DTYPE))
    weights = design.design_weights(
        precision, free, pairs, information, 500.0, weight_matrix, steps=3000
    )
    total = precision + 500.0 * design._laplacian(count, pairs, weights * information)
    covariance = design._inverse(total, free)
    gain = information * design._pair_quadratic(covariance @ weight_matrix @ covariance, pairs)
    mean = float(weights @ gain)
    assert float(gain.max()) <= mean * 1.01
    support = weights > 1e-3
    assert gain[support].tolist() == pytest.approx([mean] * int(support.sum()), rel=0.02)


def _uniform_posterior(count: int, games: float = 0.0) -> design.Posterior:
    """A field of equal players, each pair with `games` drawn games."""
    wins = torch.full((count, count), games / 2, dtype=DTYPE)
    wins.fill_diagonal_(0.0)
    return design.fit_posterior(wins, 0)


def test_a_plan_spreads_its_blocks_over_pairs_and_counts_earlier_games():
    posterior = _uniform_posterior(6)
    targets = design.Targets(torch.full((6,), 40.0, dtype=DTYPE))
    allowed = ~torch.eye(6, dtype=torch.bool)
    plan = design.plan_batch(posterior, targets, allowed, design.Layout(10, 16))
    assert len(plan.blocks) == 10 and all(i < j for i, j in plan.blocks)
    assert len(set(plan.blocks)) >= 5  # a distribution, not one pair
    assert {player for pair in plan.blocks for player in pair} == set(range(6))
    assert 0.0 < plan.after < plan.criterion and sum(plan.weights.values()) == pytest.approx(1.0)
    # a pair with many games already gets fewer blocks than one with none
    wins = torch.zeros((6, 6), dtype=DTYPE)
    wins[2, 3] = wins[3, 2] = 400.0
    played = design.plan_batch(
        design.fit_posterior(wins, 0), targets, allowed, design.Layout(10, 16)
    )
    assert played.blocks.count((2, 3)) < max(played.blocks.count(pair) for pair in played.blocks)


def test_a_plan_keeps_to_the_players_per_batch_and_the_lanes():
    posterior = _uniform_posterior(8)
    targets = design.Targets(torch.full((8,), 40.0, dtype=DTYPE))
    allowed = ~torch.eye(8, dtype=torch.bool)
    for players in (2, 3, 5):
        plan = design.plan_batch(posterior, targets, allowed, design.Layout(6, 16, players))
        assert len({player for pair in plan.blocks for player in pair}) <= players
    # lanes: group 0 (players 1 to 3) holds at most 3 slots of a batch
    groups = (None, 0, 0, 0, 1, 1, 1, 1)
    plan = design.plan_batch(
        posterior, targets, allowed, design.Layout(6, 16, None, groups, (3, 9))
    )
    held = [groups[index] for pair in plan.blocks for index in pair]
    assert held.count(0) <= 3 and held.count(1) <= 9
    with pytest.raises(ValueError, match="no allowed pair fits"):
        design.plan_batch(posterior, targets, allowed, design.Layout(6, 16, None, groups, (1, 1)))


def test_the_triangle_share_plays_every_pair_of_decision_players_first():
    posterior = _uniform_posterior(6, games=50.0)
    targets = design.Targets(torch.full((6,), 30.0, dtype=DTYPE))
    allowed = ~torch.eye(6, dtype=torch.bool)
    plan = design.plan_batch(
        posterior, targets, allowed, design.Layout(8, 16), triangle_share=0.5, decision=(1, 2, 3)
    )
    assert set(plan.blocks[:3]) == {(1, 2), (1, 3), (2, 3)}


# ---- the saturated anchor pair (the 672-of-1152 allocation)


SATURATED = torch.tensor([0.0, 1500.0, 1450.0], dtype=DTYPE)  # scripted, two strong arms


def test_the_old_allocation_spent_its_games_on_the_saturated_anchor_pair():
    """The reproduction, at the old allocator's arithmetic: a virtual-game prior's
    information vanishes with a sweep, so the strong player's error never falls and the
    greedy trace keeps taking its anchor pair (most of the games went to it)."""
    generator = torch.Generator().manual_seed(0)
    play = _play(SATURATED, 96, generator)
    wins = torch.zeros((3, 3), dtype=DTYPE)
    spent = collections.Counter()
    for _ in range(6):  # 6 batches of 2 blocks of 96 games: 1152 games
        ratings, _ = fit_ratings(wins, 0, 1.0)
        probability = torch.sigmoid((ratings[:, None] - ratings[None, :]) / SCALE)
        games = wins + wins.T + torch.tensor([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
        chosen = []
        for _ in range(2):  # each block greedily, as the old allocate_blocks did
            best, value = None, -1.0
            for pair in ((0, 1), (0, 2), (1, 2)):
                added = games.clone()
                for first, second in chosen + [pair]:
                    added[first, second] += 96
                    added[second, first] += 96
                weight = added * probability * (1 - probability)
                fisher = (torch.diag(weight.sum(1)) - weight)[1:, 1:]
                trace = float(torch.linalg.inv(fisher).trace())
                if best is None or -trace > value:
                    best, value = pair, -trace
            chosen.append(best)
        for (first, second), (won, _, lost) in zip(chosen, play(chosen).tolist(), strict=True):
            wins[first, second] += won
            wins[second, first] += lost
            spent[(first, second)] += 96
    saturated = max(spent[(0, 1)], spent[(0, 2)])
    assert saturated >= 672 and spent[(1, 2)] <= 96


def test_the_saturated_anchor_pair_gives_way_to_the_contrast_that_matters():
    """The same field, rated by the tournament: the two arms' difference (a named contrast)
    takes most of the games, and the anchor pairs no more than a block each once they are
    found out of reach."""
    names = ["scripted", "a", "b"]
    config = tournament.TournamentConfig(
        ArenaConfig(blocks=2, block_environments=96),
        target_stderr=45.0,
        budget=1152,
        contrasts=(("a", "b"),),
        contrast_stderr=20.0,
    )
    logged = []
    ratings, stderr, pair_games = tournament.fit_field(
        _play(SATURATED, 96, torch.Generator().manual_seed(0)),
        names,
        0,
        tournament.allowed_pairs(names),
        config,
        logged.append,
        "saturated",
    )
    assert float(pair_games[1, 2]) >= 384
    assert float(pair_games[0, 1] + pair_games[0, 2]) <= 192
    assert all(math.isfinite(float(value)) for value in (*ratings, *stderr))
    assert any("out of reach" in line for line in logged)
    difference = float(stderr[1] ** 2 + stderr[2] ** 2)  # finite, within the prior
    assert difference < 2 * design.PRIOR_STDERR**2


def test_a_hopeless_field_stops_rather_than_spend_its_budget():
    names = ["scripted", "a", "b"]
    config = tournament.TournamentConfig(
        ArenaConfig(blocks=2, block_environments=96), target_stderr=45.0, budget=1152
    )
    logged = []
    _, _, pair_games = tournament.fit_field(
        _play(SATURATED, 96, torch.Generator().manual_seed(0)),
        names,
        0,
        tournament.allowed_pairs(names),
        config,
        logged.append,
        "saturated",
    )
    assert float(pair_games.sum()) / 2 < 1152 and "stopped" in logged[-1]


def test_a_ladder_is_rated_through_its_neighbours():
    """A ladder of checkpoints 150 Elo apart, the top ones far above the anchor: the games
    go mostly to close pairs, and every target is met within the budget."""
    truth = torch.tensor([150.0 * index for index in range(10)], dtype=DTYPE)
    names = ["scripted"] + [f"c{index}" for index in range(1, 10)]
    config = tournament.TournamentConfig(
        ArenaConfig(blocks=8, block_environments=64), target_stderr=50.0, budget=40_000
    )
    ratings, stderr, pair_games = tournament.fit_field(
        _play(truth, 64, torch.Generator().manual_seed(1)),
        names,
        0,
        tournament.allowed_pairs(names),
        config,
        lambda line: None,
        "ladder",
    )
    assert float(stderr.max()) <= 50.0
    assert ((ratings - truth).abs() <= 4.0 * stderr + 1e-9).all()
    distance = (truth[:, None] - truth[None, :]).abs()
    close = float(pair_games[distance <= 450.0].sum())
    assert close > 0.8 * float(pair_games.sum())
