"""The rating tournament's statistics (league spec 6, `bnb.league.design`): the
maximum-likelihood fit with no prior and its bounds, the targets, the continuous A-optimal
design and the batch plan with its players-per-batch and lane limits, the out-of-reach
rule, and the saturated anchor pair that the earlier greedy allocation kept playing."""

import collections
import json
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


# ---- the fit


CONNECTED = {(0, 1): (30, 50), (1, 2): (40, 40), (0, 2): (20, 60), (2, 3): (70, 10)}


def test_the_fit_is_the_plain_maximum_likelihood_on_a_connected_field():
    """No prior: on a field where every player won and lost some games, the fit is the
    Bradley-Terry maximum (the gradient of the plain log-likelihood vanishes there), its
    covariance the inverse of the negative Hessian, and it agrees with `fit_ratings` without
    virtual games."""
    wins = _wins(CONNECTED, 4)
    posterior = design.fit_posterior(wins, 0)
    assert posterior.status == (design.RATED,) * 4

    def log_likelihood(free: torch.Tensor) -> torch.Tensor:
        theta = torch.cat([torch.zeros(1, dtype=DTYPE), free])
        return (wins * torch.nn.functional.logsigmoid(theta[:, None] - theta[None, :])).sum()

    at = (posterior.ratings[1:] / SCALE).clone().requires_grad_()
    (gradient,) = torch.autograd.grad(log_likelihood(at), at)
    assert gradient.abs().max() < 1e-9
    hessian = torch.autograd.functional.hessian(log_likelihood, at.detach())
    covariance = torch.linalg.inv(-hessian) * SCALE**2
    assert posterior.covariance[1:, 1:].flatten().tolist() == pytest.approx(
        covariance.flatten().tolist(), rel=1e-8
    )
    ratings, stderr = fit_ratings(wins, 0, prior_games=0.0)
    assert posterior.ratings.tolist() == pytest.approx(ratings.tolist(), abs=1e-6)
    assert posterior.stderr.tolist() == pytest.approx(stderr.tolist(), rel=1e-8)
    assert float(posterior.ratings[0]) == 0.0 and float(posterior.stderr[0]) == 0.0


def test_a_sweep_is_reported_as_a_bound():
    """A player who beat the anchor in every game has no finite maximum: it is reported as
    a lower bound, the rating at which its sweep has probability BOUND_LEVEL, with no
    standard error; one who lost every game, as an upper bound. The rated players' fit is
    the fit without them."""
    for games in (1, 96, 5760):
        posterior = design.fit_posterior(_wins({(0, 1): (0, games)}, 2), 0)
        expected = SCALE * math.log(design.BOUND_LEVEL ** (1 / games)) - SCALE * math.log(
            1 - design.BOUND_LEVEL ** (1 / games)
        )
        assert posterior.status[1] == design.ABOVE
        assert float(posterior.ratings[1]) == pytest.approx(expected, abs=1e-6)
        assert math.isinf(float(posterior.stderr[1]))
    wins = torch.zeros((6, 6), dtype=DTYPE)
    wins[:4, :4] = _wins(CONNECTED, 4)
    wins[4, 2] = 96.0  # 4 beat player 2 in every game
    wins[0, 5] = 50.0  # the anchor beat 5 in every game
    posterior = design.fit_posterior(wins, 0)
    alone = design.fit_posterior(_wins(CONNECTED, 4), 0)
    assert posterior.status[4:] == (design.ABOVE, design.BELOW)
    assert posterior.ratings[:4].tolist() == pytest.approx(alone.ratings.tolist(), abs=1e-9)
    assert posterior.stderr[:4].tolist() == pytest.approx(alone.stderr.tolist(), rel=1e-9)
    assert float(posterior.ratings[4]) > float(posterior.ratings[2])
    assert float(posterior.ratings[5]) < 0.0
    row = tournament.rating_columns(posterior, 4)
    assert row["stderr"] is None and row["status"] == design.ABOVE
    assert tournament.describe_rating(row).startswith("> ")


def test_a_group_joined_by_sweeps_keeps_its_own_differences():
    """Two players who split their games and both swept the anchor: their shift is a
    bound, their difference keeps its maximum likelihood and its standard error, and a
    contrast between them can be met."""
    wins = _wins({(0, 1): (0, 96), (0, 2): (0, 96), (1, 2): (60, 36)}, 3)
    posterior = design.fit_posterior(wins, 0)
    assert posterior.status == (design.RATED, design.ABOVE, design.ABOVE)
    difference = float(posterior.ratings[1] - posterior.ratings[2])
    assert difference == pytest.approx(SCALE * math.log(60 / 36), abs=1e-6)
    contrast = design.Targets(torch.full((3,), math.inf, dtype=DTYPE), ((1, 2, 100.0),))
    (error,), _ = contrast.standard_errors(posterior)
    assert float(error) == pytest.approx(SCALE / math.sqrt(96 * 0.625 * 0.375), rel=1e-6)
    assert contrast.met(posterior)
    # a player who played only the undetermined: no bound, and its target is unmet
    wins = _wins({(0, 1): (40, 40), (2, 3): (10, 20)}, 4)
    posterior = design.fit_posterior(wins, 0)
    assert posterior.status[2:] == (design.UNDETERMINED,) * 2
    assert not design.Targets(torch.full((4,), 500.0, dtype=DTYPE)).met(posterior)


# ---- the targets


def test_targets_list_players_then_contrasts_and_say_when_met():
    targets = design.Targets(torch.tensor([math.inf, 40.0, 20.0], dtype=DTYPE), ((1, 2, 30.0),))
    rows, goals = targets.rows(0)
    assert rows.tolist() == [[0, 1, 0], [0, 0, 1], [0, 1, -1]] and goals.tolist() == [40, 20, 30]
    assert targets.labels(["scripted", "a", "b"], 0) == ["a", "b", "a - b"]
    wins = _wins({(0, 1): (50, 50), (1, 2): (50, 50), (0, 2): (50, 50)}, 3)
    assert design.Targets(torch.full((3,), 40.0, dtype=DTYPE)).met(design.fit_posterior(wins, 0))
    assert not targets.met(design.fit_posterior(wins, 0))
    # a bound is left out of the check; an unplayed player is unmet
    loose = design.Targets(torch.full((4,), 40.0, dtype=DTYPE))
    swept = torch.zeros((4, 4), dtype=DTYPE)
    swept[:3, :3] = wins
    swept[3, 0] = 96.0
    assert loose.met(design.fit_posterior(swept, 0))
    swept[3, 0] = 0.0
    assert not loose.met(design.fit_posterior(swept, 0))


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
    """The same field, rated by the tournament: both arms sweep the anchor and are reported
    as one group above a bound; their difference (a named contrast) has a standard error
    within the group, takes most of the games and is met, and the anchor pairs get no more
    than a block each."""
    names = ["scripted", "a", "b"]
    config = tournament.TournamentConfig(
        ArenaConfig(blocks=2, block_environments=96),
        target_stderr=45.0,
        budget=1152,
        contrasts=(("a", "b"),),
        contrast_stderr=20.0,
    )
    logged = []
    posterior, pair_games = tournament.fit_field(
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
    assert posterior.status[1:] == (design.ABOVE, design.ABOVE)  # both swept the anchor
    contrast = design.Targets(torch.full((3,), math.inf, dtype=DTYPE), ((1, 2, 20.0),))
    (error,), _ = contrast.standard_errors(posterior)
    assert float(error) <= 20.0 and "stopped: every target is met" in logged[-1]


def test_a_hopeless_field_stops_rather_than_spend_its_budget():
    names = ["scripted", "a", "b"]
    config = tournament.TournamentConfig(
        ArenaConfig(blocks=2, block_environments=96), target_stderr=45.0, budget=1152
    )
    logged = []
    posterior, pair_games = tournament.fit_field(
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
    posterior, pair_games = tournament.fit_field(
        _play(truth, 64, torch.Generator().manual_seed(1)),
        names,
        0,
        tournament.allowed_pairs(names),
        config,
        lambda line: None,
        "ladder",
    )
    ratings, stderr = posterior.ratings, posterior.stderr
    assert posterior.status == (design.RATED,) * 10 and float(stderr.max()) <= 50.0
    assert ((ratings - truth).abs() <= 4.0 * stderr + 1e-9).all()
    distance = (truth[:, None] - truth[None, :]).abs()
    close = float(pair_games[distance <= 450.0].sum())
    assert close >= 0.75 * float(pair_games.sum())  # 0.80 to 0.825 over seeds 1 to 8


def test_a_swept_player_is_planned_against_stronger_opponents():
    """A new player who swept the anchor: it is planned at its bound plus MARGIN_CAP, so
    its next games go to the rated players above the anchor nearest that, not back to the
    anchor (the saturated pair)."""
    names = ["scripted", "x", "y", "z", "new"]
    truth = torch.tensor([0.0, 300.0, 700.0, 1100.0], dtype=DTYPE)
    wins = torch.zeros((5, 5), dtype=DTYPE)
    for first, second in ((0, 1), (1, 2), (2, 3), (0, 2)):  # a rated ladder, many games
        share = float(torch.sigmoid((truth[first] - truth[second]) / SCALE))
        wins[first, second] = 2000 * share
        wins[second, first] = 2000 * (1 - share)
    wins[4, 0] = 96.0  # the new player won all 96 games against the anchor
    posterior = design.fit_posterior(wins, 0)
    assert posterior.status[4] == design.ABOVE and posterior.status[:4] == (design.RATED,) * 4
    targets = design.Targets(torch.full((5,), 45.0, dtype=DTYPE))
    assert targets.met(posterior)  # the rated players are met, the bound left out
    allowed = ~torch.eye(5, dtype=torch.bool)
    plan = design.plan_batch(posterior, targets, allowed, design.Layout(2, 96))
    # the bound is about 600, so it is planned at about 800: y (700) is the nearest
    assert float(posterior.ratings[4]) == pytest.approx(600.0, abs=5.0)
    assert [(names[first], names[second]) for first, second in plan.blocks] == [("y", "new")] * 2


# ---- the layout probe (bench/rating_layout_probe.py)


def test_the_probe_reads_layouts_and_more_blocks_plan_more_information():
    from bench import rating_layout_probe as probe

    assert probe.parse_layouts("2x96,8x128x16") == [(2, 96, 32), (8, 128, 16)]
    with pytest.raises(ValueError, match="blocks x environments"):
        probe.parse_layouts("2")
    few, _ = probe.reference_information(2, 512, None)
    many, _ = probe.reference_information(8, 128, None)
    assert many > few > 0.0  # the same games over more pairs


@pytest.mark.usefixtures("global_generator")
def test_the_probe_measures_a_tiny_layout_on_the_cpu(tmp_path):
    from bench import rating_layout_probe as probe
    from bnb.contracts import GameConfig
    from bnb.league.matches import player

    game = GameConfig(2, 2, 1500.0, episode_seconds=2.0, respawn_seconds=1.0)
    out = tmp_path / "probe.jsonl"
    field = [player("scripted"), player("random"), player("rung_T2")]
    summaries = probe.measure(
        field, game, [(2, 4, 16)], torch.device("cpu"), 1, False, None, out, lambda line: None
    )
    assert summaries[0]["games_per_batch"] == 8 and summaries[0]["games_per_second"] > 0.0
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert lines[0]["games"] == 8 and "summary" in lines[-1]
