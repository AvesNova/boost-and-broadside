"""Ratings with a cyclic part (`bnb.league.cyclic`, mElo-2k): it finds a rock-paper-scissors
field that Bradley-Terry cannot fit, and adds nothing to a transitive one."""

import pytest
import torch

from bnb.league import cyclic


def _counts(z: torch.Tensor, games: int, seed: int) -> torch.Tensor:
    """(K, K, 3) wins, draws (none) and losses of `games` games per pair drawn at log-odds z."""
    generator = torch.Generator().manual_seed(seed)
    count = z.shape[0]
    counts = torch.zeros((count, count, 3), dtype=torch.float64)
    for i in range(count):
        for j in range(i + 1, count):
            won = float(
                torch.binomial(
                    torch.tensor(float(games), dtype=torch.float64),
                    torch.sigmoid(z[i, j]).double(),
                    generator=generator,
                )
            )
            counts[i, j] = torch.tensor([won, 0.0, games - won])
            counts[j, i] = torch.tensor([games - won, 0.0, won])
    return counts


def _rock_paper_scissors() -> torch.Tensor:
    """Six players: a transitive spread plus a strong three-way cycle."""
    ratings = torch.tensor([0.6, 0.3, 0.0, -0.1, -0.4, -0.4], dtype=torch.float64)
    angle = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.float64) * 2.0943951
    vectors = 1.2 * torch.stack([angle.cos(), angle.sin()], 1)
    return cyclic.log_odds(ratings, vectors)


def test_the_log_odds_are_antisymmetric_and_bradley_terry_at_k_zero():
    ratings = torch.tensor([0.5, -0.2, 0.1], dtype=torch.float64)
    vectors = torch.randn((3, 4), dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    z = cyclic.log_odds(ratings, vectors)
    assert torch.allclose(z, -z.T)
    plain = cyclic.log_odds(ratings, vectors[:, :0])
    assert torch.allclose(plain, ratings[:, None] - ratings[None, :])


def test_a_transitive_field_is_recovered_and_gains_nothing_from_a_cycle():
    ratings = torch.linspace(-1.0, 1.0, 6, dtype=torch.float64)
    counts = _counts(cyclic.log_odds(ratings, torch.zeros((6, 0))), 400, 0)
    wins = counts[..., 0] + 0.5 * counts[..., 1]
    model = cyclic.fit(wins, counts.sum(-1), 0)
    assert torch.allclose(model.ratings, ratings - ratings.mean(), atol=0.08)
    result = cyclic.cross_validate(counts, (0, 1), (1.0, 10.0), folds=4, links=("logit",))
    for name, model in result["models"].items():
        assert model["gain_over_bradley_terry"] < 2.0 * model["gain_stderr"] + 1e-9, name


def test_a_cycle_is_found_out_of_sample():
    z = _rock_paper_scissors()
    counts = _counts(z, 300, 1)
    result = cyclic.cross_validate(counts, (0, 1), (0.1, 1.0), folds=4)
    assert result["best_transitive"].startswith("link=")
    best = result["models"][result["best"]]
    assert ", k=1," in result["best"]
    assert best["gain_over_bradley_terry"] > 5.0 * best["gain_stderr"]
    wins = counts[..., 0] + 0.5 * counts[..., 1]
    model = cyclic.fit(wins, counts.sum(-1), 1, 0.1)
    assert torch.allclose(model.log_odds(), z, atol=0.4)  # 300 games a pair: about 0.15 noise
    parts = cyclic.decomposition(model)
    assert parts["cyclic_share"] > 0.5
    plain = cyclic.fit(wins, counts.sum(-1), 0)
    off = cyclic.residuals(counts, plain.probability()).abs()
    assert (off > 3.0).sum() > 6  # Bradley-Terry misses the cycle's pairs
    assert (cyclic.residuals(counts, model.probability()).abs() > 3.5).sum() == 0


def test_the_outcomes_are_every_game_once():
    counts = torch.zeros((3, 3, 3), dtype=torch.float64)
    counts[0, 1] = torch.tensor([3.0, 1.0, 2.0])
    counts[1, 0] = torch.tensor([2.0, 1.0, 3.0])
    counts[1, 2] = torch.tensor([0.0, 0.0, 4.0])
    counts[2, 1] = torch.tensor([4.0, 0.0, 0.0])
    first, second, result, fold = cyclic.outcomes(counts, 2)
    assert len(result) == 10 and set(fold.tolist()) == {0, 1}
    assert float(result[(first == 0) & (second == 1)].sum()) == pytest.approx(3.5)
    assert float(result[(first == 1) & (second == 2)].sum()) == 0.0


def test_a_bent_link_is_fitted_without_a_cycle():
    """Results whose margins follow a power link (gamma 0.6): the power link recovers the
    exponent and predicts held-out games better than Bradley-Terry, with no cycle, and no
    triple goes round by majority."""
    ratings = torch.linspace(-2.5, 2.5, 7, dtype=torch.float64)
    z = ratings[:, None] - ratings[None, :]
    bent = torch.sign(z) * z.abs() ** 0.6
    counts = _counts(bent, 400, 2)
    wins = counts[..., 0] + 0.5 * counts[..., 1]
    model = cyclic.fit(wins, counts.sum(-1), 0, link="power")
    assert model.gamma == pytest.approx(0.6, abs=0.1)
    result = cyclic.cross_validate(counts, (0,), (), folds=4)
    power = result["models"]["link=power, k=0, penalty=0"]
    assert power["gain_over_bradley_terry"] > 3.0 * power["gain_stderr"]
    assert result["best_transitive"] == "link=power, k=0, penalty=0"
    assert cyclic.majority_cycles(counts) == []


def test_a_majority_cycle_is_counted_once():
    counts = torch.zeros((3, 3, 3), dtype=torch.float64)
    for a, b, won in ((0, 1, 7), (1, 2, 6), (2, 0, 9)):
        counts[a, b] = torch.tensor([won, 0.0, 10.0 - won])
        counts[b, a] = torch.tensor([10.0 - won, 0.0, won])
    assert cyclic.majority_cycles(counts) == [(0, 1, 2, 0.6)]
    counts[2, 0], counts[0, 2] = counts[0, 2].clone(), counts[2, 0].clone()
    assert cyclic.majority_cycles(counts) == []
