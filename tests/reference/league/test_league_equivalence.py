"""v1's rating estimators through the interface: the full Bradley-Terry fit, one player
against known opponents, the ladder fit with the gauge held, the live rating, and the
offline game allocation."""

import math

import torch

from league_helpers import case, check

# The goldens are each estimator's converged optimum; an implementation must reach it to
# 1e-6 relative (about 1e-3 rating points at the gauge's scale), whatever its algorithm.
CONVERGED = dict(relative=1e-6, absolute=1e-6)


def check_ratings(actual, expected, name):
    """Elementwise, with +-inf and nan (a sweep, no games) required exactly."""
    actual = torch.as_tensor(actual, dtype=torch.float64).reshape(expected.shape)
    special = ~torch.isfinite(expected)
    assert torch.equal(torch.isnan(actual), torch.isnan(expected)), f"{name}: nan pattern differs"
    infinite = torch.isinf(expected)
    assert torch.equal(actual[infinite], expected[infinite]), f"{name}: infinities differ"
    check(actual[~special], expected[~special], name, **CONVERGED)


def test_bradley_terry_with_prior(arm):
    golden = case("league_ratings")
    ratings, stderr = arm.fit_ratings(golden.inputs["wins"], golden.config["anchor"],
                                      golden.config["prior_games"])
    check(ratings, golden.outputs["ratings"], "ratings", **CONVERGED)
    check(stderr, golden.outputs["stderr"], "stderr", **CONVERGED)


def test_bradley_terry_without_prior(arm):
    golden = case("league_ratings_unshrunk")
    ratings, stderr = arm.fit_ratings(golden.inputs["wins"], golden.config["anchor"],
                                      golden.config["prior_games"])
    check(ratings, golden.outputs["ratings"], "ratings", **CONVERGED)
    check(stderr, golden.outputs["stderr"], "stderr", **CONVERGED)


def test_one_player_against_known_opponents(arm):
    golden = case("league_single")
    inputs = golden.inputs
    results = [arm.rate_against(inputs["opponent_ratings"], wins, losses)
               for wins, losses in zip(inputs["wins"], inputs["losses"])]
    check_ratings([float(r) for r, _ in results], golden.outputs["rating"], "rating")
    check_ratings([float(s) for _, s in results], golden.outputs["stderr"], "stderr")


def test_ladder_holds_the_gauge(arm):
    golden = case("league_ladder")
    inputs = golden.inputs
    ratings = arm.fit_ladder(inputs["wins"], inputs["ties"], inputs["fixed"],
                             inputs["fixed_ratings"], golden.config["prior_games"])
    check(ratings, golden.outputs["ratings"], "ratings", **CONVERGED)
    fixed = inputs["fixed"]
    check(torch.as_tensor(ratings, dtype=torch.float64)[fixed], inputs["fixed_ratings"][fixed],
          "fixed ratings", relative=0.0, absolute=0.0)


def test_live_rating(arm):
    golden = case("league_live")
    inputs = golden.inputs
    results = [arm.rate_live(inputs["opponent_ratings"], wins, losses, ties)
               for wins, losses, ties in zip(inputs["wins"], inputs["losses"], inputs["ties"])]
    check_ratings([float(r) for r, _ in results], golden.outputs["rating"], "rating")
    check_ratings([float(s) for _, s in results], golden.outputs["stderr"], "stderr")
    assert all(math.isinf(float(s)) for r, s in results if math.isinf(float(r)))


def test_game_allocation(arm):
    golden = case("league_allocation")
    config, inputs = golden.config, golden.inputs
    allocation = arm.allocate(inputs["pair_games"], inputs["ratings"], config["anchor"],
                              config["budget"], config["minimum_per_pair"])
    check(torch.as_tensor(allocation, dtype=torch.int64), golden.outputs["allocation"].long(),
          "allocation")
    seed = arm.allocate(torch.zeros_like(inputs["pair_games"]), torch.zeros_like(inputs["ratings"]),
                        config["anchor"], config["seed_budget"], 0)
    check(torch.as_tensor(seed, dtype=torch.int64), golden.outputs["seed_allocation"].long(),
          "seed allocation")
