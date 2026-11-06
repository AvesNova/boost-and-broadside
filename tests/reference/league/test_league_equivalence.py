"""v1's rating estimators through the interface: the full Bradley-Terry fit and the offline
game allocation."""

import torch
from league_helpers import case, check

# The goldens are each estimator's converged optimum; an implementation must reach it to
# 1e-6 relative (about 1e-3 rating points at the gauge's scale), whatever its algorithm.
CONVERGED = dict(relative=1e-6, absolute=1e-6)


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
