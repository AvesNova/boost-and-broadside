"""v1's rating estimator through the interface: the full Bradley-Terry fit."""

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
