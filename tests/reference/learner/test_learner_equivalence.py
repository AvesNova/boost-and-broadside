"""v1's per-level returns, team mixing and loss terms, through the interface."""

from learner_helpers import TIGHT, case, check


def test_per_level_gae(arm):
    golden = case("learner_gae")
    inputs = golden.inputs
    advantages, returns = arm.per_level_gae(
        inputs["rewards"], inputs["values"], inputs["terminated"], inputs["next_value"],
        inputs["next_done"], inputs["gamma"], inputs["lam"])
    check(advantages, golden.outputs["advantages"], "advantages", **TIGHT)
    check(returns, golden.outputs["returns"], "returns", **TIGHT)


def test_team_aggregate(arm):
    golden = case("learner_team_aggregate")
    inputs = golden.inputs
    advantage, total = arm.team_aggregate(inputs["team"], inputs["alive"], inputs["advantages"],
                                          inputs["returns"], inputs["team_spirit"])
    check(advantage, golden.outputs["advantage"], "advantage", **TIGHT)
    check(total, golden.outputs["total_return"], "total_return", **TIGHT)


def test_policy_and_entropy(arm):
    golden = case("learner_policy")
    inputs = golden.inputs
    policy = arm.policy_loss(inputs["logp"], inputs["old_logp"], inputs["advantage"],
                             golden.config["return_scale"], inputs["mask"], golden.config["clip"])
    check(policy, golden.outputs["policy"], "policy", **TIGHT)
    check(arm.entropy_term(inputs["entropy"], inputs["mask"]), golden.outputs["entropy"],
          "entropy", **TIGHT)


def test_critic_losses(arm):
    golden = case("learner_critic")
    inputs = golden.inputs
    per_level = arm.critic_losses(
        inputs["local_logits"], inputs["outcome_logits"], inputs["local_returns"],
        inputs["outcome_targets"], inputs["valued"], inputs["bins"],
        golden.config["local_levels"].tolist(), golden.config["outcome_level"])
    check(per_level, golden.outputs["per_level"], "per_level", **TIGHT)


def test_shot_cross_entropy(arm):
    golden = case("learner_shot_cross_entropy")
    inputs = golden.inputs
    parts, outcome = arm.shot_cross_entropy(
        inputs["trajectory_logits"], inputs["trajectory_target"], inputs["outcome_logits"],
        inputs["outcome_target"], golden.config["runs"].tolist())
    check(parts, golden.outputs["trajectory"], "trajectory", **TIGHT)
    check(outcome, golden.outputs["outcome"], "outcome", **TIGHT)
