"""The learner fixture interface over bnb (see README.md).

Each method is a thin translation onto `bnb.learn`. The fixtures' masked means are the
learner's masked sums over the mask's own count (the learner divides by minibatch totals);
the fixtures' `(T, B, N)` is the learner's `(L, R, S)`.
"""

from bnb.learn import losses, returns


class LearnerArm:
    def per_level_gae(self, rewards, values, terminated, next_value, next_done, gamma, lam):
        return returns.per_level_gae(rewards, values, terminated, next_value, next_done,
                                     gamma, lam)

    def team_aggregate(self, team, alive, advantages, level_returns, team_spirit):
        return returns.team_aggregate(team, alive, advantages, level_returns, team_spirit)

    def policy_loss(self, logp, old_logp, advantage, return_scale, mask, clip):
        per_token = losses.surrogate(logp, old_logp, advantage, return_scale, clip)
        return losses.masked_sum(per_token, mask) / mask.sum()

    def entropy_term(self, entropy, mask):
        return -losses.masked_sum(entropy, mask) / mask.sum()

    def critic_losses(self, local_logits, outcome_logits, local_returns, outcome_targets,
                      valued, bins, local_levels, outcome_level):
        steps = outcome_logits[..., 0].numel()
        return losses.critic_losses(local_logits, outcome_logits, local_returns,
                                    outcome_targets, valued, bins, tuple(local_levels),
                                    outcome_level, valued.sum(), steps)

    def shot_cross_entropy(self, trajectory_logits, trajectory_target, outcome_logits,
                           outcome_target, runs):
        return losses.shot_cross_entropy(trajectory_logits, trajectory_target, outcome_logits,
                                         outcome_target, tuple(map(tuple, runs)))


def make_arm():
    return LearnerArm()
