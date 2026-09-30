"""Behavior-cloning intent, as an overlay on the RL profile.

BC pretrains the policy against the stochastic scripted controller: the
controller supplies supervised action targets on every environment, no policy
gradient is taken, and no roster opponent plays a rollout.  The critic and the
next-state head train alongside so RL inherits more than an actor.

Imitation has to happen in the environment RL continues in, so everything BC's
objective does not require is RL's value *by construction* rather than by
restatement.  This module is the complete list of differences:

* ``objective.next_state_coef`` -- full-strength next-state prediction while
  there is a dense supervised signal to learn the trunk from.
* ``optimizer.total_timesteps`` -- BC owns its budget and stops when imitation
  saturates, not when RL's curriculum ends.
* five schedule entries, each commented below.

The overlay is why there is no test policing that list.  A shared value cannot
drift here without drifting in RL too, which is the property the deleted
``tests/config/test_bc_profile.py`` spent 181 lines checking by hand.
"""

from dataclasses import replace

from boost_and_broadside.config.schedule_spec import hold
from boost_and_broadside.profiles.rl import RL_PROFILE

BC_SCHEDULE_SPEC = replace(
    RL_PROFILE.schedule_spec,
    # Warm up to the project learning rate, then hold.  RL's decay tail is keyed
    # to keypoints at 100M and 500M steps -- the end of *its* budget -- and
    # means nothing on BC's own, much longer one.
    learning_rate=((0, 1e-7, "linear"), (6_000_000, 3e-4, "hold")),
    # No policy gradient: the scripted controller supplies supervised action
    # targets and never takes a side in the rollout.
    policy_gradient_coef=hold(0.0),
    # In BC this is the policy head's only learning signal, deliberately
    # balanced one-to-one against the next-state auxiliary BC also weights at
    # 1.0.  RL's 2.0 is the strength of an *auxiliary* imitation term carried
    # alongside a live policy gradient.
    behavior_cloning_coef=hold(0.93),
    # 18.9 looks large and is not: the value loss is a Huber on normalized
    # returns whose raw value is ~0.04, so the weighted term lands near 0.8 --
    # beside behaviour cloning's. The number is large because the value head
    # was absorbing almost all of its own gradient and passing almost none to
    # the trunk, which is the thing being corrected.
    value_function_coef=hold(18.9),
    # League opposition disabled: no roster opponent plays a BC rollout.  The
    # Elo evaluator still runs -- BC's own scripted win rate is what decays the
    # cloning weight -- and it rates against the same derived rungs RL uses.
    league_fraction=hold(0.0),
    # A KL trust region early-stops epochs when the policy moves away from the
    # one that produced the rollout.  Under supervision that movement is the
    # objective, so the PPO stopping criterion does not apply.
    target_kl=hold(None),
)

BC_PROFILE = replace(
    RL_PROFILE,
    name="bc",
    schedule_spec=BC_SCHEDULE_SPEC,
    # --- Gradient-share balance, set from a measured decomposition ----------
    # Coefficients chosen so each term takes a target share of the trunk's
    # pre-clip gradient, from `benchmarks/gradient_decomposition.py` on the
    # 6.4M-step pilot (docs/internal/grad-decomposition-pilot-sep2026.json).
    # The gradient of `c * L` is exactly `c * grad L`, so one measurement gives
    # every coefficient at once:  c_new = c_old * (target share / measured) * G.
    #
    # Targets: bc 0.30, value 0.25, next_state 0.15, enemy_action 0.10,
    # outcome 0.08, density 0.07. The objective takes the plurality; value gets
    # a real share because BC shapes the trunk for *action*-relevant features
    # and the critic needs *outcome*-relevant ones, which nothing else supplies
    # -- run 748's explained variance plateaued at 0.574 with value holding
    # 0.001 of the trunk. Regularizers are left out of share-targeting: their
    # scale is meaningful in the nats they live in.
    #
    # G puts the total pre-clip norm near 0.4 against max_grad_norm 1.0, so the
    # clip becomes a spike-catcher instead of binding every step by a factor of
    # 100. That rescale is only safe because Adam's eps is 1e-8; at the old
    # 1e-5 it would have pushed a third of the network further into SGD.
    #
    # Provisional, and known to be. They come from a head whose sigma has not
    # calibrated (hidden-enemy position read z^2 = 11 at 6.4M), which matters
    # most for next_state, whose beta-NLL weighting is sigma-dependent.
    # Re-measure near 30M and adjust once; `train/clip_fire_rate` and the term
    # shares are the readouts.
    next_state_coef=2.9e-4,
    enemy_action_coef=1.0,
    outcome_categorical_coef=0.052,
    global_density_coef=4.9,
    # Beta-NLL weighting on the next-state likelihood. At zero the gradient is
    # r / sigma**2, which put 7,421x more of itself on the channels and
    # visibility classes the head already predicts best -- 80% on allies and
    # 0.1% on hidden enemies, which is the bucket the belief plane exists for.
    # At 0.5 the per-token gradient is the standardized residual r / sigma,
    # which calibration pins near one everywhere, and the measured spread
    # across all 33 (channel, class) cells falls to 2.6x. Sigma still trains;
    # only the weighting is detached.
    next_state_beta=0.5,
    # The only stop condition, and the same budget RL carries. With cloning held
    # at full strength there is no self-terminating gate left here -- the 2B
    # placeholder meant "runs until imitation saturates", and saturation is now
    # something to read off ``loss/behavioral_cloning_kl`` rather than something
    # the run detects.
    total_timesteps=500_000_000,
    # Cloning never decays here. In RL the decay withdraws a warm start as the
    # policy outgrows it; in a pretraining run it is the *only* signal training
    # the actor, and RL's 0.45 would switch it off at roughly teacher parity --
    # exactly where a clone becomes worth keeping. Worse, with the policy
    # gradient already at zero, a decayed BC weight leaves nothing training the
    # actor at all: ``_actor_entropy_coef`` then drops entropy to zero to stop
    # the policy walking back to uniform, and the run spends its remaining
    # budget refining a critic for a frozen actor.
    #
    # Stop this run on a plateau in ``loss/behavioral_cloning_kl`` instead.
    bc_winrate_target=None,
)
