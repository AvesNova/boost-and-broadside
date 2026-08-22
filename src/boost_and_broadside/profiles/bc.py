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

from boost_and_broadside.config.schedule_spec import constant_spec, linear_spec
from boost_and_broadside.profiles.rl import RL_PROFILE

BC_SCHEDULE_SPEC = replace(
    RL_PROFILE.schedule_spec,
    # Warm up to the project learning rate, then hold.  RL's decay tail is keyed
    # to keypoints at 100M and 500M steps -- the end of *its* budget -- and
    # means nothing on BC's own, much longer one.
    learning_rate=linear_spec((0, 1e-7), (6_000_000, 3e-4)),
    # No policy gradient: the scripted controller supplies supervised action
    # targets and never takes a side in the rollout.
    policy_gradient_coef=constant_spec(0.0),
    # In BC this is the policy head's only learning signal, deliberately
    # balanced one-to-one against the next-state auxiliary BC also weights at
    # 1.0.  RL's 2.0 is the strength of an *auxiliary* imitation term carried
    # alongside a live policy gradient.
    behavior_cloning_coef=constant_spec(1.0),
    # League opposition disabled: no roster opponent plays a BC rollout.  The
    # Elo evaluator still runs -- BC's own scripted win rate is what decays the
    # cloning weight -- and it rates against the same derived rungs RL uses.
    league_fraction=constant_spec(0.0),
    # A KL trust region early-stops epochs when the policy moves away from the
    # one that produced the rollout.  Under supervision that movement is the
    # objective, so the PPO stopping criterion does not apply.
    target_kl=constant_spec(None),
)

BC_PROFILE = replace(
    RL_PROFILE,
    name="bc",
    schedule_spec=BC_SCHEDULE_SPEC,
    next_state_coef=1.0,
    total_timesteps=2_000_000_000,
)
