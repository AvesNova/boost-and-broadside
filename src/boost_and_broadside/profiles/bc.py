"""Behavior-cloning intent, as an overlay on the RL profile.

BC pretrains the policy against the stochastic scripted controller: the
controller supplies supervised action targets on every environment, no policy
gradient is taken, and no roster opponent plays a rollout.  The critic and the
next-state head train alongside so RL inherits more than an actor.

Imitation has to happen in the environment RL continues in, so everything BC's
objective does not require is RL's value *by construction* rather than by
restatement.  This module is the complete list of differences:

* the auxiliary weights -- set for this profile's own gradient balance, which
  has no policy gradient in it (see the balance block below).
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
    # In BC this is the policy head's only learning signal. RL's 2.0 is the
    # strength of an *auxiliary* imitation term carried alongside a live policy
    # gradient. Set by the gradient-share balance in BC_PROFILE.
    behavior_cloning_coef=hold(0.017),
    # Per reward level: the critic sums its five levels' cross-entropies, each
    # at weight one (COMPONENT_VALUE_WEIGHTS). Set by the balance in BC_PROFILE.
    value_function_coef=hold(0.18),
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
    # --- Gradient-share balance ----------------------------------------------
    # Every term is in nats per token, so each weight is an exchange rate
    # between nats of different things. The weights are chosen so each term
    # takes a target share of the trunk's pre-clip gradient, measured with
    # `benchmarks/gradient_decomposition.py`. The gradient of `c * L` is
    # exactly `c * grad L`, so one measurement gives every weight at once:
    #     c_new = c_old * target share / measured share
    # Only the ratios matter: Adam divides out a common scale, and the global
    # clip (max_grad_norm 1.0, several times the ~0.2 typical norm) is a spike
    # guard, not a target.
    #
    # Targets: bc 0.30, value 0.30 (about 0.06 per level), next_state 0.20,
    # enemy_action 0.12, density 0.08. The objective takes the plurality.
    # Value gets a real share because RL inherits this critic, and BC shapes
    # the trunk for *action*-relevant features where the critic needs
    # *outcome*-relevant ones -- run 748's explained variance plateaued at
    # 0.574 with value holding 0.001 of the trunk. Next-state rose from 0.15
    # because hidden-ship beliefs are now built from its outputs (plan section
    # 8.3), so its quality is an input's quality. Regularizers are left out:
    # their scale is meaningful in the nats they live in.
    #
    # Two pilots, each solved from the median share over checkpoints past the
    # warmup:
    # * 754 (docs/internal/bc-pilot-754-oct2026.json), 13.8-14.7M steps, from
    #   first guesses bc 0.042, value 0.007, next_state 0.0023, enemy_action
    #   0.07, density 0.03. It priced damage per health point, so value was not
    #   solved from it.
    # * 755 (docs/internal/bc-pilot-755-oct2026.json), 10.3-19.7M steps, from
    #   bc 0.025, value 0.007, next_state 0.0030, enemy_action 0.063, density
    #   0.024. Measured: bc 0.45, value 0.012, next_state 0.28, enemy_action
    #   0.14, density 0.10. With damage fixed, value had lost the gradient the
    #   mispricing gave it.
    # Inside value the outcome still draws next to nothing of the trunk
    # (~1e-5 against 3e-3 to 1e-2 for each per-ship level), and its
    # explained variance sits near 0.35. Its per-level weight is the knob if
    # the next run shows the same.
    #
    # Shares drift by design: BC's teacher targets are soft, so its gradient
    # falls toward zero as imitation converges, while value and next-state
    # keep a noise floor. Measure in one window, after the 6M warmup and once
    # next-state's zero-initialised residual has stopped growing its share.
    next_state_coef=0.0021,
    enemy_action_coef=0.055,
    global_density_coef=0.020,
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
