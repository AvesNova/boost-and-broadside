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
    behavior_cloning_coef=hold(0.025),
    # Per reward level: the critic sums its five levels' cross-entropies, each
    # at weight one (COMPONENT_VALUE_WEIGHTS). Set by the balance in BC_PROFILE.
    value_function_coef=hold(0.007),
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
    # between nats of different things. A term's share of the trunk's pre-clip
    # gradient is measured with `benchmarks/gradient_decomposition.py`; the
    # gradient of `c * L` is exactly `c * grad L`, so one measurement gives
    # every weight for a chosen set of shares:
    #     c_new = c_old * target share / measured share
    # Only the ratios matter: Adam divides out a common scale, and the global
    # clip (max_grad_norm 1.0, several times the typical norm) is a spike guard,
    # not a target.
    #
    # The balance between the first five terms is pilot 755's, measured over
    # 10.3-19.7M steps (docs/internal/bc-pilot-755-oct2026.json):
    #     bc 0.45, next_state 0.28, enemy_action 0.14, density 0.10, value 0.01
    # Cloning takes the plurality and next-state the next share: hidden-ship
    # beliefs are built from its outputs (plan section 8.3), so its quality is
    # an input's quality.
    #
    # The counterfactual shot heads take 0.12 of it, about enemy_action's
    # share, and the other terms keep 755's ratios at 0.88 of their shares.
    # Solved from pilot 759, which ran the 755 weights plus first guesses for
    # the shots (docs/internal/bc-pilot-759-oct2026.json). Its median shares
    # over 10.3-19.7M were
    #     bc 0.43, next_state 0.20, shot_outcome 0.14, density 0.11,
    #     enemy_action 0.07, shot_trajectory 0.04, value 0.01
    # against targets bc 0.40, next_state 0.25, enemy_action 0.12, density
    # 0.09, shot_trajectory 0.07, shot_outcome 0.05. The cloning and value
    # weights are held and the others solved relative to cloning. The outcome
    # gets the smaller shot share because its gradient is mostly sampling
    # noise: its two half-batches agreed at a half-cosine of 0.04-0.49 and a
    # signal fraction of -0.27-0.72, where next-state's was 0.91-0.96 -- whether
    # one shot hits is close to a coin flip per minibatch.
    #
    # Value stays near zero on the trunk, by measurement rather than intent. The
    # check run 756 (docs/internal/bc-check-756-oct2026.json) solved value up to
    # a 0.30 share, and at 19.7M steps against 755: live Elo 530 vs 913, BC KL
    # 0.88 vs 0.49, hidden-enemy next-state cross-entropy 2.38 vs 1.90, for
    # explained variance that moved both ways (zone capture 0.92 vs 0.84,
    # kill/death 0.61 vs 0.85). About 85% of that value gradient was the damage
    # level's: the noisiest returns, pushed into features cloning and
    # next-state use. The critic's heads learn their own levels well without
    # the trunk (0.80-0.93 explained variance per ship level on 755), and RL is
    # where the critic is trained in earnest.
    #
    # The outcome level draws next to nothing of the trunk in every run (~1e-3
    # of the damage level's) and its explained variance sits at 0.35-0.46.
    # A mirrored self-play match may simply be unpredictable until late; the
    # per-level weight in COMPONENT_VALUE_WEIGHTS is the knob if RL shows
    # otherwise.
    #
    # Shares drift by design: BC's teacher targets are soft, so its gradient
    # falls toward zero as imitation converges, while value and next-state
    # keep a noise floor. Measure in one window, after the 6M warmup and once
    # next-state's zero-initialised residual has stopped growing its share.
    next_state_coef=0.0041,
    enemy_action_coef=0.11,
    global_density_coef=0.021,
    shot_trajectory_coef=0.0035,
    shot_outcome_coef=0.019,
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
