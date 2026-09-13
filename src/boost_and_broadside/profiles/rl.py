"""The reinforcement-learning profile.

Designed to follow an explicit BC pretraining checkpoint, but also runnable
from scratch.  The opponent mix stays half self-play and half an Elo-proximity
league draw; roster membership, rather than a schedule phase, provides the
curriculum.

There is one RL profile rather than a field-free one and a fielded one.
``num_fields`` is a sequence length, not an architecture: it sets the token
count ``N + M`` and no weight shape depends on it, so a "model that does not
support fields" was never a thing this registry had to represent.  Zero fields
remains a reachable *configuration* -- it is what run 682 trained under and how
that run is still evaluated -- but it is a value to set, not a profile to pick.
"""

from boost_and_broadside.config.core import FrontlineConfig
from boost_and_broadside.config.defaults import (
    COMPONENT_GAMMAS_PER_TICK,
    COMPONENT_LAMBDAS_PER_TICK,
    ELO_EVAL,
    LIVE_REFERENCE_PROBABILITIES,
    MODEL_CONFIG,
    REWARDS,
    SHIP_CONFIG,
    make_rl_schedule_spec,
)
from boost_and_broadside.config.schema import LaunchSizingSpec, ProfileSpec
from boost_and_broadside.constants import DEFAULT_MAX_BULLETS_PER_SHIP
from boost_and_broadside.env.frontline import frontline_ship_config

RL_PROFILE = ProfileSpec(
    name="rl",
    ship_config=frontline_ship_config(SHIP_CONFIG),
    model_config=MODEL_CONFIG,
    # --- Environment ---
    num_ships=8,
    num_fields=10,
    max_bullets=DEFAULT_MAX_BULLETS_PER_SHIP,
    max_episode_steps=9_000,
    # Frontline physics and decisions both run at 30 Hz. The 128-step rollout
    # spans 4.3 seconds without repeating field/visibility work inside a decision.
    action_repeat=1,
    spawn_resource_spread=0.0,
    vision_range=1024.0,
    zones_occlude=False,
    frontline=FrontlineConfig(
        zone_radius=330.0,
        zone_ring_radius=1200.0,
        playable_radius=2600.0,
        capture_seconds=8.0,
        # Off. Standing on a point cost 2 health/s while paying nothing until the
        # meter completed, which made contesting a zone strictly dominated early
        # and compounded the shaping bias run 735 exploited. The capture tier now
        # supplies the pressure that this was standing in for.
        defense_damage_per_second=0.0,
        respawn_health=25.0,
        spawn_heal_per_second=12.0,
        enemy_spawn_damage_per_second=8.0,
        boundary_damage_per_second=5.0,
        boundary_damage_per_pixel_second=0.05,
        front_win_threshold=3,
    ),
    # --- Rollout shape ---
    # 12M, after run 733 tested 24M and measured worse.
    #
    # The idea was to buy back the batch the Frontline observation spent: zones
    # plus ten fields took the per-decision token count from run 731's 12 (8
    # ships + 4 fields) to 24, so a fixed budget bought half the decisions. The
    # error was in believing that could be fixed by spending more tokens. At a
    # fixed epoch count the two quantities are reciprocal in the budget --
    #
    #     batch per step   = logical_batch / num_minibatches
    #     steps per sample = num_minibatches * epochs / logical_batch
    #
    # -- so doubling it bought a 2x batch per optimizer step by giving up half
    # the optimizer steps per sample, and the critic is what paid. Measured at
    # 13M steps: 1,856 optimizer steps and 0.538 explained variance at 12M,
    # against 992 and 0.352 at 24M. Live Elo tracked slightly below too.
    #
    # Nor was it a large-batch run that would repay the slow start later. Run 731
    # has almost exactly the 24M optimizer geometry -- ~1M decisions per update,
    # the same steps per sample -- and reached 0.45 explained variance by 13M
    # where 733 was at 0.35. 733 was below both references, not on a different
    # trajectory through them.
    #
    # What 731 actually had was twice the decisions inside the same token budget,
    # because a decision cost it half as many tokens. That axis is real and still
    # open: routing map objects through K/V memory instead of the trunk, and
    # sizing the batch on trunk tokens rather than observation tokens, raises
    # decisions per update without touching the reciprocal above. Spending more
    # tokens cannot substitute for it.
    logical_batch_tokens=12_000_000,
    num_steps=128,
    num_minibatches=32,
    # --- Objective ---
    paradigm="ego_pass",
    schedule_spec=make_rl_schedule_spec(),
    rewards=REWARDS,
    next_state_coef=0.2,
    windowed_loss_coef=0.1,
    # --- Discounts, per physics tick ---
    gamma_per_tick=0.99,
    gae_lambda_per_tick=0.95,
    component_gammas_per_tick=COMPONENT_GAMMAS_PER_TICK,
    component_lambdas_per_tick=COMPONENT_LAMBDAS_PER_TICK,
    # --- Optimizer, scalers, budget ---
    clip_coef=0.15,
    max_grad_norm=1.0,
    # 500M rather than 1B, which is where the schedules finish rather than an
    # arbitrary truncation: the learning rate reaches its 1.5e-4 floor at exactly
    # 500M and holds, and shaping_scale finishes its taper at 400M. Past that the
    # run is pure incremental self-play at fixed coefficients, and run 719's
    # calibrated curve prices it accordingly -- 500M captures 94.4% of the 1B
    # result, and the last 500M bought +37 Elo for 51 hours, against 0.06 h/Elo
    # over the first 200M. Run 731 reproduced the same saturation profile under
    # different physics and different reward weights.
    total_timesteps=500_000_000,
    return_ema_alpha=0.005,
    # A divide-by-zero guard, and nothing more.  At the previous 1.0 it bound 8
    # of 12 components on every update of run 719 -- including the win pair --
    # compressing their critic targets by up to 121x and their critic gradients
    # by four orders of magnitude.  The outlier problem that motivated the large
    # floor is now handled where it belongs, by ``value_huber_delta``.
    #
    # 1e-3 rather than 1e-2 because the guard has to sit far below every live
    # component's spread, not just below it.  Measured on run 719's logged return
    # histograms, the narrowest component (field_death) has a 4-sigma span of
    # 0.0127: twelve times this floor, but only 1.3x a floor of 1e-2.
    return_min_span=1e-3,
    advantage_min_rms=1e-4,
    # Squared error inside one normalized unit, linear outside.  Per-component
    # normalization necessarily exposes heavy tails -- a sparse component is a
    # spike at zero with rare large excursions -- and bounding their gradient
    # here keeps one component's tail from setting the whole critic's step.
    value_huber_delta=1.0,
    # --- League and live evaluation ---
    league_size=20,
    league_slots=4,
    # Rung ratings are derived, not stated: the live gauge pins random at 0 and
    # scripted at 1000 and places a rung at 1000·p.  It is defined rather than
    # fitted, so it says nothing about how hard any particular field count is,
    # and live ratings do not compare across environments.
    live_reference_probabilities=LIVE_REFERENCE_PROBABILITIES,
    elo_milestone_gap=200.0,
    elo_temperature=200.0,
    league_uniform_sampling=False,
    elo_eval=ELO_EVAL,
    bc_winrate_target=0.45,
    # --- Persistence and logging ---
    checkpoint_dir="checkpoints",
    histogram_interval=10,
    log_interval=10,
    # --- Machine sizing ---
    # Two micro-batches per shard minibatch, not five. The divisor was set when
    # the PPO update ran eager, where finer chunking was measurably faster on a
    # VRAM-constrained card. Compiling `evaluate_actions` inverted that: a
    # compiled pass carries far less per-call overhead, so fewer and larger
    # passes win. Measured on an RTX 4070 Laptop, update phase 26.38 -> 19.19 s
    # per epoch and +9.4% end-to-end throughput, for 3897 MiB allocated against
    # 2225 -- which still leaves about 2.4 GB of the card free.
    #
    # It must also divide the minibatch evenly. An uneven split gives dynamo two
    # shapes, which sends it dynamic, and inductor then fails outright in the
    # RG-LRU scan's power-of-two padding. `compile_policy` pins `dynamic=False`
    # so that is a bounded recompile rather than a crash, but an even divisor
    # avoids the second graph entirely.
    launch=LaunchSizingSpec(rollout_tokens=4_000_000, microbatches_per_minibatch=2),
)
