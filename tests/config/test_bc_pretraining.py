"""The behavior-cloning profile, as a pretraining artifact.

BC exists to produce a policy RL can start from, so the properties that matter
are the ones that decide how good that policy gets -- not how gracefully it
hands over, which is RL's problem.
"""

import pytest

from boost_and_broadside.config.resolve import resolve_profile
from boost_and_broadside.profiles.bc import BC_PROFILE
from boost_and_broadside.profiles.rl import RL_PROFILE


@pytest.fixture(scope="module")
def bc():
    return resolve_profile(BC_PROFILE)


@pytest.fixture(scope="module")
def rl():
    return resolve_profile(RL_PROFILE)


def test_cloning_never_decays_during_pretraining(bc, rl) -> None:
    """RL withdraws the warm start as the policy outgrows it. A pretraining run
    has nothing else training its actor, so withdrawing it stops the run early
    at roughly teacher parity -- the point where the clone becomes worth
    keeping."""
    assert bc.train_config.bc_winrate_target is None
    assert rl.train_config.bc_winrate_target == 0.45


def test_nothing_else_trains_the_actor_so_the_weight_has_to_hold(bc) -> None:
    """Why the above is load-bearing rather than a preference: with the policy
    gradient at zero, a decayed cloning weight leaves the actor with no gradient
    at all."""
    schedule = bc.train_config.schedule
    assert schedule.policy_gradient_coef(0) == 0.0
    assert schedule.behavior_cloning_coef(0) > 0.0


def test_the_ppo_stopping_criterion_is_off(bc) -> None:
    """Under supervision, movement away from the rollout policy is the
    objective, so a KL trust region would early-stop the thing being trained."""
    assert bc.train_config.schedule.target_kl(0) is None


def test_the_critic_and_next_state_heads_train_alongside(bc) -> None:
    """RL is meant to inherit more than an actor: a critic that already explains
    the returns is what makes early advantages mean anything."""
    assert bc.train_config.schedule.value_function_coef(0) > 0.0
    assert bc.train_config.next_state_coef > 0.0


def test_the_architecture_matches_rl_exactly(bc, rl) -> None:
    """``--pretrain-from`` loads weights by name into RL's model. Any divergence
    in width, entity counts, or head set is a failed load rather than a warning,
    so the overlay must not change them."""
    assert bc.model_config == rl.model_config
    assert bc.ship_config == rl.ship_config
    bc_env = bc.train_config.scales[0].env_config
    rl_env = rl.train_config.scales[0].env_config
    assert bc_env.num_ships == rl_env.num_ships
    assert bc_env.num_fields == rl_env.num_fields
    assert (bc_env.frontline is None) == (rl_env.frontline is None)
    # The categorical outcome head is built only when something trains it, so a
    # mismatch here would hand RL a checkpoint missing those weights entirely.
    assert (bc.train_config.outcome_categorical_coef > 0.0) == (
        rl.train_config.outcome_categorical_coef > 0.0
    )


def test_a_none_target_is_accepted_but_a_nonsense_one_is_not() -> None:
    """The sentinel has to stay distinguishable from a bad number."""
    import dataclasses

    config = resolve_profile(RL_PROFILE).train_config
    dataclasses.replace(config, bc_winrate_target=None)
    with pytest.raises(ValueError, match="bc_winrate_target"):
        dataclasses.replace(config, bc_winrate_target=0.0)
    with pytest.raises(ValueError, match="bc_winrate_target"):
        dataclasses.replace(config, bc_winrate_target=1.5)
