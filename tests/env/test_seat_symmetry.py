"""Neither side may be able to tell which seat it is in.

Team 0 and Team 1 play the same game. A model handed the canonical team-0 view
must not be able to recover which physical side produced it, and both sides must
receive the same information about their own ships and about the enemy.

The regression these pin down: the rollout passed the *observation's* ``TEAM_ID``
channel to the pending-action privacy mask. Hidden ships are zeroed there and
zero is also Team 0's real id, so once every enemy was out of sight the channel
read as all zeros -- Team 0 then privatised nothing and saw every enemy's pending
command, while Team 1 privatised everything and could not see its own. Only
``state.ship_team_id`` is a correct source for that mask.

``spawn_reveal`` is on in every one of these, matching the training profiles.
With it off, a ship can be permanently unseen by one side, the belief tracker
never promotes it, and its channels stay at their masked zeros -- which produces
apparent asymmetries that are a property of the test config, not the code.
"""

from __future__ import annotations

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import EnvConfig, ShipConfig
from boost_and_broadside.config.defaults import REWARDS
from boost_and_broadside.constants import (
    PRIVATE_POWER_ACTION,
    PRIVATE_SHOOT_ACTION,
    PRIVATE_TURN_ACTION,
)
from boost_and_broadside.env.observation import ObsKey
from boost_and_broadside.env.wrapper import YemongEnvWrapper
from boost_and_broadside.runtime.actions import PendingActionState
from boost_and_broadside.train.rl.belief import DualBeliefTracker
from boost_and_broadside.train.rl.features import build_standard_coordinator
from boost_and_broadside.train.rl.opponents import flip_team_obs

NUM_SHIPS = 6
NUM_ENVS = 4
# Far enough in that ships have scattered and most enemies are out of sight. At
# reset every ship is revealed, which hides exactly this class of bug.
WARMUP_STEPS = 12


def _env_config() -> EnvConfig:
    return EnvConfig(
        num_ships=NUM_SHIPS,
        max_bullets=4,
        max_episode_steps=400,
        vision_range=300.0,
        spawn_reveal=True,
    )


def _wrapper() -> tuple[YemongEnvWrapper, ShipConfig, EnvConfig]:
    ship_config = ShipConfig()
    env_config = _env_config()
    wrapper = YemongEnvWrapper(
        num_envs=NUM_ENVS,
        ship_config=ship_config,
        env_config=env_config,
        rewards=REWARDS,
        device=torch.device("cpu"),
    )
    return wrapper, ship_config, env_config


def _private_triple() -> torch.Tensor:
    return torch.tensor([PRIVATE_POWER_ACTION, PRIVATE_TURN_ACTION, PRIVATE_SHOOT_ACTION])


def _run_to_midgame(wrapper: YemongEnvWrapper, ship_config: ShipConfig):
    """Step with the scripted controller until enemies are genuinely hidden."""
    scripted = StochasticScriptedAgent(ship_config, StochasticAgentConfig())
    action_state = PendingActionState.allocate(NUM_ENVS, NUM_SHIPS, torch.device("cpu"))
    obs = wrapper.reset(seed=17)
    for _ in range(WARMUP_STEPS):
        selected = scripted.get_actions(wrapper.env.state, wrapper.last_visibility.ship).int()
        obs, _, dones, truncated, info = wrapper.step(action_state.applied_action())
        action_state.commit(selected, info["actuator_contiguous"], dones | truncated)
        action_state.write_observation(obs, wrapper.env.state.ship_team_id, NUM_SHIPS)
    return obs


def test_observation_team_id_is_not_authoritative() -> None:
    """Characterisation: the raw observation's TEAM_ID is masked, the state's is not.

    Pinned rather than assumed, because this is the trap the bug fell into. Any
    consumer that needs ground truth must read ``state.ship_team_id``.
    """
    wrapper, ship_config, _ = _wrapper()
    obs = _run_to_midgame(wrapper, ship_config)

    true_team = wrapper.env.state.ship_team_id
    observed_team = obs["team_id"][:, :NUM_SHIPS].long()
    hidden = ~wrapper.last_visibility.ship[:, 0].bool()

    assert hidden.any(), "warmup hid nothing; the test cannot bind"
    # Every ship hidden from Team 0 reads as team 0 there, whatever side it is on.
    assert torch.all(observed_team[hidden] == 0)
    assert not torch.equal(observed_team, true_team.long())


def test_pending_action_privacy_hides_enemies_and_spares_own_ships() -> None:
    """Own ships keep their pending command; enemies are private. Both views."""
    wrapper, ship_config, _ = _wrapper()
    obs = _run_to_midgame(wrapper, ship_config)

    true_team = wrapper.env.state.ship_team_id
    private = _private_triple()

    for observer, view in ((0, obs.data), (1, obs.team1_data)):
        pending = view[ObsKey.PREVIOUS_ACTION][:, :NUM_SHIPS]
        is_private = (pending == private).all(-1)
        own = true_team == observer
        enemy = true_team != observer
        assert own.any() and enemy.any()
        assert not is_private[own].any(), f"observer {observer} cannot see its own commands"
        assert is_private[enemy].all(), f"observer {observer} can read enemy commands"


def test_pending_action_privacy_survives_the_team_flip() -> None:
    """After flip_team the canonical team-1 view must read like a team-0 view."""
    wrapper, ship_config, _ = _wrapper()
    obs = _run_to_midgame(wrapper, ship_config)

    flipped = flip_team_obs(obs.for_team(1), NUM_SHIPS)
    pending = flipped.data[ObsKey.PREVIOUS_ACTION][:, :NUM_SHIPS]
    is_private = (pending == _private_triple()).all(-1)

    true_team = wrapper.env.state.ship_team_id
    own = true_team == 1  # team 1's own ships, relabelled to 0 by the flip
    enemy = true_team == 0

    assert not is_private[own].any(), "team 1 cannot see its own commands after the flip"
    assert is_private[enemy].all(), "team 1 can read enemy commands after the flip"


def _composed_views(wrapper, ship_config, env_config, steps: int):
    """Drive both sides with one controller and return the composed seat views."""
    scripted = StochasticScriptedAgent(ship_config, StochasticAgentConfig())
    coordinator = build_standard_coordinator(ship_config)
    beliefs = DualBeliefTracker(
        NUM_ENVS,
        NUM_SHIPS,
        ship_config.dt * env_config.action_repeat,
        coordinator,
        torch.device("cpu"),
    )
    action_state = PendingActionState.allocate(NUM_ENVS, NUM_SHIPS, torch.device("cpu"))
    raw = wrapper.reset(seed=23)
    for _ in range(steps):
        beliefs.compose(raw)
        selected = scripted.get_actions(wrapper.env.state, wrapper.last_visibility.ship).int()
        raw, _, dones, truncated, info = wrapper.step(action_state.applied_action())
        action_state.commit(selected, info["actuator_contiguous"], dones | truncated)
        action_state.write_observation(raw, wrapper.env.state.ship_team_id, NUM_SHIPS)
        beliefs.reset(dones | truncated)
    composed = beliefs.compose(raw)
    return composed.for_team(0), flip_team_obs(composed.for_team(1), NUM_SHIPS)


def test_composed_seat_labels_are_exactly_self_relative() -> None:
    """Each seat's composed view must label its own ships 0 and the enemy 1.

    Exact rather than distributional: a seat that mislabels even one slot has a
    different picture of the board than its opponent does.
    """
    wrapper, ship_config, env_config = _wrapper()
    view0, view1 = _composed_views(wrapper, ship_config, env_config, WARMUP_STEPS)
    true_team = wrapper.env.state.ship_team_id.long()

    assert torch.equal(view0.data[ObsKey.TEAM_ID][:, :NUM_SHIPS].long(), true_team)
    assert torch.equal(view1.data[ObsKey.TEAM_ID][:, :NUM_SHIPS].long(), 1 - true_team)


def test_composed_views_agree_on_ships_both_sides_can_see() -> None:
    """A ship visible to both seats must look identical from either one."""
    wrapper, ship_config, env_config = _wrapper()
    view0, view1 = _composed_views(wrapper, ship_config, env_config, WARMUP_STEPS)

    visibility = wrapper.last_visibility.ship
    both = visibility[:, 0].bool() & visibility[:, 1].bool()
    assert both.any(), "no ship was visible to both sides; the test cannot bind"

    for key in (ObsKey.HEALTH, ObsKey.POS, ObsKey.VEL, ObsKey.POWER, ObsKey.ALIVE):
        a = view0.data[key][:, :NUM_SHIPS]
        b = view1.data[key][:, :NUM_SHIPS]
        mask = both
        while mask.dim() < a.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(a)
        torch.testing.assert_close(a[mask], b[mask], msg=f"{key.name} differs between seats")
