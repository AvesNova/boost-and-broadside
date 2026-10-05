"""Which reward components may be estimated once per team rather than per ship.

``GLOBAL_VALUE_COMPONENTS`` is a claim about the *rewards*, not a modelling
preference: a component belongs there only if every ship on a side is paid the
same number, because only then is the return the same number and one estimate
enough. This checks the claim against real rollouts rather than trusting the
set, so a component added to it for convenience fails here.
"""

import pytest
import torch

from boost_and_broadside.config.core import EnvConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.rewards import (
    GLOBAL_VALUE_COMPONENTS,
    REWARD_COMPONENT_NAMES,
    build_reward_components,
)
from boost_and_broadside.profiles import PROFILES

NUM_ENVS = 24
STEPS = 700  # far enough in that the fleets reach the zones; see the zone test


@pytest.fixture(scope="module")
def rollout():
    """Per-component per-ship rewards over a scripted Frontline rollout."""
    from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
    from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
    from boost_and_broadside.env.perception import team_visibility_from_state

    profile = PROFILES["rl"]
    config = EnvConfig(
        num_ships=10,
        num_fields=4,
        max_bullets=profile.max_bullets,
        max_episode_steps=profile.max_episode_steps,
        frontline=profile.frontline,
        vision_range=profile.vision_range,
    )
    env = TensorEnv(NUM_ENVS, profile.ship_config, config, "cpu")
    env.reset(seed=3)
    agent = StochasticScriptedAgent(profile.ship_config, StochasticAgentConfig())
    components = build_reward_components(profile.rewards, profile.ship_config)
    by_name = {component.name: component for component in components}

    collected: dict[str, list[torch.Tensor]] = {name: [] for name in by_name}
    teams: list[torch.Tensor] = []
    for _ in range(STEPS):
        visibility = team_visibility_from_state(env.state, profile.ship_config, config, False)
        actions = agent.get_actions(env.state, visibility.ship)
        previous = env.state.clone() if hasattr(env.state, "clone") else env.state
        team_id = env.state.ship_team_id.clone()
        dones, truncated = env.tick(actions)
        for name, component in by_name.items():
            collected[name].append(
                component.compute(previous, actions, env.state, dones | truncated).clone()
            )
        teams.append(team_id)
    return (
        {name: torch.stack(values) for name, values in collected.items()},
        torch.stack(teams),
    )


def _is_uniform_per_team(reward: torch.Tensor, team_id: torch.Tensor) -> bool:
    """Whether every ship on a side is paid the same, at every step and env."""
    for team in (0, 1):
        mask = team_id == team  # (T, B, N)
        for step in range(reward.shape[0]):
            for env in range(reward.shape[1]):
                values = reward[step, env][mask[step, env]]
                if values.numel() and float(values.max() - values.min()) > 1e-6:
                    return False
    return True


def test_the_set_names_components_that_exist(rollout):
    rewards, _ = rollout
    assert GLOBAL_VALUE_COMPONENTS <= set(REWARD_COMPONENT_NAMES)
    assert GLOBAL_VALUE_COMPONENTS <= set(rewards)


@pytest.mark.parametrize("name", sorted(GLOBAL_VALUE_COMPONENTS))
def test_a_global_component_pays_every_teammate_the_same(rollout, name):
    """The condition for estimating it once off the global token."""
    rewards, teams = rollout
    assert _is_uniform_per_team(rewards[name], teams), (
        f"{name!r} is in GLOBAL_VALUE_COMPONENTS but does not pay every ship on "
        f"a side the same; a single team-level estimate would be wrong for it"
    )


def test_the_zone_rewards_are_not_uniform_and_so_stay_per_ship(rollout):
    """The split between the ships that showed up and the ones that did not
    *is* the zone levels' credit assignment. If this ever passes as
    uniform, either the reward changed or the rollout never contested a zone --
    both worth knowing before anything is moved."""
    rewards, teams = rollout
    contested = [
        name
        for name in ("zone_capture", "zone_progress")
        if name in rewards and float(rewards[name].abs().sum()) > 0.0
    ]
    assert contested, "no zone reward was paid in this rollout; the test proves nothing"
    for name in contested:
        assert not _is_uniform_per_team(rewards[name], teams)
        assert name not in GLOBAL_VALUE_COMPONENTS
