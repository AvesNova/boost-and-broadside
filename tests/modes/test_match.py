"""The shared match loop: perspective, bullet axis, and side assignment.

These three used to be decided independently in every evaluation loop, and the
loops disagreed. What a policy is shown is not something the caller can see in
its output — a policy handed the wrong side's view, or an observation without the
bullets it trained on, plays worse and says nothing — so each property is pinned
by watching what actually reaches the policy.
"""

import torch

from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.observation import BulletObsKey, ObsKey
from boost_and_broadside.evaluation.agents import ResolvedAgent
from boost_and_broadside.evaluation.match import MatchRunner, agent_view, merge_team_actions
from boost_and_broadside.train.rl.policy_io import PolicyBundle, build_policy

SHIP_CONFIG = ShipConfig()
ENV_CONFIG = EnvConfig(num_ships=4, max_bullets=4, max_episode_steps=32)


class _Recorder:
    """Delegates to a policy while keeping every observation it was handed."""

    def __init__(self, policy):
        self.__dict__["_policy"] = policy
        self.__dict__["seen"] = []

    def get_action_and_value(self, obs, hidden):
        self.seen.append(obs)
        return self._policy.get_action_and_value(obs, hidden)

    def __getattr__(self, name):
        return getattr(self.__dict__["_policy"], name)


def _policy_agent(reads_bullets: bool = True, paradigm: str = "ego_pass", record: bool = False):
    model_config = ModelConfig(
        d_model=32,
        n_heads=4,
        n_yemong_blocks=1,
        n_bullet_cross_per_block=1 if reads_bullets else 0,
    )
    policy = build_policy(
        model_config,
        SHIP_CONFIG,
        num_value_components=3,
        num_ships=ENV_CONFIG.num_ships,
        team_pma_k=(),
    ).eval()
    bundle = PolicyBundle(
        policy=policy,
        model_config=model_config,
        ship_config=SHIP_CONFIG,
        env_config=ENV_CONFIG,
        field_map_config=None,
        num_value_components=3,
        team_pma_k=(),
        paradigm=paradigm,
    )
    return ResolvedAgent("policy", _Recorder(policy) if record else policy, bundle=bundle)


def _runner(agents, team0, team1, num_envs=2):
    env = TensorEnv(num_envs, SHIP_CONFIG, ENV_CONFIG, torch.device("cpu"))
    runner = MatchRunner(
        env,
        agents,
        team0_index=torch.tensor(team0),
        team1_index=torch.tensor(team1),
        ship_config=SHIP_CONFIG,
        num_ships=ENV_CONFIG.num_ships,
    )
    runner.init_hidden()
    env.reset()
    return runner


class TestPerspective:
    def test_an_ego_pass_policy_playing_team_one_is_shown_the_mirror(self):
        agent = _policy_agent()
        runner = _runner([agent, ResolvedAgent("random", None)], team0=[1, 0], team1=[0, 1])
        obs = runner.observe()

        view = agent_view(agent, obs, ENV_CONFIG.num_ships, runner.team1_index == 0)

        N = ENV_CONFIG.num_ships
        # Env 0: the policy plays team 1 and sees mirrored ships and bullets.
        assert torch.equal(view[ObsKey.TEAM_ID][0, :N], 1 - obs[ObsKey.TEAM_ID][0, :N])
        assert torch.equal(
            view.bullets[BulletObsKey.TEAM_ID][0], 1 - obs.bullets[BulletObsKey.TEAM_ID][0]
        )
        # Env 1: it plays team 0 and sees the world as it is.
        assert torch.equal(view[ObsKey.TEAM_ID][1, :N], obs[ObsKey.TEAM_ID][1, :N])

    def test_a_shared_pass_policy_is_never_mirrored(self):
        """It trained on both sides, so mirroring would hand it the wrong one."""
        agent = _policy_agent(paradigm="shared_pass")
        runner = _runner([agent], team0=[0, 0], team1=[0, 0])
        obs = runner.observe()

        assert agent_view(agent, obs, ENV_CONFIG.num_ships, runner.team1_index == 0) is obs

    def test_scripted_agents_are_never_mirrored(self):
        """They read the raw state, where teams are unambiguous."""
        agent = ResolvedAgent("scripted", None)
        obs = _runner([agent], team0=[0, 0], team1=[0, 0]).observe()

        assert agent_view(agent, obs, ENV_CONFIG.num_ships, torch.tensor([True, True])) is obs


class TestBulletAxis:
    def test_the_observation_carries_bullets_when_any_agent_reads_them(self):
        reader = _policy_agent(reads_bullets=True)
        blind = _policy_agent(reads_bullets=False)

        assert _runner([reader, blind], [0, 0], [1, 1]).include_bullets
        assert not _runner([blind, blind], [0, 0], [1, 1]).include_bullets

    def test_a_bullet_reading_policy_is_handed_the_bullet_axis(self):
        agent = _policy_agent(reads_bullets=True, record=True)
        runner = _runner([agent, ResolvedAgent("random", None)], team0=[0, 0], team1=[1, 1])

        runner.step()

        assert agent.agent.seen and all(obs.bullets is not None for obs in agent.agent.seen)


class TestSideAssignment:
    def test_each_ship_acts_on_its_own_teams_agent(self):
        team_id = torch.tensor([[0, 0, 1, 1]])
        team0 = torch.full((1, 4, 3), 7)
        team1 = torch.full((1, 4, 3), 9)

        merged = merge_team_actions(team0, team1, team_id)

        assert merged[0, :, 0].tolist() == [7, 7, 9, 9]

    def test_a_policy_only_acts_in_the_environments_it_was_assigned(self):
        agent = _policy_agent(record=True)
        # Present in env 0 only; env 1 is random against random.
        runner = _runner(
            [agent, ResolvedAgent("random", None)], team0=[0, 1], team1=[1, 1], num_envs=2
        )

        runner.step()

        assert [int(obs[ObsKey.TEAM_ID].shape[0]) for obs in agent.agent.seen] == [1]
