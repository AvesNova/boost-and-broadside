"""Stepping a batch of environments with agents assigned per side.

Every evaluation path in the project — win-rate collection, the all-pairs Elo
sweep, the post-hoc calibration tournament, and video capture — was its own copy
of the same five steps: build the observation, take each agent's view of it,
merge the two sides' actions by team, step, and reset what finished. The copies
drifted. Some attached the bullet axis to the observation and some did not; some
gave a team-1 policy its own ego view and some left it playing as the enemy.

The loops themselves are genuinely different — they differ in how environments
map to agents and in what they count — so this owns the five steps and leaves the
loop, and the tally, to the caller.
"""

from dataclasses import replace

import torch

from boost_and_broadside.config import EnvConfig, FieldMapConfig, ShipConfig
from boost_and_broadside.env.observation import YemongObservation, observation_from_state
from boost_and_broadside.evaluation.agents import (
    ResolvedAgent,
    agents_read_bullets,
    get_actions,
    init_hidden,
    reset_done_envs,
)
from boost_and_broadside.evaluation.environment import create_evaluation_env


def merge_team_actions(
    team0_actions: torch.Tensor, team1_actions: torch.Tensor, team_id: torch.Tensor
) -> torch.Tensor:
    """Select each ship's action from the agent controlling its team."""
    return torch.where((team_id == 0).unsqueeze(-1), team0_actions, team1_actions)


def agent_is_ego_pass(agent: ResolvedAgent) -> bool:
    """Whether this agent's weights were trained from team 0's perspective only.

    Taken from the checkpoint's own provenance, because a field can mix runs.
    Checkpoints written before the paradigm was recorded default to ego_pass,
    which every run in this project's history has used.
    """
    bundle = getattr(agent, "bundle", None)
    return True if bundle is None else bundle.paradigm != "shared_pass"


def agent_view(
    agent: ResolvedAgent,
    obs: YemongObservation,
    num_ships: int,
    as_team1: torch.Tensor,
) -> YemongObservation:
    """Return the observation from ``agent``'s own side of the match.

    An ego_pass policy only ever learned to act as team 0, so wherever it plays
    team 1 it must see mirrored team IDs — ships and bullets alike. Agents that
    read the raw state rather than the observation (scripted, random) are
    unaffected, and a shared_pass policy was trained on both sides already.
    """
    if agent.kind != "policy" or not agent_is_ego_pass(agent):
        return obs
    return obs.flip_team(num_ships, mask=as_team1)


class MatchRunner:
    """Steps one batch of environments with agents assigned to each side.

    Args:
        env:         The TensorEnv to step.
        agents:      Every agent that can act in this batch.
        team0_index: (B,) index into ``agents`` — who controls team 0 in each env.
        team1_index: (B,) index into ``agents`` — likewise for team 1.
        ship_config: Physics constants used to build observations.
        num_ships:   N in this batch's environments.

    The two index tensors may name the same agent, which is how a policy plays
    itself. Policies act on a slice of the batch — the environments they appear
    in on either side — because each carries one recurrent state sized to exactly
    those. Scripted and random agents read the raw state and are cheapest run over
    the whole batch, so they are.
    """

    def __init__(
        self,
        env,
        agents: list[ResolvedAgent],
        team0_index: torch.Tensor,
        team1_index: torch.Tensor,
        ship_config: ShipConfig,
        num_ships: int,
    ) -> None:
        self.env = env
        self.agents = agents
        self.team0_index = team0_index
        self.team1_index = team1_index
        self.ship_config = ship_config
        self.num_ships = num_ships
        self.num_envs = int(team0_index.shape[0])
        self.device = team0_index.device
        self.active = [
            ((team0_index == index) | (team1_index == index)).nonzero(as_tuple=True)[0]
            for index in range(len(agents))
        ]
        # One observation serves every agent, so the bullet axis is attached
        # whenever *any* of them reads it: a policy that ignores bullets is
        # unaffected by their presence, while one that reads them and is handed an
        # observation without them plays blind with nothing to show for it.
        self.include_bullets = agents_read_bullets(*agents)
        self._arange = torch.arange(self.num_envs, device=self.device)

    @property
    def num_tokens(self) -> int:
        """Entity tokens per env. Read from the config, not the state, so hidden
        state can be allocated before the first reset."""
        return self.num_ships + self.env.env_config.num_fields

    def init_hidden(self) -> None:
        """Allocate each policy's recurrent state over the envs it plays in."""
        for agent, active in zip(self.agents, self.active):
            init_hidden(agent, int(active.numel()), self.num_tokens, self.device)

    def observe(self) -> YemongObservation:
        """Build the observation for the current state."""
        return observation_from_state(
            self.env.state, self.ship_config, include_bullets=self.include_bullets
        )

    def actions(self, obs: YemongObservation) -> torch.Tensor:
        """Every ship's action, taken from the agent that controls its team."""
        per_agent = torch.zeros(
            len(self.agents),
            self.num_envs,
            self.num_ships,
            3,
            dtype=torch.int32,
            device=self.device,
        )
        # A calibration field carries a rung of semi-random players built on one
        # shared scripted agent. Computing its actions once per step instead of
        # once per rung is the difference between one scripted pass and a dozen.
        scripted_cache: dict[int, torch.Tensor] = {}
        random_draw: torch.Tensor | None = None
        for index, agent in enumerate(self.agents):
            active = self.active[index]
            if agent.kind == "semi_random":
                key = id(agent.agent.scripted_agent)
                if key not in scripted_cache:
                    scripted_cache[key] = agent.agent.scripted_agent.get_actions(self.env.state)
                scripted = scripted_cache[key]
                if random_draw is None:
                    random_draw = agent.agent.random_actions_like(scripted)
                per_agent[index] = agent.agent.mix_actions(scripted, random_draw).int()
                continue
            if agent.kind != "policy":
                # Reads the state, not the observation: one vectorized call covers
                # the batch, and the merge discards the envs it does not control.
                per_agent[index] = get_actions(
                    agent, None, self.env.state, self.num_envs, self.num_ships, self.device
                ).int()
                continue
            if active.numel() == 0:
                continue
            view = agent_view(
                agent, obs.slice_envs(active), self.num_ships, self.team1_index[active] == index
            )
            per_agent[index, active] = get_actions(
                agent, view, self.env.state, int(active.numel()), self.num_ships, self.device
            ).int()
        return merge_team_actions(
            per_agent[self.team0_index, self._arange],
            per_agent[self.team1_index, self._arange],
            self.env.state.ship_team_id,
        )

    def step(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Observe, act, and advance one tick. Returns (dones, truncated)."""
        return self.env.step(self.actions(self.observe()))

    def reset_finished(self, done_any: torch.Tensor, options: dict | None = None) -> None:
        """Reset finished environments and the recurrent state riding on them."""
        if not bool(done_any.any()):
            return
        self.env.reset_envs(done_any, options=options)
        for agent, active in zip(self.agents, self.active):
            if agent.kind == "policy" and active.numel():
                reset_done_envs(agent, done_any[active], self.num_tokens)


def evaluate_matchup(
    agent0: ResolvedAgent,
    agent1: ResolvedAgent,
    n0: int,
    n1: int,
    num_envs: int,
    ship_config: ShipConfig,
    env_config: EnvConfig,
    device: str,
    *,
    field_map_config: FieldMapConfig | None = None,
) -> tuple[int, int, int, float]:
    """Run parallel games to completion and return wins, ties, and mean length."""
    num_ships = n0 + n1
    torch_device = torch.device(device)
    env = create_evaluation_env(
        num_envs,
        ship_config,
        replace(env_config, num_ships=num_ships),
        torch_device,
        field_map_config=field_map_config,
    )
    results = torch.zeros(num_envs, dtype=torch.int32, device=torch_device)
    episode_lengths = torch.zeros(num_envs, dtype=torch.int64, device=torch_device)
    finished = torch.zeros(num_envs, dtype=torch.bool, device=torch_device)

    runner = MatchRunner(
        env,
        [agent0, agent1],
        team0_index=torch.zeros(num_envs, dtype=torch.long, device=torch_device),
        team1_index=torch.ones(num_envs, dtype=torch.long, device=torch_device),
        ship_config=ship_config,
        num_ships=num_ships,
    )
    runner.init_hidden()
    reset_options = {"team_sizes": (n0, n1)}
    env.reset(options=reset_options)

    while not finished.all():
        dones, truncated = runner.step()
        done_any = dones | truncated
        newly_done = done_any & ~finished
        if newly_done.any():
            episode_lengths[newly_done] = env.state.step_count[newly_done].long()
            alive = env.state.ship_alive
            team = env.state.ship_team_id
            team0_alive = (alive & (team == 0)).any(dim=1)
            team1_alive = (alive & (team == 1)).any(dim=1)
            team0_won = newly_done & team0_alive & ~team1_alive
            team1_won = newly_done & team1_alive & ~team0_alive
            results[team0_won] = 0
            results[team1_won] = 1
            results[newly_done & ~team0_won & ~team1_won] = 2
            finished |= newly_done
        runner.reset_finished(done_any, options=reset_options)

    results_cpu = results.cpu()
    return (
        int((results_cpu == 0).sum()),
        int((results_cpu == 1).sum()),
        int((results_cpu == 2).sum()),
        float(episode_lengths.cpu().float().mean()),
    )
