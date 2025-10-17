from typing import Any
import numpy as np
import torch

from .agents import Agent


class Squad:
    """Represents a squad of ships controlled by an agent"""

    def __init__(self, squad_id: str, agent: Agent, ship_ids: list[int], team_id: int):
        """
        Initialize a squad

        Args:
            squad_id: Unique identifier for the squad
            agent: Agent controlling this squad
            ship_ids: List of ship IDs in this squad
            team_id: Team ID this squad belongs to
        """
        self.squad_id = squad_id
        self.agent = agent
        self.ship_ids = ship_ids
        self.team_id = team_id
        self.active = True  # Whether this squad is still active in the episode

    def get_ship_count(self) -> int:
        """Get the number of ships in this squad"""
        return len(self.ship_ids)

    def get_active_ships(self, obs: dict[str, np.ndarray]) -> list[int]:
        """Get list of active (alive) ships in this squad"""
        if "alive" not in obs:
            return self.ship_ids

        active_ships = []
        for ship_id in self.ship_ids:
            if ship_id < obs["alive"].shape[0] and obs["alive"][ship_id, 0] > 0:
                active_ships.append(ship_id)

        return active_ships


class Team:
    """Represents a team containing one or more squads"""

    def __init__(self, team_id: int, name: str | None = None):
        """
        Initialize a team

        Args:
            team_id: Unique identifier for the team
            name: Optional display name for the team
        """
        self.team_id = team_id
        self.name = name or f"Team {team_id}"
        self.squads: dict[str, Squad] = {}

    def add_squad(self, squad: Squad) -> None:
        """Add a squad to this team"""
        self.squads[squad.squad_id] = squad

    def get_active_ships(self, obs: dict[str, np.ndarray]) -> list[int]:
        """Get all active ships from all squads in this team"""
        active_ships = []
        for squad in self.squads.values():
            if squad.active:
                active_ships.extend(squad.get_active_ships(obs))
        return active_ships

    def is_active(self, obs: dict[str, np.ndarray]) -> bool:
        """Check if this team has any active ships"""
        return len(self.get_active_ships(obs)) > 0


class AgentManager:
    """Manages agents, squads, and teams for game coordination"""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the agent manager

        Args:
            config: Configuration dictionary containing agent and team setup
        """
        self.config = config
        self.teams: dict[int, Team] = {}
        self.squads: dict[str, Squad] = {}
        self.agents: dict[str, Agent] = {}
        self.ship_to_squad: dict[int, str] = {}  # Maps ship ID to squad ID
        self.ship_to_team: dict[int, int] = {}  # Maps ship ID to team ID

        # Initialize teams and squads from config
        self._initialize_from_config()

    def _initialize_from_config(self) -> None:
        """Initialize teams, squads, and agents from configuration"""
        teams_config = self.config.get("teams", {})

        for team_id, team_config in teams_config.items():
            # Convert string team_id to int if needed
            team_id = int(team_id)
            team_name = team_config.get("name", f"Team {team_id}")

            # Create team
            team = Team(team_id, team_name)
            self.teams[team_id] = team

            # Create squads for this team
            squads_config = team_config.get("squads", {})
            for squad_id, squad_config in squads_config.items():
                self._create_squad(team_id, squad_id, squad_config)

    def _create_squad(
        self, team_id: int, squad_id: str, squad_config: dict[str, Any]
    ) -> None:
        """
        Create a squad from configuration

        Args:
            team_id: ID of the team this squad belongs to
            squad_id: ID for the squad
            squad_config: Configuration for the squad
        """
        # Get agent configuration
        agent_config = squad_config.get("agent", {})
        agent_type = agent_config.get("type", "scripted")

        # Get ship configuration
        ship_ids = squad_config.get("ship_ids", [])
        if not ship_ids:
            raise ValueError(f"Squad {squad_id} has no ships configured")

        # Create agent with proper parameters
        agent = AgentFactory.create_agent_with_params(
            agent_type=agent_type,
            agent_id=squad_id,
            team_id=team_id,
            squad=ship_ids,
            config=agent_config,
        )
        self.agents[squad_id] = agent

        # Create squad
        squad = Squad(squad_id, agent, ship_ids, team_id)
        self.squads[squad_id] = squad

        # Add to team
        self.teams[team_id].add_squad(squad)

        # Update mappings
        for ship_id in ship_ids:
            self.ship_to_squad[ship_id] = squad_id
            self.ship_to_team[ship_id] = team_id

    def get_agent_for_ship(self, ship_id: int) -> Agent | None:
        """Get the agent that controls a specific ship"""
        squad_id = self.ship_to_squad.get(ship_id)
        if squad_id:
            return self.agents.get(squad_id)
        return None

    def get_squad_for_ship(self, ship_id: int) -> Squad | None:
        """Get the squad that contains a specific ship"""
        squad_id = self.ship_to_squad.get(ship_id)
        if squad_id:
            return self.squads.get(squad_id)
        return None

    def get_team_for_ship(self, ship_id: int) -> Team | None:
        """Get the team that contains a specific ship"""
        team_id = self.ship_to_team.get(ship_id)
        if team_id:
            return self.teams.get(team_id)
        return None

    def get_active_agents(self, obs: dict[str, np.ndarray]) -> dict[str, Agent]:
        """Get all agents with active ships"""
        active_agents = {}
        for squad_id, squad in self.squads.items():
            if squad.active and len(squad.get_active_ships(obs)) > 0:
                active_agents[squad_id] = squad.agent
        return active_agents

    def get_actions(
        self, obs: dict[str, np.ndarray]
    ) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
        """
        Get actions from all active agents

        Args:
            obs: Current observation

        Returns:
            Tuple of (actions_by_ship, agent_info)
        """
        actions_by_ship: dict[int, np.ndarray] = {}
        agent_info: dict[str, Any] = {}

        for squad_id, squad in self.squads.items():
            if not squad.active:
                continue

            # Get active ships for this squad
            active_ships = squad.get_active_ships(obs)
            if not active_ships:
                continue

            # Get observation for this squad's ships
            squad_obs = self._extract_squad_obs(obs, active_ships)

            # Get actions from agent
            actions = squad.agent.get_actions(squad_obs)

            # Store agent info
            agent_info[squad_id] = {}

            # Map actions back to individual ships
            squad_actions = self._map_actions_to_ships(actions, active_ships)
            actions_by_ship.update(squad_actions)

        return actions_by_ship, agent_info

    def _extract_squad_obs(
        self, obs: dict[str, np.ndarray], ship_ids: list[int]
    ) -> dict[str, np.ndarray]:
        """Extract observation for a specific set of ships"""
        squad_obs: dict[str, np.ndarray] = {}

        for key, value in obs.items():
            if isinstance(value, np.ndarray) and len(value.shape) >= 1:
                # Extract rows for the specified ships
                squad_obs[key] = value[ship_ids]
            else:
                # Non-array observations are shared
                squad_obs[key] = value

        return squad_obs

    def _map_actions_to_ships(
        self, actions: dict[int, torch.Tensor], ship_ids: list[int]
    ) -> dict[int, torch.Tensor]:
        """Map agent actions to individual ships"""
        ship_actions: dict[int, torch.Tensor] = {}

        for ship_id in ship_ids:
            if ship_id in actions:
                ship_actions[ship_id] = actions[ship_id]
            else:
                # Default to no action if ship not in actions
                ship_actions[ship_id] = torch.zeros(6, dtype=torch.float32)

        return ship_actions

    def reset(self) -> None:
        """Reset all agents for a new episode"""
        for agent in self.agents.values():
            agent.reset()

        # Reset squad active status
        for squad in self.squads.values():
            squad.active = True

    def get_team_status(self, obs: dict[str, np.ndarray]) -> dict[int, dict[str, Any]]:
        """Get status information for all teams"""
        team_status: dict[int, dict[str, Any]] = {}

        for team_id, team in self.teams.items():
            active_ships = team.get_active_ships(obs)
            team_status[team_id] = {
                "name": team.name,
                "active": team.is_active(obs),
                "active_ships": len(active_ships),
                "total_ships": sum(
                    squad.get_ship_count() for squad in team.squads.values()
                ),
                "squads": list(team.squads.keys()),
            }

        return team_status

    def check_victory(self, obs: dict[str, np.ndarray]) -> tuple[bool, int | None]:
        """
        Check if there's a victory condition

        Returns:
            Tuple of (game_over, winning_team_id)
        """
        active_teams = [
            team_id for team_id, team in self.teams.items() if team.is_active(obs)
        ]

        if len(active_teams) <= 1:
            # Game over, either one team won or it's a draw
            winner = active_teams[0] if active_teams else None
            return True, winner

        return False, None
