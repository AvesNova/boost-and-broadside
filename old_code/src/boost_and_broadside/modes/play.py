from omegaconf import DictConfig, OmegaConf

from boost_and_broadside.game_coordinator import GameCoordinator


def play(cfg: DictConfig) -> None:
    """
    Run the game in play mode with rendering.

    This mode allows for human interaction (if enabled) and visualization
    of agent behaviors. It sets up the teams based on configuration
    and runs a single episode.

    Args:
        cfg: Configuration dictionary containing:
            - team1: Type of agent for team 1.
            - team2: Type of agent for team 2.
            - human_player: Boolean to enable human control.
            - agents: Agent configurations.
            - train.model.transformer: Transformer settings (fallback).
    """
    print("Starting play mode...")

    # Allow config modification to dynamically setup teams
    OmegaConf.set_struct(cfg, False)

    # Get team configurations - Fail fast if missing
    team1_type = cfg.team1
    team2_type = cfg.team2
    human_player = cfg.human_player

    print(f"Team 1: {team1_type}")
    print(f"Team 2: {team2_type}")
    print(f"Human Player: {human_player}")

    # Helper to create agent config
    def get_agent_config(agent_type: str) -> dict:
        # Use current global config as default for transformer-based agents
        transformer_config = OmegaConf.to_container(cfg.model, resolve=True)
        if isinstance(transformer_config, dict) and "num_actions" in transformer_config:
            del transformer_config["num_actions"]

        base_config = {
            "agent_id": "player",
            "team_id": 0,
            "squad": [],
            **transformer_config,  # type: ignore
        }
        return {"agent_type": agent_type, "agent_config": base_config}

    # Setup Team 1
    if team1_type in cfg.agents:
        cfg.agents.team1_agent = cfg.agents[team1_type]
    else:
        cfg.agents.team1_agent = get_agent_config(team1_type)

    # Setup Team 2
    if team2_type in cfg.agents:
        cfg.agents.team2_agent = cfg.agents[team2_type]
    else:
        cfg.agents.team2_agent = get_agent_config(team2_type)

    # Set teams for the game
    cfg.collect.teams = ["team1_agent", "team2_agent"]

    # 2048 steps
    cfg.collect.max_episode_length = 2048

    # Create game coordinator
    coordinator = GameCoordinator(cfg)

    # Add human player if requested
    if human_player:
        # Assuming ship 0 is always the first ship of team 0
        coordinator.env.add_human_player(0)
        print("Human control enabled for Ship 0")

    import time
    import pygame  # Added import for pygame.time.wait

    blue_wins = 0
    red_wins = 0
    ties = 0

    try:
        while True:
            # Run a single episode
            coordinator.reset(game_mode="nvn")
            time, terminated, truncated, steps, winners, win_reason = coordinator.step()

            # Use final observation right before the auto-reset
            final_obs = getattr(coordinator, "final_obs", coordinator.obs_history[-1])
            print(f"Final Obs Alive Mask: {final_obs['alive']}")

            alive_teams = coordinator._get_teams_from_obs(final_obs)

            if len(winners) == 1 and winners[0] == 0:
                blue_wins += 1
                outcome = f"Blue Wins! ({win_reason})"
            elif len(winners) == 1 and winners[0] == 1:
                red_wins += 1
                outcome = f"Red Wins! ({win_reason})"
            else:
                ties += 1
                outcome = f"TIE ({win_reason})"

            print(f"\nEpisode Finished in {steps} steps ({time:.1f}s): {outcome}")
            print(
                f"Alive indices: Blue: {alive_teams.get(0, [])}, Red: {alive_teams.get(1, [])}"
            )
            print(f"Standings -> Blue: {blue_wins} | Red: {red_wins} | Ties: {ties}\n")

            print("Restarting next episode in 1 second...")
            pygame.time.wait(1000)

            # Re-init pygame since a ship might trigger close if it was closed via GUI
            # actually we don't want to close between episodes, we just loop.

    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    except Exception as e:
        print(f"Error during play: {e}")
        raise
    finally:
        # Clean up
        coordinator.close()
        print("Play mode ended")
