from omegaconf import DictConfig, OmegaConf

from src.game_coordinator import GameCoordinator


def play(cfg: DictConfig) -> None:
    """
    Play mode for running the game with rendering

    Args:
        cfg: Configuration dictionary
    """
    print("Starting play mode...")

    # Allow config modification
    OmegaConf.set_struct(cfg, False)

    # Get team configurations
    team1_type = cfg.get("team1", "scripted")
    team2_type = cfg.get("team2", "scripted")
    human_player = cfg.get("human_player", True)

    print(f"Team 1: {team1_type}")
    print(f"Team 2: {team2_type}")
    print(f"Human Player: {human_player}")

    # Helper to create agent config
    def get_agent_config(agent_type):
        # Default config for agents that need it
        base_config = {
            "agent_id": "player",
            "team_id": 0,
            "squad": [],
            # Transformer defaults
            "token_dim": 12,
            "embed_dim": 64,
            "num_heads": 4,
            "num_layers": 3,
            "max_ships": 8,
        }
        return {"agent_type": agent_type, "agent_config": base_config}

    # Setup Team 1
    if team1_type in cfg.agents:
        cfg.agents["team1_agent"] = cfg.agents[team1_type]
    else:
        cfg.agents["team1_agent"] = get_agent_config(team1_type)

    # Setup Team 2
    if team2_type in cfg.agents:
        cfg.agents["team2_agent"] = cfg.agents[team2_type]
    else:
        cfg.agents["team2_agent"] = get_agent_config(team2_type)

    # Set teams for the game
    cfg.collect.teams = ["team1_agent", "team2_agent"]

    # Create game coordinator
    coordinator = GameCoordinator(cfg)

    # Add human player if requested
    if human_player:
        # Assuming ship 0 is always the first ship of team 0
        coordinator.env.add_human_player(0)
        print("Human control enabled for Ship 0")

    try:
        # Run a single episode
        coordinator.reset(game_mode="nvn")

        coordinator.step()

    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    except Exception as e:
        print(f"Error during play: {e}")
        raise
    finally:
        # Clean up
        coordinator.close()
        print("Play mode ended")
