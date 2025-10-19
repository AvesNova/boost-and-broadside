from omegaconf import DictConfig

from src.game_coordinator import GameCoordinator


def play(cfg: DictConfig) -> None:
    """
    Play mode for running the game with rendering

    Args:
        cfg: Configuration dictionary
    """
    print("Starting play mode...")

    # Create game coordinator
    coordinator = GameCoordinator(cfg)

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
