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
        print("Running episode...")
        summary = coordinator.run_episode(num_steps=1000)

        # Print summary
        print(f"Episode completed:")
        print(f"  Steps: {summary['steps']}")
        print(
            f"  Winner: Team {summary['winner']}"
            if summary["winner"] is not None
            else "  Winner: Draw"
        )

    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    except Exception as e:
        print(f"Error during play: {e}")
        raise
    finally:
        # Clean up
        coordinator.close()
        print("Play mode ended")
