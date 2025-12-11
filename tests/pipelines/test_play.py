import pytest

from modes.play import play


def test_play_pipeline(default_config):
    """Test the play pipeline in headless mode."""
    # Update config for testing
    cfg = default_config.copy()
    cfg.mode = "play"
    cfg.environment.render_mode = "none"  # Headless
    cfg.human_player = False  # Disable human player for automation
    cfg.collect.max_episode_length = 10  # Short episode

    # Ensure play mode uses the agents defined in conftest
    cfg.team1 = "scripted"
    cfg.team2 = "scripted"

    # We need to make sure play() respects these configs
    # play.py uses GameCoordinator which takes render_mode override
    # But play() function itself might need adjustment if it hardcodes things
    # Looking at play.py, it passes cfg to GameCoordinator.
    # It also calls coordinator.step() which runs until termination.
    # We set max_episode_length to 10 so it should terminate quickly.

    try:
        play(cfg)
    except Exception as e:
        pytest.fail(f"Play pipeline failed: {e}")
