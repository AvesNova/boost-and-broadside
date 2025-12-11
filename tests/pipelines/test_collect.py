import os
from modes.collect import collect


def test_collect_pipeline(default_config, tmp_path):
    """Test the data collection pipeline."""
    # Update config for testing
    cfg = default_config.copy()
    cfg.mode = "collect"
    cfg.collect.num_workers = 1
    cfg.collect.max_episode_length = 10  # Short episode

    print(f"DEBUG: Env world_size: {cfg.environment.world_size}")
    print(f"DEBUG: Agent world_size: {cfg.agents.scripted.agent_config.world_size}")

    # Use tmp_path for output
    # The collect_data function likely uses hydra's output directory or a configured path
    # We might need to patch the output directory or check where it writes
    # Assuming it writes to 'data/bc_pretraining' relative to CWD

    # Let's change CWD to tmp_path to avoid polluting the project
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        # Run collection
        collect(cfg)

        # Verify output
        # Expecting data/bc_pretraining/YYYYMMDD_HHMMSS/worker_0/data_final.pkl
        # Since we don't know the exact timestamp, we check for the structure

        base_dir = os.path.join(tmp_path, "data", "bc_pretraining")
        assert os.path.exists(base_dir)

        # Find the run directory
        run_dirs = os.listdir(base_dir)
        assert len(run_dirs) > 0
        run_dir = os.path.join(base_dir, run_dirs[0])

        # Check for worker directory
        worker_dir = os.path.join(run_dir, "worker_0")
        assert os.path.exists(worker_dir)

        # Check for data file
        data_file = os.path.join(worker_dir, "data_final.pkl")
        assert os.path.exists(data_file)

    finally:
        os.chdir(original_cwd)
