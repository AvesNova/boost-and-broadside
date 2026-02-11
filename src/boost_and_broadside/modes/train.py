from pathlib import Path
from omegaconf import DictConfig

from boost_and_broadside.modes.collect import collect
from boost_and_broadside.train.bc_trainer import train_bc
from boost_and_broadside.train.rl_trainer import train_rl


def train(cfg: DictConfig) -> None:
    """
    Main training pipeline entry point.

    Orchestrates the training pipeline:
    1. Data Collection (optional)
    2. BC Training (optional)
    3. RL Training (optional)

    Args:
        cfg: Configuration dictionary.
    """

    # Pipeline flags - Fail fast if not present in config (no defaults)
    # Note: We can allow defaults for these specific flags if we want,
    # but strictly speaking per style guide we should enforce them in config.
    # However, for backward compatibility with simple "mode=train" we might want to check config structure.
    # But let's stick to the plan: No defaults in code.
    # If the user wants to run defaults, they should be in config.yaml.

    run_collect = cfg.train.run_collect
    run_bc = cfg.train.run_bc
    run_rl = cfg.train.run_rl

    bc_data_path = cfg.train.bc_data_path
    bc_model_path = None

    # 1. Data Collection
    if run_collect:
        print("\n=== Starting Pipeline Step 1: Data Collection ===")
        collected_data_path = collect(cfg)
        if collected_data_path:
            bc_data_path = str(collected_data_path)
            # Update config for subsequent steps
            cfg.train.bc_data_path = bc_data_path
            print(f"Pipeline: Using newly collected data at {bc_data_path}")
        else:
            print("Pipeline: Data collection failed or returned no path.")

    # 2. BC Training
    if run_bc:
        print("\n=== Starting Pipeline Step 2: BC Training ===")
        # Ensure we have data
        if not bc_data_path:
            print("Pipeline Error: No BC data path specified or collected.")
            return

        bc_model_path = train_bc(cfg)
        if bc_model_path:
            print(f"Pipeline: BC training finished. Model saved at {bc_model_path}")
        else:
            print("Pipeline: BC training failed or disabled.")

    # 2.5. World Model Training
    if cfg.train.get("run_world_model", False):
        print("\n=== Starting Pipeline Step 2.5: World Model Training ===")
        from boost_and_broadside.train.train_world_model import train_world_model

        train_world_model(cfg)

    # 3. RL Training
    if run_rl:
        print("\n=== Starting Pipeline Step 3: RL Training ===")

        # Determine pretrained model path
        pretrained_path = None

        # Priority 1: Model from immediate BC step
        if bc_model_path:
            pretrained_path = bc_model_path
            print(f"Pipeline: Using BC model from current run: {pretrained_path}")

        # Priority 2: Configured pretrained path
        elif cfg.train.rl.pretrained_model_path:
            pretrained_path = Path(cfg.train.rl.pretrained_model_path)
            print(f"Pipeline: Using configured pretrained model: {pretrained_path}")

        train_rl(cfg, pretrained_model_path=pretrained_path)
