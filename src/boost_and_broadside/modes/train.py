from omegaconf import DictConfig
from boost_and_broadside.modes.collect import collect


def train(cfg: DictConfig) -> None:
    """
    Main training pipeline entry point.

    Orchestrates the training pipeline:
    1. Data Collection (optional)
    2. World Model Training (mandatory/optional)

    Args:
        cfg: Configuration dictionary.
    """
    run_collect = cfg.train.run_collect

    # 1. Data Collection
    if run_collect:
        print("\n=== Starting Pipeline Step 1: Data Collection ===")
        collected_path = collect(cfg)
        if collected_path:
             cfg.train.bc_data_path = str(collected_path)
             print(f"Pipeline: Using newly collected data at {collected_path}")

    # 2. World Model Training
    if cfg.train.get("run_world_model", True):
        print("\n=== Starting Pipeline Step 2: World Model Training ===")
        from boost_and_broadside.train.train_world_model import train_world_model
        train_world_model(cfg)
