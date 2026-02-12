"""
World model evaluation script.

Loads a trained world model and evaluates its ability to generate
realistic rollouts from initial conditions.
"""

import logging

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from boost_and_broadside.train.data_loader import create_continuous_data_loader # Use new loader if needed, or keep for now

log = logging.getLogger(__name__)


def eval_world_model(cfg: DictConfig) -> None:
    """
    Evaluate a trained world model.

    Args:
        cfg: Hydra configuration object.
    """
    log.info("Starting World Model Evaluation...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # 1. Load Data (Validation set)
    data_path = cfg.train.bc_data_path
    if data_path is None:
        from boost_and_broadside.train.data_loader import get_latest_data_path
        data_path = get_latest_data_path()

    log.info(f"Loading data from {data_path}")
    
    from boost_and_broadside.train.data_loader import create_continuous_data_loader
    _, val_loader = create_continuous_data_loader(
        data_path,
        batch_size=1,
        seq_len=cfg.model.get("seq_len", 96),
        validation_split=0.2,
        num_workers=0,
        world_size=tuple(cfg.environment.world_size)
    )

    # Get a sample
    batch = next(iter(val_loader))
    sample_states = batch["states"]
    state_dim = sample_states.shape[-1]
    # action_dim = sample_actions.shape[-1] # This is 3, but we use 12 for one-hot
    # We will hardcode action_dim to 12 as per model requirement

    # 2. Load Model
    from boost_and_broadside.utils.model_finder import find_most_recent_model
    from boost_and_broadside.eval.rollout_metrics import compute_rollout_metrics
    from boost_and_broadside.train.world_model.setup import create_model

    model_path = find_most_recent_model("world_model")
    if model_path is None:
        log.error("No world model found!")
        return

    log.info(f"Loading model from {model_path}")

    # Use the setup helper instead of manual instantiation
    model = create_model(cfg, data_path, device)
    
    # Load state dict
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        log.warning(f"Direct load failed: {e}. Trying via 'model_state_dict' key.")
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
             model.load_state_dict(checkpoint["model_state_dict"])
        else:
             raise e
             
    model.eval()

    # 3. Evaluation Loop
    log.info("Starting evaluation...")

    # Get env config
    env_config = OmegaConf.to_container(cfg.environment, resolve=True)
    env_config["render_mode"] = "none"

    # Compute Rollout MSE
    metrics = compute_rollout_metrics(
        model,
        env_config,
        device,
        num_scenarios=4,  # Configurable?
        max_steps=128,
    )

    mse_sim = metrics["mse_sim"]
    mse_dream = metrics["mse_dream"]

    log.info(
        f"Evaluation Result - Average Rollout MSE: Sim={mse_sim:.6f}, Dream={mse_dream:.6f}"
    )

    # Create simple CSV with single value
    csv_path = "eval_rollout_metrics.csv"
    with open(csv_path, "w") as f:
        f.write("metric,value\n")
        f.write(f"rollout_mse_sim,{mse_sim:.6f}\n")
        f.write(f"rollout_mse_dream,{mse_dream:.6f}\n")
    log.info(f"Saved metric to {csv_path}")
