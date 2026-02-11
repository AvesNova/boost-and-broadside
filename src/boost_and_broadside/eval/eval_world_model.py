"""
World model evaluation script.

Loads a trained world model and evaluates its ability to generate
realistic rollouts from initial conditions.
"""

import logging

import torch
from omegaconf import DictConfig

from boost_and_broadside.agents.world_model import WorldModel
from boost_and_broadside.train.data_loader import load_bc_data, create_unified_data_loaders
from omegaconf import OmegaConf

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
    data = load_bc_data(data_path)

    # Create loader to get dimensions and sample
    # We use the long validation loader for evaluation
    _, _, _, val_loader = create_unified_data_loaders(
        data,
        short_batch_size=cfg.world_model.short_batch_size,
        long_batch_size=1,  # Batch size 1 for evaluation
        short_batch_len=cfg.world_model.short_batch_len,
        long_batch_len=cfg.world_model.long_batch_len,
        batch_ratio=cfg.world_model.batch_ratio,
        validation_split=0.2,
        num_workers=0,
    )

    # Get a sample
    sample_states, sample_actions, _, _, _ = next(iter(val_loader))
    state_dim = sample_states.shape[-1]
    # action_dim = sample_actions.shape[-1] # This is 3, but we use 12 for one-hot
    # We will hardcode action_dim to 12 as per model requirement

    # 2. Load Model
    from boost_and_broadside.utils.model_finder import find_most_recent_model
    from boost_and_broadside.eval.rollout_metrics import compute_rollout_metrics

    model_path = find_most_recent_model("world_model")
    if model_path is None:
        log.error("No world model found!")
        return

    log.info(f"Loading model from {model_path}")

    # Force action_dim to 12 (3+7+2 one-hot)
    action_dim = 12

    model = WorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=cfg.world_model.embed_dim,
        n_layers=cfg.world_model.n_layers,
        n_heads=cfg.world_model.n_heads,
        max_ships=cfg.world_model.n_ships,
        max_context_len=cfg.world_model.context_len,
    ).to(device)

    model.load_state_dict(torch.load(model_path))
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
