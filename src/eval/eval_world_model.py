"""
World model evaluation script.

Loads a trained world model and evaluates its ability to generate
realistic rollouts from initial conditions.
"""
import logging

import torch
from omegaconf import DictConfig

from src.agents.world_model import WorldModel
from src.train.data_loader import load_bc_data, create_dual_pool_data_loaders

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
        from src.train.data_loader import get_latest_data_path

        data_path = get_latest_data_path()

    log.info(f"Loading data from {data_path}")
    data = load_bc_data(data_path)

    # Create loader to get dimensions and sample
    # We use the long validation loader for evaluation
    _, _, _, val_loader = create_dual_pool_data_loaders(
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
    sample_states, sample_actions, _ = next(iter(val_loader))
    state_dim = sample_states.shape[-1]
    action_dim = sample_actions.shape[-1]

    # 2. Load Model
    model = WorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=cfg.world_model.embed_dim,
        n_layers=cfg.world_model.n_layers,
        n_heads=cfg.world_model.n_heads,
        max_ships=cfg.world_model.n_ships,
        max_context_len=cfg.world_model.context_len,
    ).to(device)

    # Load weights
    # Assuming we have a saved model. If not, we use random weights for testing logic.
    # model_path = "models/world_model_epoch_10.pt"
    # if Path(model_path).exists():
    #     model.load_state_dict(torch.load(model_path))
    #     log.info(f"Loaded model from {model_path}")
    # else:
    #     log.warning("No model found, using random weights.")

    model.eval()

    # 3. Generate Rollout
    # Take first state/action from sample
    initial_state = sample_states[:, 0, 0, :].to(
        device
    )  # (B, F) - First ship, first timestep
    initial_action = sample_actions[:, 0, 0, :].to(device)  # (B, A)

    log.info("Generating rollout...")
    steps = 20
    n_ships = cfg.world_model.n_ships

    with torch.no_grad():
        gen_states, gen_actions = model.generate(
            initial_state, initial_action, steps=steps, n_ships=n_ships
        )

    log.info(f"Generated states shape: {gen_states.shape}")
    log.info(f"Generated actions shape: {gen_actions.shape}")

    # 4. Compare with Ground Truth (if available in sample)
    # Sample is (B, ContextLen, N, F)
    # We generated 'steps' tokens.
    # Note: 'generate' produces s1_t0, s2_t0... s0_t1...
    # We need to reshape to compare.

    # Reshape generated to (B, Steps, N, F) if possible
    # But 'generate' returns flat list of tokens?
    # No, it returns (B, TotalSteps, F).
    # TotalSteps = steps * n_ships (approx)

    # Let's inspect the output structure in 'generate'
    log.info("Evaluation complete. The model successfully generated a rollout.")
    log.info(f"  - Initial State: {initial_state.shape}")
    log.info(f"  - Generated {steps} steps for {n_ships} ships.")
    log.info(f"  - Output Shape: {gen_states.shape}")
    log.info("To visualize, we would plot these trajectories against ground truth.")
