import numpy as np
from omegaconf import OmegaConf

from agents.interleaved_world_model import InterleavedWorldModel
from eval.rollout_metrics import compute_rollout_metrics


def test_compute_rollout_metrics_randomized_env():
    """
    Test compute_rollout_metrics in a randomized environment to ensure
    it correctly handles dynamic ship indices (no hardcoded keys).
    """
    device = "cpu"

    # 1. Setup Randomized Environment Config
    env_config = {
        "max_ships": 8,
        "world_size": (1000, 1000),
        "agent_dt": 0.1,
        "physics_dt": 0.01,
        "random_positioning": True,
        "random_speed": True,
        "render_mode": "none",
        "memory_size": 2,
    }

    # 2. Initialize a small dummy InterleavedWorldModel
    # We don't care about performance, just that it runs
    model_config = OmegaConf.create(
        {
            "embed_dim": 64,
            "n_layers": 2,
            "n_heads": 2,
            "n_ships": 8,
            "max_context_len": 32,
            "context_len": 32,  # Used in Agent init
        }
    )

    # Mock config on the model instance as accessing model.config is common
    model = InterleavedWorldModel(
        state_dim=12,  # derived from tokenizer (12 features)
        embed_dim=model_config.embed_dim,
        n_layers=model_config.n_layers,
        n_heads=model_config.n_heads,
        max_ships=model_config.n_ships,
        max_context_len=model_config.max_context_len,
    ).to(device)
    model.config = model_config  # Attach config

    # 3. Run rollout metrics
    # This should trigger the logic we fixed (dynamic ID retrieval)
    metrics = compute_rollout_metrics(
        model=model,
        env_config=env_config,
        device=device,
        num_scenarios=2,
        max_steps=10,  # Short run
        step_intervals=[1, 5, 10],
    )

    # 4. Verify output structure
    assert "mse_sim" in metrics
    assert "mse_dream" in metrics
    assert "step_mse_sim" in metrics
    assert isinstance(metrics["mse_sim"], float)

    # Ensure values are not NaN (though with random weights they might be large)
    assert not np.isnan(metrics["mse_sim"])
