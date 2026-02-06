
import pytest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
from omegaconf import OmegaConf

from train.world_model.trainer import Trainer
from train.world_model.validator import Validator
from train.world_model.logger import MetricLogger

@pytest.fixture
def mock_components(tmp_path):
    model = MagicMock()
    model.parameters.return_value = [torch.tensor(0.0)]
    optimizer = MagicMock()
    scheduler = MagicMock()
    scaler = MagicMock()
    swa_model = MagicMock()
    
    # Validator with controllable behavior
    validator = MagicMock()
    validator.validate_validation_set.return_value = {
        "val_loss": 0.5, "error_power": 0.1, "error_turn": 0.1, "error_shoot": 0.1,
        "preds_p": [], "targets_p": []
    }
    validator.validate_autoregressive.return_value = {
        "val_rollout_mse_state": 0.05,
        "val_rollout_mse_step": [0.01, 0.02, 0.03] # List for heatmap
    }
    
    logger = MagicMock()
    
    cfg = OmegaConf.create({
        "world_model": {
            "epochs": 10,
            "batch_ratio": 4,
            "noise_scale": 0.1,
            "validation": {
                "max_batches": 10,
                "heavy_eval_freq": 2
            }
        },
        "train": {"amp": False},
        "wandb": {
            "enabled": True,
            "project": "test_project",
            "entity": "test_entity",
            "entity": "test_entity",
            "group": "test_group"
        },
        "environment": {
            "world_size": [1000.0, 1000.0]
        }
    })
    
    device = torch.device("cpu")
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    
    # Patch wandb to avoid starting a real run
    with patch("wandb.init"), patch("wandb.log"), patch("wandb.plot.line"), patch("wandb.Table"):
         yield {
            "model": model, "optimizer": optimizer, "scheduler": scheduler,
            "scaler": scaler, "swa_model": swa_model, "logger": logger,
            "validator": validator, "cfg": cfg, "device": device, "run_dir": run_dir
        }

def test_heavy_eval_freq(mock_components):
    """Test that heavy eval only runs on specific epochs."""
    with patch("torch.save"), \
         patch("train.world_model.trainer.calculate_action_counts", return_value={"power": np.ones(3), "turn": np.ones(7), "shoot": np.ones(2)}), \
         patch("train.world_model.trainer.compute_class_weights", return_value=torch.ones(3)), \
         patch("train.world_model.trainer.apply_turn_exceptions", side_effect=lambda x: x), \
         patch("train.world_model.trainer.normalize_weights", side_effect=lambda w, c: w):
        
        trainer = Trainer(**mock_components, data_path="dummy_path")
        
        # Disable SWA for this test to focus on _validate_epoch logic inside train loop
        # But effectively we just call _validate_epoch directly
        
        # Epoch 0 (1st epoch) -> 1 % 2 != 0 -> Not Heavy
        trainer._validate_epoch(0, [], [])
        mock_components["validator"].validate_autoregressive.assert_not_called()
        
        # Epoch 1 (2nd epoch) -> 2 % 2 == 0 -> Heavy
        trainer._validate_epoch(1, [], [])
        mock_components["validator"].validate_autoregressive.assert_called_once()
        
        # Check that logger received the AR metrics
        # The last call to log_epoch should contain Val_AR metrics
        args, _ = mock_components["logger"].log_epoch.call_args
        metrics = args[0]
        assert "Val_AR/val_rollout_mse_state" in metrics
        assert "Val_AR/val_rollout_mse_step" in metrics
        assert metrics["Val_AR/val_rollout_mse_step"] == [0.01, 0.02, 0.03]

def test_logger_heatmap_logic(mock_components):
    """Test that logger converts lists to wandb Line Plots."""
    logger = MetricLogger(mock_components["cfg"], mock_components["run_dir"])
    
    with patch("wandb.log") as mock_wandb_log, patch("wandb.plot.line") as mock_wandb_line:
        epoch_metrics = {
            "epoch": 2, 
            "global_step": 100,
            "Val_AR/val_rollout_mse_step": [0.1, 0.2, 0.3]
        }
        
        logger.log_epoch(epoch_metrics, 1)
        
        # Verify wandb.plot.line was called
        mock_wandb_line.assert_called_once()
        
        # Verify wandb.log was called with the plot
        args, _ = mock_wandb_log.call_args
        log_dict = args[0]
        assert "Val_AR/val_rollout_mse_step" in log_dict
        
def test_max_batches_limit(mock_components):
    """Test that Validator respects max_batches."""
    # We need a real validator instance for this, not a mock
    model = MagicMock()
    device = torch.device("cpu")
    cfg = mock_components["cfg"]
    
    validator = Validator(model, device, cfg)
    
    # Mock Loader: Infinite iterator
    class MockLoader:
        def __iter__(self):
            while True:
                # Return dummy batch structure (Size 9 tuple)
                yield {
                    "states": torch.zeros(1, 11, 2, 16),
                    "actions": torch.zeros(1, 11, 2, 3), # +1 for target shifting (T+1)
                    "target_actions": torch.zeros(1, 11, 2, 3), 
                    "team_ids": torch.zeros(1, 11, 2, dtype=torch.long),
                    "seq_idx": torch.zeros(1, 11),
                    "loss_mask": torch.ones(1, 11, 2),
                    "pos": torch.zeros(1, 11, 2, 2), # Dummy Pos
                    "vel": None,
                    "rewards": torch.zeros(1, 11, 2),
                    "returns": torch.zeros(1, 11, 2)
                }
    
    loader = MockLoader()
    
    # Should stop after 10 batches (cfg.world_model.validation.max_batches = 10)
    # Each batch takes some processing. We mock model return to avoid error.
    model.return_value = (
        torch.zeros(1, 10, 2, 16), # Pred States
        torch.zeros(1, 10, 2, 12), # Pred Actions
        torch.zeros(1, 10, 1),     # Pred Values
        torch.zeros(1, 10, 1)      # Pred Rewards
    )
    model.get_loss.return_value = (torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), {})
    
    # Run
    metrics = validator.validate_validation_set([loader])
    
    # How to verify? 
    # Since we can't easily spy on the loop count without adding side effects, 
    # we can trust that it finished (didn't run forever).
    # But to be sure, we can spy on model calls.
    assert model.call_count == 10
