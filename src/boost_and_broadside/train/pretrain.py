"""
World model training script.

Trains a transformer-based world model to predict future states and actions
using masked reconstruction and denoising objectives.
"""

import logging
from pathlib import Path
from datetime import datetime
import torch
from omegaconf import DictConfig, OmegaConf

from boost_and_broadside.train.data_loader import load_bc_data
from boost_and_broadside.train.swa import SWAModule

# New Modules
from boost_and_broadside.train.world_model.setup import (
    create_model, create_optimizer, create_scheduler, get_data_loaders
)
from boost_and_broadside.train.world_model.logger import MetricLogger
from boost_and_broadside.train.world_model.validator import Validator
from boost_and_broadside.train.world_model.trainer import Trainer

log = logging.getLogger(__name__)

def pretrain(cfg: DictConfig) -> None:
    """
    Train the Yemong world model (Pretraining).
    """
    log.info("Starting Yemong Pretraining...")

    # 1. Enable TF32 globally
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Setup Run Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("models/world_model") / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {run_dir}")
    OmegaConf.save(cfg, run_dir / "config.yaml")

    # Setup File Logging
    log_file = run_dir / "train.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO) # Ensure info is captured
    
    log.info(f"Logging to {log_file}")

    # 3. Load Data
    data_path = load_bc_data(cfg.train.bc_data_path)
    
    # 4. Create Components
    model = create_model(cfg, data_path, device)
    optimizer = create_optimizer(model, cfg)
    
    # Loaders
    train_loader, val_loader = get_data_loaders(cfg, data_path)
    
    # Calculate total steps for scheduler estimate
    acc_steps = cfg.train.get("gradient_accumulation_steps", 1)
    
    # If range test, override defaults for more diagnostic resolution
    sched_cfg = cfg.model.get("scheduler", None)
    is_range_test = sched_cfg and "range_test" in getattr(sched_cfg, "type", "")
    
    if is_range_test:
        log.info(f"LR Range Test: Using configured gradient_accumulation_steps={acc_steps}")
        
    train_batches = len(train_loader)
    # Prefer cfg.train.epochs, fallback to model if needed.
    epochs = cfg.train.get("epochs", cfg.model.get("epochs", 100))
    total_est_steps = int(train_batches / acc_steps) * epochs
    
    total_sched_steps = total_est_steps
    if is_range_test:
        total_sched_steps = min(total_est_steps, sched_cfg.range_test.steps)
        log.info(f"LR Range Test: Scaling scheduler over {total_sched_steps} steps")

    scheduler = create_scheduler(optimizer, cfg, total_sched_steps)
    
    # Amp Scaler
    use_amp = cfg.train.get("amp", False) and device.type == 'cuda'
    # Disable scaler for BFloat16 as it's not needed and not implemented for BF16 in many PyTorch versions
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    
    # SWA
    swa_model = SWAModule(model)
    
    # Logger
    logger = MetricLogger(cfg, run_dir)
    
    # Validator
    validator = Validator(model, device, cfg)
    
    # 5. Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        swa_model=swa_model,
        logger=logger,
        validator=validator,
        cfg=cfg,
        device=device,
        run_dir=run_dir,
        data_path=data_path # Pass resolved path for re-loading
    )
    
    # 6. Run
    try:
        trainer.train(train_loader, val_loader)
    finally:
        logger.close()
        
    log.info("World Model training complete.")
