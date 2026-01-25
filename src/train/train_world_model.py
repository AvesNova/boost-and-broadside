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

from train.data_loader import load_bc_data
from train.swa import SWAModule

# New Modules
from train.world_model.setup import (
    create_model, create_optimizer, create_scheduler, get_data_loaders
)
from train.world_model.logger import MetricLogger
from train.world_model.validator import Validator
from train.world_model.trainer import Trainer

log = logging.getLogger(__name__)

def train_world_model(cfg: DictConfig) -> None:
    """
    Train the world model.
    """
    log.info("Starting World Model training... (Refactored Pipeline)")

    # 1. Enable TF32 globally
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
    train_short, train_long, val_short, val_long = get_data_loaders(cfg, data_path)
    
    # Calculate total steps for scheduler estimate
    acc_steps = cfg.world_model.get("gradient_accumulation_steps", 1)
    short_batches = len(train_short)
    ratio = cfg.world_model.batch_ratio
    prob_short = ratio / (ratio + 1)
    # Estimate total micro steps
    total_est_steps = int(short_batches / prob_short / acc_steps) * cfg.world_model.epochs
    
    scheduler = create_scheduler(optimizer, cfg, total_est_steps)
    
    # Amp Scaler
    use_amp = cfg.train.get("amp", False) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
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
        trainer.train(train_short, train_long, val_short, val_long)
    finally:
        logger.close()
        
    log.info("World Model training complete.")
