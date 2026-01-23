import logging
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import h5py
import os
from omegaconf import DictConfig

from agents.interleaved_world_model import InterleavedWorldModel
from train.data_loader import load_bc_data, create_unified_data_loaders

log = logging.getLogger(__name__)

def create_model(cfg: DictConfig, data_path: str, device: torch.device) -> InterleavedWorldModel:
    """Initialize the Interleaved World Model."""
    # Get dimensions from data
    with h5py.File(data_path, "r") as f:
        if "token_dim" in f.attrs:
            state_dim = int(f.attrs["token_dim"])
        else:
            state_dim = f["tokens"].shape[-1]
    
    log.info(f"Initialized World Model with state_dim={state_dim}")

    model = InterleavedWorldModel(
        state_dim=state_dim,
        embed_dim=cfg.world_model.embed_dim,
        n_layers=cfg.world_model.n_layers,
        n_heads=cfg.world_model.n_heads,
        max_ships=cfg.world_model.n_ships,
        max_context_len=cfg.world_model.context_len,
        use_relational_head=cfg.world_model.get("use_relational_head", True),
    ).to(device)
    
    # Compile
    if os.name != 'nt':
        log.info("Compiling model...")
        model = torch.compile(model)
    else:
        log.info("Skipping torch.compile on Windows")
        
    return model

def create_optimizer(model: InterleavedWorldModel, cfg: DictConfig) -> optim.Optimizer:
    """Create AdamW optimizer."""
    return optim.AdamW(model.parameters(), lr=cfg.world_model.learning_rate, fused=True)

def create_scheduler(optimizer: optim.Optimizer, cfg: DictConfig, total_steps: int) -> lr_scheduler.LRScheduler | None:
    """Create learning rate scheduler."""
    sched_cfg = cfg.world_model.get("scheduler", None)
    if not sched_cfg:
        return None
        
    if "range_test" in sched_cfg.type:
        range_cfg = sched_cfg.range_test
        log.info(f"Initializing Exponential Range Test Scheduler: {range_cfg.start_lr} -> {range_cfg.end_lr} over {total_steps} steps")

        for param_group in optimizer.param_groups:
            param_group['lr'] = range_cfg.start_lr
            
        if range_cfg.start_lr == 0:
             raise ValueError("Start LR cannot be 0 for exponential range test.")
             
        total_multiplier = range_cfg.end_lr / range_cfg.start_lr
        
        def lr_lambda(step):
            s = min(step, total_steps)
            progress = s / total_steps
            return total_multiplier ** progress
            
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    elif sched_cfg.type == "warmup_constant":
        warmup_cfg = sched_cfg.warmup
        log.info(f"Initializing Warmup Constant Scheduler: {warmup_cfg.start_lr} -> {cfg.world_model.learning_rate} over {warmup_cfg.steps} steps")
        
        target_lr = cfg.world_model.learning_rate
        start_lr = warmup_cfg.start_lr
        warmup_steps = warmup_cfg.steps
        
        def lr_lambda(step):
            if step < warmup_steps:
                return (start_lr + (target_lr - start_lr) * (step / warmup_steps)) / target_lr
            return 1.0 
            
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    return None

def get_data_loaders(cfg: DictConfig, data_path: str):
    """Create unified data loaders."""
    return create_unified_data_loaders(
        data_path,
        short_batch_size=cfg.world_model.short_batch_size,
        long_batch_size=cfg.world_model.long_batch_size,
        short_batch_len=cfg.world_model.short_batch_len,
        long_batch_len=cfg.world_model.long_batch_len,
        batch_ratio=cfg.world_model.batch_ratio,
        validation_split=0.2,
        num_workers=cfg.world_model.get("num_workers", 4),
        prefetch_factor=cfg.world_model.get("prefetch_factor", 2),
        world_size=tuple(cfg.environment.world_size),
    )
