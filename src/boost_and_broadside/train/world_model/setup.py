import logging
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import h5py
import os
from omegaconf import DictConfig

# New Model
from boost_and_broadside.agents.mamba_bb import MambaBB
from boost_and_broadside.train.data_loader import create_continuous_data_loader

log = logging.getLogger(__name__)

class MambaConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def create_model(cfg: DictConfig, data_path: str, device: torch.device) -> MambaBB:
    """Initialize the MambaBB World Model."""
    # Get dimensions from data
    with h5py.File(data_path, "r") as f:
        if "token_dim" in f.attrs:
            state_dim = int(f.attrs["token_dim"])
        else:
            state_dim = f["tokens"].shape[-1]
    
    log.info(f"Initialized MambaBB with input_dim={state_dim}")

    # Map Hydra Config to Model Config
    model_cfg = MambaConfig(
        input_dim=state_dim,
        d_model=cfg.world_model.embed_dim,
        n_layers=cfg.world_model.n_layers,
        n_heads=cfg.world_model.n_heads,
        action_dim=12, # 3+7+2
        target_dim=state_dim, # Delta prediction
        loss_type=cfg.world_model.loss.get("type", "fixed")
    )

    # Keep model in Float32 for stability (Autocast will handle mixed precision)
    # forcing pure BFloat16 weights without careful optimizer setup causes instability/NaNs
    model = MambaBB(model_cfg).to(device)
    
    # Compile
    import platform
    is_wsl = 'wsl' in platform.release().lower() or 'microsoft' in platform.release().lower()
    
    if os.name != 'nt' and not is_wsl and cfg.train.get("compile", True):
        log.info("Compiling model...")
        try:
             model = torch.compile(model)
        except Exception as e:
             log.warning(f"Torch compile failed: {e}. Running eager.")
    else:
        log.info("Skipping torch.compile (Windows/WSL)")
        
    return model

def create_optimizer(model: MambaBB, cfg: DictConfig) -> optim.Optimizer:
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

def get_data_loaders(cfg: DictConfig, data_path: str, min_skill: float = 0.0):
    """Create Continuous Data Loaders."""
    # We ignore short/long splits and return just Train/Val Continuous loaders.
    # To match Trainer expectations (train_short, train_long, val_short, val_long),
    # we might need to return them as duplicates or placeholders?
    # Trainer loop: train_short_loader, train_long_loader.
    # I'll update Trainer to just take 'train_loader'.
    # But init signature of Trainer expects them.
    # For now, let's return train_loader, None, val_loader, None
    # And update Trainer to handle None.
    
    train_loader, val_loader = create_continuous_data_loader(
        data_path,
        batch_size=cfg.world_model.batch_size,
        seq_len=cfg.world_model.get("seq_len", 96),
        validation_split=0.2,
        num_workers=cfg.world_model.get("num_workers", 4),
        world_size=tuple(cfg.environment.world_size),
        min_skill=min_skill
    )
    
    return train_loader, val_loader
