import logging
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import h5py
import os
from omegaconf import DictConfig

# New Model
# New Model
# from boost_and_broadside.agents.mamba_bb import MambaBB # Removed
from boost_and_broadside.train.data_loader import create_continuous_data_loader

log = logging.getLogger(__name__)

class MambaConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def create_model(cfg: DictConfig, data_path: str, device: torch.device):
    """Initialize the Yemong Model."""
    import hydra
    import h5py
    
    # Get dimensions from data
    with h5py.File(data_path, "r") as f:
        if "token_dim" in f.attrs:
            state_dim = int(f.attrs["token_dim"])
        else:
            state_dim = f["tokens"].shape[-1]
    
    log.info(f"Initialized Model with input_dim={state_dim}")

    # Inject dynamic dims into config if they are placeholders
    # Hydra instantiation will use cfg.model
    # We might need to override input_dim/target_dim if they are not set correctly in yaml
    # But usually yaml has defaults.
    
    # Force input_dim match data
    if "input_dim" in cfg.model:
        cfg.model.input_dim = state_dim
    if "target_dim" in cfg.model:
        cfg.model.target_dim = state_dim

    # Instantiate via Hydra
    log.info(f"Instantiating model target: {cfg.model._target_}")
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model = model.to(device)
    
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

def create_optimizer(model: torch.nn.Module, cfg: DictConfig) -> optim.Optimizer:
    """Create AdamW optimizer."""
    # Check cfg.train first, then cfg.model, then default
    lr = cfg.train.get("learning_rate", cfg.model.get("learning_rate", 1e-4))
    return optim.AdamW(model.parameters(), lr=lr, fused=True)

def create_scheduler(optimizer: optim.Optimizer, cfg: DictConfig, total_steps: int) -> lr_scheduler.LRScheduler | None:
    """Create learning rate scheduler."""
    sched_cfg = cfg.model.get("scheduler", None)
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
        lr = cfg.train.get("learning_rate", cfg.model.get("learning_rate", 1e-4))
        log.info(f"Initializing Warmup Constant Scheduler: {warmup_cfg.start_lr} -> {lr} over {warmup_cfg.steps} steps")
        
        target_lr = lr
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
    # cfg.train has batch_size? or cfg.model? 
    # Usually batch_size is training param, but was in world_model.
    # Let's check both or prefer model for mismatch safety.
    batch_size = cfg.train.get("batch_size", cfg.model.get("batch_size", 32))

    train_loader, val_loader = create_continuous_data_loader(
        data_path,
        batch_size=batch_size,
        seq_len=cfg.model.get("seq_len", 96),
        validation_split=0.2,
        num_workers=cfg.train.get("num_workers", 4),
        world_size=tuple(cfg.environment.world_size),
        min_skill=min_skill
    )
    
    return train_loader, val_loader
