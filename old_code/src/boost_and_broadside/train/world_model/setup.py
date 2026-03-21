import logging
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import h5py
import os
from omegaconf import DictConfig, OmegaConf

# New Model
# New Model
from boost_and_broadside.train.data_loader import create_continuous_data_loader

log = logging.getLogger(__name__)

def automate_num_reward_components(cfg: DictConfig):
    """
    Automatically set num_reward_components in cfg.model based on the top-level rewards config.
    Also injects component names into RewardLoss if present.
    """
    if "rewards" in cfg:
        reward_names = list(cfg.rewards.keys())
        num_rewards = len(reward_names)
        log.info(f"Implicitly setting num_reward_components = {num_rewards} from 'rewards' config.")
        
        OmegaConf.set_struct(cfg.model, False)
        cfg.model.num_reward_components = num_rewards
        
        # Also inject component names into RewardLoss config for descriptive logging
        if "loss" in cfg.model and "losses" in cfg.model.loss:
             for loss_cfg in cfg.model.loss.losses:
                  if "RewardLoss" in loss_cfg._target_:
                       loss_cfg.component_names = reward_names
                       log.info(f"Injected reward names into RewardLoss config: {reward_names}")

def create_model(cfg: DictConfig, data_path: str, device: torch.device):
    """Initialize the Yemong Model."""
    import hydra
    import h5py
    
    # Dims are now typically defined in YAML or constants
    # We only log what was found in HDF5 for debugging
    with h5py.File(data_path, "r") as f:
        attr_dim = int(f.attrs.get("token_dim", -1))
        log.info(f"HDF5 'token_dim' attribute: {attr_dim}")
    
    # We trust the config values which should be aligned with the current code
    # (STATE_DIM=5, TARGET_DIM=7)
    log.info(f"Instantiating model with input_dim={cfg.model.get('input_dim')} and target_dim={cfg.model.get('target_dim')}")

    # Pass stats_path to model if raw_stats.csv exists in the data directory
    stats_path = os.path.join(os.path.dirname(data_path), "raw_stats.csv")
    if os.path.exists(stats_path):
        log.info(f"Adding stats_path to model config: {stats_path}")
        OmegaConf.set_struct(cfg.model, False)
        cfg.model.stats_path = stats_path

    # Automate num_reward_components if rewards section exists
    automate_num_reward_components(cfg)

    # Instantiate via Hydra
    log.info(f"Instantiating model target: {cfg.model._target_}")
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model = model.to(device)
    
    # Compile
    import platform
    is_wsl = 'wsl' in platform.release().lower() or 'microsoft' in platform.release().lower()
    
    if os.name != 'nt' and not is_wsl and cfg.train.get("compile", True):
        mode = cfg.train.get("compile_mode", "default")
        log.info(f"Compiling model with mode='{mode}'...")
        try:
             model = torch.compile(model, mode=mode)
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
        min_skill=min_skill,
        reward_config=cfg.get("rewards", None)
    )
    
    return train_loader, val_loader
