import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path

from boost_and_broadside.core.config import ShipConfig
from boost_and_broadside.env2.env import TensorEnv
from boost_and_broadside.env2.gpu_wrapper import GPUEnvWrapper
from boost_and_broadside.train.evolve.trainer import EvoTrainer

def train_evolve(cfg: DictConfig) -> None:
    print("=== Starting Evolutionary Training (GPU Native) ===")
    
    print("Initializing Base Environment...")
    env_cfg = OmegaConf.to_container(cfg.environment, resolve=True)
    
    valid_keys = ShipConfig.__annotations__.keys()
    ship_cfg_dict = {k: v for k, v in env_cfg.items() if k in valid_keys}
    ship_config = ShipConfig(**ship_cfg_dict)
    
    if "world_size" in env_cfg:
        ship_config.world_size = tuple(env_cfg["world_size"])

    # Num Envs is not strictly needed for the initial pass since we override it in eval, 
    # but we create one dummy environment for device alignment
    env = TensorEnv(
        num_envs=1,
        config=ship_config,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_ships=8,
        max_bullets=env_cfg.get("max_bullets", 20),
        max_episode_steps=cfg.train.evolve.get("max_episode_steps", 1024),
    )
    
    print("Initializing EvoTrainer...")
    trainer = EvoTrainer(cfg, env)
    
    trainer.train()
