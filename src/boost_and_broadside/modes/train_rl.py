
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from boost_and_broadside.core.config import ShipConfig
from boost_and_broadside.env2.env import TensorEnv
from boost_and_broadside.env2.gpu_wrapper import GPUEnvWrapper
from boost_and_broadside.models.yemong.scaffolds import YemongDynamics
from boost_and_broadside.train.rl.ppo import PPOTrainer

def train_rl(cfg: DictConfig) -> None:
    print("=== Starting RL Training (GPU Native) ===")
    
    # 1. Setup Environment
    print("Initializing Environment...")
    env_cfg = OmegaConf.to_container(cfg.environment, resolve=True)
    
    # Filter config for ShipConfig
    valid_keys = ShipConfig.__annotations__.keys()
    # Also include fields with defaults that might not be in annotations dict directly (dataclass specific)
    # But __annotations__ usually has all fields.
    ship_cfg_dict = {k: v for k, v in env_cfg.items() if k in valid_keys}
    
    # Create ShipConfig
    ship_config = ShipConfig(**ship_cfg_dict)
    
    # Override world size if explicitly set
    if "world_size" in env_cfg:
        ship_config.world_size = tuple(env_cfg["world_size"])

    # Num Envs from PPO config
    num_envs = cfg.train.ppo.num_envs
    
    # Create TensorEnv
    # We use the max_ships from model config to ensure compatibility
    max_ships = cfg.model.max_ships
    
    env = TensorEnv(
        num_envs=num_envs,
        config=ship_config,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_ships=max_ships,
        max_bullets=env_cfg.get("max_bullets", 20)
    )
    
    # Wrap for GPU-native RL Interface
    env = GPUEnvWrapper(env)
    
    # 2. Setup Agent
    print("Initializing YemongDynamics...")
    agent = YemongDynamics(cfg.model)
    agent.to(env.device)
    
    # 3. Setup Trainer
    print("Initializing PPOTrainer...")
    trainer = PPOTrainer(cfg, env, agent)
    
    # 4. Train
    print("Starting Training Loop...")
    trainer.train()
