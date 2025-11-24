import torch.nn as nn
from pathlib import Path
from omegaconf import OmegaConf

from .human import HumanAgent
from .scripted import ScriptedAgent
from .team_transformer_agent import TeamTransformerAgent
from .replay import ReplayAgent
from .dummy import DummyAgent
from .random_agent import RandomAgent
from src.utils.model_finder import find_most_recent_model, find_best_model


def create_agent(agent_type: str, agent_config: dict) -> nn.Module:
    match agent_type:
        case "human":
            return HumanAgent(**agent_config)
        case "scripted":
            return ScriptedAgent(**agent_config)
        case "team_transformer_agent":
            return TeamTransformerAgent(**agent_config)
        case "replay_agent":
            return ReplayAgent(**agent_config)
        case "dummy":
            return DummyAgent(**agent_config)
        case "random":
            return RandomAgent(**agent_config)
        case "most_recent_bc":
            model_path = find_most_recent_model("bc")
            if not model_path:
                print("Warning: No BC model found for 'most_recent_bc' agent. Using random initialization.")
                return TeamTransformerAgent(**agent_config)
            
            # Try to load config from run directory
            run_dir = Path(model_path).parent
            config_path = run_dir / "config.yaml"
            
            final_config = agent_config.copy()
            if config_path.exists():
                print(f"Loading agent config from {config_path}")
                saved_cfg = OmegaConf.load(config_path)
                # Extract transformer config
                transformer_cfg = OmegaConf.to_container(saved_cfg.train.model.transformer, resolve=True)
                # Remove keys that are not arguments to TeamTransformerAgent
                if "num_actions" in transformer_cfg:
                    del transformer_cfg["num_actions"]
                final_config.update(transformer_cfg)
            else:
                print(f"Warning: No config.yaml found in {run_dir}. Using current defaults.")
            
            final_config["model_path"] = model_path
            return TeamTransformerAgent(**final_config)

        case "most_recent_rl":
            model_path = find_most_recent_model("rl")
            if not model_path:
                print("Warning: No RL model found for 'most_recent_rl' agent. Using random initialization.")
                return TeamTransformerAgent(**agent_config)
            
            # Try to load config from run directory
            run_dir = Path(model_path).parent
            config_path = run_dir / "config.yaml"
            
            final_config = agent_config.copy()
            if config_path.exists():
                print(f"Loading agent config from {config_path}")
                saved_cfg = OmegaConf.load(config_path)
                # Extract transformer config
                transformer_cfg = OmegaConf.to_container(saved_cfg.train.model.transformer, resolve=True)
                final_config.update(transformer_cfg)
            else:
                print(f"Warning: No config.yaml found in {run_dir}. Using current defaults.")
            
            final_config["model_path"] = model_path
            return TeamTransformerAgent(**final_config)

        case "best_bc":
            model_path = find_best_model("bc")
            if not model_path:
                print("Warning: No BC model found for 'best_bc' agent. Using random initialization.")
                return TeamTransformerAgent(**agent_config)
            
            # Try to load config from run directory
            run_dir = Path(model_path).parent
            config_path = run_dir / "config.yaml"
            
            final_config = agent_config.copy()
            if config_path.exists():
                print(f"Loading agent config from {config_path}")
                saved_cfg = OmegaConf.load(config_path)
                # Extract transformer config
                transformer_cfg = OmegaConf.to_container(saved_cfg.train.model.transformer, resolve=True)
                final_config.update(transformer_cfg)
            else:
                print(f"Warning: No config.yaml found in {run_dir}. Using current defaults.")
            
            final_config["model_path"] = model_path
            return TeamTransformerAgent(**final_config)

        case "best_rl":
            model_path = find_best_model("rl")
            if not model_path:
                print("Warning: No RL model found for 'best_rl' agent. Using random initialization.")
                return TeamTransformerAgent(**agent_config)
            
            # Try to load config from run directory
            run_dir = Path(model_path).parent
            config_path = run_dir / "config.yaml"
            
            final_config = agent_config.copy()
            if config_path.exists():
                print(f"Loading agent config from {config_path}")
                saved_cfg = OmegaConf.load(config_path)
                # Extract transformer config
                transformer_cfg = OmegaConf.to_container(saved_cfg.train.model.transformer, resolve=True)
                final_config.update(transformer_cfg)
            else:
                print(f"Warning: No config.yaml found in {run_dir}. Using current defaults.")
            
            final_config["model_path"] = model_path
            return TeamTransformerAgent(**final_config)
        case _:
            raise TypeError(f"Unknown agent type: : {agent_type}")
