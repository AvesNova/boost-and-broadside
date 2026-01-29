import torch
import gymnasium as gym
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from agents.sb3_adapter import TeamTransformerSB3Policy
from env2.sb3_wrapper import SB3Wrapper
from env2.coordinator_wrapper import TensorEnvWrapper


def create_sb3_env(cfg: DictConfig) -> gym.Env:
    """
    Helper function to create the SB3 environment.

    Args:
        cfg: Configuration dictionary.

    Returns:
        Wrapped Gymnasium environment.
    """
    env_config = dict(cfg.environment)
    env_config["render_mode"] = "none"  # Force no rendering for training

    base_env = TensorEnvWrapper(**env_config)
    return SB3Wrapper(base_env, cfg)


def train_rl(cfg: DictConfig, pretrained_model_path: Path | None = None) -> None:
    """
    RL training using Stable Baselines3 PPO.

    Args:
        cfg: Configuration dictionary.
        pretrained_model_path: Path to pretrained BC model weights (optional).
    """
    print("\nStarting RL training...")

    rl_config = cfg.train.rl

    # Access required params to ensure they exist
    _ = rl_config.learning_rate
    _ = rl_config.n_steps
    _ = rl_config.batch_size
    _ = rl_config.n_epochs
    _ = rl_config.gamma
    _ = rl_config.gae_lambda
    _ = rl_config.clip_range
    _ = rl_config.ent_coef
    _ = rl_config.vf_coef
    _ = rl_config.max_grad_norm
    _ = rl_config.total_timesteps

    # Create environment
    n_envs = rl_config.get(
        "n_envs", 1
    )  # This one can have a default as it's often optional/local
    print(f"Creating {n_envs} parallel environments...")

    # Check if we need FrameStack (for World Model)
    use_world_model = rl_config.get("policy_type", "transformer") == "world_model"
    stack_size = rl_config.get("context_len", 1) if use_world_model else 0

    def create_env_with_wrapper(cfg):
        env = create_sb3_env(cfg)
        if use_world_model:
            # SB3 env returns Dict observations: {"tokens": ...}
            # FrameStack works on Box/Discrete, but for Dict it stacks each key?
            # Gym's FrameStack supports Dict since v0.26+ approx?
            # Actually standard gym FrameStack might throw on Dict.
            # However, SB3 VecFrameStack handles it.
            # But making it here allows per-env stacking.

            # Let's rely on VecFrameStack if possible, but VecFrameStack from SB3 stacks primarily on Channel dim for Images.
            # We want simple stacking.
            # Let's use `gym.wrappers.FrameStack`.
            # IMPORTANT: SB3 env wrapper returns Dict `{"tokens": (N,F)}`.
            # FrameStack will make it `{"tokens": (k, N, F)}`.
            # This is exactly what we want.
            return gym.wrappers.FrameStackObservation(env, stack_size)
        return env

    env = make_vec_env(
        create_env_with_wrapper,
        n_envs=n_envs,
        env_kwargs={"cfg": cfg},
        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("models/rl") / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"RL Output directory: {run_dir}")

    # Save config immediately
    OmegaConf.save(cfg, run_dir / "config.yaml")

    # Initialize PPO
    if use_world_model:
        from train.ppo_world_model import WorldModelPPO
        from agents.sb3_world_model_adapter import WorldModelSB3Policy

        # World Model Config
        wm_config = OmegaConf.to_container(cfg.world_model, resolve=True)
        # Update with overrides from RL config if any (e.g. context_len)
        if "context_len" in rl_config:
            wm_config["context_len"] = rl_config.context_len

        policy_kwargs = {
            "model_config": wm_config,
            "freeze_backbone": rl_config.get("freeze_world_model", False),
            "aux_loss_config": {
                "mask_ratio": wm_config.get("mask_ratio", 0.15),
                "noise_scale": wm_config.get("noise_scale", 0.1),
            },
        }

        AlgorithmClass = WorldModelPPO
        PolicyClass = WorldModelSB3Policy
        extra_algorithm_kwargs = {"aux_loss_coef": rl_config.get("aux_loss_coef", 1.0)}

        print("Using WorldModelPPO with WorldModelSB3Policy")

    else:
        # Standard Transformer Policy
        AlgorithmClass = PPO
        PolicyClass = TeamTransformerSB3Policy

        model_config = OmegaConf.to_container(cfg.train.model.transformer, resolve=True)
        policy_kwargs = {
            "model_config": model_config,
        }
        extra_algorithm_kwargs = {}
        print("Using Standard PPO with TeamTransformerSB3Policy")

    model = AlgorithmClass(
        PolicyClass,
        env,
        learning_rate=rl_config.learning_rate,
        n_steps=rl_config.n_steps,
        batch_size=rl_config.batch_size,
        n_epochs=rl_config.n_epochs,
        gamma=rl_config.gamma,
        gae_lambda=rl_config.gae_lambda,
        clip_range=rl_config.clip_range,
        ent_coef=rl_config.ent_coef,
        vf_coef=rl_config.vf_coef,
        max_grad_norm=rl_config.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(run_dir),
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        **extra_algorithm_kwargs,
    )

    # Load pretrained weights if available
    if pretrained_model_path and pretrained_model_path.exists():
        print(f"Loading pretrained weights from {pretrained_model_path}")
        try:
            state_dict = torch.load(pretrained_model_path, map_location=model.device)

            if use_world_model:
                # Load into world_model
                model.policy.world_model.load_state_dict(state_dict, strict=False)
                # Note: value head will remain random / initialized
            else:
                model.policy.transformer_model.load_state_dict(state_dict)

            print("Successfully loaded pretrained weights")
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=rl_config.n_steps,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="rl_model",
    )

    # Train
    total_timesteps = rl_config.total_timesteps
    print(f"Training for {total_timesteps} timesteps...")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # Save final model
        final_path = run_dir / "final_rl_model.zip"
        model.save(final_path)
        print(f"Saved final RL model to {final_path}")

        # Also save the inner transformer state dict for compatibility with our agents
        if use_world_model:
            torch.save(
                model.policy.world_model.state_dict(),
                run_dir / "final_world_model.pth",
            )
            print(f"Saved World Model weights to {run_dir / 'final_world_model.pth'}")
        else:
            torch.save(
                model.policy.transformer_model.state_dict(),
                run_dir / "final_rl_transformer.pth",
            )
            print(
                f"Saved transformer weights to {run_dir / 'final_rl_transformer.pth'}"
            )

        env.close()
