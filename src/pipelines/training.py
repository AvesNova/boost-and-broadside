"""
Training pipeline module - handles BC, RL, and full training pipelines
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

from omegaconf import DictConfig, OmegaConf

# Import shared utilities
# Removed old config import - now using Hydra DictConfig
from ..utils import (
    setup_directories,
    setup_logging,
    generate_run_name,
    save_config_copy,
    InterruptHandler,
)
from ..model_utils import (
    load_model,
    save_model,
    create_model,
    create_ppo_model,
    create_bc_model,
    transfer_weights,
    ModelMetadata,
)
from ..cli_args import (
    get_training_arguments,
    get_rl_training_arguments,
    get_full_training_arguments,
)

# Import existing training functions
from ..bc_training import train_bc_model
from ..rl_wrapper import create_unified_rl_env
from ..transformer_policy import create_team_ppo_model

# Import parallel processing utilities
from ..parallel_rl_wrapper import create_parallel_rl_env


class TrainingPipeline:
    """Handles all training-related operations"""

    @staticmethod
    def add_subparsers(subparsers):
        """Add training subcommands to the argument parser"""
        train_parser = subparsers.add_parser("train", help="Training operations")
        train_subparsers = train_parser.add_subparsers(
            dest="train_command", help="Training command"
        )

        # BC training
        bc_parser = train_subparsers.add_parser("bc", help="Behavior cloning training")
        for arg in get_training_arguments():
            bc_parser.add_argument(*arg["args"], **arg["kwargs"])

        # RL training
        rl_parser = train_subparsers.add_parser(
            "rl", help="Reinforcement learning training"
        )
        for arg in get_rl_training_arguments():
            rl_parser.add_argument(*arg["args"], **arg["kwargs"])

        # Full pipeline
        full_parser = train_subparsers.add_parser(
            "full", help="Complete training pipeline"
        )
        for arg in get_full_training_arguments():
            full_parser.add_argument(*arg["args"], **arg["kwargs"])

        return train_parser

    @staticmethod
    def execute(cfg: DictConfig) -> int:
        """Execute the appropriate training command with DictConfig"""
        try:
            with InterruptHandler("Training interrupted by user"):
                # If config is nested under 'train', extract it
                if "train" in cfg and isinstance(cfg.train, DictConfig):
                    cfg = cfg.train

                # Get command from config or from command structure
                train_command = None

                # Try to get from train_command first (directly from config)
                if "train_command" in cfg:
                    train_command = cfg.train_command
                # Try to get from command (from CLI args)
                elif "command" in cfg:
                    train_command = cfg.command
                # Fallback to training.mode
                elif cfg.get("training", {}).get("mode"):
                    train_command = cfg.training.mode

                if train_command == "bc":
                    return TrainingPipeline._train_bc(cfg)
                elif train_command == "rl":
                    return TrainingPipeline._train_rl(cfg)
                elif train_command == "full":
                    return TrainingPipeline._train_full(cfg)
                else:
                    print(f"Unknown training command: {train_command}")
                    return 1
        except Exception as e:
            print(f"Error during training: {e}")
            return 1

    @staticmethod
    def _train_bc(cfg: DictConfig) -> int:
        """Execute behavior cloning training with DictConfig"""
        # Generate run name if not provided
        run_name = cfg.get("run_name") or generate_run_name("bc")

        print("=" * 60)
        print("PHASE 1: BEHAVIOR CLONING PRETRAINING")
        print("=" * 60)
        print(f"Run name: {run_name}")

        # Setup directories and logging
        checkpoint_dir, log_dir = setup_directories(f"{run_name}_bc")
        logger = setup_logging(run_name, log_dir=log_dir)

        # Save config copy
        save_config_copy(OmegaConf.to_container(cfg, resolve=True), checkpoint_dir)

        # Check if BC data exists
        data_dir = Path(cfg.data_collection.bc_data.output_dir)
        data_files = list(data_dir.glob("*.pkl*"))

        if not data_files:
            print("No BC training data found. Please run data collection first:")
            print("python main.py collect bc")
            return 1

        print(f"Found BC training data: {len(data_files)} files")

        # Create BC model
        model_config = cfg.model.transformer
        bc_model = create_bc_model(model_config)

        # Train BC model
        bc_config = cfg.model.bc
        trained_model_path = train_bc_model(
            model=bc_model,
            data_files=data_files,
            config=bc_config,
            output_dir=checkpoint_dir,
            run_name=f"{run_name}_bc",
        )

        # Create and save metadata
        metadata = ModelMetadata(
            model_type="bc",
            config=model_config,
            training_stats={"final_model_path": trained_model_path},
            description=f"Behavior cloning model trained with {len(data_files)} episodes",
        )
        save_model(bc_model, trained_model_path, metadata)

        print(f"BC pretraining complete! Model saved to: {trained_model_path}")
        return 0

    @staticmethod
    def _train_rl(cfg: DictConfig) -> int:
        """Execute reinforcement learning training with DictConfig"""
        # Generate run name if not provided
        run_name = cfg.get("run_name") or generate_run_name("rl")

        print("=" * 60)
        print("PHASE 2: REINFORCEMENT LEARNING TRAINING")
        print("=" * 60)
        print(f"Run name: {run_name}")

        bc_model = cfg.get("bc_model")
        if bc_model:
            print(f"Initializing from BC model: {bc_model}")
        else:
            print("Training from scratch")

        # Setup directories and logging
        checkpoint_dir, log_dir = setup_directories(f"{run_name}_rl")
        logger = setup_logging(run_name, log_dir=log_dir)

        # Save config copy
        save_config_copy(OmegaConf.to_container(cfg, resolve=True), checkpoint_dir)

        # Create training environment
        env_config = cfg.environment
        training_config = cfg.training.rl

        # Check if parallel processing is enabled
        parallel_config = cfg.get("parallel_processing", {})
        if parallel_config.get("enabled", False):
            # Use parallel environments
            num_envs = parallel_config["rl_training"]["num_envs"]
            print(f"Using parallel RL training with {num_envs} environments")

            train_env = create_parallel_rl_env(
                env_config=env_config,
                num_envs=num_envs,
                learning_team_id=training_config["learning_team_id"],
                opponent_config=training_config["opponent"],
            )

            # Wrap for SB3
            from stable_baselines3.common.vec_env import SubprocVecEnv

            def make_env():
                return create_unified_rl_env(
                    env_config=env_config,
                    learning_team_id=training_config["learning_team_id"],
                    opponent_config=training_config["opponent"],
                )

            train_env = SubprocVecEnv([make_env for _ in range(num_envs)])
        else:
            # Use single environment
            train_env = create_unified_rl_env(
                env_config=env_config,
                learning_team_id=training_config["learning_team_id"],
                opponent_config=training_config["opponent"],
            )

            # Wrap environments for SB3
            from stable_baselines3.common.monitor import Monitor
            from stable_baselines3.common.vec_env import DummyVecEnv

            train_env = Monitor(train_env, str(log_dir / "train"))
            train_env = DummyVecEnv([lambda: train_env])

        # Create evaluation environment (always vs scripted for consistent metrics)
        eval_opponent_config = OmegaConf.to_container(
            training_config["opponent"], resolve=True
        )
        eval_opponent_config["type"] = "scripted"
        eval_opponent_config["scripted_mix_ratio"] = 1.0

        eval_env = create_unified_rl_env(
            env_config=env_config,
            learning_team_id=training_config["learning_team_id"],
            opponent_config=eval_opponent_config,
        )

        # Wrap evaluation environment
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv

        eval_env = Monitor(eval_env, str(log_dir / "eval"))
        eval_env = DummyVecEnv([lambda: eval_env])

        # Create PPO model
        model_config = cfg.model.transformer
        ppo_config = cfg.model.ppo
        team_assignments = {0: [0, 1], 1: [2, 3]}  # Will be updated by nvn

        if bc_model:
            # Initialize PPO with BC weights
            model = TrainingPipeline._create_ppo_from_bc(
                train_env,
                bc_model,
                model_config,
                training_config["learning_team_id"],
                ppo_config,
                team_assignments,
            )
        else:
            # Create fresh PPO model
            model = create_ppo_model(
                env=train_env,
                transformer_config=model_config,
                team_id=training_config["learning_team_id"],
                team_assignments=team_assignments,
                ppo_config=ppo_config,
            )

        # Setup callbacks with periodic checkpointing
        checkpoint_freq = parallel_config.get("rl_training", {}).get(
            "checkpoint_frequency", 25000
        )

        # Convert training_config to dict for callbacks
        training_config_dict = OmegaConf.to_container(training_config, resolve=True)
        training_config_dict["checkpoint_freq"] = checkpoint_freq

        callbacks = TrainingPipeline._setup_callbacks(
            training_config_dict, checkpoint_dir, log_dir, train_env
        )

        # Ensure GPU utilization
        import torch

        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            print(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )

            # Set GPU memory fraction if specified
            gpu_memory_fraction = parallel_config.get("rl_training", {}).get(
                "gpu_memory_fraction", 0.9
            )
            if gpu_memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
                print(f"GPU memory fraction set to {gpu_memory_fraction}")

        # Start training
        print(
            f"Starting RL training for {training_config['total_timesteps']:,} timesteps..."
        )

        try:
            from stable_baselines3 import PPO

            model.learn(
                total_timesteps=training_config["total_timesteps"],
                callback=callbacks,
                progress_bar=True,
            )

            # Save final model
            final_model_path = checkpoint_dir / "final_rl_model"
            model.save(str(final_model_path))

            # Create and save metadata
            metadata = ModelMetadata(
                model_type="ppo",
                config={
                    "transformer": model_config,
                    "ppo": ppo_config,
                    "team_id": training_config["learning_team_id"],
                    "team_assignments": team_assignments,
                },
                training_stats={
                    "total_timesteps": training_config["total_timesteps"],
                    "bc_model": bc_model,
                },
                description=f"PPO model trained for {training_config['total_timesteps']} timesteps",
            )
            save_model(model, str(final_model_path), metadata)

            print(f"RL training complete! Final model saved to: {final_model_path}")
            return 0

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            interrupted_path = checkpoint_dir / "interrupted_rl_model"
            model.save(str(interrupted_path))
            print(f"Interrupted model saved to: {interrupted_path}")
            return 1

        finally:
            train_env.close()
            eval_env.close()

    @staticmethod
    def _train_full(cfg: DictConfig) -> int:
        """Execute the complete training pipeline with DictConfig"""
        # Generate run name if not provided
        run_name = cfg.get("run_name") or generate_run_name("full")

        print("=" * 60)
        print(f"UNIFIED TRAINING PIPELINE: {run_name}")
        print("=" * 60)

        bc_model_path = None

        # Phase 1: BC Pretraining (optional)
        if not cfg.get("skip_bc", False):
            # Create a modified config for BC training
            bc_cfg = cfg.copy()
            bc_cfg.run_name = f"{run_name}_bc"
            bc_cfg.training.mode = "bc"

            result = TrainingPipeline._train_bc(bc_cfg)
            if result != 0:
                print("BC training failed or skipped")
            else:
                bc_model_path = f"checkpoints/{run_name}_bc/best_bc_model.pt"
        else:
            print("Skipping BC pretraining phase")

        # Phase 2: RL Training
        # Create a modified config for RL training
        rl_cfg = cfg.copy()
        rl_cfg.run_name = f"{run_name}_rl"
        rl_cfg.training.mode = "rl"
        if bc_model_path:
            rl_cfg.bc_model = bc_model_path

        result = TrainingPipeline._train_rl(rl_cfg)
        if result != 0:
            return result

        # Phase 3: Final Evaluation
        print("=" * 60)
        print("FINAL MODEL EVALUATION")
        print("=" * 60)

        rl_model_path = f"checkpoints/{run_name}_rl/final_rl_model"
        TrainingPipeline._evaluate_final_model(rl_model_path, cfg, run_name)

        print("=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"BC Model: {bc_model_path or 'None'}")
        print(f"RL Model: {rl_model_path}")

        return 0

    @staticmethod
    def _create_ppo_from_bc(
        env,
        bc_model_path: str,
        transformer_config: dict,
        team_id: int,
        ppo_config: dict,
        team_assignments: dict,
    ):
        """Create PPO model initialized with BC weights"""
        model = create_ppo_model(
            env=env,
            transformer_config=transformer_config,
            team_id=team_id,
            team_assignments=team_assignments,
            ppo_config=ppo_config,
        )

        # Load BC model
        bc_model = load_model(bc_model_path, "bc")

        # Transfer weights from BC model to PPO policy
        try:
            # Get the transformer from PPO policy
            ppo_transformer = model.policy.get_transformer_model()

            # Copy weights from BC model
            success = transfer_weights(bc_model, ppo_transformer, strict=False)

            if success:
                print("Successfully initialized PPO with BC weights")
            else:
                print(
                    "Warning: Could not transfer BC weights, using random initialization"
                )

        except Exception as e:
            print(f"Warning: Could not transfer BC weights: {e}")
            print("Continuing with random initialization")

        return model

    @staticmethod
    def _setup_callbacks(training_config, checkpoint_dir, log_dir, train_env):
        """Setup training callbacks with periodic checkpointing"""
        from stable_baselines3.common.callbacks import (
            EvalCallback,
            CheckpointCallback,
            BaseCallback,
        )
        from ..callbacks import SelfPlayCallback
        import torch
        import os

        callbacks = []

        # Enhanced checkpoint callback with GPU memory management
        class EnhancedCheckpointCallback(CheckpointCallback):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def _on_step(self) -> bool:
                result = super()._on_step()

                # Clear GPU cache after checkpointing
                if result and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                return result

        # Checkpoint callback
        checkpoint_freq = training_config.get("checkpoint_freq", 25000)
        checkpoint_callback = EnhancedCheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(checkpoint_dir),
            name_prefix="rl_model",
            save_replay_buffer=True,
        )
        callbacks.append(checkpoint_callback)

        # GPU memory monitoring callback
        class GPUMemoryCallback(BaseCallback):
            def __init__(self, verbose: int = 0):
                super().__init__(verbose)

            def _on_step(self) -> bool:
                if torch.cuda.is_available() and self.n_calls % 1000 == 0:
                    memory_allocated = torch.cuda.memory_allocated() / 1e9
                    memory_reserved = torch.cuda.memory_reserved() / 1e9

                    if self.verbose > 0:
                        print(
                            f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB"
                        )

                return True

        gpu_callback = GPUMemoryCallback(verbose=1)
        callbacks.append(gpu_callback)

        # Evaluation callback
        eval_callback = EvalCallback(
            train_env.envs[
                0
            ],  # Use train_env for evaluation since eval_env isn't properly set up
            best_model_save_path=str(checkpoint_dir / "best_model"),
            log_path=str(log_dir),
            eval_freq=training_config["eval_freq"],
            n_eval_episodes=training_config["eval_episodes"],
            deterministic=True,
        )
        callbacks.append(eval_callback)

        # Self-play callback (if using self-play)
        if training_config["opponent"]["type"] in ["self_play", "mixed"]:
            selfplay_callback = SelfPlayCallback(
                env_wrapper=train_env.envs[0],
                save_freq=training_config["selfplay_update_freq"],
                min_save_steps=training_config["min_steps_before_selfplay"],
            )
            callbacks.append(selfplay_callback)

        return callbacks

    @staticmethod
    def _evaluate_final_model(model_path: str, cfg: DictConfig, run_name: str):
        """Evaluate the final trained model with DictConfig"""
        from ..collect_data import evaluate_model

        eval_config = OmegaConf.to_container(cfg.evaluation, resolve=True)
        eval_config["model_config"] = OmegaConf.to_container(
            cfg.model.transformer, resolve=True
        )

        stats = evaluate_model(model_path, eval_config)

        # Save evaluation results
        from pathlib import Path
        import yaml

        results_dir = Path(f"results/{run_name}")
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / "final_evaluation.yaml"
        with open(results_file, "w") as f:
            yaml.dump(stats, f)

        print(f"Evaluation results saved to: {results_file}")
