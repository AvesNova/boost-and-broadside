"""
Custom callbacks for training monitoring and self-play management.
"""

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import wandb


class SelfPlayCallback(BaseCallback):
    """
    Callback to manage self-play memory updates during training.
    """

    def __init__(
        self,
        env_wrapper,
        save_freq: int = 20000,
        min_save_steps: int = 50000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.env_wrapper = env_wrapper
        self.save_freq = save_freq
        self.min_save_steps = min_save_steps
        self.last_save_step = 0

    def _on_step(self) -> bool:
        """Called after each environment step"""

        # Save model to memory periodically
        if (
            self.num_timesteps >= self.min_save_steps
            and self.num_timesteps - self.last_save_step >= self.save_freq
        ):

            # Extract the transformer model from the SB3 policy
            transformer_model = self.model.policy.get_transformer_model()

            # Get the actual environment from Monitor wrapper
            actual_env = (
                self.env_wrapper.env
                if hasattr(self.env_wrapper, "env")
                else self.env_wrapper
            )

            # Add to self-play memory
            if hasattr(actual_env, "add_model_to_memory"):
                actual_env.add_model_to_memory(transformer_model)

                self.last_save_step = self.num_timesteps

                if self.verbose >= 1:
                    print(
                        f"Added model to self-play memory at step {self.num_timesteps}"
                    )
                    # Access selfplay_opponent through the actual environment
                    if hasattr(actual_env, "selfplay_opponent"):
                        print(
                            f"Memory size: {len(actual_env.selfplay_opponent.model_memory)}"
                        )
                    print(f"Win rate: {actual_env.get_win_rate():.3f}")

        # Log metrics
        # Get the actual environment from Monitor wrapper
        actual_env = (
            self.env_wrapper.env
            if hasattr(self.env_wrapper, "env")
            else self.env_wrapper
        )
        if hasattr(actual_env, "get_win_rate"):
            self.logger.record("self_play/win_rate", actual_env.get_win_rate())
            if hasattr(actual_env, "selfplay_opponent"):
                self.logger.record(
                    "self_play/memory_size",
                    len(actual_env.selfplay_opponent.model_memory),
                )
            self.logger.record("self_play/episode_count", actual_env.episode_count)

        return True


class EvalAgainstScriptedCallback(BaseCallback):
    """
    Callback to evaluate against scripted agents periodically.
    """

    def __init__(
        self,
        eval_freq: int = 50000,
        n_eval_episodes: int = 20,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.last_eval_step = 0

    def _on_step(self) -> bool:
        """Called after each environment step"""

        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self._evaluate_against_scripted()
            self.last_eval_step = self.num_timesteps

        return True

    def _evaluate_against_scripted(self):
        """Run evaluation against scripted agents"""
        from .rl_wrapper import UnifiedRLWrapper

        # Create evaluation environment (vs scripted only)
        eval_env = UnifiedRLWrapper(
            env_config={
                "world_size": (1200, 800),
                "max_ships": 4,
                "agent_dt": 0.04,
                "physics_dt": 0.02,
            },
            team_id=0,
            team_assignments={0: [0, 1], 1: [2, 3]},
            opponent_type="scripted",
            scripted_mix_ratio=1.0,  # Always scripted
        )

        try:
            # Evaluate the model
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
                return_episode_rewards=True,
            )

            # Calculate win rate (assume positive reward = win)
            wins = sum(1 for r in episode_rewards if r > 0)
            win_rate = wins / len(episode_rewards)

            # Log results
            self.logger.record("eval_scripted/mean_reward", np.mean(episode_rewards))
            self.logger.record("eval_scripted/std_reward", np.std(episode_rewards))
            self.logger.record("eval_scripted/mean_ep_length", np.mean(episode_lengths))
            self.logger.record("eval_scripted/win_rate", win_rate)

            if self.verbose >= 1:
                print(f"Scripted eval at step {self.num_timesteps}:")
                print(f"  Win rate: {win_rate:.3f}")
                print(f"  Mean reward: {np.mean(episode_rewards):.2f}")
                print(f"  Mean episode length: {np.mean(episode_lengths):.1f}")

            # Log to wandb if available
            if wandb.run:
                wandb.log(
                    {
                        "eval_scripted/mean_reward": np.mean(episode_rewards),
                        "eval_scripted/win_rate": win_rate,
                        "eval_scripted/mean_ep_length": np.mean(episode_lengths),
                        "timesteps": self.num_timesteps,
                    }
                )

        except Exception as e:
            print(f"Error during scripted evaluation: {e}")
        finally:
            eval_env.close()


class WandbCallback(BaseCallback):
    """
    Callback for logging to Weights & Biases.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        """Called after each environment step"""

        # Only log if wandb is initialized
        if not wandb.run:
            return True

        # Log training metrics every 1000 steps
        if self.num_timesteps % 1000 == 0:
            # Get metrics from logger
            log_data = {}

            # Add timesteps
            log_data["timesteps"] = self.num_timesteps

            # Log rollout metrics if available
            if hasattr(self.model, "_last_obs") and hasattr(
                self.training_env, "get_attr"
            ):
                try:
                    # Get environment statistics
                    win_rates = self.training_env.get_attr("get_win_rate")
                    if win_rates:
                        log_data["train/win_rate"] = np.mean([wr() for wr in win_rates])

                    episode_counts = self.training_env.get_attr("episode_count")
                    if episode_counts:
                        log_data["train/episode_count"] = np.mean(episode_counts)

                except Exception:
                    pass  # Ignore errors in metric collection

            # Log learning rate
            if hasattr(self.model, "learning_rate"):
                if callable(self.model.learning_rate):
                    log_data["train/learning_rate"] = self.model.learning_rate(1.0)
                else:
                    log_data["train/learning_rate"] = self.model.learning_rate

            # Log if we have data
            if log_data:
                wandb.log(log_data)

        return True


class DetailedLoggingCallback(BaseCallback):
    """
    Callback for detailed logging of training metrics.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        """Called after each environment step"""

        if self.num_timesteps % self.log_freq == 0:
            self._log_detailed_metrics()

        return True

    def _log_detailed_metrics(self):
        """Log detailed training metrics"""

        # Model-specific metrics
        if hasattr(self.model.policy, "features_extractor"):
            # Log transformer attention weights if available
            transformer = self.model.policy.get_transformer_model()

            # You could add attention visualization here
            # For now, just log model info
            total_params = sum(p.numel() for p in transformer.parameters())
            trainable_params = sum(
                p.numel() for p in transformer.parameters() if p.requires_grad
            )

            self.logger.record("model/total_params", total_params)
            self.logger.record("model/trainable_params", trainable_params)

        # Training progress
        if hasattr(self.model, "num_timesteps"):
            progress = self.model.num_timesteps / getattr(
                self.model, "_total_timesteps", 1
            )
            self.logger.record("train/progress", progress)

        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            self.logger.record("system/gpu_memory_allocated_gb", memory_allocated)
            self.logger.record("system/gpu_memory_reserved_gb", memory_reserved)
