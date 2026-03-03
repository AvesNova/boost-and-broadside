
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import wandb
from omegaconf import DictConfig, OmegaConf

from boost_and_broadside.train.rl.buffer import GPUBuffer
from boost_and_broadside.core.constants import StateFeature
from boost_and_broadside.train.batch_processor import BatchProcessor
# We assume YemongDynamics is importable and has the right interface
# from boost_and_broadside.models.yemong.scaffolds import YemongDynamics

class PPOTrainer:
    def __init__(self, cfg: DictConfig, env, agent):
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.batch_size = int(cfg.train.ppo.num_envs * cfg.train.ppo.num_steps)
        self.minibatch_size = int(self.batch_size // cfg.train.ppo.num_minibatches)
        self.num_iterations = cfg.train.ppo.total_timesteps // self.batch_size
        
        # Experiment Tracking
        self.run_name = f"{cfg.project_name}_{cfg.mode}_{int(time.time())}"
        if cfg.wandb.enabled:
             wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                config=OmegaConf.to_container(cfg, resolve=True),
                name=self.run_name,
                monitor_gym=False,
                save_code=True
             )
             wandb.watch(self.agent, log="all", log_freq=100)
        self.writer = SummaryWriter(f"runs/{self.run_name}")
        
        # Optimization
        # Separate LR for Mamba if needed, but for now standard Adam
        self.optimizer = optim.Adam(
            self.agent.parameters(), 
            lr=cfg.train.ppo.learning_rate, 
            eps=1e-5
        )
        
        # Buffer Initialization using Environment Sample
        print("Initializing Buffer...")
        sample_obs, _ = self.env.reset()
        obs_shapes = {}
        for k, v in sample_obs.items():
            # v is tensor (num_envs, ...)
            # We want shape (...)
            obs_shapes[k] = v.shape[1:] 
            
        print(f"Observation Shapes: {obs_shapes}")
        
        # Determine max_ships from state shape
        if 'state' in sample_obs:
             # state shape is (B, N, F)
             self.max_ships = sample_obs['state'].shape[1]
        else:
             self.max_ships = cfg.model.max_ships

        # Action Shape
        # Yemong outputs (B, N, 3) for sampled actions
        # We assume max_ships is dim 1 of state?
        # Or we can infer from sample_obs['prev_action'] if available
        if 'prev_action' in sample_obs:
             action_shape = sample_obs['prev_action'].shape[1:]
        else:
             # Fallback 
             action_shape = (cfg.model.max_ships, 3)

        self.buffer = GPUBuffer(
            num_steps=cfg.train.ppo.num_steps,
            num_envs=cfg.train.ppo.num_envs,
            obs_shapes=obs_shapes,
            action_shape=action_shape,
            gamma=cfg.train.ppo.gamma,
            gae_lambda=cfg.train.ppo.gae_lambda,
            device=self.device
        )
        
    def train(self):
        # Initial Reset
        next_obs, _ = self.env.reset()
        
        # Init Mamba State
        conv_state_map = self.agent.allocate_inference_cache(
            batch_size=self.cfg.train.ppo.num_envs * self.max_ships, 
            max_seqlen=self.cfg.train.ppo.num_steps
        )
        
        # We need to structure the state as {layer: (conv, ssm)}
        # allocate_inference_cache returns exactly that (from my edit to scaffolds.py)
        next_mamba_state = conv_state_map # It is a dict {i: (conv, ssm)}
        
        next_done = torch.zeros(self.cfg.train.ppo.num_envs, device=self.device)
        
        global_step = 0
        start_time = time.time()
        
        for iteration in range(1, self.num_iterations + 1):
            
            # --- ROLLOUT PHASE ---
            self.buffer.reset()
            
            # 1. Reshape Mamba State for Storage: (B*N, ...) -> (B, N, ...)
            # This allows Buffer to slice by Env Index correctly
            stored_state = {}
            for l_idx, (conv, ssm) in next_mamba_state.items():
                # conv: (B*N, D, W) -> (B, N, D, W)
                c_vs = conv.view(self.cfg.train.ppo.num_envs, self.max_ships, *conv.shape[1:])
                # ssm: (B*N, D, N) -> (B, N, D, N)
                s_vs = ssm.view(self.cfg.train.ppo.num_envs, self.max_ships, *ssm.shape[1:])
                stored_state[l_idx] = (c_vs, s_vs)
                
            self.buffer.store_initial_state(stored_state)
            
            # Anneal LR
            if self.cfg.train.ppo.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lrnow = frac * self.cfg.train.ppo.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            # Accumulators for Episode Stats
            iter_episode_returns = []
            iter_episode_lengths = []

            for step in range(0, self.cfg.train.ppo.num_steps):
                global_step += self.cfg.train.ppo.num_envs
                
                with torch.no_grad():
                    # RECURRENT INFERENCE
                    # Add Time dimension (T=1) for Mamba
                    obs_input = {k: v.unsqueeze(1) for k, v in next_obs.items()}
                    
                    if getattr(self.agent, 'is_interleaved', False):
                        # Step 1: Predict Action and Value from State
                        action, logprob, _, value, next_mamba_state, *_ = self.agent.get_action_and_value(
                            obs_input, next_mamba_state, step_type="state"
                        )
                        action_sq = action.squeeze(1)
                        logprob_sq = logprob.squeeze(1)
                        value_sq = value.squeeze(1)
                        
                    else:
                        action, logprob, _, value, next_mamba_state, *_ = self.agent.get_action_and_value(
                            obs_input, next_mamba_state
                        )
                        action_sq = action.squeeze(1)
                        logprob_sq = logprob.squeeze(1)
                        value_sq = value.squeeze(1)
                    
                # Execute Step
                next_obs_new, reward, terminated, truncated, info = self.env.step(action_sq)
                done = torch.logical_or(terminated, truncated).float()
                
                with torch.no_grad():
                    if getattr(self.agent, 'is_interleaved', False):
                        # Step 2: Update Mamba memory with Action token
                        _, _, _, _, next_mamba_state, *_ = self.agent.get_action_and_value(
                            obs_input, next_mamba_state, action=action, step_type="action"
                        )
                
                # Track Episodic Return
                if "final_info" in info:
                    for item in info["final_info"]:
                        if "episode" in item:
                            ep_ret = item["episode"]["r"]
                            ep_len = item["episode"]["l"]
                            iter_episode_returns.append(ep_ret)
                            iter_episode_lengths.append(ep_len)

                # Buffer Add
                self.buffer.add(next_obs, action_sq, logprob_sq, reward, next_done, value_sq)
                
                next_obs = next_obs_new
                next_done = done
                
                # Mamba State Reset on Done
                # next_mamba_state is (B*N, ...).
                # done is (B,). We need to mask all ships for done envs.
                if next_done.any():
                    done_indices = torch.nonzero(next_done, as_tuple=True)[0]
                    # Map done envs to flat indices: range(env*N, (env+1)*N)
                    # Or reshape, mask, flatten.
                    for layer_idx in next_mamba_state:
                         conv, ssm = next_mamba_state[layer_idx]
                         # Reshape to (B, N, ...)
                         c_view = conv.view(self.cfg.train.ppo.num_envs, self.max_ships, *conv.shape[1:])
                         s_view = ssm.view(self.cfg.train.ppo.num_envs, self.max_ships, *ssm.shape[1:])
                         
                         c_view[done_indices] = 0
                         s_view[done_indices] = 0
                         # No need to flatten back, view shares memory

            # --- BOOTSTRAP PHASE ---
            with torch.no_grad():
                obs_input = {k: v.unsqueeze(1) for k, v in next_obs.items()}
                _, _, _, next_value, _ = self.agent.get_action_and_value(obs_input, next_mamba_state)
                next_value = next_value.squeeze(1)
                # Compute GAE
                self.buffer.compute_gae(next_value, next_done)
                
            # --- UPDATE PHASE ---
            # Accumulators for Loss Stats
            clipfracs = []
            loss_stats = {
                "pg_loss": [],
                "v_loss": [],
                "entropy_loss": [],
                "old_approx_kl": [],
                "approx_kl": [],
                "s_loss": [],
                "r_loss": [],
                "explained_var": []
            }
            
            # Explained Variance (calculated once per update usually, but can average)
            y_pred, y_true = self.buffer.values.flatten().cpu().numpy(), self.buffer.returns.flatten().cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            loss_stats["explained_var"].append(explained_var)
            
            for epoch in range(self.cfg.train.ppo.update_epochs):
                
                iterator = self.buffer.get_minibatch_iterator(self.cfg.train.ppo.num_minibatches)
                
                for mb_obs, mb_next_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values, mb_dones, mb_rewards, mb_initial_state in iterator:
                    
                    # mb_obs is (T, B_chunk, N, F).
                    # We need (B_chunk, T, N, F)
                    obs_input = {k: v.permute(1, 0, *range(2, v.ndim)).contiguous() for k, v in mb_obs.items()}
                    
                    # mb_initial_state: {idx: (conv, ssm)} where conv is (B_chunk, N, ...) from storage reshape
                    # Mamba update needs flattened state: (B_chunk*N, ...)
                    flat_mamba_state = {}
                    current_batch_size = next(iter(mb_obs.values())).shape[1] # B_chunk
                    
                    for l_idx, (conv, ssm) in mb_initial_state.items():
                         # conv: (B_chunk, N, ...) -> (B_chunk*N, ...)
                         c_flat = conv.reshape(current_batch_size * self.max_ships, *conv.shape[2:])
                         s_flat = ssm.reshape(current_batch_size * self.max_ships, *ssm.shape[2:])
                         flat_mamba_state[l_idx] = (c_flat, s_flat)

                    # mb_actions: (T, B, N, 3) -> (B, T, N, 3)
                    mb_actions_perm = mb_actions.permute(1, 0, 2, 3).contiguous()
                    
                    # Permute Targets to (B, T, ...)
                    # (T, B, N) -> (B, T, N)
                    mb_logprobs = mb_logprobs.permute(1, 0, 2).contiguous()
                    mb_advantages = mb_advantages.permute(1, 0, 2).contiguous()
                    mb_returns = mb_returns.permute(1, 0, 2).contiguous()
                    mb_values = mb_values.permute(1, 0, 2).contiguous()
                    
                    mb_rewards_perm = mb_rewards.permute(1, 0, 2).contiguous() # (B, T, N)
                    mb_next_obs_perm = mb_next_obs['state'].permute(1, 0, 2, 3).contiguous() # (B, T, N, F)
                    mb_dones_perm = mb_dones.permute(1, 0).contiguous().long() # (B, T)

                    # Construct seq_idx for Mamba2
                    # We want seq_idx to change at step t if t was a reset (start of new episode).
                    # mb_dones[t-1]=1 => step t is new episode => seq_idx[t] != seq_idx[t-1]
                    # So we shift dones right by 1.
                    dones_shifted = torch.roll(mb_dones_perm, 1, dims=1)
                    dones_shifted[:, 0] = 0 # First step state is handled by passing 'flat_mamba_state'
                    seq_idx = torch.cumsum(dones_shifted, dim=1).int()

                    # PARALLEL SCAN INFERENCE (entire sequence)
                    # Returns: action(None), logprob, entropy, value, mamba_state(None), next_state_pred, reward_pred
                    _, new_logprob, entropy, new_value, _, next_state_pred, reward_pred, *extras = self.agent.get_action_and_value(
                        obs_input, flat_mamba_state, action=mb_actions_perm, seq_idx=seq_idx
                    )
                    
                    logratio = new_logprob - mb_logprobs
                    ratio = logratio.exp()
                    
                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.cfg.train.ppo.clip_coef).float().mean().item()]
                        
                        loss_stats["old_approx_kl"].append(old_approx_kl.item())
                        loss_stats["approx_kl"].append(approx_kl.item())
                        
                    # Normalize Advantage
                    if self.cfg.train.ppo.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                        
                    # Policy Loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 
                        1 - self.cfg.train.ppo.clip_coef, 
                        1 + self.cfg.train.ppo.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    # Value Loss
                    v_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()
                    
                    # Entropy Loss
                    entropy_loss = entropy.mean()
                    
                    # --- Auxiliary Losses ---
                    # Use BatchProcessor for standardized targets
                    mb_curr_obs_perm = mb_obs['state'].permute(1, 0, 2, 3).contiguous()  # (B, T, N, F)
                    mb_next_obs_perm = mb_next_obs['state'].permute(1, 0, 2, 3).contiguous()
                    mb_curr_pos = mb_obs['pos'].permute(1, 0, 2, 3).contiguous()
                    mb_next_pos = mb_next_obs['pos'].permute(1, 0, 2, 3).contiguous()
                    
                    W, H = self.cfg.environment.world_size
                    target_states, target_pairwise = BatchProcessor.compute_targets(
                        mb_curr_obs_perm, mb_next_obs_perm, mb_curr_pos, mb_next_pos, W, H, self.device
                    )
                    
                    B, T, N, F_dim = mb_next_obs_perm.shape
                    
                    # Masks
                    valid_mask = (1.0 - mb_dones_perm.float())  # (B, T), zero at done steps
                    input_alive = mb_curr_obs_perm[..., StateFeature.HEALTH] > 0
                    target_alive = mb_next_obs_perm[..., StateFeature.HEALTH] > 0
                    
                    # Extract Pairwise and Components from extras if available
                    pairwise_pred = extras[0] if len(extras) > 0 else None
                    reward_components = extras[1] if len(extras) > 1 else None
                    
                    # Compute Auxiliary Losses via model.get_loss()
                    loss_out = self.agent.get_loss(
                        pred_states=next_state_pred.view(B, T, N, -1),
                        pred_actions=None, # PPO update doesn't output logits for Behavioral Cloning by default
                        target_states=target_states,
                        target_actions=mb_actions_perm,
                        loss_mask=valid_mask,
                        lambda_state=self.cfg.train.ppo.get("lambda_state", 0.5), # Configurable RL weights
                        lambda_actions=self.cfg.train.ppo.get("lambda_actions", 0.0), # Off by default
                        pred_values=new_value.unsqueeze(-1) if new_value.ndim == 2 else new_value,
                        pred_rewards=reward_pred,
                        target_returns=mb_returns[..., 0:1],
                        target_rewards=mb_rewards_perm[..., 0:1],
                        lambda_value=self.cfg.train.ppo.get("lambda_value", self.cfg.train.ppo.vf_coef),
                        lambda_reward=self.cfg.train.ppo.get("lambda_reward", 0.5),
                        weights_power=None, weights_turn=None, weights_shoot=None, weights_flat=None,
                        target_alive=target_alive,
                        input_alive=input_alive,
                        pairwise_pred=pairwise_pred,
                        target_pairwise=target_pairwise,
                        lambda_pairwise=self.cfg.train.ppo.get("lambda_pairwise", 0.0),
                        reward_components=reward_components
                    )
                    
                    s_loss = loss_out.get("state_loss", torch.tensor(0.0))
                    v_loss = loss_out.get("value_loss", torch.tensor(0.0))
                    r_loss = loss_out.get("reward_loss", torch.tensor(0.0))
                    
                    # Accumulate Stats
                    loss_stats["pg_loss"].append(pg_loss.item())
                    loss_stats["v_loss"].append(v_loss.item())
                    loss_stats["entropy_loss"].append(entropy_loss.item())
                    loss_stats["s_loss"].append(s_loss.item())
                    loss_stats["r_loss"].append(r_loss.item())

                    # Total Loss = PG + Entropy + Weighted Model Aux Losses (Value, Reward, State, etc)
                    loss = pg_loss - self.cfg.train.ppo.ent_coef * entropy_loss + loss_out["loss"]
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.cfg.train.ppo.max_grad_norm)
                    self.optimizer.step()
                
                # Check KL divergence at end of epoch or use accumulated mean?
                # Early stopping usually checks last batch? Or average?
                # Using last batch KL for early stopping logic is standard.
                if self.cfg.train.ppo.target_kl is not None and approx_kl > self.cfg.train.ppo.target_kl:
                    break

            # Logging at the END of Iteration (aggregated)
            if iteration % self.cfg.wandb.log_interval == 0:
                sps = int(global_step / (time.time() - start_time))
                
                # Mean Losses
                mean_losses = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in loss_stats.items()}
                mean_clipfrac = np.mean(clipfracs) if len(clipfracs) > 0 else 0.0
                
                # Log Losses
                self.writer.add_scalar("losses/policy_loss", mean_losses["pg_loss"], global_step)
                self.writer.add_scalar("losses/value_loss", mean_losses["v_loss"], global_step)
                self.writer.add_scalar("losses/entropy", mean_losses["entropy_loss"], global_step)
                self.writer.add_scalar("losses/old_approx_kl", mean_losses["old_approx_kl"], global_step)
                self.writer.add_scalar("losses/approx_kl", mean_losses["approx_kl"], global_step)
                self.writer.add_scalar("losses/clipfrac", mean_clipfrac, global_step)
                self.writer.add_scalar("losses/explained_variance", explained_var, global_step) # Use last calculation
                self.writer.add_scalar("losses/state_loss", mean_losses["s_loss"], global_step)
                self.writer.add_scalar("losses/reward_loss", mean_losses["r_loss"], global_step)
                self.writer.add_scalar("charts/SPS", sps, global_step)
                
                # Log Returns
                if len(iter_episode_returns) > 0:
                    mean_ep_ret = np.mean(iter_episode_returns)
                    mean_ep_len = np.mean(iter_episode_lengths)
                    self.writer.add_scalar("charts/episodic_return", mean_ep_ret, global_step)
                    self.writer.add_scalar("charts/episodic_length", mean_ep_len, global_step)
                else:
                    mean_ep_ret = 0.0 # Placeholder
                    mean_ep_len = 0.0

                print(f"Iteration {iteration}: SPS={sps} P_Loss={mean_losses['pg_loss']:.3f} V_Loss={mean_losses['v_loss']:.3f} S_Loss={mean_losses['s_loss']:.3f} R_Loss={mean_losses['r_loss']:.3f} Ep_Ret={mean_ep_ret:.1f}")

                if self.cfg.wandb.enabled:
                    log_dict = {
                        "losses/policy_loss": mean_losses["pg_loss"],
                        "losses/value_loss": mean_losses["v_loss"],
                        "losses/entropy": mean_losses["entropy_loss"],
                        "losses/approx_kl": mean_losses["approx_kl"],
                        "losses/explained_variance": explained_var,
                        "losses/state_loss": mean_losses["s_loss"],
                        "losses/reward_loss": mean_losses["r_loss"],
                        "charts/SPS": sps,
                        "global_step": global_step
                    }
                    if len(iter_episode_returns) > 0:
                        log_dict["charts/episodic_return"] = mean_ep_ret
                        log_dict["charts/episodic_length"] = mean_ep_len
                        
                    wandb.log(log_dict)
