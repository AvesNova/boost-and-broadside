import torch
from boost_and_broadside.core.constants import StateFeature, TargetFeature, TARGET_DIM

class BatchProcessor:
    @staticmethod
    def compute_targets(input_states, next_states, pos_curr, pos_next, W, H, device):
        # Toroidal Delta Pos
        d_pos = pos_next - pos_curr
        d_pos[..., 0] = d_pos[..., 0] - torch.round(d_pos[..., 0] / W) * W
        d_pos[..., 1] = d_pos[..., 1] - torch.round(d_pos[..., 1] / H) * H
        
        # State Deltas
        d_state = next_states - input_states
        
        # Pairwise Targets
        pos_next_i = pos_next.unsqueeze(-2)   # (..., N, 1, 2)
        pos_next_j = pos_next.unsqueeze(-3)   # (..., 1, N, 2)
        target_pairwise_pos = pos_next_j - pos_next_i  # (..., N, N, 2)
        
        d_vel = d_state[..., StateFeature.VX:StateFeature.VY+1]
        d_vel_i = d_vel.unsqueeze(-2)
        d_vel_j = d_vel.unsqueeze(-3)
        target_pairwise_vel = d_vel_j - d_vel_i # (..., N, N, 2)
        
        target_pairwise = torch.cat([target_pairwise_pos, target_pairwise_vel], dim=-1)
        
        # Construct Target Vector
        target_states = torch.zeros((*d_state.shape[:-1], TARGET_DIM), device=device, dtype=d_state.dtype)
        target_states[..., TargetFeature.DX:TargetFeature.DY+1] = d_pos
        target_states[..., TargetFeature.DVX] = d_state[..., StateFeature.VX]
        target_states[..., TargetFeature.DVY] = d_state[..., StateFeature.VY]
        target_states[..., TargetFeature.DHEALTH] = d_state[..., StateFeature.HEALTH]
        target_states[..., TargetFeature.DPOWER] = d_state[..., StateFeature.POWER]
        target_states[..., TargetFeature.DANG_VEL] = d_state[..., StateFeature.ANG_VEL]
        
        return target_states, target_pairwise

    @staticmethod
    def process_batch(batch_data, config, device, use_amp=False):
        """
        Processes raw batch data into standardized model inputs, targets, and masks.
        Handles temporal slicing, toroidal distance wrapping, and delta computation.
        
        Args:
            batch_data (dict): Raw batch dictionary (e.g., from ContinuousView or GPUBuffer)
                               Expected keys: states, actions, team_ids, seq_idx, loss_mask, pos
                               Optional keys: reset_mask, rewards, returns
            config (DictConfig): Model configuration containing world_size.
            device (torch.device): Device to place tensors on.
            use_amp (bool): Whether to cast state inputs to bfloat16.
            
        Returns:
            dict: Containing 'inputs', 'targets', 'masks', and 'extras'
        """
        dtype = torch.float32
        input_dtype = torch.bfloat16 if use_amp else torch.float32
        
        # 1. Base Variables
        states = batch_data["states"].to(device, dtype=input_dtype, non_blocking=True)
        actions = batch_data["actions"].to(device, non_blocking=True)
        team_ids = batch_data["team_ids"].to(device, non_blocking=True)
        seq_idx = batch_data["seq_idx"].to(device, non_blocking=True)
        loss_mask = batch_data["loss_mask"].to(device, non_blocking=True)
        pos_all = batch_data["pos"].to(device, dtype=dtype, non_blocking=True)
        
        # 2. Slicing Inputs [0:-1] and Targets [1:]
        input_states = states[:, :-1]
        next_states = states[:, 1:]
        
        target_actions = actions[:, :-1]
        
        zero_action = torch.zeros_like(actions[:, :1])
        input_actions = torch.cat([zero_action, actions[:, :-1]], dim=1)[:, :-1]
        
        loss_mask_slice = loss_mask[:, 1:]
        pos_curr = pos_all[:, :-1]
        pos_next = pos_all[:, 1:]
        
        # 3. Deltas and Targets
        if "environment" in config and "world_size" in config.environment:
            W, H = config.environment.world_size
        else:
            W, H = 1000, 1000 # Fallback
            
        target_states, target_pairwise = BatchProcessor.compute_targets(
            input_states, next_states, pos_curr, pos_next, W, H, device
        )
        
        # Velocity for Relational Trunk (Current Vx, Vy)
        vel = input_states[..., StateFeature.VX : StateFeature.VY+1]
        
        # Alive Masks
        alive = input_states[..., StateFeature.HEALTH] > 0
        target_alive = next_states[..., StateFeature.HEALTH] > 0
        
        # Return Dict
        result = {
            "inputs": {
                "state": input_states,
                "prev_action": input_actions,
                "pos": pos_curr,
                "vel": vel,
                "team_ids": team_ids[:, :-1],
                "seq_idx": seq_idx[:, :-1],
                "alive": alive,
                "target_actions": target_actions
            },
            "targets": {
                "target_states": target_states,
                "target_actions": target_actions,
                "target_pairwise": target_pairwise,
                "target_alive": target_alive
            },
            "masks": {
                "loss_mask": loss_mask_slice,
                "input_alive": alive
            },
            "extras": {
                # Add optional RL keys if present
            }
        }
        
        if "reset_mask" in batch_data:
            result["inputs"]["reset_mask"] = batch_data["reset_mask"][:, :-1].to(device, non_blocking=True)
            
        if "rewards" in batch_data:
            r = batch_data["rewards"].to(device, non_blocking=True)[:, :-1]
            if r.dim() == 2: r = r.unsqueeze(-1)
            result["targets"]["target_rewards"] = r
            
        if "returns" in batch_data:
            ret = batch_data["returns"].to(device, non_blocking=True)[:, :-1]
            if ret.dim() == 2: ret = ret.unsqueeze(-1)
            result["targets"]["target_returns"] = ret
            
        return result
