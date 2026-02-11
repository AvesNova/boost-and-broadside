import random
import torch

from boost_and_broadside.core.constants import NORM_VELOCITY

def get_rollout_length(epoch: int, cfg) -> int:
    """
    Determine rollout length for the current epoch based on schedule.
    """
    config = cfg.model.rollout
    if not config.enabled or epoch < config.start_epoch:
        return 0

    # Calculate progress in ramp (0.0 to 1.0)
    ramp_progress = (epoch - config.start_epoch) / max(1, config.ramp_epochs)
    ramp_progress = min(1.0, max(0.0, ramp_progress))

    # Linear interpolation of max length
    current_max = (
        config.max_len_start
        + (config.max_len_end - config.max_len_start) * ramp_progress
    )
    current_max = int(current_max)

    if current_max < 1:
        return 0

    # Sample from uniform distribution [1, current_max]
    return random.randint(1, current_max)


def perform_rollout(model, input_states, input_actions, input_pos, team_ids, rollout_len):
    """
    Perform closed-loop rollout on the batch.
    Modifies input_states, input_actions, and input_pos in-place.
    
    Args:
        model: MambaBB
        input_states: (B, T, N, D)
        input_actions: (B, T, N, 3) - Discrete indices
        input_pos: (B, T, N, 2) - Float32 Position
        team_ids: (B, T, N) 
        rollout_len: int
    """
    batch_size, time_steps, num_ships, _ = input_states.shape
    min_context = 4

    if rollout_len <= 0 or time_steps <= min_context + rollout_len + 1:
        return

    # Pick a random start time
    start_t = random.randint(min_context, time_steps - rollout_len - 1)
    
    device = input_states.device
    
    model.eval()
    with torch.no_grad():
        # Current window variables
        curr_states = input_states.clone()
        curr_actions = input_actions.clone()
        curr_pos = input_pos.clone()
        
        for i in range(rollout_len):
            t = start_t + i
            
            # Sliding window context for speed
            ctx_start = max(0, t - 32) 
            
            # Slice context
            s_in = curr_states[:, ctx_start : t + 1]
            p_in = curr_pos[:, ctx_start : t + 1]
            a_in = curr_actions[:, ctx_start : t + 1].clone() 
            # a_in at t is currently ground truth or old value. We want to predict it.
            a_in[:, -1] = 0 
            
            if team_ids.ndim == 3:
                tm_in = team_ids[:, ctx_start : t + 1]
            else:
                tm_in = team_ids # Assume (B, N)
                
            # Extract features for relational encoder
            # States: [Health(0), Power(1), Vx(2), Vy(3), AngVel(4)]
            # Target: [dx, dy, dVx, dVy, dHealth, dPower, dAngVel]
            pos = p_in
            vel = s_in[..., 2:4] # Vx, Vy
            alive = s_in[..., 0] > 0
                
            pred_s, pred_a_logits, _, _, _ = model(
                state=s_in, 
                prev_action=a_in, 
                pos=pos,
                vel=vel,
                att=None,
                team_ids=tm_in,
                alive=alive
            )
            
            # 1. Update Action at t
            last_a_logits = pred_a_logits[:, -1] # (B, N, 12)
            
            # Argmax
            p_idx = last_a_logits[..., 0:3].argmax(dim=-1)
            t_idx = last_a_logits[..., 3:10].argmax(dim=-1)
            s_idx = last_a_logits[..., 10:12].argmax(dim=-1)
            
            next_action = torch.stack([p_idx, t_idx, s_idx], dim=-1).float() # (B, N, 3)
            
            # Update curr_actions at t
            curr_actions[:, t] = next_action
            input_actions[:, t] = next_action # Update external tensor
            
            # 2. Update State S_{t+1}
            # The model already produced pred_s (S_{t+1}) using the context and dummy action.
            # But the spec says: Actor Pass (Phase 1) runs first, then World Model Pass (Phase 2).
            # In MambaBB.forward, both are computed. 
            # However, the `pred_s` was computed using `prev_action` which we set to 0.
            # We should re-run the forward with the predicted action for better consistency
            # if we want strictly Phase 2 after Phase 1.
            
            a_in[:, -1] = next_action
            
            # Phase 2: World Model Pass
            # Predict S_{t+1} given Action_t
            pred_s, _, _, _, _ = model(
                state=s_in, 
                prev_action=a_in, 
                pos=pos,
                vel=vel,
                att=None,
                team_ids=tm_in,
                alive=alive
            )
            
            delta_target = pred_s[:, -1] # (B, N, 7)
            
            # 2. Update State S_{t+1}
            # Delta Target Layout: [dx, dy, dVx, dVy, dH, dP, dAV]
            # State Layout: [H, P, Vx, Vy, AV]
            
            curr_s = curr_states[:, t] # (B, N, 5)
            next_state = curr_s.clone()
            
            # Update normalized components
            next_state[..., 0] = (curr_s[..., 0] + delta_target[..., 4]).clamp(0, 1) # Health
            next_state[..., 1] = (curr_s[..., 1] + delta_target[..., 5]).clamp(0, 1) # Power
            next_state[..., 2] = (curr_s[..., 2] + delta_target[..., 2])             # Vx
            next_state[..., 3] = (curr_s[..., 3] + delta_target[..., 3])             # Vy
            next_state[..., 4] = (curr_s[..., 4] + delta_target[..., 6])             # AngVel
            
            # Update curr_states at t+1 if valid
            if t + 1 < time_steps:
                curr_states[:, t+1] = next_state
                input_states[:, t+1] = next_state
            
            # 3. Update Position (Model-Predicted Deltas)
            # Maintain in F32
            pos_t = curr_pos[:, t].float()
            dx = delta_target[..., 0].float()
            dy = delta_target[..., 1].float()
            
            pos_next = pos_t + torch.stack([dx, dy], dim=-1)
            
            # Wrap around world (Toroidal)
            # Use cfg if available or hardcode standard 1024
            # For pretraining we often use 1024 or 1200x800
            # Let's use 1024.0 as default if not passed.
            world_size = [1024.0, 1024.0] 
            pos_next[..., 0] = torch.remainder(pos_next[..., 0], world_size[0])
            pos_next[..., 1] = torch.remainder(pos_next[..., 1], world_size[1])
            
            # Update curr_pos at t+1
            if t + 1 < time_steps:
                curr_pos[:, t + 1] = pos_next
                input_pos[:, t + 1] = pos_next
            
