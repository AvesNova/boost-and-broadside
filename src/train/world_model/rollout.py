import random
import torch

def get_rollout_length(epoch: int, cfg) -> int:
    """
    Determine rollout length for the current epoch based on schedule.
    """
    config = cfg.world_model.rollout
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


def perform_rollout(model, input_states, input_actions, team_ids, rollout_len):
    """
    Perform closed-loop rollout on the batch.
    Modifies input_states and input_actions in-place.
    
    Args:
        model: MambaBB
        input_states: (B, T, N, D)
        input_actions: (B, T, N, 3) - Discrete indices
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
        
        for i in range(rollout_len):
            t = start_t + i
            
            # Sliding window context for speed
            ctx_start = max(0, t - 32) 
            
            # Slice context
            s_in = curr_states[:, ctx_start : t + 1]
            a_in = curr_actions[:, ctx_start : t + 1].clone() 
            # a_in at t is currently ground truth or old value. We want to predict it.
            a_in[:, -1] = 0 
            
            if team_ids.ndim == 3:
                tm_in = team_ids[:, ctx_start : t + 1]
            else:
                tm_in = team_ids # Assume (B, N)
                
            # Extract features for relational encoder
            pos = s_in[..., 3:5]
            vel = s_in[..., 5:7]
            att = s_in[..., 10:12]
            alive = s_in[..., 1] > 0
                
            pred_s, pred_a_logits = model(
                state=s_in, 
                prev_action=a_in, 
                pos=pos,
                vel=vel,
                att=att,
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
            
            pred_s, _ = model(
                state=s_in, 
                prev_action=a_in, 
                pos=pos,
                vel=vel,
                att=att,
                team_ids=tm_in,
                alive=alive
            )
            
            next_state = pred_s[:, -1] # (B, N, D)
            
            # --- Analytic Attitude Update (Fixing "Dreaming" Mode) ---
            # Predicted attitude is garbage because we don't train it.
            # We must calculate it from Velocity + Action.
            
            # 1. Get Predicted Velocity (Indices 5,6)
            pred_vx = next_state[..., 5]
            pred_vy = next_state[..., 6]
            speed = torch.sqrt(pred_vx**2 + pred_vy**2 + 1e-6).unsqueeze(-1)
            
            # Base Attitude (Direction of Motion)
            base_att_cos = pred_vx.unsqueeze(-1) / speed
            base_att_sin = pred_vy.unsqueeze(-1) / speed
            
            # 2. Turn Offsets (Radians)
            # 0:0, 1:-5, 2:+5, 3:-15, 4:+15
            turn_idx = t_idx # (B, N) from above
            
            offsets = torch.zeros_like(base_att_cos)
            
            # Hardcoded standard config angles (deg2rad)
            deg5 = 0.0872665
            deg15 = 0.261799
            
            # 1: Left (-5)
            offsets[turn_idx == 1] = -deg5
            # 2: Right (+5)
            offsets[turn_idx == 2] = deg5
            # 3: Sharp Left (-15)
            offsets[turn_idx == 3] = -deg15
            # 4: Sharp Right (+15)
            offsets[turn_idx == 4] = deg15
            
            # 3. Rotate
            # cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
            # sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
            
            cos_off = torch.cos(offsets)
            sin_off = torch.sin(offsets)
            
            new_cos = base_att_cos * cos_off - base_att_sin * sin_off
            new_sin = base_att_sin * cos_off + base_att_cos * sin_off
            
            # 4. Overwrite in next_state (Indices 10,11)
            next_state[..., 10] = new_cos.squeeze(-1)
            next_state[..., 11] = new_sin.squeeze(-1)
            
            # Update curr_states at t+1
            if t + 1 < time_steps:
                curr_states[:, t + 1] = next_state
                input_states[:, t + 1] = next_state # Update external tensor
