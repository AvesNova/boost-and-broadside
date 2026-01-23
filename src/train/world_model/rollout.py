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
        model: InterleavedWorldModel
        input_states: (B, T, N, D)
        input_actions: (B, T, N, 3) - Discrete indices
        team_ids: (B, N) or (B, T, N)
        rollout_len: int
    """
    batch_size, time_steps, num_ships, _ = input_states.shape
    min_context = 4

    if rollout_len <= 0 or time_steps <= min_context + rollout_len + 1:
        return

    # Pick a random start time
    start_t = random.randint(min_context, time_steps - rollout_len - 1)
    
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
                tm_in = team_ids
                
            pred_s, pred_a_logits, _ = model(
                s_in, 
                a_in, 
                tm_in, 
                noise_scale=0.0,
                return_embeddings=False
            )
            
            # pred_a_logits is (B, T_window, N, 12).
            # We want the LAST step (corresponding to t).
            last_a_logits = pred_a_logits[:, -1] # (B, N, 12)
            
            # Argmax
            p_idx = last_a_logits[..., 0:3].argmax(dim=-1)
            t_idx = last_a_logits[..., 3:10].argmax(dim=-1)
            s_idx = last_a_logits[..., 10:12].argmax(dim=-1)
            
            next_action = torch.stack([p_idx, t_idx, s_idx], dim=-1).float() # (B, N, 3)
            
            # Update curr_actions at t
            curr_actions[:, t] = next_action
            input_actions[:, t] = next_action # Update external tensor
            
            # 2. Predict State S_{t+1}
            a_in[:, -1] = next_action
            
            pred_s, _, _ = model(
                s_in,
                a_in,
                tm_in,
                noise_scale=0.0
            )
            
            # pred_s is (B, T_window, N, D).
            # Corresponds to S_{next}.
            next_state = pred_s[:, -1] # (B, N, D)
            
            # Update curr_states at t+1
            if t + 1 < time_steps:
                curr_states[:, t + 1] = next_state
                input_states[:, t + 1] = next_state # Update external tensor
