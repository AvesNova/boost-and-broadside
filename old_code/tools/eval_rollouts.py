
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import datetime
from pathlib import Path
from omegaconf import OmegaConf

from boost_and_broadside.agents.mamba_bb import MambaBB, MambaConfig, InferenceParams
from boost_and_broadside.train.data_loader import load_bc_data, create_unified_data_loaders
from boost_and_broadside.utils.model_finder import find_most_recent_model
from boost_and_broadside.eval.rollout_metrics import compute_rollout_metrics
from boost_and_broadside.core.constants import NORM_VELOCITY

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def extract_features(state_tokens):
    """
    Extracts vel, att from tokenized state (9-dim).
    Layout: [0]Team, [1]Health, [2]Power, [3,4]Vel, [5,6]Att, [7]Shoot, [8]AngVel
    """
    # Vel: [3, 4]
    vel_rec = state_tokens[..., 3:5] * NORM_VELOCITY
    
    # Att: [5, 6]
    att_rec = state_tokens[..., 5:7]
    
    return vel_rec, att_rec

def evaluate_rollouts(data_path, dataset_name, model, device, max_steps=50, num_episodes=5):
    """
    Evaluates model on a dataset using Dreaming (AR) and Teacher Forcing (TF).
    """
    log.info(f"Evaluating on {dataset_name} from {data_path}...")
    
    # Load Data
    data = load_bc_data(data_path)
    
    # Create loader
    train_loader_short, train_loader_long, val_loader_short, val_loader_long = create_unified_data_loaders(
        data,
        short_batch_size=1,
        long_batch_size=1,
        short_batch_len=max_steps + 10,
        long_batch_len=max_steps + 10,
        batch_ratio=0.0, 
        validation_split=0.2 if dataset_name == "Validation" else 0.0,
        num_workers=0
    )
    
    loader = val_loader_long if dataset_name == "Validation" else train_loader_long
    if not loader:
         log.warning(f"No data found for {dataset_name}")
         return None, None
         
    iterator = iter(loader)
    
    mse_dream_per_step = []
    mse_tf_per_step = []
    
    count = 0
    with torch.no_grad():
        for i in range(num_episodes):
            try:
                batch = next(iterator)
            except StopIteration:
                break
                
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            pos_gt = batch["pos"].to(device)
            alive = batch["action_masks"].to(device)
            
            B, T, N, D = states.shape
            if T < max_steps + 1: continue
            
            # --- DREAMING LOOP ---
            curr_s = states[:, 0, :, :]
            curr_a_prev = torch.zeros(B, N, 3, device=device)
            curr_pos = pos_gt[:, 0, :, :]
            
            episode_errors = []
            inference_params = InferenceParams(max_batch_size=B*N, max_seqlen=max_steps+10) if InferenceParams else None
            actor_cache = None
            
            for t in range(max_steps):
                v, at = extract_features(curr_s)
                
                # Forward
                pred_s, pred_a_logits, _, _, new_cache = model(
                    state=curr_s.unsqueeze(1),
                    prev_action=curr_a_prev.unsqueeze(1),
                    pos=curr_pos.unsqueeze(1),
                    vel=v.unsqueeze(1),
                    att=at.unsqueeze(1) if at is not None else None,
                    inference_params=inference_params,
                    actor_cache=actor_cache
                )
                actor_cache = new_cache
                
                # Update state
                next_s_pred = pred_s.squeeze(1)
                
                # Sample Action
                probs = pred_a_logits.squeeze(1)
                p_idx = probs[..., 0:3].argmax(-1)
                t_idx = probs[..., 3:10].argmax(-1)
                s_idx = probs[..., 10:12].argmax(-1)
                next_a_pred = torch.stack([p_idx, t_idx, s_idx], dim=-1).float()
                
                # Record Error
                gt_next_s = states[:, t+1, :, :]
                mask = alive[:, t+1, :].unsqueeze(-1)
                mse = ((next_s_pred - gt_next_s)**2 * mask).sum() / (mask.sum() + 1e-6)
                episode_errors.append(mse.item())
                
                # Update for next step
                curr_s = next_s_pred
                curr_a_prev = next_a_pred
                curr_pos = pos_gt[:, t+1, :, :] # Anchor to GT Pos
                
            mse_dream_per_step.append(np.array(episode_errors))
            
            # --- TEACHER FORCING LOOP ---
            tf_states = states[:, 0:max_steps, :, :]
            tf_pos = pos_gt[:, 0:max_steps, :, :]
            tf_actions_data = actions[:, 0:max_steps-1, :, :]
            tf_prev_actions = torch.cat([torch.zeros(B, 1, N, 3, device=device), tf_actions_data], dim=1)
            
            v_tf, at_tf = extract_features(tf_states)
            
            pred_s_tf, _, _, _, _ = model(
                state=tf_states,
                prev_action=tf_prev_actions,
                pos=tf_pos,
                vel=v_tf,
                att=at_tf,
                inference_params=None,
                actor_cache=None
            )
            
            gt_next_s_tf = states[:, 1:max_steps+1, :, :]
            mask_tf = alive[:, 1:max_steps+1, :].unsqueeze(-1)
            
            tf_errors = []
            for t in range(max_steps):
                step_mse = ((pred_s_tf[:, t] - gt_next_s_tf[:, t])**2 * mask_tf[:, t]).sum() / (mask_tf[:, t].sum() + 1e-6)
                tf_errors.append(step_mse.item())
            
            mse_tf_per_step.append(np.array(tf_errors))
            count += 1
            if count >= num_episodes: break
                
    return np.mean(mse_dream_per_step, axis=0), np.mean(mse_tf_per_step, axis=0)

def analyze_feature_errors(data_path, model, device, max_steps=50, num_episodes=20):
    """Detailed error analysis for each token dimension."""
    log.info(f"Analyzing feature errors on {num_episodes} episodes...")
    data = load_bc_data(data_path)
    _, _, _, val_loader_long = create_unified_data_loaders(
        data, short_batch_size=1, long_batch_size=1, 
        short_batch_len=max_steps + 10, long_batch_len=max_steps + 10,
        batch_ratio=0.0, validation_split=0.2, num_workers=0
    )
    
    loader = val_loader_long if val_loader_long else []
    iterator = iter(loader)
    D = model.config.input_dim
    sum_err = torch.zeros(D, device=device)
    sum_sq_err = torch.zeros(D, device=device)
    sum_abs_err = torch.zeros(D, device=device)
    sum_delta = torch.zeros(D, device=device)
    sum_sq_delta = torch.zeros(D, device=device)
    total_count = torch.zeros(D, device=device)
    
    count = 0
    with torch.no_grad():
        for i in range(num_episodes):
            try: batch = next(iterator)
            except StopIteration: break
                
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            pos_gt = batch["pos"].to(device)
            alive = batch["action_masks"].to(device)
            
            B, T, N, _ = states.shape
            if T < max_steps + 1: continue
            
            tf_states = states[:, 0:max_steps, :, :]
            tf_pos = pos_gt[:, 0:max_steps, :, :]
            tf_actions_data = actions[:, 0:max_steps-1, :, :]
            tf_prev_actions = torch.cat([torch.zeros(B, 1, N, 3, device=device), tf_actions_data], dim=1)
            
            v_tf, at_tf = extract_features(tf_states)
            pred_s_tf, _, _, _, _ = model(
                state=tf_states, prev_action=tf_prev_actions, 
                pos=tf_pos, vel=v_tf, att=at_tf, 
                inference_params=None, actor_cache=None
            )
            
            gt_next_s_tf = states[:, 1:max_steps+1, :, :]
            gt_curr_s_tf = states[:, 0:max_steps, :, :]
            mask = alive[:, 1:max_steps+1, :].unsqueeze(-1)
            
            # Error = Pred - GT
            diff = (pred_s_tf - gt_next_s_tf) * mask
            sum_err += diff.sum(dim=(0, 1, 2))
            sum_sq_err += (diff**2).sum(dim=(0, 1, 2))
            sum_abs_err += diff.abs().sum(dim=(0, 1, 2))
            
            # Raw Delta = GT_{t+1} - GT_t
            delta = (gt_next_s_tf - gt_curr_s_tf) * mask
            sum_delta += delta.sum(dim=(0, 1, 2))
            sum_sq_delta += (delta**2).sum(dim=(0, 1, 2))
            
            total_count += mask.sum(dim=(0, 1, 2)).expand(D)
            count += 1
            if count >= num_episodes: break
            
    return sum_err.cpu().numpy(), sum_sq_err.cpu().numpy(), sum_abs_err.cpu().numpy(), sum_delta.cpu().numpy(), sum_sq_delta.cpu().numpy(), total_count.cpu().numpy()

def evaluate_live_rollouts(model, device, num_scenarios=10, max_steps=50):
    log.info(f"Running Live Evaluation (Sim vs Dream) on {num_scenarios} scenarios...")
    env_config = {
        "num_envs": 1, "render_mode": "none", "world_size": [1024.0, 1024.0],
        "max_ships": 8, "max_steps": max_steps + 10, "bg_mode": "random", "reward_mode": "dense",
    }
    metrics = compute_rollout_metrics(
        model, env_config, device, num_scenarios=num_scenarios, max_steps=max_steps, step_intervals=None
    )
    return metrics["full_mse_sim"], metrics["full_mse_dream"]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to best_world_model.pth")
    parser.add_argument("--run_dir", type=str, help="Run directory (e.g. models/world_model/run_...)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Logic to find model path
    model_path = args.model_path
    if not model_path and args.run_dir:
        model_path = os.path.join(args.run_dir, "best_world_model.pth")
    if not model_path:
        model_path = find_most_recent_model("world_model")
        
    if not model_path or not os.path.exists(model_path):
        log.error(f"Model path not found: {model_path}")
        return
    
    log.info(f"Loading model from {model_path}")
    model_dir = os.path.dirname(model_path)
    cfg_path = os.path.join(model_dir, "config.yaml")
    cfg = OmegaConf.load(cfg_path)
    
    mamba_cfg = MambaConfig(
        input_dim=9, d_model=cfg.world_model.embed_dim, n_layers=cfg.world_model.n_layers, n_heads=cfg.world_model.n_heads,
        action_dim=12, target_dim=9, loss_type=getattr(cfg.world_model, "loss_type", "fixed"), max_context_len=int(cfg.world_model.seq_len)
    )
    mamba_cfg.embed_dim = mamba_cfg.d_model
    model = MambaBB(mamba_cfg).to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {k[10:] if k.startswith("_orig_mod.") else k: v for k, v in state_dict.items()}
    if "state_encoder.0.weight" in new_state_dict:
        ckpt_dim = new_state_dict["state_encoder.0.weight"].shape[1]
        if ckpt_dim != mamba_cfg.input_dim:
            mamba_cfg.input_dim = ckpt_dim
            if "world_head.3.weight" in new_state_dict: mamba_cfg.target_dim = new_state_dict["world_head.3.weight"].shape[0]
            model = MambaBB(mamba_cfg).to(device)
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    results = {}
    bc_data_path = cfg.train.bc_data_path if cfg.train.bc_data_path and os.path.exists(cfg.train.bc_data_path) else None
    if not bc_data_path:
        import glob
        h5_files = glob.glob("data/**/*.h5", recursive=True) + glob.glob("data/**/*.hdf5", recursive=True)
        h5_files.sort(key=os.path.getmtime, reverse=True)
        bc_files = [f for f in h5_files if "bc_pretraining" in f]
        bc_data_path = bc_files[0] if bc_files else (h5_files[0] if h5_files else None)

    if bc_data_path:
        d_mse, tf_mse = evaluate_rollouts(bc_data_path, "Validation", model, device, max_steps=50)
        if d_mse is not None:
            results["Dataset_Dream"] = d_mse
            results["Dataset_TF"] = tf_mse
    
    sim_mse, dream_mse = evaluate_live_rollouts(model, device, num_scenarios=20, max_steps=50)
    results["Live_Sim"] = sim_mse
    results["Live_Dream"] = dream_mse
    
    # Save Outputs
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"outputs/rollout_eval/run_{now}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot
    plt.figure(figsize=(10, 6))
    colors = {'Dataset_TF': 'green', 'Dataset_Dream': 'blue', 'Live_Sim': 'orange', 'Live_Dream': 'red'}
    styles = {'Dataset_TF': '--', 'Dataset_Dream': ':', 'Live_Sim': '-', 'Live_Dream': '-'}
    labels = {'Dataset_TF': 'Dataset: TF', 'Dataset_Dream': 'Dataset: Dream', 'Live_Sim': 'Live: Sim', 'Live_Dream': 'Live: Dream'}
    
    for key, mse_arr in results.items():
        arr = mse_arr.cpu().numpy() if isinstance(mse_arr, torch.Tensor) else mse_arr
        plt.plot(np.arange(1, len(arr) + 1), arr, label=labels.get(key, key), color=colors.get(key, 'black'), linestyle=styles.get(key, '-'))
    plt.xlabel('Step'); plt.ylabel('MSE'); plt.title('MambaBB Eval'); plt.legend(); plt.grid(True, alpha=0.3); plt.yscale('log')
    plt.savefig(out_dir / 'rollout_eval_full.png'); plt.close()
    
    # CSVs
    with open(out_dir / 'summary.csv', 'w') as f:
        f.write("Benchmark,Step_1,Step_10,Step_50\n")
        for key, arr in results.items():
            arr = arr.cpu().numpy() if isinstance(arr, torch.Tensor) else arr
            f.write(f"{labels.get(key, key)},{arr[0]:.6f},{arr[9] if len(arr)>9 else 0:.6f},{arr[49] if len(arr)>49 else 0:.6f}\n")
    
    if bc_data_path:
        sum_err, sum_sq_err, sum_abs_err, sum_delta, sum_sq_delta, t_count = analyze_feature_errors(bc_data_path, model, device, max_steps=50, num_episodes=50)
        
        # Calculate Mean and Std
        # Error metrics
        mean_err = sum_err / (t_count + 1e-6)
        std_err = np.sqrt(np.maximum(0, (sum_sq_err / (t_count + 1e-6)) - (mean_err**2)))
        mae_err = sum_abs_err / (t_count + 1e-6)
        
        # Raw Delta metrics
        mean_delta = sum_delta / (t_count + 1e-6)
        std_delta = np.sqrt(np.maximum(0, (sum_sq_delta / (t_count + 1e-6)) - (mean_delta**2)))
        
        feature_labels = ["0:Team", "1:Health", "2:Power", "3:Vel_X", "4:Vel_Y", "5:Att_X", "6:Att_Y", "7:Shoot", "8:AngVel", "9:Misc1", "10:Misc2"]
        with open(out_dir / 'feature_errors.csv', 'w') as f:
            f.write("Feature_ID,Feature_Name,Err_Bias,Err_Std,Err_MAE,Delta_Bias,Delta_Std,Count\n")
            for i in range(len(mean_err)):
                lbl = feature_labels[i] if i < len(feature_labels) else f"Dim_{i}"
                f.write(f"{i},{lbl},{mean_err[i]:.8f},{std_err[i]:.8f},{mae_err[i]:.8f},{mean_delta[i]:.8f},{std_delta[i]:.8f},{int(t_count[i])}\n")
        log.info(f"Archived outputs to {out_dir}")

if __name__ == "__main__":
    main()
