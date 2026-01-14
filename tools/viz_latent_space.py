
import sys
from pathlib import Path
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pacmap
import yaml
import argparse

# Add src to sys.path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agents.world_model import WorldModel
from train.data_loader import load_bc_data, create_bc_data_loader
from utils.tensor_utils import to_one_hot
from env.constants import PowerActions, TurnActions, ShootActions

def load_config(run_dir: Path):
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Ensure train section exists
    if "train" not in config:
        config["train"] = {}
    return config

def load_model(run_dir: Path, config: dict, device: str = "cpu"):
    wm_config = config.get("world_model", {})
    
    # Extract model params
    # Note: Using default args from WorldModel __init__ if not in config
    model = WorldModel(
        state_dim=12, # hardcoded based on known state dim 
        action_dim=12, # hardcoded based on known action dim (3+7+2)
        embed_dim=wm_config.get("embed_dim", 128),
        n_layers=wm_config.get("n_layers", 4),
        n_heads=wm_config.get("n_heads", 4),
        max_ships=wm_config.get("n_ships", 8),
        max_context_len=wm_config.get("context_len", 128),
        dropout=0.0 # No dropout for inference
    )
    
    # Find checkpoint
    checkpoint_path = run_dir / "best_world_model.pth"
    if not checkpoint_path.exists():
        # Fallback to latest epoch
        checkpoints = list(run_dir.glob("world_model_epoch_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {run_dir}")
        # Sort by epoch number
        checkpoint_path = sorted(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))[-1]
    
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model



def extract_embeddings(model, data, device, max_batches=10, seq_len=96):
    embeddings_list = []
    actions_list = []
    values_list = []
    ship_ids_list = []
    timesteps_list = []
    
    print(f"Extracting embeddings using {device}...")
    
    # Process sequential episodes
    team_0 = data["team_0"]
    all_tokens = team_0["tokens"]
    all_actions = team_0["actions"]
    episode_lengths = data["episode_lengths"]
    
    count = 0
    start_idx = 0
    
    for i, length in enumerate(episode_lengths):
        length = length.item()
        end_idx = start_idx + length
        
        # Only process if long enough for sequence
        if length >= seq_len:
            # Slice first seq_len steps (Teacher Forcing window)
            # tokens: [start_idx, start_idx + seq_len]
            tokens = all_tokens[start_idx : start_idx + seq_len]
            
            # actions: shift right. 
            # First action input is 0.
            # Then actions from start_idx to start_idx + seq_len - 1
            raw_actions = all_actions[start_idx : start_idx + seq_len - 1]
            zeros = torch.zeros(1, *raw_actions.shape[1:], dtype=raw_actions.dtype, device=raw_actions.device)
            p_actions = torch.cat([zeros, raw_actions], dim=0) # (T, N, 3)
            
            # Batch dim add
            tokens = tokens.unsqueeze(0).to(device) # (1, T, N, F)
            p_actions = p_actions.unsqueeze(0).to(device) # (1, T, N, 3)
            
            # To One Hot
            input_actions_oh = to_one_hot(p_actions) # (1, T, N, 12)
            
            with torch.no_grad():
                 _, pred_actions, pred_val, _, _, embeddings = model(
                    tokens, 
                    input_actions_oh, 
                    return_embeddings=True
                )
            
            # Flatten: (B, T, N, E) -> (B*T*N, E)
            B, T, N, E = embeddings.shape
            
            embeddings_list.append(embeddings.reshape(-1, E).cpu().numpy())
            
            # Create ship IDs matching the flatten order
            # (B, T, N) where each entry is 0..N-1
            ship_ids = torch.arange(N, device=device).view(1, 1, N).expand(B, T, N)
            ship_ids_list.append(ship_ids.reshape(-1).cpu().numpy())
            
            # Create Timesteps matching flattne order
            # (B, T, N) where each entry is 0..T-1
            timesteps = torch.arange(T, device=device).view(1, T, 1).expand(B, T, N)
            timesteps_list.append(timesteps.reshape(-1).cpu().numpy())
            
            # Get target actions for coloring
            # We want the actions that occur at this step (or intended action).
            # The input was p_actions (previous).
            # The target for classification is usually the *next* action, or the action taken at this state.
            # In BC, given State_t, Action_{t-1}, we predict Action_t.
            # The coloring should likely be Action_t (the ground truth action at this step).
            # which is all_actions[start_idx : start_idx + seq_len]
            target_actions = all_actions[start_idx : start_idx + seq_len].unsqueeze(0).to(device)
            actions_list.append(target_actions.reshape(-1, 3).cpu().numpy())
            
            # Flatten predicted value: (B, T, N) -> (B*T*N)
            values_list.append(pred_val.reshape(-1).cpu().numpy())
            
            count += 1
            if count >= max_batches:
                break
        
        # Move to next episode
        start_idx = end_idx
    
    print(f"Processed {count} episodes.")

    return (
        np.concatenate(embeddings_list, axis=0),
        np.concatenate(actions_list, axis=0),
        np.concatenate(values_list, axis=0),
        np.concatenate(ship_ids_list, axis=0),
        np.concatenate(timesteps_list, axis=0)
    )

def plot_with_legend(projections, labels, label_map, title, filename):
    plt.figure(figsize=(10, 8))
    
    # Get unique labels
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    
    # Use qualitative colormaps designed for distinct categories
    if num_labels <= 10:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(num_labels)]
    elif num_labels <= 20:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i) for i in range(num_labels)]
    else:
        # Fallback to evenly spaced colors for many categories
        cmap = plt.get_cmap("nipy_spectral")
        colors = [cmap(i/num_labels) for i in range(num_labels)]
         
    for i, label_val in enumerate(unique_labels):
        mask = labels == label_val
        label_name = label_map.get(int(label_val), str(int(label_val)))
        plt.scatter(
            projections[mask, 0], 
            projections[mask, 1], 
            c=[colors[i]], 
            s=12, # Larger for better visibility
            alpha=0.7,
            label=label_name,
            edgecolors='none' # Cleaner visuals
        )
        
    # Improve legend position and style
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., frameon=False)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=200) # Higher DPI
    plt.close()

def plot_continuous(projections, values, title, filename):
    plt.figure(figsize=(10, 8))
    # Use 'turbo' or 'plasma' for better perceptual uniformity and vibrancy
    scatter = plt.scatter(projections[:, 0], projections[:, 1], c=values, cmap='turbo', s=2, alpha=0.6)
    cbar = plt.colorbar(scatter, label="Value")
    cbar.solids.set_alpha(1) # Ensure colorbar is opaque
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def plot_and_save(projections, actions, values, ship_ids, timesteps, method_name, save_dir, model_name):
    # Action indices
    # actions column 0 is power
    # actions column 1 is turn
    # actions column 2 is shoot
    
    power_actions = actions[:, 0]
    turn_actions = actions[:, 1]
    shoot_actions = actions[:, 2]
    
    # Mappings
    power_map = {v.value: v.name.title().replace("_", " ") for v in PowerActions}
    turn_map = {v.value: v.name.title().replace("_", " ") for v in TurnActions}
    shoot_map = {v.value: v.name.title().replace("_", " ") for v in ShootActions}
    
    # Ship & Team Mappings
    unique_ships = np.unique(ship_ids)
    ship_map = {int(s): f"Ship {int(s)}" for s in unique_ships}
    
    # Assume 4v4 split: 0-3 Ally, 4-7 Enemy
    # Or generically first half vs second half
    max_ship_id = int(np.max(ship_ids))
    half_point = (max_ship_id + 1) // 2
    team_ids = (ship_ids >= half_point).astype(int)
    team_map = {0: "Ally (Team 0)", 1: "Enemy (Team 1)"}
    
    # 1. Power Actions
    plot_with_legend(
        projections, 
        power_actions, 
        power_map, 
        f"{model_name} - {method_name} - Power Actions", 
        save_dir / f"{method_name}_power.png"
    )

    # 2. Turn Actions
    plot_with_legend(
        projections, 
        turn_actions, 
        turn_map, 
        f"{model_name} - {method_name} - Turn Actions", 
        save_dir / f"{method_name}_turn.png"
    )
    
    # 3. Shoot Actions
    plot_with_legend(
        projections, 
        shoot_actions, 
        shoot_map, 
        f"{model_name} - {method_name} - Shoot Actions", 
        save_dir / f"{method_name}_shoot.png"
    )
    
    # 4. Values
    plot_continuous(
        projections, 
        values, 
        f"{model_name} - {method_name} - Value", 
        save_dir / f"{method_name}_value.png"
    )
    
    # 5. Ship IDs
    plot_with_legend(
        projections,
        ship_ids,
        ship_map,
        f"{model_name} - {method_name} - Ship ID",
        save_dir / f"{method_name}_ship.png"
    )
    
    # 6. Team IDs
    plot_with_legend(
        projections,
        team_ids,
        team_map,
        f"{model_name} - {method_name} - Team ID",
        save_dir / f"{method_name}_team.png"
    )
    
    # 7. Timesteps
    plot_continuous(
        projections,
        timesteps,
        f"{model_name} - {method_name} - Timestep",
        save_dir / f"{method_name}_timestep.png"
    )

def generate_report(model_name, run_dir, config, output_dir):
    report_path = output_dir / "report.md"
    
    # Metadata
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    wm_config = config.get("world_model", {})
    
    with open(report_path, "w") as f:
        f.write(f"# Latent Space Visualization Report: {model_name}\n\n")
        f.write(f"**Date Generated:** {timestamp}\n\n")
        f.write(f"**Model Path:** `{run_dir}`\n\n")
        
        f.write("## Model Configuration\n")
        f.write(f"- **Embed Dim:** {wm_config.get('embed_dim', 'N/A')}\n")
        f.write(f"- **Layers:** {wm_config.get('n_layers', 'N/A')}\n")
        f.write(f"- **Heads:** {wm_config.get('n_heads', 'N/A')}\n")
        f.write(f"- **Context Len:** {wm_config.get('context_len', 'N/A')}\n\n")
        
        f.write("## Visualizations\n")
        f.write("### PCA Projections\n")
        f.write(f"![PCA Power](PCA_power.png)\n")
        f.write(f"![PCA Turn](PCA_turn.png)\n")
        f.write(f"![PCA Shoot](PCA_shoot.png)\n")
        f.write(f"![PCA Value](PCA_value.png)\n")
        f.write(f"![PCA Ship ID](PCA_ship.png)\n")
        f.write(f"![PCA Team ID](PCA_team.png)\n")
        f.write(f"![PCA Timestep](PCA_timestep.png)\n\n")
        
        f.write("### PaCMAP Projections\n")
        f.write(f"![PaCMAP Power](PaCMAP_power.png)\n")
        f.write(f"![PaCMAP Turn](PaCMAP_turn.png)\n")
        f.write(f"![PaCMAP Shoot](PaCMAP_shoot.png)\n")
        f.write(f"![PaCMAP Value](PaCMAP_value.png)\n")
        f.write(f"![PaCMAP Ship ID](PaCMAP_ship.png)\n")
        f.write(f"![PaCMAP Team ID](PaCMAP_team.png)\n")
        f.write(f"![PaCMAP Timestep](PaCMAP_timestep.png)\n")
    
    print(f"Report generated at: {report_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, default="models/interesting")
    parser.add_argument("--output_dir", type=str, default="outputs/latent_viz")
    parser.add_argument("--max_batches", type=int, default=256)
    parser.add_argument("--test-run", action="store_true", help="Run quick test on single model")
    parser.add_argument("--latest", action="store_true", help="Process only the most recent model")
    parser.add_argument("--data_path", type=str, default=None, help="Path to specific data file")
    args = parser.parse_args()
    
    base_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Find run directories
    run_dirs = [d for d in base_dir.iterdir() if d.is_dir() and (d / "config.yaml").exists()]
    
    if args.latest and run_dirs:
        # Sort by name (timestamp) and take the last one
        run_dirs.sort(key=lambda d: d.name)
        run_dirs = [run_dirs[-1]]
        print(f"Selected most recent run: {run_dirs[0].name}")

    if args.test_run and not args.latest:
        run_dirs = run_dirs[:1]
        args.max_batches = 1
        print("Test run: limiting to 1 model and 1 batch.")
        
    print(f"Found {len(run_dirs)} models to process.")
    
    for run_dir in run_dirs:
        model_name = run_dir.name
        print(f"\nProcessing {model_name}...")
        
        try:
            config = load_config(run_dir)
            
            # Load data specific to this model
            data_path = config.get("train", {}).get("bc_data_path", None)
            if data_path:
                 # Check if path relative to config or project root. Assuming project root.
                 pass
            else:
                 # Use latest if not found, but we should probably just use latest anyway if not testing reproduction
                 pass
            
            # For simplicity and robustness, lets just load the global latest data 
            # or the one specified in args if user wants consistency.
            # But adhering to config is better.
            
            if args.data_path:
                 data_to_load = args.data_path
            elif data_path:
                data_to_load = data_path.replace("\\", "/") # fix windows
            else:
                data_to_load = None # Load latest
            
            print(f"Loading data from: {data_to_load if data_to_load else 'LATEST'}")
            data = load_bc_data(data_to_load)

            model = load_model(run_dir, config, device)
            
            # Extract embeddings with manual sequential slicing
            # Use max_batches as number of episodes to process
            embeddings, actions, values, ship_ids, timesteps = extract_embeddings(
                model, 
                data, 
                device, 
                max_batches=args.max_batches,
                seq_len=96 # As requested
            )
            
            # Subsample for plotting if too large (> 10k points)
            if len(embeddings) > 10000:
                indices = np.random.choice(len(embeddings), 10000, replace=False)
                embeddings = embeddings[indices]
                actions = actions[indices]
                values = values[indices]
                ship_ids = ship_ids[indices]
                timesteps = timesteps[indices]
            
            print(f"Projecting {len(embeddings)} points...")
            
            model_out_dir = output_dir / model_name
            model_out_dir.mkdir(parents=True, exist_ok=True)
            
            # PCA
            print("Running PCA...")
            pca = PCA(n_components=2)
            pca_proj = pca.fit_transform(embeddings)
            plot_and_save(pca_proj, actions, values, ship_ids, timesteps, "PCA", model_out_dir, model_name)
            
            # PaCMAP
            print("Running PaCMAP...")
            # Use PCA initialization for PaCMAP for stability/speed if high dim? 
            # Or just run directly. PaCMAP is fast.
            embedding_learner = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0)
            pacmap_proj = embedding_learner.fit_transform(embeddings, init="pca")
            plot_and_save(pacmap_proj, actions, values, ship_ids, timesteps, "PaCMAP", model_out_dir, model_name)
            
            # Generate Report
            generate_report(model_name, run_dir, config, model_out_dir)
            
            print(f"Done processing {model_name}")
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
