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
import h5py

# Add src to sys.path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agents.interleaved_world_model import InterleavedWorldModel
from train.data_loader import load_bc_data
from utils.tensor_utils import to_one_hot
from core.constants import (
    PowerActions, 
    TurnActions, 
    ShootActions,
    MAX_SHIPS,
    STATE_DIM,
    ACTION_DIM,
    TOTAL_ACTION_LOGITS
)


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

    state_dim = wm_config.get("token_dim", STATE_DIM)
    
    # Model uses flattened action dim for HEADS (3+7+2=12), but INPUT is discrete (3)
    # The InterleavedWorldModel handles embedding internally.
    
    model = InterleavedWorldModel(
        state_dim=state_dim,
        embed_dim=wm_config.get("embed_dim", 128),
        n_layers=wm_config.get("n_layers", 4),
        n_heads=wm_config.get("n_heads", 4),
        max_ships=wm_config.get("n_ships", MAX_SHIPS),
        max_context_len=wm_config.get("context_len", 128),
        dropout=0.0,  # No dropout for inference
    )

    # Find checkpoint
    checkpoint_path = run_dir / "best_world_model.pth"
    if not checkpoint_path.exists():
        # Fallback to latest epoch
        checkpoints = list(run_dir.glob("world_model_epoch_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {run_dir}")
        # Sort by epoch number
        checkpoint_path = sorted(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))[
            -1
        ]

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def extract_embeddings(model, h5_path, device, max_batches=10, seq_len=96):
    embeddings_list = []
    actions_list = []
    values_list = []
    ship_ids_list = []
    team_ids_list = []  # Added
    timesteps_list = []
    alive_list = []

    print(f"Extracting embeddings using {device} from {h5_path}...")

    with h5py.File(h5_path, "r") as f:
        episode_lengths = f["episode_lengths"][:]
        all_tokens = f["tokens"]
        all_actions = f["actions"]
        
        has_team_ids = "team_ids" in f
        all_team_ids = f["team_ids"] if has_team_ids else None
        
        count = 0
        
        # Pre-calculate starts
        ep_starts = np.zeros_like(episode_lengths)
        ep_starts[1:] = np.cumsum(episode_lengths[:-1])

        for i, length in enumerate(episode_lengths):
            start_idx = int(ep_starts[i])
            # end_idx = start_idx + length # Unused

            # Only process if long enough for sequence
            if length >= seq_len:
                # Slice first seq_len steps
                tokens_np = all_tokens[start_idx : start_idx + seq_len]
                
                # actions: shift right.
                raw_actions_np = all_actions[start_idx : start_idx + seq_len - 1]
                
                # Zeros for first step
                zeros = np.zeros((1, *raw_actions_np.shape[1:]), dtype=raw_actions_np.dtype)
                p_actions_np = np.concatenate([zeros, raw_actions_np], axis=0) # (T, N, 3)

                # Team IDs
                num_ships = tokens_np.shape[1]
                if has_team_ids:
                    team_ids_np = all_team_ids[start_idx : start_idx + seq_len]
                else:
                    # Default: first half 0, second half 1
                    tid_frame = np.zeros((num_ships,), dtype=np.int64)
                    tid_frame[num_ships//2:] = 1
                    team_ids_np = np.tile(tid_frame, (seq_len, 1)) # (T, N)

                # Convert to Tensor
                tokens = torch.from_numpy(tokens_np).unsqueeze(0).to(device) # (1, T, N, D)
                p_actions = torch.from_numpy(p_actions_np).unsqueeze(0).to(device) # (1, T, N, 3)
                input_team_ids = torch.from_numpy(team_ids_np).unsqueeze(0).to(device) # (1, T, N)

                # Cast actions to long
                if p_actions.dtype != torch.long:
                    p_actions = p_actions.long()
                    
                # Ensure tokens are float
                if tokens.dtype != torch.float32:
                    tokens = tokens.float()
                    
                # Ensure inputs are contiguous if possible, usually fine

                with torch.no_grad():
                    # model forward returns: pred_states, pred_actions, pred_val (if exists), latents, 12d, pred_rel
                    outputs = model(
                        tokens, 
                        p_actions, 
                        input_team_ids, 
                        return_embeddings=True
                    )
                    # output indices: 
                    # 0: pred_states
                    # 1: pred_actions
                    # 2: _ (return_embeddings=True usually returns tuple of 6?)
                    # In InterleavedWorldModel forward:
                    # return pred_states, pred_actions, None, latents, features_12d, pred_relational
                    # So index 3 is latents.
                    embeddings = outputs[3]
                    
                    # Assume no value head
                    pred_val = torch.zeros(tokens.shape[0], tokens.shape[1], tokens.shape[2], device=device)

                # Flatten: (B, T, N, E) -> (B*T*N, E)
                B_dim, T_dim, N_dim, E_dim = embeddings.shape

                embeddings_list.append(embeddings.reshape(-1, E_dim).cpu().numpy())

                # Create ship IDs matching the flatten order
                ship_ids = torch.arange(N_dim, device=device).view(1, 1, N_dim).expand(B_dim, T_dim, N_dim)
                ship_ids_list.append(ship_ids.reshape(-1).cpu().numpy())

                # Create Team IDs (Flattened)
                # Team IDs input was (1, T, N)
                team_ids_list.append(input_team_ids.reshape(-1).cpu().numpy())

                # Create Timesteps matching flatten order
                timesteps = torch.arange(T_dim, device=device).view(1, T_dim, 1).expand(B_dim, T_dim, N_dim)
                timesteps_list.append(timesteps.reshape(-1).cpu().numpy())

                # Get target actions for coloring (actions at T)
                # all_actions[start_idx : start_idx + seq_len]
                target_actions_np = all_actions[start_idx : start_idx + seq_len]
                actions_list.append(target_actions_np.reshape(-1, 3))

                # Get alive status from health (index 1)
                # tokens_np is (T, N, D)
                health = tokens_np[..., 1]
                is_alive = (health > 0).astype(np.float32)
                alive_list.append(is_alive.reshape(-1))

                # Value
                values_list.append(pred_val.reshape(-1).cpu().numpy())

                count += 1
                if count >= max_batches:
                    break

        print(f"Processed {count} episodes.")

    return (
        np.concatenate(embeddings_list, axis=0),
        np.concatenate(actions_list, axis=0),
        np.concatenate(values_list, axis=0),
        np.concatenate(ship_ids_list, axis=0),
        np.concatenate(team_ids_list, axis=0), # Added
        np.concatenate(timesteps_list, axis=0),
        np.concatenate(alive_list, axis=0),
    )


def plot_with_legend(projections, labels, label_map, title, filename, is_enemy=None):
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
        colors = [cmap(i / num_labels) for i in range(num_labels)]

    for i, label_val in enumerate(unique_labels):
        mask = labels == label_val
        key = int(label_val)
        label_name = label_map.get(key, str(key))

        # Plot parts based on enemy status if available
        if is_enemy is not None:
            # Allies (dots)
            ally_mask = mask & (is_enemy == 0)
            if np.any(ally_mask):
                plt.scatter(
                    projections[ally_mask, 0],
                    projections[ally_mask, 1],
                    c=[colors[i]],
                    s=12,
                    alpha=0.7,
                    label=label_name
                    if i == 0
                    or label_name not in plt.gca().get_legend_handles_labels()[1]
                    else None,
                    marker="o",
                    edgecolors="none",
                )

            # Enemies (crosses)
            enemy_mask = mask & (is_enemy == 1)
            if np.any(enemy_mask):
                has_label = label_name in plt.gca().get_legend_handles_labels()[1]
                plt.scatter(
                    projections[enemy_mask, 0],
                    projections[enemy_mask, 1],
                    c=[colors[i]],
                    s=20, 
                    alpha=0.7,
                    label=label_name if not has_label else None,
                    marker="+",
                    linewidths=1.0,
                )
        else:
            plt.scatter(
                projections[mask, 0],
                projections[mask, 1],
                c=[colors[i]],
                s=12,
                alpha=0.7,
                label=label_name,
                edgecolors="none",
            )

    handles, labels_leg = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))
    plt.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        frameon=False,
    )

    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_continuous(projections, values, title, filename, is_enemy=None):
    plt.figure(figsize=(10, 8))

    vmin, vmax = np.min(values), np.max(values)

    if is_enemy is not None:
        # Allies
        mask_ally = is_enemy == 0
        if np.any(mask_ally):
            plt.scatter(
                projections[mask_ally, 0],
                projections[mask_ally, 1],
                c=values[mask_ally],
                cmap="turbo",
                s=2,
                alpha=0.6,
                vmin=vmin,
                vmax=vmax,
                marker="o",
            )

        # Enemies
        mask_enemy = is_enemy == 1
        if np.any(mask_enemy):
            plt.scatter(
                projections[mask_enemy, 0],
                projections[mask_enemy, 1],
                c=values[mask_enemy],
                cmap="turbo",
                s=10, 
                alpha=0.7, 
                vmin=vmin,
                vmax=vmax,
                marker="+",
                linewidths=0.5,
            )

        sm = plt.cm.ScalarMappable(
            cmap="turbo", norm=plt.Normalize(vmin=vmin, vmax=vmax)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, label="Value", ax=plt.gca())
    else:
        scatter = plt.scatter(
            projections[:, 0], projections[:, 1], c=values, cmap="turbo", s=2, alpha=0.6
        )
        cbar = plt.colorbar(scatter, label="Value")

    cbar.solids.set_alpha(1)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_and_save(
    projections,
    actions,
    values,
    ship_ids,
    team_ids,
    timesteps,
    alive_status,
    method_name,
    save_dir,
    model_name,
    suffix="",
):
    # Action indices: (Power, Turn, Shoot)
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

    team_map = {0: "Ally (Team 0)", 1: "Enemy (Team 1)"}
    alive_map = {0: "Dead", 1: "Alive"}

    title_suffix = f" ({suffix.replace('_', ' ').strip()})" if suffix else ""

    # 1. Power Actions
    plot_with_legend(
        projections,
        power_actions,
        power_map,
        f"{model_name} - {method_name} - Power Actions{title_suffix}",
        save_dir / f"{method_name}_power{suffix}.png",
        is_enemy=team_ids,
    )

    # 2. Turn Actions
    plot_with_legend(
        projections,
        turn_actions,
        turn_map,
        f"{model_name} - {method_name} - Turn Actions{title_suffix}",
        save_dir / f"{method_name}_turn{suffix}.png",
        is_enemy=team_ids,
    )

    # 3. Shoot Actions
    plot_with_legend(
        projections,
        shoot_actions,
        shoot_map,
        f"{model_name} - {method_name} - Shoot Actions{title_suffix}",
        save_dir / f"{method_name}_shoot{suffix}.png",
        is_enemy=team_ids,
    )

    # 4. Values (Might be 0 if model doesn't output it)
    plot_continuous(
        projections,
        values,
        f"{model_name} - {method_name} - Value{title_suffix}",
        save_dir / f"{method_name}_value{suffix}.png",
        is_enemy=team_ids,
    )

    # 5. Ship IDs
    plot_with_legend(
        projections,
        ship_ids,
        ship_map,
        f"{model_name} - {method_name} - Ship ID{title_suffix}",
        save_dir / f"{method_name}_ship{suffix}.png",
        is_enemy=team_ids,
    )

    # 6. Team IDs
    plot_with_legend(
        projections,
        team_ids,
        team_map,
        f"{model_name} - {method_name} - Team ID{title_suffix}",
        save_dir / f"{method_name}_team{suffix}.png",
        is_enemy=team_ids,
    )

    # 7. Timesteps
    plot_continuous(
        projections,
        timesteps,
        f"{model_name} - {method_name} - Timestep{title_suffix}",
        save_dir / f"{method_name}_timestep{suffix}.png",
        is_enemy=team_ids,
    )

    # 8. Alive Status
    plot_with_legend(
        projections,
        alive_status,
        alive_map,
        f"{model_name} - {method_name} - Alive Status{title_suffix}",
        save_dir / f"{method_name}_alive{suffix}.png",
        is_enemy=team_ids,
    )


def generate_report(model_name, run_dir, config, output_dir, passes):
    report_path = output_dir / "report.md"
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

        for p in passes:
            suffix = p["suffix"]
            name = p["name"]
            f.write(f"## Visualizations ({name})\n")

            f.write("### PCA Projections\n")
            f.write(f"![PCA Power](PCA_power{suffix}.png)\n")
            f.write(f"![PCA Turn](PCA_turn{suffix}.png)\n")
            f.write(f"![PCA Shoot](PCA_shoot{suffix}.png)\n")
            f.write(f"![PCA Value](PCA_value{suffix}.png)\n")
            f.write(f"![PCA Ship](PCA_ship{suffix}.png)\n")
            f.write(f"![PCA Team](PCA_team{suffix}.png)\n")
            f.write(f"![PCA Timestep](PCA_timestep{suffix}.png)\n")
            f.write(f"![PCA Alive](PCA_alive{suffix}.png)\n\n")
            
            f.write("### PaCMAP Projections\n")
            f.write(f"![PaCMAP Power](PaCMAP_power{suffix}.png)\n")
            f.write(f"![PaCMAP Turn](PaCMAP_turn{suffix}.png)\n")
            f.write(f"![PaCMAP Shoot](PaCMAP_shoot{suffix}.png)\n")
            f.write(f"![PaCMAP Value](PaCMAP_value{suffix}.png)\n")
            f.write(f"![PaCMAP Ship](PaCMAP_ship{suffix}.png)\n")
            f.write(f"![PaCMAP Team](PaCMAP_team{suffix}.png)\n")
            f.write(f"![PaCMAP Timestep](PaCMAP_timestep{suffix}.png)\n")
            f.write(f"![PaCMAP Alive](PaCMAP_alive{suffix}.png)\n\n")
            
    print(f"Report generated at: {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, default="models/world_model")
    parser.add_argument("--output_dir", type=str, default="outputs/latent_viz")
    parser.add_argument("--max_batches", type=int, default=100) # Reduced default
    parser.add_argument(
        "--test-run", action="store_true", help="Run quick test on single model"
    )
    parser.add_argument(
        "--latest", action="store_true", help="Process only the most recent model"
    )
    parser.add_argument(
        "--data_path", type=str, default=None, help="Path to specific HDF5 data file"
    )
    args = parser.parse_args()

    base_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Find run directories
    run_dirs = [
        d for d in base_dir.iterdir() if d.is_dir() and (d / "config.yaml").exists()
    ]

    if args.latest and run_dirs:
        run_dirs.sort(key=lambda d: d.stat().st_mtime)
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

            if args.data_path:
                data_path = args.data_path
            else:
                data_path = load_bc_data() # Gets latest by default

            print(f"Loading data from: {data_path}")
            # Validating data path exists
            if not Path(data_path).exists():
                print(f"Data path {data_path} does not exist. Skipping.")
                continue

            model = load_model(run_dir, config, device)

            embeddings, actions, values, ship_ids, team_ids, timesteps, alive_status = (
                extract_embeddings(
                    model,
                    data_path,
                    device,
                    max_batches=args.max_batches,
                    seq_len=config.get("world_model", {}).get("context_len", 96),
                )
            )

            raw_data = {
                "embeddings": embeddings,
                "actions": actions,
                "values": values,
                "ship_ids": ship_ids,
                "team_ids": team_ids,
                "timesteps": timesteps,
                "alive_status": alive_status,
            }

            passes = [
                {
                    "name": "All Data",
                    "suffix": "",
                    "sample_limit": 10000,
                    "mask": np.ones(len(embeddings), dtype=bool),
                },
                {
                    "name": "Alive Only",
                    "suffix": "_alive",
                    "sample_limit": 10000,
                    "mask": raw_data["alive_status"] == 1,
                },
                {
                    "name": "Ally Alive",
                    "suffix": "_ally_alive",
                    "sample_limit": 10000,
                    "mask": (raw_data["alive_status"] == 1)
                    & (raw_data["team_ids"] == 0),
                },
                {
                    "name": "Enemy Alive",
                    "suffix": "_enemy_alive",
                    "sample_limit": 10000,
                    "mask": (raw_data["alive_status"] == 1)
                    & (raw_data["team_ids"] == 1),
                },
            ]

            model_out_dir = output_dir / model_name
            model_out_dir.mkdir(parents=True, exist_ok=True)

            for p in passes:
                print(f"\nRunning pass: {p['name']}")
                mask = p["mask"]
                suffix = p["suffix"]

                if np.sum(mask) == 0:
                    print(f"Skipping {p['name']} due to empty mask.")
                    continue

                # Filter
                curr_emb = raw_data["embeddings"][mask]
                curr_act = raw_data["actions"][mask]
                curr_val = raw_data["values"][mask]
                curr_sid = raw_data["ship_ids"][mask]
                curr_tid = raw_data["team_ids"][mask]
                curr_time = raw_data["timesteps"][mask]
                curr_alive = raw_data["alive_status"][mask]

                # Subsample for plotting if too large
                sample_limit = p.get("sample_limit", 10000)
                if len(curr_emb) > sample_limit:
                    indices = np.random.choice(
                        len(curr_emb), sample_limit, replace=False
                    )
                    curr_emb = curr_emb[indices]
                    curr_act = curr_act[indices]
                    curr_val = curr_val[indices]
                    curr_sid = curr_sid[indices]
                    curr_tid = curr_tid[indices]
                    curr_time = curr_time[indices]
                    curr_alive = curr_alive[indices]

                print(f"Projecting {len(curr_emb)} points...")

                # PCA
                print("Running PCA...")
                pca = PCA(n_components=2)
                pca_proj = pca.fit_transform(curr_emb)
                plot_and_save(
                    pca_proj,
                    curr_act,
                    curr_val,
                    curr_sid,
                    curr_tid,
                    curr_time,
                    curr_alive,
                    "PCA",
                    model_out_dir,
                    model_name,
                    suffix=suffix,
                )

                # PaCMAP
                print("Running PaCMAP...")
                try:
                    embedding_learner = pacmap.PaCMAP(
                        n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0
                    )
                    pacmap_proj = embedding_learner.fit_transform(curr_emb, init="pca")
                    plot_and_save(
                        pacmap_proj,
                        curr_act,
                        curr_val,
                        curr_sid,
                        curr_tid,
                        curr_time,
                        curr_alive,
                        "PaCMAP",
                        model_out_dir,
                        model_name,
                        suffix=suffix,
                    )
                except Exception as e:
                    print(f"PaCMAP failed: {e}. Skipping.")

            # Generate Report
            generate_report(model_name, run_dir, config, model_out_dir, passes)

            print(f"Done processing {model_name}")

        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
