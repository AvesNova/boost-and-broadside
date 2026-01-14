"""
World model training script.

Trains a transformer-based world model to predict future states and actions
using masked reconstruction and denoising objectives.
"""

import logging
from pathlib import Path

from datetime import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agents.world_model import WorldModel
from train.data_loader import load_bc_data, create_unified_data_loaders
from utils.tensor_utils import to_one_hot
from eval.metrics import compute_dreaming_error, compute_controlling_error
from eval.rollout_metrics import compute_rollout_metrics

log = logging.getLogger(__name__)





def to_one_hot(actions: torch.Tensor) -> torch.Tensor:
    """
    Convert discrete action indices to concatenated one-hot vectors.

    Args:
        actions: (..., 3) tensor of action indices [power, turn, shoot]

    Returns:
        (..., 12) tensor of one-hot actions
    """
    power = actions[..., 0].long()
    turn = actions[..., 1].long()
    shoot = actions[..., 2].long()

    power_oh = F.one_hot(power, num_classes=3)
    turn_oh = F.one_hot(turn, num_classes=7)
    shoot_oh = F.one_hot(shoot, num_classes=2)

    return torch.cat([power_oh, turn_oh, shoot_oh], dim=-1).float()


def train_world_model(cfg: DictConfig) -> None:
    """
    Train the world model.

    Args:
        cfg: Hydra configuration object.
    """
    log.info("Starting World Model training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Load Data
    data_path = cfg.train.bc_data_path

    log.info(f"Loading data from {data_path}")
    data = load_bc_data(data_path)

    # Initialize Model
    # Get dimensions from data
    # We need to peek at data to get dimensions.
    # Let's just use hardcoded dimensions or get them from config/data structure.
    # The data loader creation is now inside the loop.
    # But we need dimensions to init model.

    # Peek at one sample
    team_0 = data["team_0"]
    sample_tokens = team_0["tokens"][0]
    # sample_actions = team_0["actions"][0] 

    state_dim = sample_tokens.shape[-1]
    # Fixed action dim for one-hot encoded actions (3 + 7 + 2)
    action_dim = 12 

    model = WorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=cfg.world_model.embed_dim,
        n_layers=cfg.world_model.n_layers,
        n_heads=cfg.world_model.n_heads,
        max_ships=cfg.world_model.n_ships,
        max_context_len=cfg.world_model.context_len,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.world_model.learning_rate)

    # Setup logging and output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("models/world_model") / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Output directory: {run_dir}")

    # Save config immediately
    OmegaConf.save(cfg, run_dir / "config.yaml")

    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(run_dir))

    csv_path = run_dir / "training_log.csv"
    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,train_state_loss,train_action_loss,train_value_loss,val_loss,rollout_mse_sim,rollout_mse_dream\n")

    # Training Loop
    epochs = cfg.world_model.epochs
    batch_ratio = cfg.world_model.batch_ratio
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Re-create data loaders each epoch to randomize pools
        train_short_loader, train_long_loader, val_short_loader, val_long_loader = (
            create_unified_data_loaders(
                data,
                short_batch_size=cfg.world_model.short_batch_size,
                long_batch_size=cfg.world_model.long_batch_size,
                short_batch_len=cfg.world_model.short_batch_len,
                long_batch_len=cfg.world_model.long_batch_len,
                batch_ratio=cfg.world_model.batch_ratio,
                validation_split=0.2,
                num_workers=0,
            )
        )

        model.train()
        total_loss = 0
        total_state_loss = 0
        total_action_loss = 0
        total_value_loss = 0

        # Create iterators
        short_iter = iter(train_short_loader)
        long_iter = iter(train_long_loader)

        # Determine number of steps
        # We run until one of the loaders is exhausted?
        # Or we determine steps based on the ratio.
        # Let's run until short loader is exhausted, as it's the dominant one.
        num_short_batches = len(train_short_loader)

        pbar = tqdm(range(num_short_batches), desc=f"Epoch {epoch+1}/{epochs}")

        short_exhausted = False

        steps = 0

        while not short_exhausted:
            # 1. Run batch_ratio short batches
            for _ in range(batch_ratio):
                try:
                    states, actions, returns, loss_mask, action_masks = next(short_iter)
                except StopIteration:
                    short_exhausted = True
                    break
                
                states = states.to(device)
                actions = actions.to(device)
                returns = returns.to(device)
                loss_mask = loss_mask.to(device)
                if action_masks is not None:
                    action_masks = action_masks.to(device)
                
                # Prepare Inputs and Targets for AR Prediction
                # Input at t: Token_t = [State_t, Action_{t-1}]
                # Target at t: Token_{t+1} = [State_{t+1}, Action_t] + Value_{t+1}
                #
                # Data Layout:
                # states: [S_0, S_1, ..., S_{T-1}]
                # actions: [A_0, A_1, ..., A_{T-1}]
                # returns: [V_0, V_1, ..., V_{T-1}]
                
                # Shift actions to get previous actions (Input component)
                # Action_{-1} is assumed 0 (padding)
                batch_size, time_steps, num_ships = states.shape[:3]
                
                prev_actions = torch.zeros_like(actions)
                prev_actions[:, 1:] = actions[:, :-1]
                prev_actions[:, 0] = 0 # First step previous action is 0
                
                # Slice indices
                # Input: Tokens 0..T-2
                input_states = states[:, :-1]
                input_prev_actions = prev_actions[:, :-1]
                
                # Target: Tokens 1..T-1
                target_states = states[:, 1:]
                target_actions = actions[:, :-1] # This IS Action_t for the input step t
                target_returns = returns[:, 1:] # Value of State_{t+1}
                
                # Adjust masks
                # Short batches: loss_mask handles padding. Just slice it.
                loss_mask_slice = loss_mask[:, 1:] # Valid if target is valid
                action_masks_slice = action_masks[:, 1:] if action_masks is not None else None

                # Convert inputs to One-Hot
                input_actions_oh = to_one_hot(input_prev_actions)

                # Input Noise Injection
                if cfg.world_model.input_noise_ratio > 0:
                     noise_mask = (torch.rand(input_states.shape[:3], device=device) < cfg.world_model.input_noise_ratio)
                     noise = torch.randn_like(input_states) * cfg.world_model.input_noise_scale
                     input_states = torch.where(noise_mask.unsqueeze(-1), input_states + noise, input_states)

                optimizer.zero_grad()

                pred_states, pred_actions, pred_value, _, _ = model(
                    input_states,
                    input_actions_oh,
                    mask_ratio=0.0,
                    noise_scale=0.0,
                    mask=None,
                )

                state_loss, action_loss, value_loss = model.get_loss(
                    pred_states=pred_states,
                    pred_actions=pred_actions,
                    pred_value=pred_value,
                    target_states=target_states,
                    target_actions=target_actions,
                    target_returns=target_returns,
                    loss_mask=loss_mask_slice,
                    action_masks=action_masks_slice,
                )

                loss = state_loss + action_loss + value_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                total_state_loss += state_loss.item()
                total_action_loss += action_loss.item()
                total_value_loss += value_loss.item()
                steps += 1
                pbar.update(1)

            if short_exhausted:
                break

            # 2. Run 1 long batch
            try:
                states, actions, returns, loss_mask, action_masks = next(long_iter)
            except StopIteration:
                # If long loader exhausted, just skip
                break
            
            states = states.to(device)
            loss_mask = loss_mask.to(device)
            returns = returns.to(device)
            action_masks = action_masks.to(device)
            actions = actions.to(device) # Raw indices

            # Shift Logic (Same as above)
            prev_actions = torch.zeros_like(actions)
            prev_actions[:, 1:] = actions[:, :-1]
            prev_actions[:, 0] = 0
            
            input_states = states[:, :-1]
            input_prev_actions = prev_actions[:, :-1]
            
            target_states = states[:, 1:]
            target_actions = actions[:, :-1]
            target_returns = returns[:, 1:]
            
            # Loss Mask Modification for Long Batches
            # Context window: 32 steps. We want to predict FROM step 32 onwards.
            # Input index 31 -> Target index 32.
            # So valid targets start from index 31 (which corresponds to target T=32)
            # Actually user said "split training into ... length 32 and long batches of length 128 with context window 96"
            # It seems user implied first 32 is context.
            # loss_mask for long batch has all 1s (usually).
            # We want to ZERO out the first 32 steps of the LOSS computation.
            # loss_mask_slice currently corresponds to Targets 1..127.
            # We want to mask out Targets 1..31. Keep 32..127.
            # So indices 0..30 of the slice should be False.
            
            loss_mask_slice = loss_mask[:, 1:]
            # Only apply context masking if sequence is long enough
            if loss_mask_slice.shape[1] > 32:
                 loss_mask_slice[:, :32] = False
            
            action_masks_slice = action_masks[:, 1:] if action_masks is not None else None

            input_actions_oh = to_one_hot(input_prev_actions)

            # Input Noise Injection
            if cfg.world_model.input_noise_ratio > 0:
                    noise_mask = (torch.rand(input_states.shape[:3], device=device) < cfg.world_model.input_noise_ratio)
                    noise = torch.randn_like(input_states) * cfg.world_model.input_noise_scale
                    input_states = torch.where(noise_mask.unsqueeze(-1), input_states + noise, input_states)

            optimizer.zero_grad()

            pred_states, pred_actions, pred_value, _, _ = model(
                input_states,
                input_actions_oh,
                mask_ratio=0.0,
                noise_scale=0.0,
                mask=None,
            )

            state_loss, action_loss, value_loss = model.get_loss(
                pred_states=pred_states,
                pred_actions=pred_actions,
                pred_value=pred_value,
                target_states=target_states,
                target_actions=target_actions,
                target_returns=target_returns,
                loss_mask=loss_mask_slice,
                action_masks=action_masks_slice,
            )

            loss = state_loss + action_loss + value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_state_loss += state_loss.item()
            total_action_loss += action_loss.item()
            total_value_loss += value_loss.item()
            
            steps += 1

            pbar.set_postfix(
                {
                    "loss": total_loss / steps,
                    "state": total_state_loss / steps,
                    "action": total_action_loss / steps,
                    "value": total_value_loss / steps,
                }
            )

        avg_loss = total_loss / steps if steps > 0 else 0
        avg_state_loss = total_state_loss / steps if steps > 0 else 0
        avg_action_loss = total_action_loss / steps if steps > 0 else 0
        avg_value_loss = total_value_loss / steps if steps > 0 else 0

        log.info(
            f"Epoch {epoch+1}: Train Loss={avg_loss:.4f} "
            f"(State={avg_state_loss:.4f}, "
            f"Action={avg_action_loss:.4f}, "
            f"Value={avg_value_loss:.4f})"
        )

        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0

        # Validate on both short and long
        for loader in [val_short_loader, val_long_loader]:
            with torch.no_grad():
                for states, actions, returns, loss_mask, action_masks in loader:
                    states = states.to(device)
                    actions = actions.to(device) # Raw indices (B, T, N, 3)
                    returns = returns.to(device)
                    loss_mask = loss_mask.to(device)
                    action_masks = action_masks.to(device)

                    # Prepare inputs/targets for validation (Same shift logic)
                    prev_actions = torch.zeros_like(actions)
                    prev_actions[:, 1:] = actions[:, :-1]
                    prev_actions[:, 0] = 0

                    input_states = states[:, :-1]
                    input_prev_actions = prev_actions[:, :-1]

                    target_states = states[:, 1:]
                    target_actions = actions[:, :-1]
                    target_returns = returns[:, 1:]

                    loss_mask_slice = loss_mask[:, 1:]
                    action_masks_slice = action_masks[:, 1:] if action_masks is not None else None

                    input_actions_oh = to_one_hot(input_prev_actions)

                    pred_states, pred_actions, pred_value, _, _ = model(
                        input_states, input_actions_oh, mask_ratio=0.0
                    )

                    state_loss, action_loss, value_loss = model.get_loss(
                        pred_states=pred_states,
                        pred_actions=pred_actions,
                        pred_value=pred_value,
                        target_states=target_states,
                        target_actions=target_actions,
                        target_returns=target_returns,
                        loss_mask=loss_mask_slice,
                        action_masks=action_masks_slice,
                    )
                    val_loss += (state_loss + action_loss + value_loss).item()
                    val_steps += 1

        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
        log.info(f"Epoch {epoch+1}: Val Loss={avg_val_loss:.4f}")

        # Log to CSV
        with open(csv_path, "a") as f:
            f.write(
                f"{epoch+1},{avg_loss:.6f},{avg_state_loss:.6f},{avg_action_loss:.6f},{avg_value_loss:.6f},{avg_val_loss:.6f},"
            )
            # Will append metrics after computing them
            # Wait, avoiding partial file write issues. Let's compute then write.
            # But the existing code writes here.
            # I will modify this block to write AFTER metrics computation.

        # Log to TensorBoard
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Loss/train_state", avg_state_loss, epoch)
        writer.add_scalar("Loss/train_action", avg_action_loss, epoch)
        writer.add_scalar("Loss/train_value", avg_value_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        
        # --- Compute and Log Rollout Metrics ---
        # Get env config from cfg
        env_config = OmegaConf.to_container(cfg.environment, resolve=True)
        # Ensure render_mode is none for headless eval
        env_config["render_mode"] = "none"
        
        # New Metrics
        rollout_metrics = compute_rollout_metrics(
            model,
            env_config,
            device,
            num_scenarios=4, # Keep it small for speed during training
            max_steps=128,
            step_intervals=[1, 2, 4, 8, 16, 32, 64, 128]
        )
        
        mse_sim = rollout_metrics["mse_sim"]
        mse_dream = rollout_metrics["mse_dream"]
        
        # Log to TensorBoard
        writer.add_scalar("Rollout/mse_sim", mse_sim, epoch)
        writer.add_scalar("Rollout/mse_dream", mse_dream, epoch)
        
        for step, mse in rollout_metrics["step_mse_sim"].items():
            writer.add_scalar(f"Rollout_Sim_Step/step_{step}", mse, epoch)
            
        for step, mse in rollout_metrics["step_mse_dream"].items():
            writer.add_scalar(f"Rollout_Dream_Step/step_{step}", mse, epoch)
            
        log.info(f"Epoch {epoch+1}: Rollout MSE Sim={mse_sim:.4f}, Dream={mse_dream:.4f}")
        
        # Finish CSV Log
        with open(csv_path, "a") as f:
            # We already wrote the first part?
            # Wait, I removed the newline in the previous edit.
            # So I can just append.
            f.write(f"{mse_sim:.6f},{mse_dream:.6f}\n")
            
        # Log Detailed Metrics to separate CSV
        detailed_csv_path = run_dir / "eval_rollout_metrics.csv"
        # If first epoch, write header
        if epoch == 0:
            with open(detailed_csv_path, "w") as f:
                # Wide format: epoch,step_0,step_1,... 
                # Note: full_mse arrays are 0-indexed (step 1 is index 0).
                # User asked for "epoch|step_1|step_2..."
                # Let's create header
                steps_header = ",".join([f"step_{i+1}" for i in range(len(rollout_metrics["full_mse_sim"]))])
                f.write(f"epoch,type,{steps_header}\n")
                
        with open(detailed_csv_path, "a") as f:
            # Sim Row
            sim_vals = ",".join([f"{v:.6f}" for v in rollout_metrics["full_mse_sim"]])
            f.write(f"{epoch+1},sim,{sim_vals}\n")
            # Dream Row
            dream_vals = ",".join([f"{v:.6f}" for v in rollout_metrics["full_mse_dream"]])
            f.write(f"{epoch+1},dream,{dream_vals}\n")
        # --------------------------------------

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), run_dir / "best_world_model.pth")
            log.info(f"Saved best model with val loss {best_val_loss:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_path = run_dir / f"world_model_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), save_path)
            log.info(f"Saved checkpoint to {save_path}")

    # Save final model
    torch.save(model.state_dict(), run_dir / "final_world_model.pth")

    # Save metadata
    metadata_path = run_dir / "model_metadata.yaml"
    metadata = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "final_metrics": {
            "train_loss": avg_loss,
            "val_loss": avg_val_loss,
            "epochs_trained": epoch + 1,
        },
    }

    OmegaConf.save(OmegaConf.create(metadata), metadata_path)

    # Close writer
    writer.close()

    log.info("World Model training complete.")
