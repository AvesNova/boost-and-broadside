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
import random
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agents.interleaved_world_model import InterleavedWorldModel
from train.data_loader import load_bc_data, create_unified_data_loaders



log = logging.getLogger(__name__)


def get_rollout_length(epoch: int, cfg: DictConfig) -> int:
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
    
    device = input_states.device
    
    with torch.no_grad():
        # Current window variables
        curr_states = input_states.clone()
        curr_actions = input_actions.clone()
        
        for i in range(rollout_len):
            t = start_t + i
            
            # Context: 0..t (inclusive of t for State input)
            # We want to predict A_t from S_t.
            # And S_{t+1} from A_t.
            
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





def train_world_model(cfg: DictConfig) -> None:
    """
    Train the world model.
    """
    log.info("Starting World Model training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Load Data
    data_path = cfg.train.bc_data_path

    log.info(f"Loading data from {data_path}")
    data_path = load_bc_data(data_path)

    # Initialize Model
    # Get dimensions from data
    import h5py

    with h5py.File(data_path, "r") as f:
        # Use attributes if available (from aggregation) or check shape
        if "token_dim" in f.attrs:
            state_dim = int(f.attrs["token_dim"])
        else:
            state_dim = f["tokens"].shape[-1]

    # Fixed action dim for one-hot encoded actions (3 + 7 + 2) is 12, but input is 3 discrete
    model = InterleavedWorldModel(
        state_dim=state_dim,
        embed_dim=cfg.world_model.embed_dim,
        n_layers=cfg.world_model.n_layers,
        n_heads=cfg.world_model.n_heads,
        max_ships=cfg.world_model.n_ships,
        max_context_len=cfg.world_model.context_len,
        dropout=0.1,
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
        f.write(
            "epoch,train_loss,train_state_loss,train_action_loss,val_loss\n"
        )

    # Training Loop
    epochs = cfg.world_model.epochs
    batch_ratio = cfg.world_model.batch_ratio
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Re-create data loaders each epoch to randomize pools
        train_short_loader, train_long_loader, val_short_loader, val_long_loader = (
            create_unified_data_loaders(
                data_path,
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
        total_error_power = 0
        total_error_turn = 0
        total_error_shoot = 0

        # Create iterators
        short_iter = iter(train_short_loader)
        long_iter = iter(train_long_loader)

        num_short_batches = len(train_short_loader)
        pbar = tqdm(range(num_short_batches), desc=f"Epoch {epoch + 1}/{epochs}")
        short_exhausted = False
        steps = 0

        while not short_exhausted:
            # 1. Run batch_ratio short batches
            for _ in range(batch_ratio):
                try:
                    (
                        states,
                        actions,
                        returns,
                        loss_mask,
                        action_masks,
                        agent_skills,
                        team_ids,
                    ) = next(short_iter)
                except StopIteration:
                    short_exhausted = True
                    break

                states = states.to(device)
                actions = actions.to(device) # Raw indices
                loss_mask = loss_mask.to(device)
                team_ids = team_ids.to(device)

                # Input: [S_0..S_{T-2}], [A_0..A_{T-2}] (Shifted actions)
                # Note: Actions from loader are [A_0..A_{T-1}]
                # We need Input Action at t to be Action_{t-1}.
                # So Input Action 0 is 0.
                
                prev_actions = torch.zeros_like(actions)
                prev_actions[:, 1:] = actions[:, :-1]
                prev_actions[:, 0] = 0

                input_states = states[:, :-1]
                input_actions = prev_actions[:, :-1]
                
                num_ships = states.shape[2]
                
                # Fix Team IDs
                if team_ids.ndim == 2 and team_ids.shape[1] != num_ships:
                     # Reconstruct standard (0,0,0,0, 1,1,1,1)
                     tid = torch.zeros((states.shape[0], num_ships), device=device, dtype=torch.long)
                     half = num_ships // 2
                     tid[:, half:] = 1
                     input_team_ids = tid
                elif team_ids.ndim == 3:
                     # (B, T, N) -> Slice time AND ships
                     input_team_ids = team_ids[:, :-1, :num_ships]
                else:
                     input_team_ids = team_ids

                target_states = states[:, 1:]
                target_actions = actions[:, :-1]
                
                loss_mask_slice = loss_mask[:, 1:]

                # Rollout Injection
                rollout_len = get_rollout_length(epoch, cfg)
                perform_rollout(model, input_states, input_actions, input_team_ids, rollout_len)

                # Train Step
                optimizer.zero_grad()

                pred_states, pred_actions, _ = model(
                    input_states,
                    input_actions,
                    input_team_ids,
                    noise_scale=cfg.world_model.noise_scale
                )

                loss, state_loss, action_loss = model.get_loss(
                    pred_states=pred_states,
                    pred_actions=pred_actions,
                    target_states=target_states,
                    target_actions=target_actions,
                    loss_mask=loss_mask_slice
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                total_state_loss += state_loss.item()
                total_action_loss += action_loss.item()
                
                with torch.no_grad():
                    valid_mask = loss_mask_slice.bool()
                    valid_pred = pred_actions[valid_mask]
                    valid_target = target_actions[valid_mask]
                    
                    p_logits = valid_pred[..., 0:3].reshape(-1, 3)
                    t_logits = valid_pred[..., 3:10].reshape(-1, 7)
                    s_logits = valid_pred[..., 10:12].reshape(-1, 2)
                    
                    p_target = valid_target[..., 0].long().reshape(-1)
                    t_target = valid_target[..., 1].long().reshape(-1)
                    s_target = valid_target[..., 2].long().reshape(-1)
                    
                    if p_target.numel() > 0:
                        total_error_power += (p_logits.argmax(-1) != p_target).float().mean().item()
                        total_error_turn += (t_logits.argmax(-1) != t_target).float().mean().item()
                        total_error_shoot += (s_logits.argmax(-1) != s_target).float().mean().item()

                steps += 1
                pbar.update(1)

            if short_exhausted:
                break

            # 2. Run 1 long batch
            try:
                (states, actions, returns, loss_mask, action_masks, agent_skills, team_ids) = next(long_iter)
            except StopIteration:
                break

            states = states.to(device)
            loss_mask = loss_mask.to(device)
            actions = actions.to(device)
            team_ids = team_ids.to(device)

            prev_actions = torch.zeros_like(actions)
            prev_actions[:, 1:] = actions[:, :-1]
            prev_actions[:, 0] = 0

            input_states = states[:, :-1]
            input_actions = prev_actions[:, :-1]
            
            num_ships = states.shape[2]
            
            # Fix Team IDs
            if team_ids.ndim == 2 and team_ids.shape[1] != num_ships:
                    # Reconstruct standard
                    tid = torch.zeros((states.shape[0], num_ships), device=device, dtype=torch.long)
                    half = num_ships // 2
                    tid[:, half:] = 1
                    input_team_ids = tid
            elif team_ids.ndim == 3:
                     # (B, T, N) -> Slice time AND ships
                     input_team_ids = team_ids[:, :-1, :num_ships]
            else:
                     input_team_ids = team_ids

            target_states = states[:, 1:]
            target_actions = actions[:, :-1]

            loss_mask_slice = loss_mask[:, 1:]
            if loss_mask_slice.shape[1] > 32:
                loss_mask_slice[:, :32] = False
            
            # Rollout
            rollout_len = get_rollout_length(epoch, cfg)
            perform_rollout(model, input_states, input_actions, input_team_ids, rollout_len)

            optimizer.zero_grad()
            
            pred_states, pred_actions, _ = model(
                input_states,
                input_actions,
                input_team_ids,
                noise_scale=cfg.world_model.noise_scale
            )
            
            loss, state_loss, action_loss = model.get_loss(
                pred_states=pred_states,
                pred_actions=pred_actions,
                target_states=target_states,
                target_actions=target_actions,
                loss_mask=loss_mask_slice
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_state_loss += state_loss.item()
            total_action_loss += action_loss.item()
            
            steps += 1
            pbar.set_postfix({
                "loss": total_loss / steps,
                "state": total_state_loss / steps,
                "action": total_action_loss / steps
            })

        avg_loss = total_loss / steps if steps > 0 else 0
        avg_state_loss = total_state_loss / steps if steps > 0 else 0
        avg_action_loss = total_action_loss / steps if steps > 0 else 0
        avg_err_p = total_error_power / steps if steps > 0 else 0
        avg_err_t = total_error_turn / steps if steps > 0 else 0
        avg_err_s = total_error_shoot / steps if steps > 0 else 0

        log.info(f"Epoch {epoch + 1}: Train Loss={avg_loss:.4f} (State={avg_state_loss:.4f}, Action={avg_action_loss:.4f})")

        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        val_error_p = 0
        val_error_t = 0
        val_error_s = 0

        for loader in [val_short_loader, val_long_loader]:
            with torch.no_grad():
                for states, actions, returns, loss_mask, _, _, team_ids in loader:
                    states = states.to(device)
                    actions = actions.to(device)
                    loss_mask = loss_mask.to(device)
                    team_ids = team_ids.to(device)

                    prev_actions = torch.zeros_like(actions)
                    prev_actions[:, 1:] = actions[:, :-1]
                    prev_actions[:, 0] = 0

                    input_states = states[:, :-1]
                    input_actions = prev_actions[:, :-1]
                    
                    num_ships = states.shape[2]
                    
                    # Fix Team IDs (Validation)
                    if team_ids.ndim == 2 and team_ids.shape[1] != num_ships:
                         tid = torch.zeros((states.shape[0], num_ships), device=device, dtype=torch.long)
                         half = num_ships // 2
                         tid[:, half:] = 1
                         input_team_ids = tid
                    elif team_ids.ndim == 3:
                         input_team_ids = team_ids[:, :-1, :num_ships]
                    else:
                         input_team_ids = team_ids

                    target_states = states[:, 1:]
                    target_actions = actions[:, :-1]
                    
                    loss_mask_slice = loss_mask[:, 1:]

                    pred_states, pred_actions, _ = model(
                        input_states, input_actions, input_team_ids, noise_scale=0.0
                    )

                    loss, state_loss, action_loss = model.get_loss(
                        pred_states=pred_states,
                        pred_actions=pred_actions,
                        target_states=target_states,
                        target_actions=target_actions,
                        loss_mask=loss_mask_slice
                    )
                    
                    val_loss += loss.item()
                    val_steps += 1
                    
                    valid_mask = loss_mask_slice.bool()
                    valid_pred = pred_actions[valid_mask]
                    valid_target = target_actions[valid_mask]
                    
                    p_logits = valid_pred[..., 0:3].reshape(-1, 3)
                    t_logits = valid_pred[..., 3:10].reshape(-1, 7)
                    s_logits = valid_pred[..., 10:12].reshape(-1, 2)
                    
                    p_target = valid_target[..., 0].long().reshape(-1)
                    t_target = valid_target[..., 1].long().reshape(-1)
                    s_target = valid_target[..., 2].long().reshape(-1)
                    
                    if p_target.numel() > 0:
                        val_error_p += (p_logits.argmax(-1) != p_target).float().mean().item()
                        val_error_t += (t_logits.argmax(-1) != t_target).float().mean().item()
                        val_error_s += (s_logits.argmax(-1) != s_target).float().mean().item()

        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
        avg_val_err_p = val_error_p / val_steps if val_steps > 0 else 0
        avg_val_err_t = val_error_t / val_steps if val_steps > 0 else 0
        avg_val_err_s = val_error_s / val_steps if val_steps > 0 else 0

        log.info(f"Epoch {epoch + 1}: Val Loss={avg_val_loss:.4f}")

        # Log to CSV
        with open(csv_path, "a") as f:
            f.write(
                f"{epoch + 1},{avg_loss:.6f},{avg_state_loss:.6f},{avg_action_loss:.6f},{avg_val_loss:.6f}\n"
            )

        # Log to TensorBoard
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Loss/train_state", avg_state_loss, epoch)
        writer.add_scalar("Loss/train_action", avg_action_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Error/train_power", avg_err_p, epoch)
        writer.add_scalar("Error/train_turn", avg_err_t, epoch)
        writer.add_scalar("Error/train_shoot", avg_err_s, epoch)
        writer.add_scalar("Error/val_power", avg_val_err_p, epoch)
        writer.add_scalar("Error/val_turn", avg_val_err_t, epoch)
        writer.add_scalar("Error/val_shoot", avg_val_err_s, epoch)

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), run_dir / "best_world_model.pth")
            log.info(f"Saved best model with val loss {best_val_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            save_path = run_dir / f"world_model_epoch_{epoch + 1}.pt"
            torch.save(model.state_dict(), save_path)
            log.info(f"Saved checkpoint to {save_path}")

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
    writer.close()
    log.info("World Model training complete.")
