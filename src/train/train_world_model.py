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

log = logging.getLogger(__name__)


def create_mixed_mask(
    batch_size: int,
    time_steps: int,
    num_ships: int,
    device: torch.device,
    mask_ratio: float = 0.15,
) -> torch.Tensor:
    """
    Create a mask using a mix of strategies:
    1. Random Token Masking (MAE-style)
    2. Block Masking (Random spans of 1-8 steps)
    3. Next-Step Masking (Masking the last token)

    Args:
        batch_size: Batch size
        time_steps: Number of time steps
        num_ships: Number of ships
        device: Device to create mask on
        mask_ratio: Base ratio for random masking

    Returns:
        Boolean mask tensor (B, T, N) where True = masked
    """
    # 1. Random Token Masking
    mask = torch.rand(batch_size, time_steps, num_ships, device=device) < mask_ratio

    # 2. Block Masking
    # For each ship in each batch, mask a random block
    # We'll do this for a subset of ships/batches to not over-mask
    # Let's say we apply block masking to 20% of sequences
    num_blocks = int(batch_size * num_ships * 0.2)
    if num_blocks > 0:
        # Randomly select batch and ship indices
        b_indices = torch.randint(0, batch_size, (num_blocks,), device=device)
        n_indices = torch.randint(0, num_ships, (num_blocks,), device=device)

        # Random block lengths (1-8)
        block_lens = torch.randint(1, 9, (num_blocks,), device=device)

        # Random start positions (ensure fit)
        start_pos = (
            torch.rand(num_blocks, device=device) * (time_steps - block_lens)
        ).long()

        for i in range(num_blocks):
            b, n, block_len, s = b_indices[i], n_indices[i], block_lens[i], start_pos[i]
            mask[b, s : s + block_len, n] = True

    # 3. Next-Step Masking
    # Always mask the last timestep for all ships to learn forward prediction
    # This is critical for the agent's primary use case
    mask[:, -1, :] = True

    return mask


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
        f.write("epoch,train_loss,train_recon_loss,train_denoise_loss,train_value_loss,val_loss\n")

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
        total_recon_loss = 0
        total_denoise_loss = 0
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
                
                # Convert actions to one-hot BEFORE moving to device 
                # (or after, but to_one_hot handles tensor)
                states = states.to(device)
                loss_mask = loss_mask.to(device)
                returns = returns.to(device)
                action_masks = action_masks.to(device)
                actions = actions.to(device) # Raw indices
                
                # Convert discrete actions to one-hot for INPUT
                actions_oh = to_one_hot(actions)

                optimizer.zero_grad()

                # Create mask (random masking for short sequences)
                mask = create_mixed_mask(
                    states.shape[0],
                    states.shape[1],
                    states.shape[2],
                    device,
                    mask_ratio=cfg.world_model.mask_ratio,
                )

                pred_states, pred_actions, pred_value, mask, _ = model(
                    states,
                    actions_oh,
                    mask_ratio=0.0,
                    noise_scale=cfg.world_model.noise_scale,
                    mask=mask,
                )

                recon_loss, denoise_loss, value_loss = model.get_loss(
                    states,
                    actions,
                    returns,
                    pred_states,
                    pred_actions,
                    pred_value,
                    mask,
                    loss_mask=loss_mask,
                    action_masks=action_masks,
                )

                loss = recon_loss + denoise_loss + value_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_denoise_loss += denoise_loss.item()
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
            
            # Convert discrete actions to one-hot for INPUT
            actions_oh = to_one_hot(actions)

            optimizer.zero_grad()

            # Create mask (random masking for long sequences too)
            mask = create_mixed_mask(
                states.shape[0],
                states.shape[1],
                states.shape[2],
                device,
                mask_ratio=cfg.world_model.mask_ratio,
            )

            pred_states, pred_actions, pred_value, mask, _ = model(
                states,
                actions_oh,
                mask_ratio=0.0,
                noise_scale=cfg.world_model.noise_scale,
                mask=mask,
            )

            recon_loss, denoise_loss, value_loss = model.get_loss(
                states, actions, returns, pred_states, pred_actions, pred_value, mask, loss_mask=loss_mask, action_masks=action_masks
            )

            loss = recon_loss + denoise_loss + value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_denoise_loss += denoise_loss.item()
            total_value_loss += value_loss.item()
            # Note: We don't increment steps here to keep pbar consistent with short batches,
            # but we do include the loss in the total.
            # Actually, let's increment steps to keep the average correct.
            steps += 1

            pbar.set_postfix(
                {
                    "loss": total_loss / steps,
                    "recon": total_recon_loss / steps,
                    "denoise": total_denoise_loss / steps,
                    "value": total_value_loss / steps,
                }
            )

        avg_loss = total_loss / steps if steps > 0 else 0
        avg_recon_loss = total_recon_loss / steps if steps > 0 else 0
        avg_denoise_loss = total_denoise_loss / steps if steps > 0 else 0
        avg_value_loss = total_value_loss / steps if steps > 0 else 0

        log.info(
            f"Epoch {epoch+1}: Train Loss={avg_loss:.4f} "
            f"(Recon={avg_recon_loss:.4f}, "
            f"Denoise={avg_denoise_loss:.4f}, "
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

                    # Convert discrete actions to one-hot for Model Input
                    actions_oh = to_one_hot(actions)

                    mask = create_mixed_mask(
                        batch_size=states.shape[0],
                        time_steps=states.shape[1],
                        num_ships=states.shape[2],
                        device=device,
                        mask_ratio=cfg.world_model.mask_ratio,
                    )

                    pred_states, pred_actions, pred_value, _, _ = model(
                        states, actions_oh, mask=mask
                    )

                    # Compute loss
                    # Pass RAW actions to get_loss as targets
                    recon_loss, denoise_loss, value_loss = model.get_loss(
                        states,
                        actions,
                        returns,
                        pred_states,
                        pred_actions,
                        pred_value,
                        mask,
                        loss_mask=loss_mask,
                        action_masks=action_masks,
                    )
                    val_loss += (recon_loss + denoise_loss + value_loss).item()
                    val_steps += 1

        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
        log.info(f"Epoch {epoch+1}: Val Loss={avg_val_loss:.4f}")

        # Log to CSV
        with open(csv_path, "a") as f:
            f.write(
                f"{epoch+1},{avg_loss:.6f},{avg_recon_loss:.6f},{avg_denoise_loss:.6f},{avg_value_loss:.6f},{avg_val_loss:.6f}\n"
            )

        # Log to TensorBoard
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Loss/train_recon", avg_recon_loss, epoch)
        writer.add_scalar("Loss/train_denoise", avg_denoise_loss, epoch)
        writer.add_scalar("Loss/train_value", avg_value_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        
        # --- Comput and Log Rollout Metrics ---
        # Get env config from cfg
        env_config = OmegaConf.to_container(cfg.environment, resolve=True)
        # Ensure render_mode is none for headless eval
        env_config["render_mode"] = "none"
        
        # 1. Dreaming (Open Loop Rollout)
        # Use validation long loader for rollout evaluation
        dream_mse = compute_dreaming_error(
            model, 
            val_long_loader, 
            device, 
            max_steps=20, # cfg.world_model.short_batch_len or similar
            num_batches=4 # Keep it fast
        )
        writer.add_scalar("Rollout/dreaming_mse", dream_mse, epoch)
        
        # 2. Controlling (Closed Loop Live Env)
        control_mse, control_reward = compute_controlling_error(
            model, 
            env_config, 
            device,
            max_episode_length=200,
            num_episodes=1
        )
        writer.add_scalar("Rollout/controlling_mse", control_mse, epoch)
        writer.add_scalar("Rollout/controlling_reward", control_reward, epoch)
        
        log.info(f"Epoch {epoch+1}: DreamMSE={dream_mse:.4f}, ControlMSE={control_mse:.4f}, ControlRew={control_reward:.2f}")
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
