"""
World model training script.

Trains a transformer-based world model to predict future states and actions
using masked reconstruction and denoising objectives.
"""
import logging
from pathlib import Path

import hydra
import torch
import torch.optim as optim
from omegaconf import DictConfig
from tqdm import tqdm

from src.agents.world_model import WorldModel
from src.train.data_loader import load_bc_data, create_dual_pool_data_loaders

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
        start_pos = (torch.rand(num_blocks, device=device) * (time_steps - block_lens)).long()
        
        for i in range(num_blocks):
            b, n, l, s = b_indices[i], n_indices[i], block_lens[i], start_pos[i]
            mask[b, s : s + l, n] = True

    # 3. Next-Step Masking
    # Always mask the last timestep for all ships to learn forward prediction
    # This is critical for the agent's primary use case
    mask[:, -1, :] = True

    return mask



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
    if data_path is None:
        from src.train.data_loader import get_latest_data_path

        data_path = get_latest_data_path()

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
    sample_actions = team_0["actions"][0]
    
    state_dim = sample_tokens.shape[-1]
    action_dim = sample_actions.shape[-1]

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

    # Training Loop
    epochs = cfg.world_model.epochs
    batch_ratio = cfg.world_model.batch_ratio

    for epoch in range(epochs):
        # Re-create data loaders each epoch to randomize pools
        train_short_loader, train_long_loader, val_short_loader, val_long_loader = create_dual_pool_data_loaders(
            data,
            short_batch_size=cfg.world_model.short_batch_size,
            long_batch_size=cfg.world_model.long_batch_size,
            short_batch_len=cfg.world_model.short_batch_len,
            long_batch_len=cfg.world_model.long_batch_len,
            batch_ratio=cfg.world_model.batch_ratio,
            validation_split=0.2,
            num_workers=0,
        )
        
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_denoise_loss = 0
        
        # Create iterators
        short_iter = iter(train_short_loader)
        long_iter = iter(train_long_loader)
        
        # Determine number of steps
        # We run until one of the loaders is exhausted?
        # Or we determine steps based on the ratio.
        # Let's run until short loader is exhausted, as it's the dominant one.
        num_short_batches = len(train_short_loader)
        num_long_batches = len(train_long_loader)
        
        # We want to run roughly num_short_batches.
        # But we need to respect the ratio.
        # Steps = num_short_batches
        
        pbar = tqdm(range(num_short_batches), desc=f"Epoch {epoch+1}/{epochs}")
        
        short_exhausted = False
        long_exhausted = False
        
        steps = 0
        
        while not short_exhausted:
            # 1. Run batch_ratio short batches
            for _ in range(batch_ratio):
                try:
                    states, actions, loss_mask = next(short_iter)
                except StopIteration:
                    short_exhausted = True
                    break
                
                states, actions, loss_mask = states.to(device), actions.to(device), loss_mask.to(device)
                
                optimizer.zero_grad()
                
                # Create mask (random masking for short sequences)
                mask = create_mixed_mask(
                    states.shape[0], states.shape[1], states.shape[2], device,
                    mask_ratio=cfg.world_model.mask_ratio
                )
                
                pred_states, pred_actions, mask, _ = model(
                    states, actions, mask_ratio=0.0, noise_scale=cfg.world_model.noise_scale, mask=mask
                )
                
                recon_loss, denoise_loss = model.get_loss(
                    states, actions, pred_states, pred_actions, mask, loss_mask=loss_mask
                )
                
                loss = recon_loss + denoise_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_denoise_loss += denoise_loss.item()
                steps += 1
                pbar.update(1)
                
            if short_exhausted:
                break
                
            # 2. Run 1 long batch
            try:
                states, actions, loss_mask = next(long_iter)
            except StopIteration:
                # If long loader exhausted, restart it?
                # Or just stop?
                # Ideally they are balanced by the pool creation logic.
                # But due to rounding, they might not be perfectly aligned.
                # Let's restart long iterator if needed, or just skip if empty.
                # If we restart, we might overfit to long samples?
                # Given the pool logic, they should be roughly proportional.
                # Let's just skip if empty.
                long_exhausted = True
                # If long is exhausted but short is not, we continue with short only?
                # Or we break?
                # Let's break to keep the ratio roughly correct.
                break
            
            states, actions, loss_mask = states.to(device), actions.to(device), loss_mask.to(device)
            
            optimizer.zero_grad()
            
            # Create mask (random masking for long sequences too?)
            # Yes, we still want to learn reconstruction/denoising on long sequences.
            mask = create_mixed_mask(
                states.shape[0], states.shape[1], states.shape[2], device,
                mask_ratio=cfg.world_model.mask_ratio
            )
            
            pred_states, pred_actions, mask, _ = model(
                states, actions, mask_ratio=0.0, noise_scale=cfg.world_model.noise_scale, mask=mask
            )
            
            recon_loss, denoise_loss = model.get_loss(
                states, actions, pred_states, pred_actions, mask, loss_mask=loss_mask
            )
            
            loss = recon_loss + denoise_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # We don't update pbar for long batch to keep it consistent with short batches count?
            # Or we should?
            # Let's just log it.
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_denoise_loss += denoise_loss.item()
            
            pbar.set_postfix({
                "loss": total_loss / (steps + 1),
                "recon": total_recon_loss / (steps + 1),
                "denoise": total_denoise_loss / (steps + 1)
            })

        avg_loss = total_loss / (steps + 1)
        log.info(
            f"Epoch {epoch+1}: Train Loss={avg_loss:.4f} "
            f"(Recon={total_recon_loss/(steps+1):.4f}, "
            f"Denoise={total_denoise_loss/(steps+1):.4f})"
        )

        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        
        # Validate on both short and long
        for loader in [val_short_loader, val_long_loader]:
            with torch.no_grad():
                for states, actions, loss_mask in loader:
                    states, actions, loss_mask = states.to(device), actions.to(device), loss_mask.to(device)

                    mask = create_mixed_mask(
                        states.shape[0], states.shape[1], states.shape[2], device,
                        mask_ratio=cfg.world_model.mask_ratio
                    )

                    pred_states, pred_actions, mask, _ = model(
                        states, actions, mask_ratio=0.0, noise_scale=0.0, mask=mask
                    )

                    recon_loss, denoise_loss = model.get_loss(
                        states, actions, pred_states, pred_actions, mask, loss_mask=loss_mask
                    )
                    val_loss += (recon_loss + denoise_loss).item()
                    val_steps += 1

        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
        log.info(f"Epoch {epoch+1}: Val Loss={avg_val_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            save_path = Path(f"models/world_model_epoch_{epoch+1}.pt")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            log.info(f"Saved model to {save_path}")

    log.info("Training complete.")
