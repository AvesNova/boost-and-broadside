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
from src.train.data_loader import load_bc_data, create_world_model_data_loader

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
    log.info(f"Loading data from {data_path}")
    log.info(f"Absolute path: {Path(data_path).resolve()}")
    data = load_bc_data(data_path)

    train_loader, val_loader = create_world_model_data_loader(
        data,
        batch_size=cfg.world_model.batch_size,
        context_len=cfg.world_model.context_len,
        validation_split=0.2,
        num_workers=0,
        use_alternating_lengths=cfg.world_model.use_alternating_lengths,
        short_batch_len=cfg.world_model.short_batch_len,
        long_batch_len=cfg.world_model.long_batch_len,
        long_batch_ratio=cfg.world_model.long_batch_ratio,
    )

    # Initialize Model
    # Get dimensions from data
    sample_tokens, sample_actions = next(iter(train_loader))
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

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_denoise_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for states, actions in pbar:
            states = states.to(device)
            actions = actions.to(device)

            if epoch == 0 and total_loss == 0:
                log.info(f"States shape: {states.shape}")
                log.info(f"Actions shape: {actions.shape}")

            optimizer.zero_grad()

            # Create mixed mask
            mask = create_mixed_mask(
                states.shape[0],  # batch_size
                states.shape[1],  # time_steps
                states.shape[2],  # num_ships
                device,
                mask_ratio=cfg.world_model.mask_ratio,
            )

            pred_states, pred_actions, mask, _ = model(
                states,
                actions,
                mask_ratio=0.0,  # We provide explicit mask
                noise_scale=cfg.world_model.noise_scale,
                mask=mask,
            )

            recon_loss, denoise_loss = model.get_loss(
                states, actions, pred_states, pred_actions, mask
            )

            loss = recon_loss + denoise_loss
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_denoise_loss += denoise_loss.item()

            pbar.set_postfix(
                {
                    "loss": loss.item(),
                    "recon": recon_loss.item(),
                    "denoise": denoise_loss.item(),
                }
            )

        avg_loss = total_loss / len(train_loader)
        log.info(
            f"Epoch {epoch+1}: Train Loss={avg_loss:.4f} "
            f"(Recon={total_recon_loss/len(train_loader):.4f}, "
            f"Denoise={total_denoise_loss/len(train_loader):.4f})"
        )

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for states, actions in val_loader:
                states = states.to(device)
                actions = actions.to(device)

                # Create validation mask (same strategy)
                mask = create_mixed_mask(
                    states.shape[0],
                    states.shape[1],
                    states.shape[2],
                    device,
                    mask_ratio=cfg.world_model.mask_ratio,
                )

                pred_states, pred_actions, mask, _ = model(
                    states,
                    actions,
                    mask_ratio=0.0,
                    noise_scale=0.0,  # No noise for validation
                    mask=mask,
                )

                recon_loss, denoise_loss = model.get_loss(
                    states, actions, pred_states, pred_actions, mask
                )
                val_loss += (recon_loss + denoise_loss).item()

        avg_val_loss = val_loss / len(val_loader)
        log.info(f"Epoch {epoch+1}: Val Loss={avg_val_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            save_path = Path(f"models/world_model_epoch_{epoch+1}.pt")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            log.info(f"Saved model to {save_path}")

    log.info("Training complete.")
