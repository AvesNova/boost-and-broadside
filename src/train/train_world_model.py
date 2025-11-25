import hydra
import torch
import torch.optim as optim
from omegaconf import DictConfig
from pathlib import Path
import logging
from tqdm import tqdm

from src.models.world_model import WorldModel
from src.train.data_loader import load_bc_data, create_world_model_data_loader

log = logging.getLogger(__name__)

def train_world_model(cfg: DictConfig):
    log.info("Starting World Model training...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Load Data
    data_path = cfg.train.bc_data_path
    if data_path is None:
        from src.train.data_loader import get_latest_data_path
        data_path = get_latest_data_path()
    
    log.info(f"Loading data from {data_path}")
    print(f"Loading data from {data_path}")
    print(f"Absolute path: {Path(data_path).resolve()}")
    data = load_bc_data(data_path)
    
    train_loader, val_loader = create_world_model_data_loader(
        data, 
        batch_size=cfg.world_model.batch_size,
        context_len=cfg.world_model.context_len,
        validation_split=0.2, # Could be in config
        num_workers=0
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
            
            pred_states, pred_actions, mask, _ = model(
                states, 
                actions, 
                mask_ratio=cfg.world_model.mask_ratio,
                noise_scale=cfg.world_model.noise_scale
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
            
            pbar.set_postfix({
                "loss": loss.item(), 
                "recon": recon_loss.item(), 
                "denoise": denoise_loss.item()
            })
            
        avg_loss = total_loss / len(train_loader)
        log.info(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f} (Recon={total_recon_loss/len(train_loader):.4f}, Denoise={total_denoise_loss/len(train_loader):.4f})")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for states, actions in val_loader:
                states = states.to(device)
                actions = actions.to(device)
                
                # For validation, maybe we don't mask? Or we mask to check reconstruction?
                # Usually we check reconstruction on validation set too.
                pred_states, pred_actions, mask, _ = model(
                    states, 
                    actions, 
                    mask_ratio=cfg.world_model.mask_ratio, # Keep masking for val loss
                    noise_scale=0.0 # No noise for validation? Or yes?
                    # Denoising task requires noise.
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
