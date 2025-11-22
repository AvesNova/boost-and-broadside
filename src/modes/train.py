import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.agents.team_transformer_agent import TeamTransformerModel
from src.train.data_loader import load_bc_data, create_bc_data_loader


def train_bc(cfg: DictConfig) -> Path | None:
    """
    Main training function for behavioral cloning.

    Args:
        cfg: Configuration dictionary
    """
    print("Starting BC training...")

    # Check if BC training is enabled
    if not cfg.train.use_bc:
        print("BC training is disabled in config")
        return

    # Get BC configuration
    # Get BC configuration
    bc_config = cfg.train.bc
    model_config = cfg.train.model.transformer

    # Load data
    print(f"Loading data from: {cfg.train.bc_data_path}")
    data = load_bc_data(cfg.train.bc_data_path)
    
    # Create data loaders
    train_loader, val_loader = create_bc_data_loader(
        data, 
        batch_size=bc_config.batch_size,
        gamma=cfg.train.rl.gamma,
        validation_split=bc_config.validation_split
    )

    # Create model
    model = TeamTransformerModel(
        token_dim=model_config.token_dim,
        embed_dim=model_config.embed_dim,
        num_heads=model_config.num_heads,
        num_layers=model_config.num_layers,
        max_ships=model_config.max_ships,
        dropout=model_config.dropout,
        use_layer_norm=model_config.use_layer_norm,
    )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=bc_config.learning_rate)

    # Loss functions
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    # Setup logging and output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("models/bc") / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {run_dir}")
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(run_dir))
    
    csv_path = run_dir / "training_log.csv"
    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,train_policy_loss,train_value_loss,train_acc,val_loss,val_acc\n")

    # Training loop
    epochs = bc_config.epochs
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_policy_loss = 0.0
        train_value_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, (tokens, actions, returns) in enumerate(train_pbar):
            # Move data to device
            tokens = tokens.to(device)  # (batch, max_ships, token_dim)
            actions = actions.to(device)  # (batch, max_ships, num_actions)
            returns = returns.to(device)  # (batch,)

            # Get action targets (one-hot to class indices)
            action_targets = torch.argmax(actions, dim=-1)  # (batch, max_ships)

            # Forward pass
            optimizer.zero_grad()
            
            # Create observation dict
            observation = {"tokens": tokens}

            # Get model predictions
            output = model(observation)
            action_logits = output["action_logits"]  # (batch, max_ships, num_actions)
            values = output["value"].squeeze(-1)  # (batch,)

            # Reshape for loss computation
            batch_size, max_ships, num_actions = action_logits.shape
            action_logits_flat = action_logits.view(-1, num_actions)
            action_targets_flat = action_targets.view(-1)

            # Compute losses
            p_loss = policy_criterion(action_logits_flat, action_targets_flat)
            v_loss = value_criterion(values, returns)
            
            loss = (bc_config.policy_weight * p_loss) + (bc_config.value_weight * v_loss)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update statistics
            train_loss += loss.item()
            train_policy_loss += p_loss.item()
            train_value_loss += v_loss.item()
            
            _, predicted = torch.max(action_logits_flat.data, 1)
            train_total += action_targets_flat.size(0)
            train_correct += (predicted == action_targets_flat).sum().item()

            # Update progress bar
            train_pbar.set_postfix(
                {
                    "Loss": f"{train_loss/(batch_idx+1):.4f}",
                    "PLoss": f"{train_policy_loss/(batch_idx+1):.4f}",
                    "VLoss": f"{train_value_loss/(batch_idx+1):.4f}",
                    "Acc": f"{100.*train_correct/train_total:.2f}%",
                }
            )

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch_idx, (tokens, actions, returns) in enumerate(val_pbar):
                # Move data to device
                tokens = tokens.to(device)
                actions = actions.to(device)
                returns = returns.to(device)

                # Get action targets
                action_targets = torch.argmax(actions, dim=-1)

                # Forward pass
                observation = {"tokens": tokens}
                output = model(observation)
                action_logits = output["action_logits"]
                values = output["value"].squeeze(-1)

                # Reshape for loss computation
                batch_size, max_ships, num_actions = action_logits.shape
                action_logits_flat = action_logits.view(-1, num_actions)
                action_targets_flat = action_targets.view(-1)

                # Compute loss
                p_loss = policy_criterion(action_logits_flat, action_targets_flat)
                v_loss = value_criterion(values, returns)
                loss = (bc_config.policy_weight * p_loss) + (bc_config.value_weight * v_loss)

                # Update statistics
                val_loss += loss.item()
                _, predicted = torch.max(action_logits_flat.data, 1)
                val_total += action_targets_flat.size(0)
                val_correct += (predicted == action_targets_flat).sum().item()

                # Update progress bar
                val_pbar.set_postfix(
                    {
                        "Loss": f"{val_loss/(batch_idx+1):.4f}",
                        "Acc": f"{100.*val_correct/val_total:.2f}%",
                    }
                )

        # Calculate epoch statistics
        avg_train_loss = train_loss / len(train_loader)
        avg_train_policy_loss = train_policy_loss / len(train_loader)
        avg_train_value_loss = train_value_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100.0 * train_correct / train_total
        val_accuracy = 100.0 * val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        # Log to CSV
        with open(csv_path, "a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.6f},{avg_train_policy_loss:.6f},{avg_train_value_loss:.6f},{train_accuracy:.2f},{avg_val_loss:.6f},{val_accuracy:.2f}\n")

        # Log to TensorBoard
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/train_policy", avg_train_policy_loss, epoch)
        writer.add_scalar("Loss/train_value", avg_train_value_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Save best model
            torch.save(model.state_dict(), run_dir / "best_bc_model.pth")
            print("  Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= bc_config.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Save final model
    torch.save(model.state_dict(), run_dir / "final_bc_model.pth")
    
    # Save metadata
    metadata_path = run_dir / "model_metadata.yaml"
    metadata = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "final_metrics": {
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "epochs_trained": epoch + 1
        }
    }
    
    OmegaConf.save(OmegaConf.create(metadata), metadata_path)
    
    # Close writer
    writer.close()

    print("BC training completed!")
    return run_dir / "final_bc_model.pth"


def train_rl(cfg: DictConfig, pretrained_model_path: Path | None = None) -> None:
    """
    RL training using Stable Baselines3 PPO.
    
    Args:
        cfg: Configuration dictionary
        pretrained_model_path: Path to pretrained BC model weights (optional)
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from env.sb3_wrapper import SB3Wrapper
    from agents.sb3_adapter import TeamTransformerSB3Policy
    from env.env import Environment

    print("\nStarting RL training...")
    
    rl_config = cfg.train.rl
    
    # Create environment
    # We need to instantiate the base Environment first
    env_config = dict(cfg.environment)
    env_config["render_mode"] = "none" # Force no rendering for training
    
    base_env = Environment(**env_config)
    env = SB3Wrapper(base_env, cfg)
    
    # Setup model config for policy
    model_config = OmegaConf.to_container(cfg.train.model.transformer, resolve=True)
    
    # Initialize PPO
    # We use our custom policy
    policy_kwargs = {
        "model_config": model_config,
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("models/rl") / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"RL Output directory: {run_dir}")
    
    model = PPO(
        TeamTransformerSB3Policy,
        env,
        learning_rate=rl_config.learning_rate,
        n_steps=rl_config.n_steps,
        batch_size=rl_config.batch_size,
        n_epochs=rl_config.n_epochs,
        gamma=rl_config.gamma,
        gae_lambda=rl_config.gae_lambda,
        clip_range=rl_config.clip_range,
        ent_coef=rl_config.ent_coef,
        vf_coef=rl_config.vf_coef,
        max_grad_norm=rl_config.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(run_dir),
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Load pretrained weights if available
    if pretrained_model_path and pretrained_model_path.exists():
        print(f"Loading pretrained BC weights from {pretrained_model_path}")
        try:
            state_dict = torch.load(pretrained_model_path, map_location=model.device)
            model.policy.transformer_model.load_state_dict(state_dict)
            print("Successfully loaded pretrained weights")
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
            
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=rl_config.n_steps,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="rl_model"
    )
    
    # Train
    total_timesteps = rl_config.get("total_timesteps", 1000000)
    print(f"Training for {total_timesteps} timesteps...")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # Save final model
        final_path = run_dir / "final_rl_model.zip"
        model.save(final_path)
        print(f"Saved final RL model to {final_path}")
        
        # Also save the inner transformer state dict for compatibility with our agents
        torch.save(
            model.policy.transformer_model.state_dict(),
            run_dir / "final_rl_transformer.pth"
        )
        print(f"Saved transformer weights to {run_dir / 'final_rl_transformer.pth'}")
        
        env.close()


def train(cfg: DictConfig) -> None:
    """
    Main training function.
    """
    bc_model_path = None
    
    # BC Training
    if cfg.train.use_bc:
        bc_model_path = train_bc(cfg)
        
    # RL Training
    if cfg.train.use_rl:
        # Use BC model if available, otherwise check config for specific path
        if bc_model_path is None and cfg.train.bc_data_path:
             # This logic is a bit flawed, bc_data_path is for data.
             # We might want a config for pretrained model path if skipping BC.
             # For now, we just pass bc_model_path.
             pass
             
        train_rl(cfg, pretrained_model_path=bc_model_path)
