"""
World model training script.

Trains a transformer-based world model to predict future states and actions
using masked reconstruction and denoising objectives.
"""

import logging
from pathlib import Path

from datetime import datetime
import os
import torch
import torch.nn.functional as F
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import random
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agents.interleaved_world_model import InterleavedWorldModel
from train.data_loader import load_bc_data, create_unified_data_loaders
from train.swa import SWAModule
import wandb



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





def validate_model(model, loaders, device, amp=True, lambda_state=1.0, lambda_action=0.01):
    """
    Run validation loop and return metrics.
    """
    model.eval()
    val_loss = torch.tensor(0.0, device=device)
    val_steps = 0
    val_error_p = torch.tensor(0.0, device=device)
    val_error_t = torch.tensor(0.0, device=device)
    val_error_s = torch.tensor(0.0, device=device)

    # loaders is a list of data loaders
    for loader in loaders:
        with torch.no_grad():
            for states, actions, returns, loss_mask, _, _, team_ids in loader:
                states = states.to(device)
                actions = actions.to(device)
                loss_mask = loss_mask.to(device)
                team_ids = team_ids.to(device)

                input_states = states[:, :-1]
                input_actions = actions[:, :-1]
                
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

                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                    pred_states, pred_actions, _ = model(
                        input_states, input_actions, input_team_ids, noise_scale=0.0
                    )

                    loss, state_loss, action_loss, _ = model.get_loss(
                        pred_states=pred_states,
                        pred_actions=pred_actions,
                        target_states=target_states,
                        target_actions=target_actions,
                        loss_mask=loss_mask_slice,
                        lambda_state=lambda_state,
                        lambda_action=lambda_action
                    )
                
                val_loss += loss
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
                    val_error_p += (p_logits.argmax(-1) != p_target).float().mean()
                    val_error_t += (t_logits.argmax(-1) != t_target).float().mean()
                    val_error_s += (s_logits.argmax(-1) != s_target).float().mean()

    avg_val_loss = val_loss.item() / val_steps if val_steps > 0 else 0.0
    avg_val_err_p = val_error_p.item() / val_steps if val_steps > 0 else 0.0
    avg_val_err_t = val_error_t.item() / val_steps if val_steps > 0 else 0.0
    avg_val_err_s = val_error_s.item() / val_steps if val_steps > 0 else 0.0

    return {
        "val_loss": avg_val_loss,
        "error_p": avg_val_err_p,
        "error_t": avg_val_err_t,
        "error_s": avg_val_err_s
    }


def train_world_model(cfg: DictConfig) -> None:
    """
    Train the world model.
    """
    log.info("Starting World Model training...")

    # 1. Enable TF32 globally
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
    ).to(device)

    # 2. Compile model
    if os.name != 'nt':
        log.info("Compiling model...")
        model = torch.compile(model)
    else:
        log.info("Skipping torch.compile on Windows (Triton usually missing)")

    # 3. Fused AdamW
    optimizer = optim.AdamW(model.parameters(), lr=cfg.world_model.learning_rate, fused=True)

    # 4. Initialize GradScaler for AMP
    scaler = torch.amp.GradScaler('cuda')

    # 5. Initialize SWA (Shadow Model on CPU)
    swa_model = SWAModule(model)

    # 5. Initialize Scheduler
    sched_cfg = cfg.world_model.get("scheduler", None)
    scheduler = None
    if sched_cfg and "range_test" in sched_cfg.type:
        range_cfg = sched_cfg.range_test
        # Estimate total steps if not provided
        if range_cfg.get("steps"):
             total_steps = range_cfg.steps
        else:
             # Estimate based on short loader length and ratio
             # Total steps ~= num_short_batches / P(short)
             # P(short) = ratio / (ratio + 1)
             short_batches = len(train_short_loader)
             ratio = cfg.world_model.batch_ratio
             prob_short = ratio / (ratio + 1)
             total_steps = int(short_batches / prob_short)
        
        log.info(f"Initializing Exponential Range Test Scheduler: {range_cfg.start_lr} -> {range_cfg.end_lr} over {total_steps} steps")

        # Use LambdaLR for Exponential Scaling
        # Formula: LR = start_lr * (end_lr / start_lr) ^ (step / total_steps)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = range_cfg.start_lr
            
        # Precompute ratio
        # Avoid division by zero if start_lr is 0 (unlikely but safe to check)
        if range_cfg.start_lr == 0:
             raise ValueError("Start LR cannot be 0 for exponential range test.")
             
        # gamma is the total multiplier to reach end_lr
        total_multiplier = range_cfg.end_lr / range_cfg.start_lr
        
        def lr_lambda(step):
            # Clip step to total_steps
            s = min(step, total_steps)
            # fraction of progress
            progress = s / total_steps
            return total_multiplier ** progress
            
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    elif sched_cfg and sched_cfg.type == "warmup_constant":
        warmup_cfg = sched_cfg.warmup
        log.info(f"Initializing Warmup Constant Scheduler: {warmup_cfg.start_lr} -> {cfg.world_model.learning_rate} over {warmup_cfg.steps} steps")
        
        # Target LR is the main learning rate from config
        target_lr = cfg.world_model.learning_rate
        start_lr = warmup_cfg.start_lr
        warmup_steps = warmup_cfg.steps
        
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return (start_lr + (target_lr - start_lr) * (step / warmup_steps)) / target_lr
            return 1.0 # Constant after warmup (relative to base LR)
            
        # Note: LambdaLR multiplies the initial LR passed to optimizer.
        # If optimizer.lr is target_lr, then we need lambda to return fraction.
        # But we set param_group['lr'] to target_lr by default in optimizer init.
        # So lambda should return factor = desired_lr / base_lr
        
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    elif sched_cfg and sched_cfg.type == "constant":
        scheduler = None # default
        
    



    # Setup logging and output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("models/world_model") / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    step_log_path = run_dir / "training_step_log.csv"
    with open(step_log_path, "w") as f:
        f.write("global_step,epoch,learning_rate,loss,state_loss,action_loss,val_loss,val_loss_swa\n")


    log.info(f"Output directory: {run_dir}")

    # Save config immediately
    OmegaConf.save(cfg, run_dir / "config.yaml")

    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(run_dir))

    csv_path = run_dir / "training_log.csv"
    with open(csv_path, "w") as f:
        f.write(
            "epoch,learning_rate,train_loss,train_state_loss,train_action_loss,val_loss,val_loss_swa\n"
        )

    # Initialize W&B
    wb_cfg = cfg.get("wandb", None)
    if wb_cfg and wb_cfg.get("enabled", False):
        wandb.init(
            project=wb_cfg.project,
            entity=wb_cfg.entity,
            name=wb_cfg.get("name") or str(run_dir.name),
            group=wb_cfg.get("group"),
            mode=wb_cfg.get("mode", "online"),
            config=OmegaConf.to_container(cfg, resolve=True),
            resume=False
        )
    
    # Logging Buffer
    log_buffer = []
    log_freq = wb_cfg.get("log_frequency", 50) if wb_cfg else 50

    # Training Loop
    epochs = cfg.world_model.epochs
    batch_ratio = cfg.world_model.batch_ratio
    best_val_loss = float("inf")
    best_val_loss_swa = float("inf")

    # Create Data Loaders (Once, to persist workers and simple train/val split)
    train_short_loader, train_long_loader, val_short_loader, val_long_loader = (
        create_unified_data_loaders(
            data_path,
            short_batch_size=cfg.world_model.short_batch_size,
            long_batch_size=cfg.world_model.long_batch_size,
            short_batch_len=cfg.world_model.short_batch_len,
            long_batch_len=cfg.world_model.long_batch_len,
            batch_ratio=cfg.world_model.batch_ratio,
            validation_split=0.2,
            num_workers=cfg.world_model.get("num_workers", 4),
            prefetch_factor=cfg.world_model.get("prefetch_factor", 2),
        )
    )

    for epoch in range(epochs):
        # Re-create data loaders each epoch to randomize pools -> MOVED OUTSIDE


        model.train()
        # 5. Initialize metrics as Tensors on GPU to avoid sync
        total_loss = torch.tensor(0.0, device=device)
        total_state_loss = torch.tensor(0.0, device=device)
        total_action_loss = torch.tensor(0.0, device=device)
        total_error_power = torch.tensor(0.0, device=device)
        total_error_turn = torch.tensor(0.0, device=device)
        total_error_shoot = torch.tensor(0.0, device=device)

        # Create iterators
        short_iter = iter(train_short_loader)
        long_iter = iter(train_long_loader)

        num_short_batches = len(train_short_loader)
        
        # Adjust pbar total for range test
        if sched_cfg and "range_test" in sched_cfg.type and sched_cfg.range_test.steps:
             pbar_total = sched_cfg.range_test.steps
        else:
             pbar_total = num_short_batches
             
        pbar = tqdm(range(pbar_total), desc=f"Epoch {epoch + 1}/{epochs}")
        # Calculate probability of short batch based on ratio
        # ratio 4 means 4:1 -> 4/5 = 0.8
        short_prob = batch_ratio / (batch_ratio + 1)
        
        # Initialize Sobol Engine for quasi-random sampling
        # Dimension 1, Scramble=True for better uniformity
        sobol_engine = torch.quasirandom.SobolEngine(dimension=1, scramble=True)
        
        steps_in_epoch = 0
        global_step_start = epoch * (total_steps if sched_cfg and "range_test" in sched_cfg.type and sched_cfg.range_test.steps else len(train_short_loader)) # Approx
        
        # We track global steps for logging
        # If range test, we use 'steps' accumulator across epochs (if multiple)
        # But range test is usually 1 epoch.
        while True:
            # Draw sample from Sobol sequence
            # draw() returns (1, 1) tensor, get item
            sample = sobol_engine.draw(1).item()
            
            is_short = sample < short_prob
            
            # Select iterator and batch type
            if is_short:
                iterator = short_iter
                loader_name = "short"
            else:
                iterator = long_iter
                loader_name = "long"

            try:
                batch_data = next(iterator)
            except StopIteration:
                # If short iterator is exhausted, we definitely stop (as per original logic).
                # If long iterator is exhausted, we also stop to keep aligned?
                # Original logic: "while not short_exhausted" and "except StopIteration: break" for long.
                # So any exhaustion ends the epoch.
                # check if we should try the other one? 
                # For safety and preventing infinite loops or imbalance, we break on any exhaustion 
                # effectively defining the epoch length by the limiting factor.
                break

            (states, actions, returns, loss_mask, action_masks, agent_skills, team_ids) = batch_data

            states = states.to(device)
            actions = actions.to(device)
            loss_mask = loss_mask.to(device)
            team_ids = team_ids.to(device)

            input_states = states[:, :-1]
            input_actions = actions[:, :-1]
            
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
            
            # For long batches, we might mask early tokens (warmup)
            # Original code ONLY did this for long batches (which was the second block)
            # We need to replicate that logic.
            if not is_short:
                if loss_mask_slice.shape[1] > 32:
                    loss_mask_slice[:, :32] = False
            
            # Rollout
            rollout_len = get_rollout_length(epoch, cfg)
            perform_rollout(model, input_states, input_actions, input_team_ids, rollout_len)

            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                pred_states, pred_actions, _ = model(
                    input_states,
                    input_actions,
                    input_team_ids,
                    noise_scale=cfg.world_model.noise_scale
                )
                
                
                loss, state_loss, action_loss, metrics = model.get_loss(
                    pred_states=pred_states,
                    pred_actions=pred_actions,
                    target_states=target_states,
                    target_actions=target_actions,
                    loss_mask=loss_mask_slice,
                    lambda_state=cfg.world_model.get("lambda_state", 1.0),
                    lambda_action=cfg.world_model.get("lambda_action", 0.01)
                )

                # Capture Metrics (if returned)
                # Note: get_loss now returns 4 values, so we unpack directly above or handle tuple
                # The updated get_loss returns (total, state, action, metrics_dict)
                # But let's be safe if it falls back for some reason (though we edited it)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.detach()
            total_state_loss += state_loss.detach()
            total_action_loss += action_loss.detach()

            # Global Step Logging
            current_lr = optimizer.param_groups[0]['lr']
            
            # Use a persistent global step
            current_step = global_step_start + steps_in_epoch
            
            # Buffer Metrics (Detach to avoid memory leak)
            if cfg.wandb.enabled:
                step_metrics = {
                    "step": current_step,
                    "epoch": epoch + 1,
                    "lr": current_lr,
                    "loss": loss.detach(),
                    "state_loss": state_loss.detach(),
                    "action_loss": action_loss.detach(),
                    "grad_norm": total_norm.detach() if isinstance(total_norm, torch.Tensor) else torch.tensor(total_norm, device=device),
                }
                
                # Add extra metrics
                for k, v in metrics.items():
                    if isinstance(v, torch.Tensor):
                        step_metrics[k] = v.detach()
                    else:
                        step_metrics[k] = torch.tensor(v, device=device)
                
                log_buffer.append(step_metrics)
                
                # Flush Buffer
                if len(log_buffer) >= log_freq:
                    # Move to CPU in one go
                    # We can't stack dicts directly, so stack values per key
                    packed = {}
                    keys = log_buffer[0].keys()
                    for k in keys:
                        # Stack tensors
                        try:
                            # Check if all are tensors
                             tensors = [x[k] for x in log_buffer]
                             packed[k] = torch.stack(tensors).cpu().tolist()
                        except:
                             # Fallback for non-tensors
                             packed[k] = [x[k] for x in log_buffer]
                    
                    # Log to WandB
                    for i in range(len(packed["step"])):
                        log_item = {k: packed[k][i] for k in packed}
                        # We must log sequentially with step
                        step_val = int(log_item.pop("step"))
                        wandb.log(log_item, step=step_val)
                    
                    log_buffer = []


            # Log every step for range test, or every 10 for normal
            log_freq_csv = 1 if (sched_cfg and "range_test" in sched_cfg.type) else 10
            
            # We need a persistent global step counter. 
            # Ideally passed in or static. For now, let's use steps_in_epoch + accumulated?
            # Actually, `scheduler.last_epoch` tracks steps for LambdaLR if stepped every batch.
            # Or just use a simple counter.
            
            # Let's write to CSV
            if steps_in_epoch % log_freq == 0:
                with open(step_log_path, "a") as f:
                     # For range test, we only care about the sequence
                     f.write(f"{steps_in_epoch},{epoch+1},{current_lr:.8f},{loss.item():.6f},{state_loss.item():.6f},{action_loss.item():.6f},,\n")

            # Check for range test limit
            if sched_cfg and "range_test" in sched_cfg.type:
                range_limit = sched_cfg.range_test.steps
                if range_limit and steps_in_epoch >= range_limit:
                    log.info(f"Reached max steps for Range Test ({range_limit}). Stopping.")
                    return # Exit training completely
            
            
            if is_short:
                # Normal mode: only update pbar on short batches (since total is num_short)
                if not (sched_cfg and "range_test" in sched_cfg.type and sched_cfg.range_test.steps):
                    pbar.update(1)
            
            # Range test mode: update on every step
            if sched_cfg and "range_test" in sched_cfg.type and sched_cfg.range_test.steps:
                pbar.update(1)

            steps_in_epoch += 1
            if steps_in_epoch % 50 == 0:
                pbar.set_postfix({
                    "loss": total_loss.item() / steps_in_epoch,
                    "state": total_state_loss.item() / steps_in_epoch,
                    "action": total_action_loss.item() / steps_in_epoch
                })

        avg_loss = total_loss.item() / steps_in_epoch if steps_in_epoch > 0 else 0
        avg_state_loss = total_state_loss.item() / steps_in_epoch if steps_in_epoch > 0 else 0
        avg_action_loss = total_action_loss.item() / steps_in_epoch if steps_in_epoch > 0 else 0
        
        avg_err_p = total_error_power.item() / steps_in_epoch if steps_in_epoch > 0 else 0
        avg_err_t = total_error_turn.item() / steps_in_epoch if steps_in_epoch > 0 else 0
        avg_err_s = total_error_shoot.item() / steps_in_epoch if steps_in_epoch > 0 else 0
        
        current_lr = optimizer.param_groups[0]['lr']
        log.info(f"Epoch {epoch + 1}: LR={current_lr:.2e} Train Loss={avg_loss:.4f} (State={avg_state_loss:.4f}, Action={avg_action_loss:.4f})")

        # Validation (Live Model)
        val_metrics = validate_model(
            model, 
            [val_short_loader, val_long_loader], 
            device,
            lambda_state=cfg.world_model.get("lambda_state", 1.0),
            lambda_action=cfg.world_model.get("lambda_action", 0.01)
        )
        avg_val_loss = val_metrics["val_loss"]
        
        log.info(f"Epoch {epoch + 1}: Val Loss={avg_val_loss:.4f}")

        # SWA Update
        if epoch >= cfg.world_model.get("swa_start_epoch", 2):
             swa_model.update_parameters(model)

        # SWA Evaluation
        avg_val_loss_swa = None
        if epoch >= cfg.world_model.get("swa_start_epoch", 2):
             # Move SWA model to GPU
             swa_model.averaged_model.to(device)
             val_metrics_swa = validate_model(
                 swa_model.averaged_model, 
                 [val_short_loader, val_long_loader], 
                 device,
                 lambda_state=cfg.world_model.get("lambda_state", 1.0),
                 lambda_action=cfg.world_model.get("lambda_action", 0.01)
             )
             swa_model.averaged_model.to('cpu') # Move back to CPU
             
             avg_val_loss_swa = val_metrics_swa["val_loss"]
             log.info(f"Epoch {epoch + 1}: SWA Val Loss={avg_val_loss_swa:.4f}")
             
             # Log SWA metrics
             writer.add_scalar("Loss/val_swa", avg_val_loss_swa, epoch)
             writer.add_scalar("Error/val_swa_power", val_metrics_swa["error_p"], epoch)
             writer.add_scalar("Error/val_swa_turn", val_metrics_swa["error_t"], epoch)
             writer.add_scalar("Error/val_swa_shoot", val_metrics_swa["error_s"], epoch)

        # Log to CSV
        current_lr = optimizer.param_groups[0]['lr']
        with open(csv_path, "a") as f:
            swa_loss_str = f"{avg_val_loss_swa:.6f}" if avg_val_loss_swa is not None else ""
            f.write(
                f"{epoch + 1},{current_lr:.8f},{avg_loss:.6f},{avg_state_loss:.6f},{avg_action_loss:.6f},{avg_val_loss:.6f},{swa_loss_str}\n"
            )

        # Log to TensorBoard
        writer.add_scalar("Train/LearningRate", current_lr, epoch)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Loss/train_state", avg_state_loss, epoch)
        writer.add_scalar("Loss/train_action", avg_action_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Error/train_power", avg_err_p, epoch)
        writer.add_scalar("Error/train_turn", avg_err_t, epoch)
        writer.add_scalar("Error/train_shoot", avg_err_s, epoch)
        writer.add_scalar("Error/val_power", val_metrics["error_p"], epoch)
        writer.add_scalar("Error/val_turn", val_metrics["error_t"], epoch)
        writer.add_scalar("Error/val_shoot", val_metrics["error_s"], epoch)

        # Save Best Models
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), run_dir / "best_world_model.pth")
            log.info(f"Saved best live model with val loss {best_val_loss:.4f}")

        if avg_val_loss_swa is not None and avg_val_loss_swa < best_val_loss_swa:
            best_val_loss_swa = avg_val_loss_swa
            torch.save(swa_model.averaged_model.state_dict(), run_dir / "best_world_model_swa.pth")
            log.info(f"Saved best SWA model with val loss {best_val_loss_swa:.4f}")

        if (epoch + 1) % 10 == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "swa_model_state_dict": swa_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "scaler_state_dict": scaler.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True)
            }
            save_path = run_dir / f"world_model_epoch_{epoch + 1}.pt"
            torch.save(checkpoint, save_path)
            log.info(f"Saved full checkpoint to {save_path}")

    # Save Final Checkpoints
    final_checkpoint = {
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "swa_model_state_dict": swa_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True)
    }
    torch.save(final_checkpoint, run_dir / "final_world_model.pt")
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
