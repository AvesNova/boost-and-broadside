import logging
import time
import torch
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import wandb

from boost_and_broadside.core.constants import PowerActions, TurnActions, ShootActions
from boost_and_broadside.utils.dataset_stats import calculate_action_counts, compute_class_weights, apply_turn_exceptions, normalize_weights

log = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, optimizer, scheduler, scaler, swa_model, logger, validator, cfg: DictConfig, device, run_dir, data_path: str):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.swa_model = swa_model
        self.logger = logger
        self.validator = validator
        self.cfg = cfg
        self.device = device
        self.run_dir = run_dir
        
        self.epochs = cfg.train.get("epochs", 1000) # Moved to train section or model? Usually train.
        # Let's assume we move generic train params to 'train' section in main config, 
        # but model specific ones stay in model.
        # For now, let's look in both or prefer 'model' for back compat if I didn't move it fully.
        # Actually, I should check where I put 'epochs' in the new configs.
        # In yemong_full.yaml I didn't put epochs. It should be in `train`.
        self.epochs = cfg.train.get("epochs", 1000) if "epochs" in cfg.train else cfg.model.get("epochs", 1000)

        self.acc_steps = cfg.train.get("gradient_accumulation_steps", 1)
        self.use_amp = cfg.train.get("amp", False) and device.type == 'cuda'
        
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_val_loss_swa = float("inf")
        self.data_path = data_path
        
        # Curriculum State
        self.curr_cfg = cfg.model.get("curriculum", {})
        self.entropy_cfg = cfg.model.get("entropy", {})
        self.loss_cfg = cfg.model.get("loss", {})
        
        self.current_min_skill = self.curr_cfg.get("min_skill_start", 0.0) if self.curr_cfg.get("enabled", False) else 0.0
        self.last_reloaded_skill = self.current_min_skill
        
        # Initialize Class Weights
        log.info("Calculating class weights for loss scaling...")
        counts = calculate_action_counts(data_path)
        
        # Get weight config
        w_cap = self.loss_cfg.get("weighted_loss_cap", 10.0)
        w_pwr = self.loss_cfg.get("weighted_loss_power", 0.5)
        
        # Calculate weights 
        dtype = torch.bfloat16 if self.use_amp else torch.float32
        w_power_full = compute_class_weights(counts["power"], cap=w_cap, power=w_pwr)
        self.w_power = w_power_full[:3].to(device, dtype=dtype)
        self.w_power = normalize_weights(self.w_power, counts["power"][:3])
        
        w_turn_full = apply_turn_exceptions(compute_class_weights(counts["turn"], cap=w_cap, power=w_pwr))
        self.w_turn = w_turn_full[:7].to(device, dtype=dtype)
        self.w_turn = normalize_weights(self.w_turn, counts["turn"][:7])
        
        w_shoot_full = compute_class_weights(counts["shoot"], cap=w_cap, power=w_pwr)
        self.w_shoot = w_shoot_full[:2].to(device, dtype=dtype)
        self.w_shoot = normalize_weights(self.w_shoot, counts["shoot"][:2])
        
        log.info(f"Action Counts - Power: {counts['power'][:3]}")
        log.info(f"Action Counts - Turn:  {counts['turn'][:7]}")
        log.info(f"Action Counts - Shoot: {counts['shoot'][:2]}")
        log.info(f"Class Weights - Power: {self.w_power.tolist()}")
        log.info(f"Class Weights - Turn:  {self.w_turn.tolist()}")
        log.info(f"Class Weights - Shoot: {self.w_shoot.tolist()}")
        
        # Log scaling effect
        log.info("Action Loss Scaling: Power=1/log(3), Turn=1/log(7), Shoot=1/log(2)")

    def _get_current_params(self, step):
        """Calculate current values for scheduled parameters."""
        # Entropy: Decay from start to end
        ent_start = self.entropy_cfg.get("lambda_start", 0.0)
        ent_end = self.entropy_cfg.get("lambda_end", 0.0)
        ent_decay = self.entropy_cfg.get("decay_steps", 1000)
        
        if step < ent_decay:
            alpha = step / ent_decay
            curr_entropy = ent_start + (ent_end - ent_start) * alpha
        else:
            curr_entropy = ent_end
            
        # Focal Gamma: Increase from start to end (or decay, depends on config)
        foc_start = self.loss_cfg.get("gamma_start", 0.0)
        foc_end = self.loss_cfg.get("gamma_end", 0.0)
        foc_decay = self.loss_cfg.get("decay_steps", 1000)
        
        if self.loss_cfg.get("use_focal_loss", False):
            if step < foc_decay:
                alpha = step / foc_decay
                curr_gamma = foc_start + (foc_end - foc_start) * alpha
            else:
                curr_gamma = foc_end
        else:
            curr_gamma = 0.0
            
        # Curriculum Skill: Decay from start (1.0) to end (0.0) typically
        curr_skill = 0.0
        if self.curr_cfg.get("enabled", False):
            skill_start = self.curr_cfg.get("min_skill_start", 0.9)
            skill_end = self.curr_cfg.get("min_skill_end", 0.0)
            skill_decay = self.curr_cfg.get("decay_steps", 1000)
            
            if step < skill_decay:
                alpha = step / skill_decay
                curr_skill = skill_start + (skill_end - skill_start) * alpha
            else:
                curr_skill = skill_end
        
        return {
            "lambda_entropy": curr_entropy,
            "focal_gamma": curr_gamma,
            "min_skill": curr_skill
        }

    def train(self, train_loader, val_loader=None):
        """Main training loop."""
        
        sched_cfg = self.cfg.model.get("scheduler", None)
        is_range_test = sched_cfg and "range_test" in sched_cfg.type

        for epoch in range(self.epochs):
            self.model.train()
            
            # Setup Iterator
            if is_range_test:
                limit = sched_cfg.range_test.steps
                pbar = tqdm(total=limit, desc="LR Range Test")
                # Use a separate iterator for the loader
                batch_iter = iter(train_loader)
            else:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
                batch_iter = pbar
            
            # Accumulators
            accumulators = self._init_accumulators()
            micro_step = 0
            steps_in_epoch = 0
            self.optimizer.zero_grad()
            self.t_last_macro = time.time()
            
            for batch_data in batch_iter:
                # 2. Process Batch
                t0 = time.time()
                step_params = self._get_current_params(self.global_step)
                step_metrics = self._train_step(batch_data, epoch, step_params)
                step_metrics["time"] = time.time() - t0
                
                # 3. Accumulate
                self._update_accumulators(accumulators, step_metrics)
                
                micro_step += 1
                
                # 4. Optimization Step
                if micro_step % self.acc_steps == 0:
                    self._optimize_step(accumulators, pbar, is_range_test, epoch)
                    self._reset_accumulators(accumulators)
                    
                    if is_range_test and self.global_step >= sched_cfg.range_test.steps:
                        log.info(f"Range test complete ({self.global_step} steps).")
                        pbar.close()
                        return

                steps_in_epoch += 1
                if steps_in_epoch % (50 * self.acc_steps) == 0:
                    pbar.set_postfix({
                         "loss": accumulators["total_loss"].item() / steps_in_epoch,
                         "state": (accumulators["total_state_loss"].item() / steps_in_epoch) if steps_in_epoch > 0 else 0,
                    })
            
            # End of Epoch
            self._log_epoch_summary(accumulators, steps_in_epoch, epoch)
            
            # Validation
            if val_loader:
                self._validate_epoch(epoch, [val_loader], ar_loader=None)
            
            # Checkpointing
            self._save_checkpoint(epoch)


    def _train_step(self, batch_data, epoch, params):
        """Performs forward pass, calculates loss, and scales gradients."""
        # Unpack Dict from ContinuousView
        # Keys: states, actions, seq_idx, reset_mask, loss_mask
        states = batch_data["states"].to(self.device, non_blocking=True)
        actions = batch_data["actions"].to(self.device, non_blocking=True)
        team_ids = batch_data["team_ids"].to(self.device, non_blocking=True)
        seq_idx = batch_data["seq_idx"].to(self.device, non_blocking=True)
        loss_mask = batch_data["loss_mask"].to(self.device, non_blocking=True)
        
        # New: Value and Reward Targets
        # Target for Value(S_t) is G_t. Input is S_t (index :-1).
        # Target for Reward(S_t, A_t) is R_t. Input is S_t.
        # So we align with inputs.
        rewards = batch_data["rewards"].to(self.device, non_blocking=True)[:, :-1]
        if rewards.dim() == 2:
             rewards = rewards.unsqueeze(-1)

        returns = batch_data["returns"].to(self.device, non_blocking=True)[:, :-1]
        if returns.dim() == 2:
             returns = returns.unsqueeze(-1)
        
        # Inputs: [0 : -1] -> Targets: [1 : End]
        # State_t -> State_{t+1}
        # Action_t (Teacher Forcing) -> Action_t (Prediction)
        
        input_states = states[:, :-1]
        target_states = states[:, 1:]
        
        # Actions:
        # Actor: S_t -> Predicts A_t.
        # So we predict actions[:, :-1].
        # Target is actions[:, :-1].
        
        # World Model: S_t + A_t -> S_{t+1}.
        # Input A is actions[:, :-1].
        
        input_actions = actions[:, :-1]
        target_actions = actions[:, :-1]
        
        loss_mask_slice = loss_mask[:, 1:] 
        
        # Pos is now explicitly passed from dataset (float32)
        pos = batch_data["pos"].to(self.device, non_blocking=True)[:, :-1]
        
        # Vel/Att from tokens
        # New Layout: Team(0), Health(1), Power(2), Vel(3,4), Att(5,6), Shoot(7), AngVel(8)
        vel = input_states[..., 3:5]
        att = input_states[..., 5:7]

        # Alive Mask (Health > 0, Health is at index 1)
        alive = input_states[..., 1] > 0
        target_alive = target_states[..., 1] > 0
        
        # Forward & Loss
        # Forward & Loss
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
             # Model Forward
             # Returns: state_pred, action_logits, value_pred, reward_pred, latent
             # Some scaffolds might return None for some outputs
             model_out = self.model(
                state=input_states,
                prev_action=input_actions,
                pos=pos,
                vel=vel,
                att=att,
                team_ids=team_ids[:, :-1],
                seq_idx=seq_idx[:, :-1], # Match temporal dim
                alive=alive,
                reset_mask=batch_data["reset_mask"][:, :-1].to(self.device, non_blocking=True) if "reset_mask" in batch_data else None
             )
             
             # Unpack with defaults if tuple length varies or just pass positional? 
             # Scaffolds are standardized to return 5 items.
             pred_states, pred_actions, pred_values, pred_rewards, _ = model_out
             
             # Get Loss
             # Scaffolds return: total_loss, state_loss, action_loss, relational_loss, metrics
             loss_out = self.model.get_loss(
                pred_states=pred_states,
                pred_actions=pred_actions,
                target_states=target_states,
                target_actions=target_actions,
                loss_mask=loss_mask_slice,
                # Pass config params. 
                # Note: Scaffolds might look up config internally too, but we pass overrides from cfg.train/model here if needed
                # For now we pass what we have, ignoring what's unused by specific scaffold
                lambda_state=self.cfg.model.get("lambda_state", 1.0),
                lambda_actions=self.cfg.model.get("lambda_actions", 0.15),
                pred_values=pred_values,
                pred_rewards=pred_rewards,
                target_returns=returns,
                target_rewards=rewards,
                lambda_value=self.cfg.model.get("lambda_value", 0.1),
                lambda_reward=self.cfg.model.get("lambda_reward", 0.1),
                weights_power=self.w_power,
                weights_turn=self.w_turn,
                weights_shoot=self.w_shoot,
                target_alive=target_alive,
                min_sigma=self.loss_cfg.get("min_sigma", 0.1)
             )
             
             loss, state_loss, action_loss, relational_loss, metrics = loss_out


        # Backward
        step_loss = loss / self.acc_steps
        self.scaler.scale(step_loss).backward()

        # Detach for logging
        return {
            "loss": loss.detach(),
            "state_loss": state_loss.detach(),
            "action_loss": action_loss.detach(),
            "relational_loss": relational_loss.detach(),
            "metrics": metrics
        }

    def _init_accumulators(self):
        # Tensors for epoch sums
        acc = {
           "total_loss": torch.tensor(0.0, device=self.device),
           "total_state_loss": torch.tensor(0.0, device=self.device),
           "total_action_loss": torch.tensor(0.0, device=self.device),
           "total_relational_loss": torch.tensor(0.0, device=self.device),
           "total_value_loss": torch.tensor(0.0, device=self.device),
           "total_reward_loss": torch.tensor(0.0, device=self.device),
           # Scalars/Values for step/epoch logging
           "acc_loss": 0.0, "acc_state": 0.0, "acc_action": 0.0, "acc_rel": 0.0,
           "acc_time": 0.0,
           "acc_errors": {} 
        }
        return acc

    def _update_accumulators(self, acc, metrics):
        # Epoch Totals
        acc["total_loss"] += metrics["loss"] * self.acc_steps # Undo div
        acc["total_state_loss"] += metrics["state_loss"]
        acc["total_action_loss"] += metrics["action_loss"]
        acc["total_relational_loss"] += metrics["relational_loss"]
        acc["total_value_loss"] += metrics["metrics"].get("value_loss", 0.0)
        acc["total_reward_loss"] += metrics["metrics"].get("reward_loss", 0.0)
        
        # Step Totals
        acc["acc_loss"] += metrics["loss"] * self.acc_steps
        acc["acc_state"] += metrics["state_loss"]
        acc["acc_action"] += metrics["action_loss"]
        acc["acc_rel"] += metrics["relational_loss"]
        acc["acc_time"] += metrics["time"]
        
        m = metrics["metrics"]
        for k, v in m.items():
            if k not in acc["acc_errors"]: acc["acc_errors"][k] = 0.0
            acc["acc_errors"][k] += v

    def _reset_accumulators(self, acc):
        acc["acc_loss"] = 0.0
        acc["acc_state"] = 0.0
        acc["acc_action"] = 0.0
        acc["acc_rel"] = 0.0
        acc["acc_time"] = 0.0
        acc["acc_errors"] = {}

    def _optimize_step(self, acc, pbar, is_range_test, epoch):
        self.scaler.unscale_(self.optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        if self.scheduler:
            self.scheduler.step()
            
        pbar.update(1)
            
        self.global_step += 1
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Prepare Step Metrics
        metrics = {
            "step": self.global_step,
            "epoch": epoch + 1,
            "lr": current_lr,
            "loss": acc["acc_loss"] / self.acc_steps,
            "state_loss": acc["acc_state"] / self.acc_steps,
            "action_loss": acc["acc_action"] / self.acc_steps,
            "relational_loss": acc["acc_rel"] / self.acc_steps,
            "param/entropy_lambda": self._get_current_params(self.global_step)["lambda_entropy"],
            "param/focal_gamma": self._get_current_params(self.global_step)["focal_gamma"],
            "param/min_skill": self._get_current_params(self.global_step)["min_skill"],
            "time/micro_batch": acc["acc_time"] / self.acc_steps, # Avg pure compute time per micro step
            "time/macro_batch": time.time() - self.t_last_macro,  # Wall time for full update (incl data load)
            "time/macro_batch": time.time() - self.t_last_macro,  # Wall time for full update (incl data load)
            "grad_norm": total_norm.detach()
        }
        
        # Reset macro timer
        self.t_last_macro = time.time()

        for k, v in acc["acc_errors"].items():
             metrics[k] = v / self.acc_steps
             
        self.logger.log_step(metrics, self.global_step)


    def _log_epoch_summary(self, acc, steps, epoch):
        if steps == 0: return

        avg_loss = acc["total_loss"].item() / steps
        avg_state = acc["total_state_loss"].item() / steps
        avg_action = acc["total_action_loss"].item() / steps
        avg_rel = acc["total_relational_loss"].item() / steps
        avg_value = acc["total_value_loss"].item() / steps
        avg_reward = acc["total_reward_loss"].item() / steps
        
        current_lr = self.optimizer.param_groups[0]['lr']
        log.info(f"Epoch {epoch + 1}: LR={current_lr:.2e} Train Loss={avg_loss:.4f} (State={avg_state:.4f}, Action={avg_action:.4f}, Rel={avg_rel:.4f}, Val={avg_value:.4f}, Rew={avg_reward:.4f})")
        
        metrics = {
            "Train/LearningRate": current_lr,
            "Loss/train": avg_loss,
            "Loss/train_state": avg_state,
            "Loss/train_action": avg_action,
            "Loss/train_relational": avg_rel,
            "Loss/train_value": avg_value,
            "Loss/train_reward": avg_reward,
            "epoch": epoch + 1,
            "global_step": self.global_step
        }
        # We could add train_error_* averages here if we tracked them per epoch
        
        self.logger.log_epoch(metrics, epoch)

        # Store for saving
        self.last_train_metrics = metrics


    def _validate_epoch(self, epoch, val_loaders, ar_loader):
        val_metrics = self.validator.validate_validation_set(val_loaders)
        
        # Heavy Eval Frequency
        val_cfg = self.cfg.model.get("validation", None)
        heavy_freq = val_cfg.heavy_eval_freq if val_cfg else 5
        is_heavy_eval = (epoch + 1) % heavy_freq == 0

        ar_metrics = {}
        if is_heavy_eval:
             ar_metrics = self.validator.validate_autoregressive(ar_loader)
        
        avg_val_loss = val_metrics["val_loss"]
        log_msg = f"Epoch {epoch + 1}: Val Loss={avg_val_loss:.4f}"
        if is_heavy_eval:
             log_msg += f" AR MSE={ar_metrics.get('val_rollout_mse_state', -1):.4f}"
        log.info(log_msg)
        
        # Merge metrics for logging
        log_metrics = {
            "Loss/val": avg_val_loss,
            "Error/val_power": val_metrics["error_power"],
            "Error/val_turn": val_metrics["error_turn"],
            "Error/val_shoot": val_metrics["error_shoot"],
            "Prob/val_power": val_metrics.get("prob_power", 0),
            "Prob/val_turn": val_metrics.get("prob_turn", 0),
            "Prob/val_shoot": val_metrics.get("prob_shoot", 0),
            "epoch": epoch + 1,
            "global_step": self.global_step
        }
        
        # Add Confusion Matrices to log (Always)
        if len(val_metrics.get("preds_p", [])) > 0:
             try:
                 log_metrics["conf_mat_power"] = wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=val_metrics["targets_p"],
                    preds=val_metrics["preds_p"],
                    class_names=[x.name for x in PowerActions]
                 )
             except Exception: pass
        if len(val_metrics.get("preds_t", [])) > 0:
             try:
                 log_metrics["conf_mat_turn"] = wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=val_metrics["targets_t"],
                    preds=val_metrics["preds_t"],
                    class_names=[x.name for x in TurnActions]
                 )
             except Exception: pass
        if len(val_metrics.get("preds_s", [])) > 0:
             try:
                 log_metrics["conf_mat_shoot"] = wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=val_metrics["targets_s"],
                    preds=val_metrics["preds_s"],
                    class_names=[x.name for x in ShootActions]
                 )
             except Exception: pass
             
        # Add AR metrics
        if is_heavy_eval:
            for k, v in ar_metrics.items():
                # Allow lists for heatmap logging
                log_metrics[f"Val_AR/{k}"] = v

        self.logger.log_epoch(log_metrics, epoch)
        
        # Save Best
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            torch.save(self.model.state_dict(), self.run_dir / "best_world_model.pth")
            log.info(f"Saved best live model with val loss {self.best_val_loss:.4f}")


    def _handle_swa(self, epoch, val_loaders):
        swa_start_condition = False
        sched_cfg = self.cfg.model.get("scheduler", None)
        if sched_cfg and sched_cfg.type == "warmup_constant":
             if self.global_step > sched_cfg.warmup.steps:
                 swa_start_condition = True
        else:
             if epoch >= self.cfg.model.get("swa_start_epoch", 2):
                 swa_start_condition = True

        if swa_start_condition:
             self.swa_model.update_parameters(self.model)
             
             # Evaluate
             self.swa_model.averaged_model.to(self.device)
             
             # SWA validation always uses default settings (respecting max_batches)
             val_metrics_swa = self.validator.validate_validation_set(val_loaders, swa_model=self.swa_model.averaged_model)
             self.swa_model.averaged_model.to('cpu')
             
             avg_val_loss_swa = val_metrics_swa["val_loss"]
             log.info(f"Epoch {epoch + 1}: SWA Val Loss={avg_val_loss_swa:.4f}")
             
             metrics = {
                 "Loss/val_swa": avg_val_loss_swa,
                 "Error/val_swa_power": val_metrics_swa["error_power"],
                 "Error/val_swa_turn": val_metrics_swa["error_turn"],
                 "Error/val_swa_shoot": val_metrics_swa["error_shoot"],
                 "epoch": epoch + 1
             }
             self.logger.log_epoch(metrics, epoch)
             
             if avg_val_loss_swa < self.best_val_loss_swa:
                self.best_val_loss_swa = avg_val_loss_swa
                torch.save(self.swa_model.averaged_model.state_dict(), self.run_dir / "best_world_model_swa.pth")
                log.info(f"Saved best SWA model with val loss {self.best_val_loss_swa:.4f}")

    def _save_checkpoint(self, epoch):
        if (epoch + 1) % 10 == 0 or (epoch + 1) == self.epochs:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "swa_model_state_dict": self.swa_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "scaler_state_dict": self.scaler.state_dict(),
                "config": OmegaConf.to_container(self.cfg, resolve=True)
            }
            name = f"world_model_epoch_{epoch + 1}.pt" if (epoch+1) < self.epochs else "final_world_model.pt"
            save_path = self.run_dir / name
            torch.save(checkpoint, save_path)
            log.info(f"Saved checkpoint to {save_path}")
