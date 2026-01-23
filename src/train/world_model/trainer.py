import logging
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import wandb
import numpy as np

from env.constants import PowerActions, TurnActions, ShootActions
from env.features import compute_pairwise_features
from train.swa import SWAModule
from train.swa import SWAModule
from train.world_model.rollout import perform_rollout, get_rollout_length
from train.world_model.setup import get_data_loaders

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
        
        self.epochs = cfg.world_model.epochs
        self.batch_ratio = cfg.world_model.batch_ratio
        self.acc_steps = cfg.world_model.get("gradient_accumulation_steps", 1)
        self.use_amp = cfg.train.get("amp", False) and device.type == 'cuda'
        
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_val_loss_swa = float("inf")
        self.data_path = data_path
        
        # Curriculum State
        self.curr_cfg = cfg.world_model.get("curriculum", {})
        self.entropy_cfg = cfg.world_model.get("entropy", {})
        self.loss_cfg = cfg.world_model.get("loss", {})
        
        self.current_min_skill = self.curr_cfg.get("min_skill_start", 0.0) if self.curr_cfg.get("enabled", False) else 0.0
        self.last_reloaded_skill = self.current_min_skill

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

    def train(self, train_short_loader, train_long_loader, val_short_loader, val_long_loader):
        """Main training loop."""
        
        sched_cfg = self.cfg.world_model.get("scheduler", None)
        is_range_test = sched_cfg and "range_test" in sched_cfg.type

        # Profiler Setup
        prof_cfg = self.cfg.get("profiler", OmegaConf.create({"enabled": False}))
        profiler = None
        if prof_cfg.enabled:
            log.info("Profiler ENABLED")
            schedule = torch.profiler.schedule(
                wait=prof_cfg.wait,
                warmup=prof_cfg.warmup,
                active=prof_cfg.active,
                repeat=prof_cfg.repeat
            )
            profiler = torch.profiler.profile(
                schedule=schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.run_dir / "profiler")),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            profiler.start()

        for epoch in range(self.epochs):
            # Check Curriculum Update
            current_params = self._get_current_params(self.global_step)
            tgt_skill = current_params["min_skill"]
            
            # Reload if skill threshold changed significantly (e.g., > 0.05)
            if abs(tgt_skill - self.last_reloaded_skill) > 0.05:
                log.info(f"Curriculum Update: Min Skill {self.last_reloaded_skill:.2f} -> {tgt_skill:.2f}. Reloading Data...")
                train_short_loader, train_long_loader, _, _ = get_data_loaders(self.cfg, self.data_path, min_skill=tgt_skill)
                # Note: We don't reload validation loaders to keep validation consistent (or should we?)
                # Usually validation set should remain static (all data) to be comparable.
                self.last_reloaded_skill = tgt_skill
                
            self.model.train()
            
            # Setup Iterators & Progress Bar
            short_iter = iter(train_short_loader)
            long_iter = iter(train_long_loader)
            num_short_batches = len(train_short_loader)

            if is_range_test and sched_cfg.range_test.steps:
                 pbar_total = sched_cfg.range_test.steps * self.acc_steps
            else:
                 total_micro = int(num_short_batches * (1 + 1/self.batch_ratio))
                 pbar_total = total_micro // self.acc_steps
                 
            pbar = tqdm(range(pbar_total), desc=f"Epoch {epoch + 1}/{self.epochs}")
            
            # Batch Pattern
            short_prob = self.batch_ratio / (self.batch_ratio + 1)
            use_sobol = self.cfg.world_model.get("use_sobol", False)
            
            def get_batch_pattern():
                if use_sobol:
                    log.info(f"Using Sobol Sampling (Ratio {self.batch_ratio})")
                    sobol_engine = torch.quasirandom.SobolEngine(dimension=1, scramble=True)
                    while True:
                        yield sobol_engine.draw(1).item() < short_prob
                else:
                    log.info(f"Using Fixed Pattern: {self.batch_ratio} Short, 1 Long")  
                    while True:
                        for _ in range(4): yield True
                        yield False
            
            batch_pattern = get_batch_pattern()
            
            # Accumulators
            accumulators = self._init_accumulators()
            micro_step = 0
            steps_in_epoch = 0
            self.optimizer.zero_grad()
            self.t_last_macro = time.time()
            
            while True:
                # 1. Get Batch
                is_short = next(batch_pattern)
                iterator = short_iter if is_short else long_iter
                
                try:
                    batch_data = next(iterator)
                except StopIteration:
                    break
                
                # 2. Process Batch
                t0 = time.time()
                # Update params for this step (micro-step level updates for smooth schedules)
                step_params = self._get_current_params(self.global_step)
                step_metrics = self._train_step(batch_data, is_short, epoch, step_params)
                step_metrics["time"] = time.time() - t0
                
                # 3. Accumulate
                self._update_accumulators(accumulators, step_metrics)
                
                micro_step += 1
                
                # 4. Optimization Step
                if micro_step % self.acc_steps == 0:
                    self._optimize_step(accumulators, pbar, is_range_test, epoch)
                    self._reset_accumulators(accumulators)
                    
                    # Update Progress Bar (Range Test)
                    if is_range_test: pbar.update(1)

                # Profiler Step
                if profiler:
                    profiler.step()

                # Check Range Test Limit
                if is_range_test and self.global_step >= sched_cfg.range_test.steps:
                     log.info("Range test complete.")
                     if profiler: profiler.stop()
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
            self._validate_epoch(epoch, [val_short_loader, val_long_loader], ar_loader=val_long_loader)
            
            # SWA
            self._handle_swa(epoch, [val_short_loader, val_long_loader])
            
            # Checkpointing
            self._save_checkpoint(epoch)

        if profiler:
            profiler.stop()

    def _train_step(self, batch_data, is_short, epoch, params):
        """Performs forward pass, calculates loss, and scales gradients."""
        (
            states, input_actions, target_actions, _,
            loss_mask, _, _, team_ids
        ) = batch_data

        states = states.to(self.device, non_blocking=True)
        input_actions = input_actions.to(self.device, non_blocking=True)
        target_actions = target_actions.to(self.device, non_blocking=True)
        loss_mask = loss_mask.to(self.device, non_blocking=True)
        team_ids = team_ids.to(self.device, non_blocking=True)
        
        # Compute Relational Features on GPU
        # Shape: (B, T, N, D) or (T, N, D) depending on batch
        rel_features = compute_pairwise_features(states, self.cfg.environment.world_size)

        input_states = states[:, :-1].clone()
        input_actions_slice = input_actions[:, :-1].clone()
        rel_features_slice = rel_features[:, :-1]
        
        num_ships = states.shape[2]
        
        # Team IDs
        if team_ids.ndim == 2 and team_ids.shape[1] != num_ships:
             tid = torch.zeros((states.shape[0], num_ships), device=self.device, dtype=torch.long)
             tid[:, num_ships//2:] = 1
             input_team_ids = tid
        elif team_ids.ndim == 3:
             input_team_ids = team_ids[:, :-1, :num_ships]
        else:
             input_team_ids = team_ids

        target_states_slice = states[:, 1:]
        target_actions_slice = target_actions[:, :-1]
        loss_mask_slice = loss_mask[:, 1:]
        
        # Masking for long batches
        if not is_short and loss_mask_slice.shape[1] > 32:
            loss_mask_slice[:, :32] = False

        # Rollout (Side Effect on inputs)
        rollout_len = get_rollout_length(epoch, self.cfg)
        perform_rollout(self.model, input_states, input_actions_slice, input_team_ids, rollout_len)

        # Forward & Loss
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
             pred_states, pred_actions, _, latents, features_12d, pred_relational = self.model(
                input_states,
                input_actions_slice,
                input_team_ids,
                relational_features=rel_features_slice,
                noise_scale=self.cfg.world_model.noise_scale,
                return_embeddings=True
             )
             
             loss, state_loss, action_loss, relational_loss, metrics = self.model.get_loss(
                pred_states=pred_states,
                pred_actions=pred_actions,
                target_states=target_states_slice,
                target_actions=target_actions_slice,
                loss_mask=loss_mask_slice,
                latents=latents,
                target_features_12d=features_12d,
                pred_relational=pred_relational,
                lambda_state=self.cfg.world_model.get("lambda_state", 1.0),
                lambda_power=self.cfg.world_model.get("lambda_power", 0.05),
                lambda_turn=self.cfg.world_model.get("lambda_turn", 0.05),
                lambda_shoot=self.cfg.world_model.get("lambda_shoot", 0.05),
                lambda_relational=self.cfg.world_model.get("lambda_relational", 0.1),
                lambda_entropy=params["lambda_entropy"],
                focal_gamma=params["focal_gamma"]
             )

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
            
        if not is_range_test:
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
            "relational_loss": acc["acc_rel"] / self.acc_steps,
            "param/entropy_lambda": self._get_current_params(self.global_step)["lambda_entropy"],
            "param/focal_gamma": self._get_current_params(self.global_step)["focal_gamma"],
            "param/min_skill": self._get_current_params(self.global_step)["min_skill"],
            "time/micro_batch": acc["acc_time"] / self.acc_steps, # Avg pure compute time per micro step
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
        
        current_lr = self.optimizer.param_groups[0]['lr']
        log.info(f"Epoch {epoch + 1}: LR={current_lr:.2e} Train Loss={avg_loss:.4f} (State={avg_state:.4f}, Action={avg_action:.4f}, Rel={avg_rel:.4f})")
        
        metrics = {
            "Train/LearningRate": current_lr,
            "Loss/train": avg_loss,
            "Loss/train_state": avg_state,
            "Loss/train_action": avg_action,
            "Loss/train_relational": avg_rel,
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
        val_cfg = self.cfg.world_model.get("validation", None)
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
        
        # Add Confusion Matrices to log (Only on heavy eval)
        if is_heavy_eval:
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
        sched_cfg = self.cfg.world_model.get("scheduler", None)
        if sched_cfg and sched_cfg.type == "warmup_constant":
             if self.global_step > sched_cfg.warmup.steps:
                 swa_start_condition = True
        else:
             if epoch >= self.cfg.world_model.get("swa_start_epoch", 2):
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
