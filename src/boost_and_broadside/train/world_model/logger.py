import logging
import torch
import wandb
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

class MetricLogger:
    def __init__(self, cfg: DictConfig, run_dir: Path):
        self.cfg = cfg
        self.run_dir = run_dir
        self.wb_cfg = cfg.get("wandb", None)
        self.log_buffer = []
        self.log_freq = self.wb_cfg.get("log_frequency", 50) if self.wb_cfg else 50
        
        # Force frequency 1 for Range Tests
        sched_cfg = cfg.model.get("scheduler", None)
        self.is_range_test = sched_cfg and "range_test" in getattr(sched_cfg, "type", "")
        if self.is_range_test:
            self.log_freq = 1
            log.info("Range Test detected: Forcing MetricLogger frequency to 1")
        
        # Setup output files
        self.step_log_path = run_dir / "training_step_log.csv"
        with open(self.step_log_path, "w") as f:
            f.write("global_step,epoch,learning_rate,loss,state_loss,action_loss,relational_loss,val_loss,val_loss_swa\n")

        self.csv_path = run_dir / "training_log.csv"
        with open(self.csv_path, "w") as f:
            f.write(
                "epoch,learning_rate,train_loss,train_state_loss,train_action_loss,train_relational_loss,val_loss,val_loss_swa\n"
            )

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(run_dir))
        
        # WandB
        if self.wb_cfg and self.wb_cfg.get("enabled", False):
            wandb.init(
                project=self.wb_cfg.project,
                entity=self.wb_cfg.entity,
                name=self.wb_cfg.get("name") or str(run_dir.name),
                group=self.wb_cfg.get("group"),
                mode=self.wb_cfg.get("mode", "online"),
                config=OmegaConf.to_container(cfg, resolve=True),
                resume=False
            )

    def log_step(self, step_metrics: dict, global_step: int):
        """Buffer and log step metrics (high frequency)."""
        # Always buffer for CSV logging
        self.log_buffer.append(step_metrics)
        
        if len(self.log_buffer) >= self.log_freq:
            self.flush_buffer()
        
        # WandB logging is handled inside flush_buffer

    def flush_buffer(self):
        """Flush buffered wandb logs."""
        if not self.log_buffer:
            return

        packed = {}
        keys = self.log_buffer[0].keys()
        for k in keys:
            try:
                tensors = [x[k] for x in self.log_buffer]
                packed[k] = torch.stack(tensors).cpu().tolist()
            except:
                packed[k] = [x[k] for x in self.log_buffer]
        
        if self.wb_cfg and self.wb_cfg.get("enabled", False):
            for i in range(len(packed["step"])):
                log_item = {k: packed[k][i] for k in packed}
                step_val = int(log_item.pop("step"))
                # Ensure native types
                wandb.log(log_item, step=step_val)
        
        # CSV Logging (Bulk)
        # We use a simple comma-separated format. For "painless" extension, 
        # we log all metrics found in the first row.
        if not hasattr(self, "_csv_headers_written"):
             self._csv_headers_written = False

        with open(self.step_log_path, "a") as f:
            for i in range(len(packed["step"])):
                row_metrics = {k: packed[k][i] for k in packed}
                
                # Write header if not done (and if we have metrics)
                if not self._csv_headers_written:
                     headers = list(row_metrics.keys())
                     # Re-write the file with headers if it was just initialized with defaults
                     # (MetricLogger.__init__ writes a default header, we might want to override)
                     with open(self.step_log_path, "w") as f2:
                          f2.write(",".join(headers) + "\n")
                     self._csv_headers_written = True
                     self._csv_column_order = headers
                
                # Write row in consistent order
                row_vals = [str(row_metrics.get(k, "")) for k in self._csv_column_order]
                f.write(",".join(row_vals) + "\n")

        self.log_buffer = []

    def log_to_csv_step(self, metrics: dict):
        """Deprecated: CSV logging is now handled in flush_buffer."""
        pass

    def log_epoch(self, epoch_metrics: dict, epoch: int):
        """Log epoch summary metrics."""
        # Flush any remaining step logs
        self.flush_buffer()
        
        # TensorBoard
        for k, v in epoch_metrics.items():
            if isinstance(v, (int, float, np.number)):
                 if "conf_mat" not in k: # Skip confusion matrices
                    self.writer.add_scalar(k, v, epoch)
        
        # CSV
        lr = epoch_metrics.get("Train/LearningRate", 0)
        train_loss = epoch_metrics.get("Loss/train", 0)
        train_state = epoch_metrics.get("Loss/train_state", 0)
        train_action = epoch_metrics.get("Loss/train_action", 0)
        train_rel = epoch_metrics.get("Loss/train_relational", 0)
        val_loss = epoch_metrics.get("Loss/val", 0)
        swa_loss = epoch_metrics.get("Loss/val_swa", "")
        if swa_loss == "": swa_loss = "" # Ensure empty string if missing
        else: swa_loss = f"{swa_loss:.6f}"

        with open(self.csv_path, "a") as f:
            f.write(
                f"{epoch + 1},{lr:.8f},{train_loss:.6f},{train_state:.6f},{train_action:.6f},{train_rel:.6f},{val_loss:.6f},{swa_loss}\n"
            )

        # WandB (Validation & Epoch Summary)
        if self.wb_cfg and self.wb_cfg.get("enabled", False):
            # Try to plot confusion matrices if present in special keys
            wandb_log = {k: v for k, v in epoch_metrics.items() if "conf_mat" not in k and "Loss/" not in k and "Train/" not in k} 
            # Note: We filter out Tensorboard style keys if we constructed them that way, 
            # but usually we pass clean keys. Let's assume the caller passes a clean dict for WandB 
            # or we handle the mapping. 
            # For simplicity, let's assume epoch_metrics contains everything we want to log to wandb 
            # properly formatted, OR we do the mapping here.
            # The original code logged specific "val_*" keys.
            
            # We will handle confusion matrices if passed as "conf_mat_power" object etc.
            if "conf_mat_power" in epoch_metrics:
                wandb_log["conf_mat_power"] = epoch_metrics["conf_mat_power"]
            if "conf_mat_turn" in epoch_metrics:
                wandb_log["conf_mat_turn"] = epoch_metrics["conf_mat_turn"]
            if "conf_mat_shoot" in epoch_metrics:
                wandb_log["conf_mat_shoot"] = epoch_metrics["conf_mat_shoot"]
            
            # Helper for line plots / heatmaps
            for k, v in epoch_metrics.items():
                if isinstance(v, (list, np.ndarray)) and "conf_mat" not in k:
                    # Log as a custom chart or line series
                    # For rollout MSE step, we want a Line Plot: Step (x) vs MSE (y)
                    # Or a Heatmap if we want to see it evolve over epochs (WandB Heatmap is usually x, y, color)
                    # Simple Line plot is better for a single epoch. 
                    # But to track over time, we might just want to dump the array and let user viz it.
                    # Or: Log a bar chart?
                    # Let's log a Line Series for this step.
                    try:
                        data = [[x, y] for x, y in enumerate(v)]
                        table = wandb.Table(data=data, columns=["step", "mse"])
                        wandb_log[k] = wandb.plot.line(table, "step", "mse", title=k)
                    except:
                        pass
                
            wandb.log(wandb_log, step=epoch_metrics.get("global_step", epoch * 100)) # Fallback step if missing

    def close(self):
        self.flush_buffer()
        self.writer.close()
