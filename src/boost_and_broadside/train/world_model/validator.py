import torch
import numpy as np
import logging
from omegaconf import DictConfig


log = logging.getLogger(__name__)

class Validator:
    def __init__(self, model, device, cfg: DictConfig):
        self.model = model
        self.device = device
        self.cfg = cfg
        self.amp = cfg.train.get("amp", False) and device.type == 'cuda'

    def validate_validation_set(self, loaders, swa_model=None, max_batches=None):
        """
        Run validation loop and return metrics.
        If swa_model is provided, uses it instead of self.model.
        """
        model_to_use = swa_model if swa_model is not None else self.model
        model_to_use.eval()
        
        val_loss = torch.tensor(0.0, device=self.device)
        val_steps = 0
        val_error_p = torch.tensor(0.0, device=self.device)
        val_error_t = torch.tensor(0.0, device=self.device)
        val_error_s = torch.tensor(0.0, device=self.device)

        # Collections for Confusion Matrix
        all_preds_p = []
        all_targets_p = []
        all_preds_t = []
        all_targets_t = []
        all_preds_s = []
        all_targets_s = []

        # Validation Limit
        if max_batches is None:
            val_cfg = self.cfg.world_model.get("validation", None)
            max_batches = val_cfg.max_batches if val_cfg else 100
        
        total_batches_processed = 0

        # loaders is a list of data loaders
        for loader in loaders:
            with torch.no_grad():
                for batch in loader:
                    if total_batches_processed >= max_batches:
                        break
                    total_batches_processed += 1
                    
                    # Unpack Dict
                # Unpack Dict
                    states = batch["states"].to(self.device)
                    actions = batch["actions"].to(self.device)
                    team_ids = batch["team_ids"].to(self.device)
                    loss_mask = batch["loss_mask"].to(self.device)
                    seq_idx = batch["seq_idx"].to(self.device)
                    
                    # New: Value and Reward Targets
                    rewards = batch["rewards"].to(self.device)[:, :-1]
                    if rewards.dim() == 2:
                         rewards = rewards.unsqueeze(-1)

                    returns = batch["returns"].to(self.device)[:, :-1]
                    if returns.dim() == 2:
                         returns = returns.unsqueeze(-1)
                    
                    # Inputs/Targets
                    input_states = states[:, :-1]
                    target_states = states[:, 1:]
                    
                    input_actions = actions[:, :-1]
                    target_actions = actions[:, :-1]
                    
                    loss_mask_slice = loss_mask[:, 1:]
                    
                    # Pos comes from batch now
                    pos = batch["pos"].to(self.device)
                    # Input pos: 0..T-1
                    pos_in = pos[:, :-1]
                    
                    # New token layout: Vel(3,4), Att(5,6)
                    vel = input_states[..., 3:5]
                    att = input_states[..., 5:7]
                    
                    alive = input_states[..., 1] > 0
                    target_alive = target_states[..., 1] > 0

                    # Cast inputs if AMP is disabled (e.g. CPU training with bfloat16 data)
                    if not self.amp and input_states.dtype == torch.bfloat16:
                         input_states = input_states.float()
                         target_states = target_states.float()
                         pos_in = pos_in.float()
                         vel = vel.float()
                         att = att.float()

                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.amp):
                        pred_states, pred_actions, pred_values, pred_rewards, _ = model_to_use(
                            state=input_states, 
                            prev_action=input_actions, 
                            pos=pos_in,
                            vel=vel,
                            att=att,
                            team_ids=team_ids[:, :-1],
                            seq_idx=seq_idx[:, :-1],
                            alive=alive
                        )

                        loss, _, _, _, metrics = model_to_use.get_loss(
                           pred_states=pred_states,
                           pred_actions=pred_actions,
                           target_states=target_states,
                           target_actions=target_actions,
                           loss_mask=loss_mask_slice,
                           lambda_state=self.cfg.world_model.get("lambda_state", 1.0),
                           lambda_power=self.cfg.world_model.get("lambda_power", 0.05),
                           lambda_turn=self.cfg.world_model.get("lambda_turn", 0.05),
                           lambda_shoot=self.cfg.world_model.get("lambda_shoot", 0.05),
                           pred_values=pred_values,
                           pred_rewards=pred_rewards,
                           target_returns=returns,
                           target_rewards=rewards,
                           lambda_value=self.cfg.world_model.get("lambda_value", 0.1),
                           lambda_reward=self.cfg.world_model.get("lambda_reward", 0.1),
                           target_alive=target_alive
                        )
                    
                    val_loss += loss
                    val_steps += 1
                    
                    valid_mask = loss_mask_slice.bool()
                    if valid_mask.sum() > 0:
                        if valid_mask.ndim == 2:
                             # Expand if mask is (B,T) and pred is (B,T,N,...)
                             valid_mask = valid_mask.unsqueeze(-1).expand_as(pred_actions[..., 0])
                        
                        # Flat mask
                        flat_mask = valid_mask.reshape(-1)
                        valid_pred = pred_actions.reshape(-1, 12)[flat_mask]
                        valid_target = target_actions.reshape(-1, 3)[flat_mask]
                        
                        p_logits = valid_pred[..., 0:3]
                        t_logits = valid_pred[..., 3:10]
                        s_logits = valid_pred[..., 10:12]
                        
                        p_target = valid_target[..., 0].long()
                        t_target = valid_target[..., 1].long()
                        s_target = valid_target[..., 2].long()
                        
                        if p_target.numel() > 0:
                            val_error_p += (p_logits.argmax(-1) != p_target).float().mean()
                            val_error_t += (t_logits.argmax(-1) != t_target).float().mean()
                            val_error_s += (s_logits.argmax(-1) != s_target).float().mean()

                            # Collect for Confusion Matrix
                            all_preds_p.append(p_logits.argmax(-1).cpu().numpy())
                            all_targets_p.append(p_target.cpu().numpy())
                            all_preds_t.append(t_logits.argmax(-1).cpu().numpy())
                            all_targets_t.append(t_target.cpu().numpy())
                            all_preds_s.append(s_logits.argmax(-1).cpu().numpy())
                            all_targets_s.append(s_target.cpu().numpy())

        avg_val_loss = val_loss.item() / val_steps if val_steps > 0 else 0.0
        avg_val_err_p = val_error_p.item() / val_steps if val_steps > 0 else 0.0
        avg_val_err_t = val_error_t.item() / val_steps if val_steps > 0 else 0.0
        avg_val_err_s = val_error_s.item() / val_steps if val_steps > 0 else 0.0

        return {
            "val_loss": avg_val_loss,
            "error_power": avg_val_err_p,
            "error_turn": avg_val_err_t,
            "error_shoot": avg_val_err_s,
            "preds_p": np.concatenate(all_preds_p) if all_preds_p else np.array([]),
            "targets_p": np.concatenate(all_targets_p) if all_targets_p else np.array([]),
            "preds_t": np.concatenate(all_preds_t) if all_preds_t else np.array([]),
            "targets_t": np.concatenate(all_targets_t) if all_targets_t else np.array([]),
            "preds_s": np.concatenate(all_preds_s) if all_preds_s else np.array([]),
            "targets_s": np.concatenate(all_targets_s) if all_targets_s else np.array([])
        }

    def validate_autoregressive(self, loader, steps=50):
        """
        Run a short autoregressive rollout.
        MambaBB gen not implemented yet.
        """
        log.warning("Autoregressive validation not implemented for MambaBB yet.")
        return {}
