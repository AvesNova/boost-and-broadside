import torch
import torch.nn.functional as F
import numpy as np
import logging
from omegaconf import DictConfig

from env.constants import PowerActions, TurnActions, ShootActions

log = logging.getLogger(__name__)

class Validator:
    def __init__(self, model, device, cfg: DictConfig):
        self.model = model
        self.device = device
        self.cfg = cfg
        self.amp = cfg.train.get("amp", False) and device.type == 'cuda'

    def validate_validation_set(self, loaders, swa_model=None):
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

        # Load weights
        lambda_state = self.cfg.world_model.get("lambda_state", 1.0)
        lambda_action = self.cfg.world_model.get("lambda_action", 0.01)
        lambda_relational = self.cfg.world_model.get("lambda_relational", 0.1)

        # loaders is a list of data loaders
        for loader in loaders:
            with torch.no_grad():
                for batch in loader:
                    (
                        states,
                        input_actions,
                        target_actions,
                        returns,
                        loss_mask,
                        action_masks,
                        agent_skills,
                        team_ids,
                        rel_features
                    ) = batch
                    
                    states = states.to(self.device)
                    input_actions = input_actions.to(self.device) # Shifted
                    target_actions = target_actions.to(self.device) # Current
                    loss_mask = loss_mask.to(self.device)
                    team_ids = team_ids.to(self.device)
                    rel_features = rel_features.to(self.device)

                    input_states = states[:, :-1]
                    input_actions_slice = input_actions[:, :-1]
                    rel_features_slice = rel_features[:, :-1]
                    
                    num_ships = states.shape[2]
                    
                    # Fix Team IDs (Validation)
                    if team_ids.ndim == 2 and team_ids.shape[1] != num_ships:
                         tid = torch.zeros((states.shape[0], num_ships), device=self.device, dtype=torch.long)
                         half = num_ships // 2
                         tid[:, half:] = 1
                         input_team_ids = tid
                    elif team_ids.ndim == 3:
                         input_team_ids = team_ids[:, :-1, :num_ships]
                    else:
                         input_team_ids = team_ids

                    target_states_slice = states[:, 1:]
                    target_actions_slice = target_actions[:, :-1]
                    
                    loss_mask_slice = loss_mask[:, 1:]

                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.amp):
                        pred_states, pred_actions, _, _, features_12d, pred_relational = model_to_use(
                            input_states, 
                            input_actions_slice, 
                            input_team_ids, 
                            relational_features=rel_features_slice,
                            noise_scale=0.0,
                            return_embeddings=True
                        )

                        loss, state_loss, action_loss, relational_loss, _ = model_to_use.get_loss(
                            pred_states=pred_states,
                            pred_actions=pred_actions,
                            target_states=target_states_slice,
                            target_actions=target_actions_slice,
                            loss_mask=loss_mask_slice,
                            target_features_12d=features_12d, 
                            pred_relational=pred_relational,
                            lambda_state=lambda_state,
                            lambda_action=lambda_action,
                            lambda_relational=lambda_relational
                        )
                    
                    val_loss += loss
                    val_steps += 1
                    
                    valid_mask = loss_mask_slice.bool()
                    valid_pred = pred_actions[valid_mask]
                    valid_target = target_actions_slice[valid_mask]
                    
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
        Run a short autoregressive rollout on one batch from the loader.
        """
        self.model.eval()
        try:
            # Get one batch
            batch = next(iter(loader))
        except StopIteration:
            return {}

        (
            states,
            input_actions,
            target_actions,
            returns,
            loss_mask,
            action_masks,
            agent_skills,
            team_ids,
            rel_features
        ) = batch
        
        # Take first B samples. Actually just use the whole batch.
        states = states.to(self.device)
        input_actions = input_actions.to(self.device)
        target_actions = target_actions.to(self.device)
        team_ids = team_ids.to(self.device)
        # rel_features = rel_features.to(self.device) # Not used for dreaming

        # We need at least steps+1 length
        if states.shape[1] < steps + 1:
            return {} # Too short

        initial_state = states[:, 0] # S0
        initial_action = input_actions[:, 0] # A_prev (dummy)
        
        # Fix Team IDs
        num_ships = states.shape[2]
        if team_ids.ndim == 2 and team_ids.shape[1] != num_ships:
                # Reconstruct standard
                tid = torch.zeros((states.shape[0], num_ships), device=self.device, dtype=torch.long)
                half = num_ships // 2
                tid[:, half:] = 1
                input_team_ids = tid
        elif team_ids.ndim == 3:
                # (B, T, N) -> Slice time AND ships. Use 0th step.
                input_team_ids = team_ids[:, 0, :num_ships]
        else:
                input_team_ids = team_ids
                
        
        # Generate
        dream_states, dream_actions = self.model.generate(
            initial_state=initial_state,
            initial_action=initial_action, # Dummy
            steps=steps,
            n_ships=num_ships,
            team_ids=input_team_ids
        )
        
        target_states_slice = states[:, 1:1+steps]
        # target_actions: In data, actions[t] is action taken at S[t].
        # Model predicts A[t] from S[t].
        # So dream_actions[0] corresponds to A[0].
        target_actions_slice = target_actions[:, 0:steps]
        
        # MSE State
        # (B, Steps, N, D)
        mse_state = F.mse_loss(dream_states, target_states_slice)
        
        # Per-step MSE (average over Batch, Ships, Dim)
        # Result: (Steps,)
        mse_state_per_step = (dream_states - target_states_slice).pow(2).mean(dim=[0, 2, 3])
        
        # Action Accuracy (Hard match of indices)
        # dream_actions is one-hot (float). target_actions is discrete indices (B, T, N, 3).
        # Convert dream to indices
        dream_p = dream_actions[..., 0:3].argmax(-1)
        dream_t = dream_actions[..., 3:10].argmax(-1)
        dream_s = dream_actions[..., 10:12].argmax(-1)
        
        target_p = target_actions_slice[..., 0].long()
        target_t = target_actions_slice[..., 1].long()
        target_s = target_actions_slice[..., 2].long()
        
        acc_p = (dream_p == target_p).float().mean()
        acc_t = (dream_t == target_t).float().mean()
        acc_s = (dream_s == target_s).float().mean()
        
        return {
            "val_rollout_mse_state": mse_state.item(),
            "val_rollout_mse_step": mse_state_per_step.tolist(), # List of floats
            "val_rollout_acc_power": acc_p.item(),
            "val_rollout_acc_turn": acc_t.item(),
            "val_rollout_acc_shoot": acc_s.item()
        }
