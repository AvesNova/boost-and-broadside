import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, List, Optional
import logging

log = logging.getLogger(__name__)

class LossModule(nn.Module):
    """
    Base interface for all loss modules.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute loss.
        
        Args:
            preds: Dictionary of model predictions.
            targets: Dictionary of ground truth targets.
            mask: Bool/Float mask (B, T, N) or (B, T). 1=Valid, 0=Ignored.
            kwargs: Additional arguments (e.g., class weights).
            
        Returns:
            Dict containing:
                "loss": Scalar weighted loss.
                "metric_name": Detached scalar for logging.
        """
        raise NotImplementedError


class CompositeLoss(LossModule):
    """
    Container for multiple loss modules.
    Sums their weighted outputs.
    """
    def __init__(self, losses: List[LossModule], loss_type: str = "fixed", weights: Optional[Dict[str, float]] = None):
        super().__init__(weight=1.0)
        self.losses = nn.ModuleList(losses)
        self.loss_type = loss_type
        
        # Uncertainty Weighting Params
        self.log_vars = None
        if loss_type == "uncertainty":
             self.log_vars = nn.ParameterDict()
             for i, loss in enumerate(losses):
                 # Use name if available, else index
                 name = getattr(loss, "name", f"loss_{i}")
                 # Check for duplicate names? For now assume unique or use index suffix
                 if name in self.log_vars: name = f"{name}_{i}"
                 self.log_vars[name] = nn.Parameter(torch.tensor(0.0))
             
    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        total_loss = torch.tensor(0.0, device=mask.device)
        metrics = {}
        
        for i, loss_mod in enumerate(self.losses):
            out = loss_mod(preds, targets, mask, **kwargs)
            
            l = out["loss"]
            
            if self.loss_type == "uncertainty" and self.log_vars is not None:
                name = getattr(loss_mod, "name", f"loss_{i}")
                if name not in self.log_vars: name = f"{name}_{i}" # fallback to match init
                
                log_var = self.log_vars[name]
                precision = torch.exp(-log_var)
                
                # Kendall & Gal: L = 0.5 * exp(-s) * loss + 0.5 * s
                # Here 'l' is already weighted by loss_mod.weight (static weight).
                # Typically, uncertainty weighting replaces static weighting.
                # But we will treat static weight as a "prior" scale if both present?
                # Usually we just use the uncertainty weight.
                # We'll apply the formula to the raw loss (which we don't have separately easily unless we trust l/weight?)
                # Actually, LossModule returns l = raw_loss * static_weight.
                # If static_weight is 1.0 (default), it's raw loss.
                
                weighted_l = 0.5 * precision * l + 0.5 * log_var
                total_loss += weighted_l
                metrics[f"sigma/{name}"] = torch.exp(0.5 * log_var).detach()
            else:
                total_loss += l
            
            # Merge metrics
            for k, v in out.items():
                if k != "loss":
                    metrics[k] = v
                    
        metrics["loss"] = total_loss
        
        # Ensure critical metrics exist for trainer compatibility
        if "pairwise_loss" not in metrics:
             metrics["pairwise_loss"] = torch.tensor(0.0, device=mask.device)
             
        return metrics


class StateLoss(LossModule):
    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        pred_states = preds.get("states")
        target_states = targets.get("states")
        
        if pred_states is None or target_states is None:
            return {"loss": torch.tensor(0.0, device=mask.device), "state_loss": torch.tensor(0.0, device=mask.device)}
            
        mask_flat = mask.reshape(-1).float()
        denom = mask_flat.sum() + 1e-6
        mse = F.mse_loss(pred_states, target_states, reduction='none')
        
        # Sum over feature dim, then mask
        s_loss = mse.mean(dim=-1).reshape(-1).mul(mask_flat).sum() / denom
        
        # Logging breakdown
        logs = {"state_loss": s_loss.detach(), "loss_sub/state_mse": s_loss.detach()}
        
        # Detailed feature logging
        with torch.no_grad():
             m_expanded = mask.unsqueeze(-1).float() if mask.ndim < pred_states.ndim else mask
             masked_mse = (mse * m_expanded).sum(dim=(0, 1, 2)) if mse.ndim == 4 else (mse * m_expanded).sum(dim=(0,1))
             feature_names = ["DX", "DY", "DVX", "DVY", "DHEALTH", "DPOWER", "DANG_VEL"]
             for i, name in enumerate(feature_names):
                  if i < mse.shape[-1]:
                       logs[f"loss_sub/state_{name}"] = (masked_mse[i] / denom).item()

        return {"loss": s_loss * self.weight, **logs}


class ActionLoss(LossModule):
    name = "actions"
    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], mask: torch.Tensor, 
                weights_power=None, weights_turn=None, weights_shoot=None, **kwargs) -> Dict[str, torch.Tensor]:
        
        pred_actions = preds.get("actions")
        target_actions = targets.get("actions")
        
        if pred_actions is None or target_actions is None:
             return {"loss": torch.tensor(0.0, device=mask.device), "action_loss": torch.tensor(0.0, device=mask.device)}

        mask_flat = mask.reshape(-1).float()
        denom = mask_flat.sum() + 1e-6
        
        l_p, l_t, l_s = pred_actions[..., 0:3], pred_actions[..., 3:10], pred_actions[..., 10:12]
        # Targets: (B, T, N, 3) -> slice
        t_p = target_actions[..., 0].long().clamp(0, 2)
        t_t = target_actions[..., 1].long().clamp(0, 6)
        t_s = target_actions[..., 2].long().clamp(0, 1)
        
        a_loss_p = (F.cross_entropy(l_p.reshape(-1, 3), t_p.reshape(-1), weight=weights_power, reduction='none') * mask_flat).sum() / denom / math.log(3)
        a_loss_t = (F.cross_entropy(l_t.reshape(-1, 7), t_t.reshape(-1), weight=weights_turn, reduction='none') * mask_flat).sum() / denom / math.log(7)
        a_loss_s = (F.cross_entropy(l_s.reshape(-1, 2), t_s.reshape(-1), weight=weights_shoot, reduction='none') * mask_flat).sum() / denom / math.log(2)
        
        a_loss = a_loss_p + a_loss_t + a_loss_s
        
        logs = {
            "action_loss": a_loss.detach(),
            "loss_sub/action_all": a_loss.detach(),
            "loss_sub/action_power": a_loss_p.detach(),
            "loss_sub/action_turn": a_loss_t.detach(),
            "loss_sub/action_shoot": a_loss_s.detach()
        }
        
        return {"loss": a_loss * self.weight, **logs}


class ValueLoss(LossModule):
    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        pred_val = preds.get("value") # (B, T, 1)
        target_ret = targets.get("returns") # (B, T, N) usually
        
        if pred_val is None or target_ret is None:
             return {"loss": torch.tensor(0.0, device=mask.device), "value_loss": torch.tensor(0.0, device=mask.device)}
             
        # Value is Team-Level. Target might be per-ship.
        # If target has extran dimension N, aggregate it.
        if target_ret.ndim == 3 and target_ret.shape[-1] > 1:
             # Weighted mean over N using mask
             # mask: (B, T, N)
             m_broad = mask.float()
             t_weighted = (target_ret * m_broad).sum(dim=-1, keepdim=True)
             d_weighted = m_broad.sum(dim=-1, keepdim=True) + 1e-6
             target_ret = t_weighted / d_weighted # (B, T, 1)
        elif target_ret.ndim == 2:
             target_ret = target_ret.unsqueeze(-1)

        # Team Mask: Valid if ANY ship is valid
        if mask.ndim == 3:
             team_mask = mask.any(dim=-1, keepdim=True).float() # (B, T, 1)
        else:
             team_mask = mask.unsqueeze(-1)

        d_glob = team_mask.sum() + 1e-6
        
        v_loss = (F.mse_loss(pred_val, target_ret.to(pred_val.dtype), reduction='none') * team_mask).sum() / d_glob
        
        logs = {"value_loss": v_loss.detach(), "loss_sub/value_mse": v_loss.detach()}
        return {"loss": v_loss * self.weight, **logs}


class RewardLoss(LossModule):
    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        pred_rew = preds.get("reward")
        target_rew = targets.get("rewards")
        
        if pred_rew is None or target_rew is None:
             return {"loss": torch.tensor(0.0, device=mask.device), "reward_loss": torch.tensor(0.0, device=mask.device)}

        # Aggregation Logic (Same as ValueLoss)
        if target_rew.ndim == 3 and target_rew.shape[-1] > 1:
             m_broad = mask.float()
             t_weighted = (target_rew * m_broad).sum(dim=-1, keepdim=True)
             d_weighted = m_broad.sum(dim=-1, keepdim=True) + 1e-6
             target_rew = t_weighted / d_weighted
        elif target_rew.ndim == 2: 
             target_rew = target_rew.unsqueeze(-1)

        if mask.ndim == 3:
             team_mask = mask.any(dim=-1, keepdim=True).float()
        else:
             team_mask = mask.unsqueeze(-1)
             
        d_glob = team_mask.sum() + 1e-6
        
        r_loss = (F.mse_loss(pred_rew, target_rew.to(pred_rew.dtype), reduction='none') * team_mask).sum() / d_glob
        
        logs = {"reward_loss": r_loss.detach(), "loss_sub/reward_mse": r_loss.detach()}
        return {"loss": r_loss * self.weight, **logs}


class PairwiseRelationalLoss(LossModule):
    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        pred_pair = preds.get("pairwise")
        target_pair = targets.get("pairwise") # (B, T, N, N, D)
        
        if pred_pair is None or target_pair is None:
            return {"loss": torch.tensor(0.0, device=mask.device), "pairwise_loss": torch.tensor(0.0, device=mask.device)}
            
        # Pairwise Mask: (B, T, N, N)
        # Both i and j must be alive/valid.
        if mask.ndim == 3:
            # mask (B, T, N) -> (B, T, N, 1) & (B, T, 1, N)
            mask_i = mask.unsqueeze(3)
            mask_j = mask.unsqueeze(2)
            pair_mask = (mask_i & mask_j).float()
        else:
            pair_mask = mask # Assume correct shape? Unlikely.
            
        denom = pair_mask.sum() + 1e-6
        pair_mask = pair_mask.unsqueeze(-1) # For feature dim
        
        mse = F.mse_loss(pred_pair, target_pair, reduction='none')
        # Sum over B, T, N, N, D
        loss = (mse * pair_mask).sum() / denom
        
        logs = {"pairwise_loss": loss.detach(), "loss_sub/pairwise_mse": loss.detach()}
        return {"loss": loss * self.weight, **logs}
