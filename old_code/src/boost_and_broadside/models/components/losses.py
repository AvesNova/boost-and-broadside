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

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
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

    def __init__(
        self,
        losses: List[LossModule],
        loss_type: str = "fixed",
        weights: Optional[Dict[str, float]] = None,
    ):
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
                if name in self.log_vars:
                    name = f"{name}_{i}"
                self.log_vars[name] = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        total_loss = torch.tensor(0.0, device=mask.device)
        metrics = {}

        for i, loss_mod in enumerate(self.losses):
            out = loss_mod(preds, targets, mask, **kwargs)

            l = out["loss"]

            if self.loss_type == "uncertainty" and self.log_vars is not None:
                name = getattr(loss_mod, "name", f"loss_{i}")
                if name not in self.log_vars:
                    name = f"{name}_{i}"  # fallback to match init

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
    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        pred_states = preds.get("states")
        target_states = targets.get("states")

        if pred_states is None or target_states is None:
            return {
                "loss": torch.tensor(0.0, device=mask.device),
                "state_loss": torch.tensor(0.0, device=mask.device),
            }

        mask_flat = mask.reshape(-1).float()
        denom = mask_flat.sum() + 1e-6
        mse = F.mse_loss(pred_states, target_states, reduction="none")

        # Sum over feature dim, then mask
        s_loss = mse.mean(dim=-1).reshape(-1).mul(mask_flat).sum() / denom

        # Logging breakdown
        logs = {"state_loss": s_loss.detach(), "loss_sub/state_mse": s_loss.detach()}

        # Detailed feature logging
        with torch.no_grad():
            m_expanded = (
                mask.unsqueeze(-1).float() if mask.ndim < pred_states.ndim else mask
            )
            masked_mse = (
                (mse * m_expanded).sum(dim=(0, 1, 2))
                if mse.ndim == 4
                else (mse * m_expanded).sum(dim=(0, 1))
            )
            feature_names = ["DX", "DY", "DVX", "DVY", "DHEALTH", "DPOWER", "DANG_VEL"]
            for i, name in enumerate(feature_names):
                if i < mse.shape[-1]:
                    logs[f"loss_sub/state_{name}"] = (masked_mse[i] / denom).item()

        return {"loss": s_loss * self.weight, **logs}


class ActionLoss(LossModule):
    name = "actions"

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        weights_power=None,
        weights_turn=None,
        weights_shoot=None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        pred_actions = preds.get("actions")
        target_actions = targets.get("actions")
        expert_action_probs = targets.get("expert_action_probs")

        if pred_actions is None or (
            target_actions is None and expert_action_probs is None
        ):
            return {
                "loss": torch.tensor(0.0, device=mask.device),
                "action_loss": torch.tensor(0.0, device=mask.device),
            }

        mask_flat = mask.reshape(-1).float()
        denom = mask_flat.sum() + 1e-6

        l_p, l_t, l_s = (
            pred_actions[..., 0:3],
            pred_actions[..., 3:10],
            pred_actions[..., 10:12],
        )

        if expert_action_probs is not None:
            # Soft targets using probabilities (3, 7, 2 = 12 dim)
            t_p_probs = expert_action_probs[..., 0:3]
            t_t_probs = expert_action_probs[..., 3:10]
            t_s_probs = expert_action_probs[..., 10:12]

            # cross_entropy supports soft labels (probabilities) directly in PyTorch when targets match pred shape
            a_loss_p = (
                (
                    F.cross_entropy(
                        l_p.reshape(-1, 3),
                        t_p_probs.reshape(-1, 3),
                        weight=weights_power,
                        reduction="none",
                    )
                    * mask_flat
                ).sum()
                / denom
                / math.log(3)
            )
            a_loss_t = (
                (
                    F.cross_entropy(
                        l_t.reshape(-1, 7),
                        t_t_probs.reshape(-1, 7),
                        weight=weights_turn,
                        reduction="none",
                    )
                    * mask_flat
                ).sum()
                / denom
                / math.log(7)
            )
            a_loss_s = (
                (
                    F.cross_entropy(
                        l_s.reshape(-1, 2),
                        t_s_probs.reshape(-1, 2),
                        weight=weights_shoot,
                        reduction="none",
                    )
                    * mask_flat
                ).sum()
                / denom
                / math.log(2)
            )
        else:
            # Hard targets using action indices
            t_p = target_actions[..., 0].long().clamp(0, 2)
            t_t = target_actions[..., 1].long().clamp(0, 6)
            t_s = target_actions[..., 2].long().clamp(0, 1)

            a_loss_p = (
                (
                    F.cross_entropy(
                        l_p.reshape(-1, 3),
                        t_p.reshape(-1),
                        weight=weights_power,
                        reduction="none",
                    )
                    * mask_flat
                ).sum()
                / denom
                / math.log(3)
            )
            a_loss_t = (
                (
                    F.cross_entropy(
                        l_t.reshape(-1, 7),
                        t_t.reshape(-1),
                        weight=weights_turn,
                        reduction="none",
                    )
                    * mask_flat
                ).sum()
                / denom
                / math.log(7)
            )
            a_loss_s = (
                (
                    F.cross_entropy(
                        l_s.reshape(-1, 2),
                        t_s.reshape(-1),
                        weight=weights_shoot,
                        reduction="none",
                    )
                    * mask_flat
                ).sum()
                / denom
                / math.log(2)
            )

        a_loss = (a_loss_p + a_loss_t + a_loss_s) / 3.0

        logs = {
            "action_loss": a_loss.detach(),
            "loss_sub/action_all": a_loss.detach(),
            "loss_sub/action_power": a_loss_p.detach(),
            "loss_sub/action_turn": a_loss_t.detach(),
            "loss_sub/action_shoot": a_loss_s.detach(),
        }

        return {"loss": a_loss * self.weight, **logs}


from boost_and_broadside.core.constants import (
    NUM_FLATTENED_ACTIONS,
    NUM_TURN_ACTIONS,
    NUM_SHOOT_ACTIONS,
)


class FlattenedActionLoss(LossModule):
    name = "actions"

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        weights_flat=None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        pred_actions = preds.get("actions")
        target_actions = targets.get("actions")
        expert_action_probs = targets.get("expert_action_probs")

        if pred_actions is None or (
            target_actions is None and expert_action_probs is None
        ):
            return {
                "loss": torch.tensor(0.0, device=mask.device),
                "action_loss": torch.tensor(0.0, device=mask.device),
            }

        mask_flat = mask.reshape(-1).float()
        denom = mask_flat.sum() + 1e-6

        # pred_actions should be (..., NUM_FLATTENED_ACTIONS)
        l_f = pred_actions

        if (
            expert_action_probs is not None
            and expert_action_probs.shape[-1] == NUM_FLATTENED_ACTIONS
        ):
            # Soft targets using 42-dim probabilities vector
            t_probs_flat = expert_action_probs.reshape(-1, NUM_FLATTENED_ACTIONS)
            a_loss = (
                (
                    F.cross_entropy(
                        l_f.reshape(-1, NUM_FLATTENED_ACTIONS),
                        t_probs_flat,
                        weight=weights_flat,
                        reduction="none",
                    )
                    * mask_flat
                ).sum()
                / denom
                / math.log(NUM_FLATTENED_ACTIONS)
            )
        else:
            # Ensure target is 1D flat index
            if target_actions.shape[-1] == 3:
                t_flat = (
                    target_actions[..., 0].long().clamp(0, 2)
                    * (NUM_TURN_ACTIONS * NUM_SHOOT_ACTIONS)
                    + target_actions[..., 1].long().clamp(0, 6) * NUM_SHOOT_ACTIONS
                    + target_actions[..., 2].long().clamp(0, 1)
                )
            else:
                t_flat = target_actions.long().squeeze(-1)

            t_flat = t_flat.clamp(0, NUM_FLATTENED_ACTIONS - 1)

            a_loss = (
                (
                    F.cross_entropy(
                        l_f.reshape(-1, NUM_FLATTENED_ACTIONS),
                        t_flat.reshape(-1),
                        weight=weights_flat,
                        reduction="none",
                    )
                    * mask_flat
                ).sum()
                / denom
                / math.log(NUM_FLATTENED_ACTIONS)
            )

        logs = {
            "action_loss": a_loss.detach(),
            "loss_sub/action_all": a_loss.detach(),
        }

        return {"loss": a_loss * self.weight, **logs}


class ValueLoss(LossModule):
    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        pred_val = preds.get("value")  # (B, T, 1)
        target_ret = targets.get("returns")  # (B, T, N) usually

        if pred_val is None or target_ret is None:
            return {
                "loss": torch.tensor(0.0, device=mask.device),
                "value_loss": torch.tensor(0.0, device=mask.device),
            }

        # Value is Team-Level. Target might be per-ship.
        # If target has extran dimension N, aggregate it.
        if target_ret.ndim == 3 and target_ret.shape[-1] > 1:
            # Weighted mean over N using mask
            # mask: (B, T, N)
            m_broad = mask.float()
            t_weighted = (target_ret * m_broad).sum(dim=-1, keepdim=True)
            d_weighted = m_broad.sum(dim=-1, keepdim=True) + 1e-6
            target_ret = t_weighted / d_weighted  # (B, T, 1)
        elif target_ret.ndim == 2:
            target_ret = target_ret.unsqueeze(-1)

        # Team Mask: Valid if ANY ship is valid
        if mask.ndim == 3:
            team_mask = mask.any(dim=-1, keepdim=True).float()  # (B, T, 1)
        else:
            team_mask = mask.unsqueeze(-1)

        d_glob = team_mask.sum() + 1e-6

        v_loss = (
            F.mse_loss(pred_val, target_ret.to(pred_val.dtype), reduction="none")
            * team_mask
        ).sum() / d_glob

        logs = {"value_loss": v_loss.detach(), "loss_sub/value_mse": v_loss.detach()}
        return {"loss": v_loss * self.weight, **logs}


class RewardLoss(LossModule):
    def __init__(
        self, weight: float = 1.0, component_names: Optional[List[str]] = None
    ):
        super().__init__(weight=weight)
        self.component_names = component_names

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Prioritize components if available for detailed logging
        pred_rew = preds.get("reward_components")
        if pred_rew is None:
            pred_rew = preds.get("reward")

        target_rew = targets.get("rewards")

        if pred_rew is None or target_rew is None:
            return {
                "loss": torch.tensor(0.0, device=mask.device),
                "reward_loss": torch.tensor(0.0, device=mask.device),
            }

        # Aggregation Logic
        # targets["rewards"] could be (B, T, N) or (B, T, N, K) or (B, T, K)
        # If N dimension exists, aggregate over it.
        # print(f"DEBUG: RewardLoss ENTRY - pred_rew: {pred_rew.shape}, target_rew (raw): {target_rew.shape}, mask: {mask.shape}")

        is_vector_reward = pred_rew.shape[-1] > 1
        num_components = pred_rew.shape[-1]

        # Helper to aggregate N dimension if present
        if target_rew.ndim == 4:  # (B, T, N, K)
            m_broad = mask.float().unsqueeze(-1)  # (B, T, N, 1)
            t_weighted = (target_rew * m_broad).sum(dim=2)  # Sum over N -> (B, T, K)
            d_weighted = m_broad.sum(dim=2) + 1e-6  # (B, T, 1)
            target_rew = t_weighted / d_weighted
        elif target_rew.ndim == 3 and target_rew.shape[-1] > 1 and not is_vector_reward:
            # (B, T, N) case for scalar prediction (legacy)
            m_broad = mask.float()
            t_weighted = (target_rew * m_broad).sum(dim=-1, keepdim=True)
            d_weighted = m_broad.sum(dim=-1, keepdim=True) + 1e-6
            target_rew = t_weighted / d_weighted

        # Ensure shapes align (B, T, K)
        if target_rew.ndim == 2:
            target_rew = target_rew.unsqueeze(-1)

        if pred_rew.shape[-1] > 1 and target_rew.shape[-1] == 1:
            # If predicting components but given a scalar target, supervise the sum of predictions
            pred_rew = pred_rew.sum(dim=-1, keepdim=True)
            is_vector_reward = False
            num_components = 1
        elif target_rew.shape[-1] > 1 and pred_rew.shape[-1] == 1:
            target_rew = target_rew.sum(dim=-1, keepdim=True)

        # Flatten for MSE
        # pred: (B, T, K) -> (B*T, K)
        # mask: (B, T) or (B, T, N) -> need team/global mask (B, T, 1)
        if mask.ndim == 3:
            team_mask = mask.any(dim=-1).float().unsqueeze(-1)  # (B, T, 1)
        else:
            team_mask = mask.unsqueeze(-1)

        d_glob = team_mask.sum() + 1e-6

        # Calculate MSE
        # (B, T, K)
        mse = F.mse_loss(pred_rew, target_rew.to(pred_rew.dtype), reduction="none")

        # Weighted sum over B, T
        weighted_mse = (mse * team_mask).sum(dim=(0, 1)) / d_glob  # (K,)

        r_loss_total = weighted_mse.sum()  # Sum over components for scalar loss

        logs = {
            "reward_loss": r_loss_total.detach(),
            "loss_sub/reward_mse": r_loss_total.detach(),
        }

        # Log individual components
        if is_vector_reward:
            for i in range(num_components):
                if i < weighted_mse.numel():
                    # Use descriptive name if available
                    name = (
                        self.component_names[i]
                        if self.component_names and i < len(self.component_names)
                        else i
                    )
                    logs[f"loss_sub/reward_{name}"] = weighted_mse[i].detach()

        return {"loss": r_loss_total * self.weight, **logs}


class PairwiseRelationalLoss(LossModule):
    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        pred_pair = preds.get("pairwise")
        target_pair = targets.get("pairwise")  # (B, T, N, N, D)

        if pred_pair is None or target_pair is None:
            return {
                "loss": torch.tensor(0.0, device=mask.device),
                "pairwise_loss": torch.tensor(0.0, device=mask.device),
            }

        # Pairwise Mask: (B, T, N, N)
        # Both i and j must be alive/valid.
        if mask.ndim == 3:
            # mask (B, T, N) -> (B, T, N, 1) & (B, T, 1, N)
            mask_i = mask.unsqueeze(3)
            mask_j = mask.unsqueeze(2)
            pair_mask = (mask_i & mask_j).float()
        else:
            pair_mask = mask  # Assume correct shape? Unlikely.

        denom = pair_mask.sum() + 1e-6
        pair_mask = pair_mask.unsqueeze(-1)  # For feature dim

        mse = F.mse_loss(pred_pair, target_pair, reduction="none")
        # Sum over B, T, N, N, D
        loss = (mse * pair_mask).sum() / denom

        logs = {"pairwise_loss": loss.detach(), "loss_sub/pairwise_mse": loss.detach()}
        return {"loss": loss * self.weight, **logs}


class SoftBinnedStateLoss(LossModule):
    """
    Soft-binned categorical prediction loss with proper per-bin-count normalisation.

    Each field's CE is divided by log(n_bins) so that a maximally-uncertain
    (uniform) prediction yields loss ≈ 1.0 regardless of bin count.

    Fields are grouped into three types:
      - state  : per-ship specs that are not team-level (health, power, pos/vel
                  angle and magnitude).  Averaged across the group.
      - value  : the team-level "value" spec.
      - reward : the team-level "reward" spec.

    Final loss = lambda_state * mean(state fields)
               + lambda_value * value_field
               + lambda_reward * reward_field

    Consumes:
        preds["soft_bin_logits"]    — List[Tensor] (B,T,N_,n_bins_i) per spec
        targets["soft_bin_targets"] — List[Tensor] same shapes, soft probabilities

    Returns zero gracefully when either key is absent.
    """

    name = "soft_bins"

    def __init__(
        self,
        weight: float = 1.0,  # kept for API compatibility; not used
        lambda_state: float = 1.0,
        lambda_value: float = 1.0,
        lambda_reward: float = 1.0,
    ):
        super().__init__(weight=weight)
        self.lambda_state = lambda_state
        self.lambda_value = lambda_value
        self.lambda_reward = lambda_reward

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        logits_list = preds.get("soft_bin_logits")
        soft_targets_list = targets.get("soft_bin_targets")

        zero = torch.tensor(0.0, device=mask.device)
        if logits_list is None or soft_targets_list is None:
            return {"loss": zero, "soft_bin_loss": zero}

        ship_mask = mask.float()  # (B,T,N)
        team_mask = mask.any(dim=-1, keepdim=True).float()  # (B,T,1)

        specs = getattr(self, "_specs", None)

        state_losses: list[torch.Tensor] = []
        value_losses: list[torch.Tensor] = []
        reward_losses: list[torch.Tensor] = []
        logs: Dict[str, torch.Tensor] = {}

        for i, (logits, soft_t) in enumerate(zip(logits_list, soft_targets_list)):
            n_bins = logits.shape[-1]
            log_bins = math.log(n_bins)  # normalisation factor

            spec = specs[i] if specs is not None and i < len(specs) else None
            is_team = (
                spec.is_team_level if spec is not None else (logits.shape[-2] == 1)
            )
            m = team_mask if is_team else ship_mask
            denom = m.sum() + 1e-6

            log_probs = F.log_softmax(logits.float(), dim=-1)
            ce = -(soft_t.float() * log_probs).sum(dim=-1)  # (B,T,N_)
            field_loss = (ce * m).sum() / denom / log_bins  # normalised to ~[0,1]

            spec_name = f"spec{i}"
            if specs is not None and i < len(specs):
                spec_name = specs[i].name

            logs[f"loss_sub/softbin_{spec_name}"] = field_loss.detach()

            # Route to appropriate group
            if specs is not None and i < len(specs):
                if specs[i].name == "value":
                    value_losses.append(field_loss)
                elif specs[i].name == "reward":
                    reward_losses.append(field_loss)
                else:
                    state_losses.append(field_loss)
            else:
                # No spec metadata: treat team-level as value/reward by position
                if is_team:
                    if len(value_losses) == 0:
                        value_losses.append(field_loss)
                    else:
                        reward_losses.append(field_loss)
                else:
                    state_losses.append(field_loss)

        def _mean(lst):
            return sum(lst) / len(lst) if lst else zero

        state_loss = _mean(state_losses)
        value_loss = _mean(value_losses)
        reward_loss = _mean(reward_losses)

        total_loss = (
            self.lambda_state * state_loss
            + self.lambda_value * value_loss
            + self.lambda_reward * reward_loss
        )

        logs["state_loss"] = state_loss.detach()
        logs["value_loss"] = value_loss.detach()
        logs["reward_loss"] = reward_loss.detach()

        logs["loss_sub/softbin_state"] = state_loss.detach()
        logs["loss_sub/softbin_value"] = value_loss.detach()
        logs["loss_sub/softbin_reward"] = reward_loss.detach()
        logs["soft_bin_loss"] = total_loss.detach()
        return {"loss": total_loss, **logs}

    def set_specs(self, specs) -> "SoftBinnedStateLoss":
        """Optional: attach spec list for per-field log names and routing."""
        self._specs = specs
        return self
