"""Plain PPO (skeleton stub): GAE, clipped policy loss, value MSE, entropy bonus.

The learner owns the optimizer and updates the model's parameters in place, which is also
what keeps the rollout's captured CUDA graph pointing at the current weights.
"""

import dataclasses

import torch
from torch.utils import _pytree

from bnb.contracts import Trajectory
from bnb.models.heads import evaluate


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    lr: float = 3e-4
    epochs: int = 4
    minibatches: int = 4
    clip: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0


def gae(reward, value, done, last_value, gamma: float, lam: float):
    """Advantages and returns (L, R, S); `done` (L, R) cuts the bootstrap after a reset."""
    adv = torch.zeros_like(reward)
    running = torch.zeros_like(last_value)
    next_value = last_value
    for t in reversed(range(reward.shape[0])):
        keep = (~done[t]).float()[..., None]
        delta = reward[t] + gamma * next_value * keep - value[t]
        running = delta + gamma * lam * keep * running
        adv[t] = running
        next_value = value[t]
    return adv, adv + value


class PPO:
    def __init__(self, model: torch.nn.Module, cfg: PPOConfig, compile: bool = True) -> None:
        self.model, self.cfg = model, cfg
        fused = next(model.parameters()).is_cuda
        self.opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5, fused=fused)
        self._loss = torch.compile(self._loss_fn, fullgraph=True, dynamic=False) if compile \
            else self._loss_fn

    def _loss_fn(self, obs, action, logp_old, adv, ret):
        logits, value = self.model(obs)
        logp, entropy = evaluate(logits, action)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ratio = torch.exp(logp - logp_old)
        clipped = torch.clamp(ratio, 1.0 - self.cfg.clip, 1.0 + self.cfg.clip)
        policy = -torch.minimum(ratio * adv, clipped * adv).mean()
        value_loss = 0.5 * (value - ret).square().mean()
        loss = policy + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy.mean()
        kl = (logp_old - logp).mean()
        stats = torch.stack([policy, value_loss, entropy.mean(), kl,
                             ((ratio - 1.0).abs() > self.cfg.clip).float().mean()])
        return loss, stats

    def update(self, traj: Trajectory, last_value: torch.Tensor) -> dict[str, float]:
        cfg = self.cfg
        with torch.no_grad():
            # The skeleton's critic is one value: the reward levels are summed (section 6
            # gives each level its own head, discount and lambda).
            adv, ret = gae(traj.reward.sum(-1), traj.value, traj.done, last_value, cfg.gamma,
                           cfg.lam)
        L, R = traj.reward.shape[:2]
        if (L * R) % cfg.minibatches:
            raise ValueError(f"{L} x {R} views do not split into {cfg.minibatches} minibatches")
        size = L * R // cfg.minibatches
        flat = _pytree.tree_map(lambda x: x.flatten(0, 1),
                                (traj.obs, traj.action, traj.logp, adv, ret))
        total = torch.zeros(5, device=adv.device)
        for _ in range(cfg.epochs):
            perm = torch.randperm(L * R, device=adv.device)
            for i in range(cfg.minibatches):
                idx = perm[i * size:(i + 1) * size]
                batch = _pytree.tree_map(lambda x: x[idx], flat)
                loss, stats = self._loss(*batch)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.opt.step()
                total += stats.detach()
        values = (total / (cfg.epochs * cfg.minibatches)).tolist()
        return dict(zip(("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"),
                        values))
