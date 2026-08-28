"""Iterated predictive latent (belief-state) model.

The trunk's final per-ship latent already contains the current physical state and
the action pending on the next transition, so predicting one step ahead is close
to a deterministic read-out and says little about the representation. What is
genuinely uncertain is what happens *several* decisions from now: where the ships
will be, and what every one of them will choose to do.

This module supplies the machinery for that objective. A single projection maps
the post-Yemong latent into a narrower predictive latent; one shared residual
transition advances that latent by one decision; and two shared heads decode a
local state transition and a factorized action distribution from it at every
horizon:

    post-Yemong latent
        -> PredictiveProjection      -> predictive latent, horizon 0
        -> heads                     -> transition t -> t+1, action at t
        -> PredictiveTransition      -> predictive latent, horizon 1
        -> heads                     -> transition t+1 -> t+2, action at t+1
        -> ...

The rollout is open-loop by construction: nothing after the projection reads a
future observation, latent, or action. Future rollout data is the *target*, never
an input, which is what makes the later horizons a belief about an uncertain
future rather than an action-conditioned simulation of a known one.

The transition and both heads are single modules reused at every horizon, so
depth costs no parameters and the dynamics a horizon-1 belief obeys are the
dynamics a horizon-11 belief obeys.
"""

import math

import torch
import torch.nn as nn

from boost_and_broadside.constants import TOTAL_ACTION_LOGITS


def initialize_head_orthogonal(head: nn.Sequential, final_gain: float = 0.01) -> None:
    """Orthogonal-init a Linear+Norm+Act+Linear head's first and last Linear layers.

    Locates layers by type instead of a fixed index, so the head's Sequential can
    grow or reorder non-Linear layers without corrupting or missing this init.

    Args:
        head: The head to initialize; must contain at least one ``nn.Linear``.
        final_gain: Gain for the output layer. The default keeps a freshly built
            head near zero, which is standard PPO practice for the policy heads
            and is what starts the predictive transition near identity.
    """
    linears = [module for module in head if isinstance(module, nn.Linear)]
    nn.init.orthogonal_(linears[0].weight, gain=math.sqrt(2))
    nn.init.zeros_(linears[0].bias)
    nn.init.orthogonal_(linears[-1].weight, gain=final_gain)
    nn.init.zeros_(linears[-1].bias)


def _prediction_head(predictive_latent_dim: int, out_dim: int) -> nn.Sequential:
    """The shared decoder shape: widen, normalize, activate, project out."""
    hidden = predictive_latent_dim * 2
    return nn.Sequential(
        nn.Linear(predictive_latent_dim, hidden),
        nn.RMSNorm(hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


class PredictiveProjection(nn.Module):
    """Maps the post-Yemong ship latent into the predictive latent space.

    Deliberately one linear layer and a norm — no hidden expansion, no
    activation. It exists to give the belief state its own space and a modest
    bottleneck, and to spare the policy latent itself from having to obey stable
    iterative dynamics. Anything deeper would let the auxiliary objective be
    solved *after* the trunk rather than pressuring the trunk to carry the
    information.

    Args:
        d_model: Width of the post-Yemong ship latent.
        predictive_latent_dim: Width of the predictive latent.
    """

    def __init__(self, d_model: int, predictive_latent_dim: int) -> None:
        super().__init__()
        self.project = nn.Linear(d_model, predictive_latent_dim)
        self.norm = nn.RMSNorm(predictive_latent_dim)
        nn.init.orthogonal_(self.project.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.project.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Args: latent (..., d_model). Returns: (..., predictive_latent_dim)."""
        return self.norm(self.project(latent))


class PredictiveTransition(nn.Module):
    """One decision step of the belief-state rollout: a residual MLP plus a norm.

    ``output = RMSNorm(latent + mlp(latent))``. The residual path is what makes
    repeated application safe — the same weights run at every horizon, so a
    non-residual block would have to be a contraction to stay bounded and an
    expansion to stay informative. The output layer is initialized small, so the
    rollout begins as (very nearly) the identity and learns to depart from it.

    Args:
        predictive_latent_dim: Width of the predictive latent.
    """

    def __init__(self, predictive_latent_dim: int) -> None:
        super().__init__()
        hidden = predictive_latent_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(predictive_latent_dim, hidden),
            nn.RMSNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, predictive_latent_dim),
        )
        self.norm = nn.RMSNorm(predictive_latent_dim)
        initialize_head_orthogonal(self.mlp)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Args: latent (..., predictive_latent_dim). Returns: the same shape."""
        return self.norm(latent + self.mlp(latent))


class PredictiveStateHead(nn.Module):
    """Decodes a *local* one-step state transition from a predictive latent.

    The prediction is always the transition out of the step the latent
    represents — horizon 0 predicts ``t -> t+1``, horizon 1 predicts
    ``t+1 -> t+2`` — never a cumulative displacement from the observed step. That
    keeps the easy, well-grounded immediate physics as an anchor while making
    every later horizon forecast a transition it cannot see the inputs of.

    Args:
        predictive_latent_dim: Width of the predictive latent.
        state_prediction_dim: Width of the FeatureCoordinator's prediction vector.
    """

    def __init__(self, predictive_latent_dim: int, state_prediction_dim: int) -> None:
        super().__init__()
        self.state_prediction_dim = state_prediction_dim
        self.net = _prediction_head(predictive_latent_dim, state_prediction_dim)
        initialize_head_orthogonal(self.net)

    def forward(self, predictive_latent: torch.Tensor) -> torch.Tensor:
        """Args: (..., predictive_latent_dim). Returns: (..., state_prediction_dim)."""
        return self.net(predictive_latent)


class PredictiveActionHead(nn.Module):
    """Decodes the factorized action distribution taken at the represented step.

    One shared head over the same [power | turn | shoot] logit layout the actor
    uses, trained by cross-entropy against the action the rollout actually took.
    It shares no weights with the actor: the actor's job is to *choose* an action
    for one ship, this head's job is to *anticipate* the choices of ships whose
    decisions the latent never made.

    Args:
        predictive_latent_dim: Width of the predictive latent.
    """

    def __init__(self, predictive_latent_dim: int) -> None:
        super().__init__()
        self.net = _prediction_head(predictive_latent_dim, TOTAL_ACTION_LOGITS)
        initialize_head_orthogonal(self.net)

    def forward(self, predictive_latent: torch.Tensor) -> torch.Tensor:
        """Args: (..., predictive_latent_dim). Returns: (..., TOTAL_ACTION_LOGITS)."""
        return self.net(predictive_latent)


class PredictiveModel(nn.Module):
    """The projection, the shared transition, and the two shared decoding heads.

    Held as one submodule so the whole auxiliary apparatus is a single thing to
    checkpoint, to exclude from the shared trunk, and to switch off.

    ``next_step_only`` strips this back to the one-step head the model carried
    before the belief state existed: no projection, no transition, no action
    head, and a state head reading the post-Yemong latent directly. It is the
    control arm — the same architecture and the same objective as run 719 —
    kept so the belief state can be measured against what it replaced rather
    than only against its own ablations. The state head is built at ``d_model``
    there, which reproduces that head exactly.

    Args:
        d_model: Width of the post-Yemong ship latent.
        predictive_latent_dim: Width of the predictive latent. Unused when
            ``next_step_only``, since nothing projects into that space.
        state_prediction_dim: Width of the FeatureCoordinator's prediction vector.
        next_step_only: Build the one-step control arm instead of the rollout.
    """

    def __init__(
        self,
        d_model: int,
        predictive_latent_dim: int,
        state_prediction_dim: int,
        next_step_only: bool = False,
    ) -> None:
        super().__init__()
        self.next_step_only = next_step_only
        # The width the heads actually read: the trunk latent itself when there
        # is no projection to narrow it.
        self.predictive_latent_dim = d_model if next_step_only else predictive_latent_dim
        if next_step_only:
            self.projection = None
            self.transition = None
            self.action_prediction_head = None
        else:
            self.projection = PredictiveProjection(d_model, predictive_latent_dim)
            self.transition = PredictiveTransition(predictive_latent_dim)
            self.action_prediction_head = PredictiveActionHead(predictive_latent_dim)
        self.state_prediction_head = PredictiveStateHead(
            self.predictive_latent_dim, state_prediction_dim
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Project a post-Yemong latent to the horizon-0 predictive latent.

        The identity when there is no projection, so callers that pair this with
        ``predict_state`` need no branch of their own.
        """
        return latent if self.next_step_only else self.projection(latent)

    def predict_state(self, predictive_latent: torch.Tensor) -> torch.Tensor:
        """The local one-step state transition believed at this horizon."""
        return self.state_prediction_head(predictive_latent)

    def predict_action_logits(self, predictive_latent: torch.Tensor) -> torch.Tensor:
        """The factorized action logits believed at this horizon."""
        if self.next_step_only:
            raise RuntimeError("the one-step control arm has no action head")
        return self.action_prediction_head(predictive_latent)

    def advance(self, predictive_latent: torch.Tensor) -> torch.Tensor:
        """Advance the belief one decision, open-loop."""
        if self.next_step_only:
            raise RuntimeError("the one-step control arm has no transition to advance")
        return self.transition(predictive_latent)


__all__ = [
    "PredictiveActionHead",
    "PredictiveModel",
    "PredictiveProjection",
    "PredictiveStateHead",
    "PredictiveTransition",
    "initialize_head_orthogonal",
]
