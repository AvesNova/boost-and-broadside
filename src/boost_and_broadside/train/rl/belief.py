"""Physical belief state for the ships one observer cannot currently see.

The environment is authoritative and computes perception. This module owns the
policy-side memory layered on top of it: a fixed-shape, GPU-resident store of
**physical** ship state per observer, which the observation builder selects
against truth when it composes that observer's legal view.

Three operations, in the order one decision runs them:

``observe``  assimilates authoritative truth for every ship in sight, voids the
             belief of anything that just spawned, and hands the builder a
             :class:`ShipBeliefSource`.
``advance``  stores the next-state head's forecast: means move by the predicted
             physical deltas, uncertainty becomes the predicted uncertainty
             outright, and the enemy-action head's distribution is recorded for
             the next view to carry.
``reset``    forgets a completed episode.

Nothing here decodes, re-encodes or substitutes anything: belief and truth are
the same eleven physical quantities, so composition is a selection.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.constants import NUM_JOINT_ACTIONS
from boost_and_broadside.env.observation import (
    ObsKey,
    ShipBeliefSource,
    YemongObservation,
    observation_from_state,
    write_pending_action_view,
)
from boost_and_broadside.env.perception import TeamVisibility
from boost_and_broadside.env.state import TensorState
from boost_and_broadside.train.rl.physical_belief import (
    ALIVE_HEALTH_EPS,
    ANGULAR_VELOCITY,
    ATTITUDE,
    COOLDOWN,
    HEALTH,
    LOCAL_LOG_INDEX,
    NEXT_STATE_OUTPUT_DIM,
    PHYSICAL_MEAN_DIM,
    PHYSICAL_UNCERTAINTY_DIM,
    POSITION_X,
    POWER,
    SHIELD_DELAY,
    VELOCITY_X,
    PhysicalNextState,
    certain_uncertainty,
    physical_means_from_state,
    unknown_uncertainty,
)


class BeliefTracker:
    """Fixed-shape, GPU-resident physical belief for one policy perspective."""

    def __init__(
        self,
        num_envs: int,
        num_ships: int,
        decision_dt: float,
        ship_config: ShipConfig,
        device: str | torch.device,
        observer_team: int = 0,
    ) -> None:
        self.num_envs = num_envs
        self.num_ships = num_ships
        self.decision_dt = float(decision_dt)
        self.ship_config = ship_config
        self.spec = PhysicalNextState.from_ship_config(ship_config)
        self.device = torch.device(device)
        if observer_team not in (0, 1):
            raise ValueError(f"observer_team must be 0 or 1, got {observer_team}")
        self.observer_team = observer_team
        self.valid = torch.zeros((num_envs, num_ships), dtype=torch.bool, device=self.device)
        self.age_steps = torch.zeros((num_envs, num_ships), dtype=torch.int32, device=self.device)
        #: The eleven physical means, in ``PHYSICAL_MEAN_NAMES`` order. Zero
        #: wherever ``valid`` is false, which is what lets the observation
        #: builder select against it without a second mask.
        self.means = torch.zeros(
            (num_envs, num_ships, PHYSICAL_MEAN_DIM), dtype=torch.float32, device=self.device
        )
        #: Thirteen log/unconstrained uncertainty terms, stated by the head
        #: rather than accumulated here. Never-observed slots hold the ceiling.
        self.uncertainty = torch.zeros(
            (num_envs, num_ships, PHYSICAL_UNCERTAINTY_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        self.action_belief = torch.full(
            (num_envs, num_ships, NUM_JOINT_ACTIONS),
            1.0 / NUM_JOINT_ACTIONS,
            device=self.device,
        )
        # The two constant uncertainty vectors, materialized once. Built here
        # rather than per call: ``torch.tensor([...], device="cuda")`` is a
        # synchronizing host copy, and these are read every decision.
        self._certain = torch.tensor(
            certain_uncertainty(), dtype=torch.float32, device=self.device
        ).view(1, 1, PHYSICAL_UNCERTAINTY_DIM)
        self._unknown = torch.tensor(
            unknown_uncertainty(), dtype=torch.float32, device=self.device
        ).view(1, 1, PHYSICAL_UNCERTAINTY_DIM)
        self.uncertainty.copy_(self._unknown.expand_as(self.uncertainty))
        # Non-finite forecasts, accumulated on device and read once per update.
        # The recursion is bounded by construction -- every unbounded channel
        # either wraps or clamps -- so a nonzero count means the head emitted a
        # NaN or an infinity, which is a signal rather than something repaired.
        self.clamp_events = torch.zeros((), dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, env_mask: torch.Tensor | None = None) -> None:
        """Forget completed episodes without touching recurrent policy history."""

        if env_mask is None:
            self.valid.zero_()
            self.age_steps.zero_()
            self.means.zero_()
            self.uncertainty.copy_(self._unknown.expand_as(self.uncertainty))
            self.action_belief.fill_(1.0 / NUM_JOINT_ACTIONS)
            return
        mask = env_mask.bool()
        self.valid[mask] = False
        self.age_steps[mask] = 0
        self.means[mask] = 0.0
        self.uncertainty[mask] = self._unknown
        self.action_belief[mask] = 1.0 / NUM_JOINT_ACTIONS

    def slice_envs(self, idx: slice | torch.Tensor) -> BeliefTracker:
        """Copy a subset into an independent tracker (used by evaluation agents)."""

        selected = BeliefTracker(
            int(self.valid[idx].shape[0]),
            self.num_ships,
            self.decision_dt,
            self.ship_config,
            self.device,
            observer_team=self.observer_team,
        )
        selected.valid.copy_(self.valid[idx])
        selected.age_steps.copy_(self.age_steps[idx])
        selected.means.copy_(self.means[idx])
        selected.uncertainty.copy_(self.uncertainty[idx])
        selected.action_belief.copy_(self.action_belief[idx])
        return selected

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    @torch.no_grad()
    def observe(self, state: TensorState, visible: torch.Tensor) -> ShipBeliefSource:
        """Assimilate what is in sight and return this observer's legal source.

        A ship that spawned this decision has its belief voided: it teleported,
        so whatever was remembered about it describes somewhere it no longer is.
        Under ``spawn_reveal`` the same decision also makes it visible, so truth
        lands in the same call and the net effect is a clean re-acquisition; with
        the reveal disabled the slot correctly falls back to "never seen".
        """

        if visible.shape != (self.num_envs, self.num_ships):
            raise ValueError(
                f"visibility must have shape {(self.num_envs, self.num_ships)}, "
                f"got {tuple(visible.shape)}"
            )
        visible = visible.bool()
        spawned = state.ship_spawned[:, : self.num_ships]
        self.valid = (self.valid & ~spawned) | visible
        valid_vector = self.valid.unsqueeze(-1)

        truth = physical_means_from_state(state)[:, : self.num_ships]
        self.means = torch.where(
            valid_vector, torch.where(visible.unsqueeze(-1), truth, self.means), 0.0
        )
        # Seeing a ship settles it: the spread becomes the finite certainty
        # floor, discarding whatever the forecast claimed. A slot nothing has
        # ever observed carries the ceiling, not a zero -- a zero log sigma reads
        # as "one unit of doubt", which is a confident claim about nothing.
        self.uncertainty = torch.where(
            visible.unsqueeze(-1),
            self._certain,
            torch.where(valid_vector, self.uncertainty, self._unknown),
        )
        self.age_steps = torch.where(
            visible,
            torch.zeros_like(self.age_steps),
            torch.where(self.valid, self.age_steps + 1, torch.zeros_like(self.age_steps)),
        )
        # A slot with no history has no prediction to offer either, so its
        # pending-command belief is the uniform distribution rather than a
        # forecast about a ship this observer has never met.
        self.action_belief = torch.where(valid_vector, self.action_belief, 1.0 / NUM_JOINT_ACTIONS)
        return self.source(state)

    def source(self, state: TensorState) -> ShipBeliefSource:
        """This observer's remembered state, in the observation's own units."""

        means = self.means
        attitude = means[..., ATTITUDE : ATTITUDE + 1]
        # Instant respawn means a remembered ship is always alive; without it,
        # believed health is the only account of whether it still is.
        alive = self.valid if state.num_zones > 0 else means[..., HEALTH] > ALIVE_HEALTH_EPS
        return ShipBeliefSource(
            pos=means[..., POSITION_X : POSITION_X + 2],
            vel=means[..., VELOCITY_X : VELOCITY_X + 2],
            att=torch.cat([torch.cos(attitude), torch.sin(attitude)], dim=-1),
            ang_vel=means[..., ANGULAR_VELOCITY : ANGULAR_VELOCITY + 1],
            shield_delay=means[..., SHIELD_DELAY : SHIELD_DELAY + 1],
            health=means[..., HEALTH : HEALTH + 1],
            power=means[..., POWER : POWER + 1],
            cooldown=means[..., COOLDOWN : COOLDOWN + 1],
            local_log_index=means[..., LOCAL_LOG_INDEX : LOCAL_LOG_INDEX + 1],
            uncertainty=self.uncertainty,
            action=self.action_belief,
            time_since_observation=(self.age_steps.float() * self.decision_dt).unsqueeze(-1),
            valid=self.valid,
            alive=alive,
            certain=self._certain,
        )

    # ------------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------------

    @torch.no_grad()
    def advance(
        self,
        prediction: torch.Tensor,
        enemy_action_logits: torch.Tensor | None = None,
    ) -> None:
        """Store the head's one-decision forecast for the next ``observe``.

        The means move by the predicted physical deltas and the uncertainty
        becomes the predicted uncertainty outright. Nothing accumulates: the head
        saw the current spread as an input and stated the next one, so summing
        forecasts would double-count what it already accounts for.
        """

        expected = (self.num_envs, self.num_ships, NEXT_STATE_OUTPUT_DIM)
        if prediction.shape != expected:
            raise ValueError(
                f"prediction must have shape {expected}, got {tuple(prediction.shape)}"
            )
        action_expected = (self.num_envs, self.num_ships, NUM_JOINT_ACTIONS)
        if enemy_action_logits is not None and enemy_action_logits.shape != action_expected:
            raise ValueError(
                f"enemy_action_logits must have shape {action_expected}, "
                f"got {tuple(enemy_action_logits.shape)}"
            )
        if enemy_action_logits is None:
            self.action_belief.fill_(1.0 / NUM_JOINT_ACTIONS)
        else:
            self.action_belief = F.softmax(enemy_action_logits.float(), dim=-1)

        prediction = prediction.float()
        # Counted against the raw forecast, before the replacement: a count taken
        # after ``nan_to_num`` reads zero for the one case that matters most.
        self.clamp_events += (~torch.isfinite(prediction)).sum()
        prediction = torch.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
        valid_vector = self.valid.unsqueeze(-1)
        self.means = torch.where(
            valid_vector, self.spec.apply_means(self.means, prediction), self.means
        )
        self.uncertainty = torch.where(
            valid_vector, prediction[..., PHYSICAL_MEAN_DIM:], self.uncertainty
        )


class DualBeliefTracker:
    """Two independent team perspectives sharing no remembered enemy truth."""

    def __init__(
        self,
        num_envs: int,
        num_ships: int,
        decision_dt: float,
        ship_config: ShipConfig,
        device: str | torch.device,
    ) -> None:
        self.team0 = BeliefTracker(
            num_envs, num_ships, decision_dt, ship_config, device, observer_team=0
        )
        self.team1 = BeliefTracker(
            num_envs, num_ships, decision_dt, ship_config, device, observer_team=1
        )

    def reset(self, env_mask: torch.Tensor | None = None) -> None:
        self.team0.reset(env_mask)
        self.team1.reset(env_mask)

    @torch.no_grad()
    def observe(
        self, state: TensorState, ship_visibility: torch.Tensor
    ) -> tuple[ShipBeliefSource, ShipBeliefSource]:
        """Assimilate both perspectives and return their sources, team 0 first.

        ``ship_visibility`` is ``TeamVisibility.ship``, ``(B, 2, N)``.
        """

        return (
            self.team0.observe(state, ship_visibility[:, 0]),
            self.team1.observe(state, ship_visibility[:, 1]),
        )

    @torch.no_grad()
    def advance(
        self,
        prediction_t0: torch.Tensor,
        prediction_t1: torch.Tensor,
        enemy_action_logits_t0: torch.Tensor | None = None,
        enemy_action_logits_t1: torch.Tensor | None = None,
    ) -> None:
        self.team0.advance(prediction_t0, enemy_action_logits_t0)
        self.team1.advance(prediction_t1, enemy_action_logits_t1)


def team_view(values: torch.Tensor, observer_team: int | torch.Tensor) -> torch.Tensor:
    """Select one observer's slice of a ``(B, 2, ...)`` per-team tensor.

    ``observer_team`` may be a ``(B,)`` tensor, for a policy that plays team 0 in
    some environments and team 1 in others -- an ego-pass league opponent or a
    tournament seat. Gathering rather than indexing is what lets such a caller
    compose one view instead of one per seat.
    """

    if not isinstance(observer_team, torch.Tensor):
        return values[:, observer_team]
    index = observer_team.long().view(-1, 1, *([1] * (values.dim() - 2)))
    return values.gather(1, index.expand(-1, 1, *values.shape[2:])).squeeze(1)


def legal_policy_view(
    tracker: BeliefTracker | None,
    state: TensorState,
    ship_config: ShipConfig,
    visibility: TeamVisibility,
    observer_team: int | torch.Tensor,
    *,
    num_ships: int,
    include_bullets: bool = False,
    pending_action: torch.Tensor | None = None,
    builder=observation_from_state,
) -> YemongObservation:
    """Compose one observer's legal observation from truth and its own belief.

    The one entry point every policy-bearing caller outside the trainer's own
    rollout uses: evaluation agents, league opponents, interactive play. Each
    gets a view built from the authoritative state and *its own* memory, rather
    than a shared view patched afterwards -- two policies watching the same game
    have different memories of it, and attributing one's to the other makes its
    behaviour unreproducible from its own weights.

    ``pending_action`` is the command physics will consume next, when the caller
    queues decisions a step ahead. Without it the view carries the command
    already spent, which is what ``state.prev_action`` holds.
    """

    visible = team_view(visibility.ship, observer_team)
    source = None if tracker is None else tracker.observe(state, visible)
    view = builder(
        state,
        ship_config,
        None,
        include_bullets,
        visible,
        None if visibility.bullet is None else team_view(visibility.bullet, observer_team),
        observer_team,
        source,
    )
    if pending_action is not None:
        write_pending_action_view(
            view.data[ObsKey.PREVIOUS_ACTION][:, :num_ships],
            pending_action,
            state.ship_team_id[:, :num_ships],
            observer_team,
            state.ship_spawned[:, :num_ships],
            belief_action=None if source is None else source.action,
        )
    return view
