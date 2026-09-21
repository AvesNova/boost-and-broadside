"""Recursive point-estimate beliefs for previously observed enemy ships.

The environment remains authoritative and produces independently masked team
views.  This module owns the policy-side memory layered on top of one such view:
visible ships refresh from truth, never-seen hidden ships stay absent, and a
previously seen hidden ship is advanced only by the policy's next-state head.
"""

import torch

from boost_and_broadside.env.observation import ObjectType, ObsKey, YemongObservation

ALIVE_HEALTH_EPS = 1.0

# Numerical ceiling on a stored belief target. Not a physical bound: no thrust,
# drag, speed or index constant appears in it, so nothing here rots when the
# simulation changes.
#
# Targets live in symlog space and ``Symlog.invert`` is ``sign(x)*expm1(|x|)``,
# so the cap has to be read through an exponential. 30 decodes to about 1.1e13
# and squares to 1.1e26, which leaves twelve orders of headroom under float32's
# 3.4e38 for the squarings downstream (``mass = n**2``, the thrust impulse's
# energy term). Physical values occupy |target| <= ~7 -- symlog of the fastest
# speed the thrust/drag equilibrium admits, about 632 px/s, is 6.45 -- so this
# sits ten orders above anything legitimate and cannot bind on a working model.
#
# It replaces ``nan_to_num``'s defaults, which were the specific reason the
# previous guard did not hold: the default ``posinf`` is float32's maximum,
# 3.4e38, and in *symlog* space that decodes to expm1(3.4e38) = inf on the very
# next compose. The old guard swapped an infinity for a value that became one
# again immediately.
BELIEF_TARGET_LIMIT = 30.0


class BeliefTracker:
    """Fixed-shape, GPU-resident belief cache for one policy perspective."""

    def __init__(
        self,
        num_envs: int,
        num_ships: int,
        decision_dt: float,
        coordinator,
        device: str | torch.device,
    ) -> None:
        self.num_envs = num_envs
        self.num_ships = num_ships
        self.decision_dt = float(decision_dt)
        self.coordinator = coordinator
        self.device = torch.device(device)
        target_dim = coordinator.total_target_dimension
        self.valid = torch.zeros((num_envs, num_ships), dtype=torch.bool, device=self.device)
        self.age_steps = torch.zeros((num_envs, num_ships), dtype=torch.int32, device=self.device)
        self.predicted_targets = torch.zeros(
            (num_envs, num_ships, target_dim), dtype=torch.float32, device=self.device
        )
        # Accumulated variance of the belief, one channel per auxiliary
        # prediction dimension. Zero while a ship is in sight and summed over
        # every forecast since it went out of it, so it grows with the hidden
        # duration rather than reporting a single step's spread.
        self.uncertainty = torch.zeros(
            (num_envs, num_ships, coordinator.total_prediction_dimension),
            dtype=torch.float32,
            device=self.device,
        )
        self.team_id = torch.zeros((num_envs, num_ships), dtype=torch.int32, device=self.device)
        self.radius = torch.zeros((num_envs, num_ships, 1), dtype=torch.float32, device=self.device)
        self.clamp_events = torch.zeros((), dtype=torch.long, device=self.device)

    def reset(self, env_mask: torch.Tensor | None = None) -> None:
        """Forget completed episodes without touching recurrent policy history elsewhere."""

        if env_mask is None:
            self.valid.zero_()
            self.age_steps.zero_()
            self.predicted_targets.zero_()
            self.uncertainty.zero_()
            self.team_id.zero_()
            self.radius.zero_()
            return
        mask = env_mask.bool()
        self.valid[mask] = False
        self.age_steps[mask] = 0
        self.predicted_targets[mask] = 0.0
        self.uncertainty[mask] = 0.0
        self.team_id[mask] = 0
        self.radius[mask] = 0.0

    def slice_envs(self, idx: slice | torch.Tensor) -> "BeliefTracker":
        """Copy a subset into an independent tracker (used by evaluation agents)."""

        selected = BeliefTracker(
            int(self.valid[idx].shape[0]),
            self.num_ships,
            self.decision_dt,
            self.coordinator,
            self.device,
        )
        selected.valid.copy_(self.valid[idx])
        selected.age_steps.copy_(self.age_steps[idx])
        selected.predicted_targets.copy_(self.predicted_targets[idx])
        selected.uncertainty.copy_(self.uncertainty[idx])
        selected.team_id.copy_(self.team_id[idx])
        selected.radius.copy_(self.radius[idx])
        return selected

    @torch.no_grad()
    def compose(self, perceived: YemongObservation) -> YemongObservation:
        """Overlay recursive predictions on hidden, previously-seen ship slots.

        The input is a single team view.  No authoritative state is accepted by
        this method, making it difficult to accidentally leak truth into a hidden
        token.  Non-predicted hidden channels, including enemy action and local
        field gradient, remain at their masked zero values.
        """

        if perceived.pos.shape[0] != self.num_envs:
            raise ValueError(
                f"belief tracker has {self.num_envs} envs, observation has {perceived.pos.shape[0]}"
            )
        n = self.num_ships
        visible = perceived[ObsKey.VISIBLE][:, :n].bool()
        hidden_belief = self.valid & ~visible

        # Cache only facts that were actually perceived. Identity and radius are
        # not predicted, but become legitimate memory after first observation.
        self.team_id = torch.where(visible, perceived[ObsKey.TEAM_ID][:, :n], self.team_id)
        self.radius = torch.where(
            visible.unsqueeze(-1), perceived[ObsKey.RADIUS][:, :n], self.radius
        )
        self.age_steps = torch.where(
            visible,
            torch.zeros_like(self.age_steps),
            torch.where(hidden_belief, self.age_steps + 1, torch.zeros_like(self.age_steps)),
        )
        self.valid |= visible
        # Seeing a ship settles it: the estimate is the observation, so whatever
        # the forecast had accumulated is discarded rather than decayed.
        self.uncertainty = torch.where(
            visible.unsqueeze(-1), torch.zeros_like(self.uncertainty), self.uncertainty
        )

        data = {key: value.clone() for key, value in perceived.items()}
        # Run the fixed-shape decode even when this batch currently has no
        # hidden beliefs. Avoiding a tensor-dependent Python branch keeps this
        # path free of GPU synchronization and friendly to torch.compile.
        raw = self.coordinator.decode_targets(self.predicted_targets)
        decoded = {
            ObsKey.POS: torch.cat([raw["position_x"], raw["position_y"]], dim=-1),
            ObsKey.VEL: raw["velocity"],
            ObsKey.ATT: raw["attitude"],
            ObsKey.ANG_VEL: raw["angular_velocity"],
            ObsKey.HEALTH: raw["health"],
            ObsKey.SHIELD_DELAY: raw["shield_delay"].clamp_min(0),
            ObsKey.POWER: raw["power"],
            ObsKey.COOLDOWN: raw["cooldown"],
            ObsKey.LOCAL_LOG_INDEX: raw["local_log_index"],
        }
        for key, belief_value in decoded.items():
            if key not in data:
                data[key] = perceived[key].clone()
            belief_value = torch.nan_to_num(belief_value)
            mask = hidden_belief.unsqueeze(-1)
            data[key][:, :n] = torch.where(mask, belief_value, data[key][:, :n])

        predicted_alive = torch.where(
            perceived[ObsKey.GAME_MODE][:, -1, 0:1] > 0,
            self.valid,
            decoded[ObsKey.HEALTH].squeeze(-1) > ALIVE_HEALTH_EPS,
        )
        data[ObsKey.ALIVE][:, :n] = torch.where(
            hidden_belief, predicted_alive, data[ObsKey.ALIVE][:, :n]
        )
        data[ObsKey.TEAM_ID][:, :n] = torch.where(
            hidden_belief, self.team_id, data[ObsKey.TEAM_ID][:, :n]
        )
        data[ObsKey.RADIUS][:, :n] = torch.where(
            hidden_belief.unsqueeze(-1), self.radius, data[ObsKey.RADIUS][:, :n]
        )
        for key in (ObsKey.PREVIOUS_ACTION, ObsKey.LOCAL_INDEX_GRADIENT):
            data[key][:, :n] = torch.where(
                hidden_belief.unsqueeze(-1),
                torch.zeros_like(data[key][:, :n]),
                data[key][:, :n],
            )
        data[ObsKey.OBJECT_TYPE][:, :n] = torch.where(
            hidden_belief,
            torch.full_like(data[ObsKey.OBJECT_TYPE][:, :n], int(ObjectType.SHIP)),
            data[ObsKey.OBJECT_TYPE][:, :n],
        )
        # Ship tokens have no zone role. Five is the explicit NONE value;
        # leaving the masked zero would falsely describe a Team-0 spawn.
        data[ObsKey.ZONE_ROLE][:, :n] = torch.where(
            hidden_belief,
            torch.full_like(data[ObsKey.ZONE_ROLE][:, :n], 5),
            data[ObsKey.ZONE_ROLE][:, :n],
        )

        data[ObsKey.BELIEF_VALID][:, :n] = self.valid
        # Map objects are static and carry no belief, so their uncertainty stays
        # zero; only the ship slots are written.
        belief_uncertainty = torch.zeros(
            (*data[ObsKey.BELIEF_VALID].shape, self.uncertainty.shape[-1]),
            dtype=torch.float32,
            device=self.uncertainty.device,
        )
        belief_uncertainty[:, :n] = self.uncertainty
        data[ObsKey.BELIEF_UNCERTAINTY] = belief_uncertainty
        data[ObsKey.TIME_SINCE_OBSERVATION][:, :n] = (
            self.age_steps.float() * self.decision_dt
        ).unsqueeze(-1)
        return YemongObservation(data=data, bullets=perceived.bullets)

    @torch.no_grad()
    def advance(
        self,
        current: YemongObservation,
        scaled_prediction: torch.Tensor,
    ) -> None:
        """Store the model's one-step forecast for the next call to ``compose``."""

        curr_targets = self.coordinator.get_target_vector(current)[:, : self.num_ships]
        forecast = self.coordinator.apply_scaled_predictions(curr_targets, scaled_prediction)
        # A hidden ship's belief is an autoregressive rollout of the next-state
        # head with nothing else bounding it, so a small bias compounds for as
        # long as the ship stays unseen. Run 734 died that way: velocity error in
        # the 30s+ hidden bucket went 99 -> 1178 px/s over ten updates and then
        # overflowed, and the non-finite logits asserted inside multinomial.
        # Variance accumulates while a ship stays unseen: one forecast's spread
        # added per step, cleared by the next sighting in ``compose``. The head
        # reports a per-step spread, so the belief's own uncertainty is the sum
        # of them and not the latest one.
        self.uncertainty = self.uncertainty + self.coordinator.prediction_variance(
            scaled_prediction
        )

        raw = forecast.float()
        # Counted against the *raw* forecast, before the replacement: nan_to_num
        # maps an infinity onto the limit exactly, so a count taken afterwards
        # reads zero for the one case that matters most. Accumulated on device
        # and read once per update -- the guard must never bind on a working
        # model, so a nonzero count is a signal rather than a repair, and
        # counting it must not cost a host sync on the hot path.
        self.clamp_events += ((~torch.isfinite(raw)) | (raw.abs() > BELIEF_TARGET_LIMIT)).sum()
        forecast = torch.nan_to_num(
            raw,
            nan=0.0,
            posinf=BELIEF_TARGET_LIMIT,
            neginf=-BELIEF_TARGET_LIMIT,
        )
        self.predicted_targets.copy_(forecast.clamp(-BELIEF_TARGET_LIMIT, BELIEF_TARGET_LIMIT))


class DualBeliefTracker:
    """Two independent team perspectives sharing no remembered enemy truth."""

    def __init__(
        self,
        num_envs: int,
        num_ships: int,
        decision_dt: float,
        coordinator,
        device: str | torch.device,
    ) -> None:
        self.team0 = BeliefTracker(num_envs, num_ships, decision_dt, coordinator, device)
        self.team1 = BeliefTracker(num_envs, num_ships, decision_dt, coordinator, device)

    def reset(self, env_mask: torch.Tensor | None = None) -> None:
        self.team0.reset(env_mask)
        self.team1.reset(env_mask)

    @torch.no_grad()
    def compose(self, perceived: YemongObservation) -> YemongObservation:
        if perceived.team1_data is None:
            raise ValueError("dual belief tracking requires independent team observations")
        team0 = self.team0.compose(perceived.for_team(0))
        team1 = self.team1.compose(perceived.for_team(1))
        return YemongObservation(
            data=team0.data,
            bullets=team0.bullets,
            team1_data=team1.data,
            team1_bullets=team1.bullets,
        )

    @torch.no_grad()
    def advance(
        self,
        current: YemongObservation,
        scaled_prediction_t0: torch.Tensor,
        scaled_prediction_t1: torch.Tensor,
    ) -> None:
        self.team0.advance(current.for_team(0), scaled_prediction_t0)
        self.team1.advance(current.for_team(1), scaled_prediction_t1)
