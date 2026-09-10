"""Recursive point-estimate beliefs for previously observed enemy ships.

The environment remains authoritative and produces independently masked team
views.  This module owns the policy-side memory layered on top of one such view:
visible ships refresh from truth, never-seen hidden ships stay absent, and a
previously seen hidden ship is advanced only by the policy's next-state head.
"""

import torch

from boost_and_broadside.env.observation import ObjectType, ObsKey, YemongObservation

ALIVE_HEALTH_EPS = 1.0


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
        self.team_id = torch.zeros((num_envs, num_ships), dtype=torch.int32, device=self.device)
        self.radius = torch.zeros((num_envs, num_ships, 1), dtype=torch.float32, device=self.device)

    def reset(self, env_mask: torch.Tensor | None = None) -> None:
        """Forget completed episodes without touching recurrent policy history elsewhere."""

        if env_mask is None:
            self.valid.zero_()
            self.age_steps.zero_()
            self.predicted_targets.zero_()
            self.team_id.zero_()
            self.radius.zero_()
            return
        mask = env_mask.bool()
        self.valid[mask] = False
        self.age_steps[mask] = 0
        self.predicted_targets[mask] = 0.0
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
            ObsKey.POWER: raw["power"],
            ObsKey.COOLDOWN: raw["cooldown"],
            ObsKey.LOCAL_LOG_INDEX: raw["local_log_index"],
        }
        for key, belief_value in decoded.items():
            belief_value = torch.nan_to_num(belief_value)
            mask = hidden_belief.unsqueeze(-1)
            data[key][:, :n] = torch.where(mask, belief_value, data[key][:, :n])

        predicted_alive = decoded[ObsKey.HEALTH].squeeze(-1) > ALIVE_HEALTH_EPS
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
        self.predicted_targets.copy_(torch.nan_to_num(forecast.float()))


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
