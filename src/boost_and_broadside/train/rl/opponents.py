"""Opponent perspectives, action overrides, league sampling, and policy averaging."""

import torch

from boost_and_broadside.env.observation import MVPObservation
from boost_and_broadside.env.state import TensorState


def slice_obs(obs: MVPObservation, start: int, end: int) -> MVPObservation:
    """Return a view of observation tensors for environments ``[start, end)``."""
    return obs.slice_envs(slice(start, end))


def slice_state(state: TensorState, start: int, end: int) -> TensorState:
    """Return a view-backed state for environments ``[start, end)``."""
    return state.slice_envs(slice(start, end))


def flip_team_obs(obs: MVPObservation, num_ships: int) -> MVPObservation:
    """Flip ship team IDs while leaving obstacle team IDs unchanged."""
    return obs.flip_team(num_ships)


class OpponentMixin:
    """Opponent-management behavior mixed into PPOTrainer."""

    def _update_avg_model(self) -> None:
        """Add the current training policy snapshot to the uniform running average."""
        first_update = self._avg_update_count == 0
        self._avg_update_count += 1
        for cumulative, parameter in zip(self._avg_param_cumsum, self._policy_module.parameters()):
            cumulative.add_(parameter.detach().float())
        for avg_parameter, cumulative in zip(
            self._avg_policy_module.parameters(), self._avg_param_cumsum
        ):
            avg_parameter.data.copy_(cumulative / self._avg_update_count)
        if first_update:
            self.roster.add_special("avg", self._global_step, 0, initial_elo=self._training_elo)

    def _opponent_obs(self, obs_slice: MVPObservation, num_ships: int) -> MVPObservation:
        """Return the observation perspective used by policy opponents."""
        return flip_team_obs(obs_slice, num_ships) if self._ego_pass else obs_slice

    def _combine_actions(
        self,
        action_t0: torch.Tensor,
        action_t1: torch.Tensor | None,
        team_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge perspective actions and build the actor-loss mask."""
        if self._ego_pass:
            team0_mask = team_id == 0
            action = torch.where(team0_mask.unsqueeze(-1), action_t0, action_t1)
            return action, team0_mask
        return action_t0.clone(), torch.ones_like(team_id, dtype=torch.bool)

    def _apply_opponent_override(
        self,
        action: torch.Tensor,
        actor_mask: torch.Tensor,
        team_id: torch.Tensor,
        start: int,
        end: int,
        opp_action: torch.Tensor,
    ) -> None:
        """Replace opponent-controlled actions in environments ``[start, end)``."""
        if self._ego_pass:
            opp_mask = team_id[start:end] == 1
        else:
            flags = self._opp_team_flag[start - self.B_self : end - self.B_self]
            opp_mask = team_id[start:end] == flags.unsqueeze(1)
        action[start:end] = torch.where(opp_mask.unsqueeze(-1), opp_action, action[start:end])
        actor_mask[start:end] &= ~opp_mask

    def _prepare_league_opponent(self, num_tokens: int) -> torch.Tensor | None:
        """Sample and prepare the league opponent for one rollout."""
        league_active = self.B_league > 0 and self._schedule_state.league_fraction > 0.0
        if not league_active:
            self.roster.evict_all_checkpoint_policies()
            self._current_league_entry = None
            self._current_league_policy = None
            return None

        entry = self.roster.sample(self._training_elo)
        self._current_league_entry = entry
        if entry is None or (entry.kind == "avg" and self._avg_update_count == 0):
            self._current_league_entry = None
            self._current_league_policy = None
            return None
        if entry.kind == "checkpoint":
            self.roster.load_policy(
                entry,
                self.model_config,
                self.coordinator,
                self.wrapper.num_active_components,
                self.wrapper.num_ships,
                self.device,
                self._compile_mode,
                team_pma_k=self._win_k,
            )
            self._current_league_policy = entry.policy
        elif entry.kind == "avg":
            self._current_league_policy = self.avg_policy
        else:
            self._current_league_policy = None
            return None
        return self._current_league_policy.initial_hidden(self.B_league, num_tokens, self.device)
