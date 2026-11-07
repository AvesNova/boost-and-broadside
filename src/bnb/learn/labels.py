"""Next-state, enemy-action and density labels (learner spec 5), built on the device per
micro-batch from the stored privileged rows and actions. Next state and enemy action are
row-aligned with the view, as the model's rows are.
"""

import torch

from bnb.contracts import Trajectory
from bnb.contracts.action import to_joint
from bnb.contracts.obs import GLOBAL, SHIP
from bnb.models.yemong.density import HexGrid
from bnb.models.yemong.inputs import truth as shifted_truth


def privileged_rows(trajectory: Trajectory) -> torch.Tensor:
    """(L, R, N, columns): every row's fog-off truth. An own ship is always visible, so its
    observed row is its truth; the enemy rows are stored."""
    own = trajectory.log_probability.shape[-1]
    return torch.cat([trajectory.obs.ships[..., :own, :], trajectory.truth], -2)


def next_rows(trajectory: Trajectory, rows: torch.Tensor) -> torch.Tensor:
    """(L, R, N, columns): the truth at t + 1; the last step's is the observation after the
    segment's last tick."""
    return torch.cat([rows[1:], trajectory.final_truth[None]], 0)


def next_state_targets(model, trajectory: Trajectory, rows: torch.Tensor) -> dict:
    """The next-state head's targets, (L, R, N, ...) per row: {"next_state": the code}, and
    with the delta head {"next_state_clipped": (L, R, N, components) bool}.

    Absolute (`YemongConfig.next_state`): the sharp state code of every row's true state at
    t + 1, in the model's frame (the true position plus the view's position offset at t).
    Delta: the code of every row's true change from t to t + 1 (`delta.DeltaCode.targets`:
    from the baseline of its true state at t), and which changes were clamped to the
    code's padded range."""
    following = next_rows(trajectory, rows)
    offset = trajectory.replay.offset
    after = shifted_truth(following, offset)
    if model.delta_code is None:
        return dict(
            next_state=model.state_code.sharp(after, following[..., SHIP["ship_type"]].long())
        )
    code, clipped = model.delta_code.targets(
        shifted_truth(rows, offset), after, rows[..., SHIP["ship_type"]].long()
    )
    return dict(next_state=code, next_state_clipped=clipped)


def rotary_targets(model, trajectory: Trajectory, rows: torch.Tensor) -> torch.Tensor:
    """(L, R, N, 2, pairs): the true (cos, sin) of every row's rotary angles at t (its true
    position in the model's frame, the view's offset added, and its true heading): the
    rotary read-out's target. Read where the row is alive at t (`trajectory.alive`)."""
    means = shifted_truth(rows, trajectory.replay.offset)
    return model.phasors(means[..., 0:2], means[..., 4])


def next_state_mask(trajectory: Trajectory) -> torch.Tensor:
    """(L, R, N): the row's belief is valid at t (own rows always), and t did not end an
    episode."""
    valid = trajectory.replay.valid
    own = trajectory.log_probability.shape[-1]
    rows = torch.cat([valid.new_ones(valid.shape[:-1] + (own,)), valid], -1)
    return rows & ~trajectory.done[..., None]


def enemy_actions(trajectory: Trajectory) -> tuple[torch.Tensor, torch.Tensor]:
    """The enemy rows' joint actions at t (L, R, N - S) and the mask: alive at t."""
    own = trajectory.log_probability.shape[-1]
    return to_joint(trajectory.action)[..., own:], trajectory.alive[..., own:]


def density_targets(grid: HexGrid, trajectory: Trajectory, rows: torch.Tensor) -> torch.Tensor:
    """(L, R, 2C): the hex counts of the living ships at t, ally cells then enemy cells, in
    world coordinates relative to the arena centre and scaled by each environment's arena
    radius."""
    own = trajectory.log_probability.shape[-1]
    is_ally = (torch.arange(rows.shape[-2], device=rows.device) < own).expand(rows.shape[:-1])
    radius = trajectory.obs.globals[..., GLOBAL["arena_radius"]][..., None]
    return grid.counts(
        rows[..., SHIP["x"]], rows[..., SHIP["y"]], is_ally, rows[..., SHIP["alive"]] > 0.5, radius
    )
