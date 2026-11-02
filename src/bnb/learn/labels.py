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
from bnb.models.yemong.state_code import ShipStateCode


def privileged_rows(trajectory: Trajectory) -> torch.Tensor:
    """(L, R, N, columns): every row's fog-off truth. An own ship is always visible, so its
    observed row is its truth; the enemy rows are stored."""
    own = trajectory.logp.shape[-1]
    return torch.cat([trajectory.obs.ships[..., :own, :], trajectory.truth], -2)


def next_rows(trajectory: Trajectory, rows: torch.Tensor) -> torch.Tensor:
    """(L, R, N, columns): the truth at t + 1; the last step's is the observation after the
    segment's last tick."""
    return torch.cat([rows[1:], trajectory.final_truth[None]], 0)


def next_state_targets(
    state_code: ShipStateCode, trajectory: Trajectory, rows: torch.Tensor
) -> torch.Tensor:
    """(L, R, N, width): the sharp code of every row's true state at t + 1, in the model's
    frame (the true position plus the view's position offset at t)."""
    following = next_rows(trajectory, rows)
    means = shifted_truth(following, trajectory.replay.offset)
    return state_code.sharp(means, following[..., SHIP["ship_type"]].long())


def next_state_mask(trajectory: Trajectory) -> torch.Tensor:
    """(L, R, N): the row's belief is valid at t (own rows always), and t did not end an
    episode."""
    valid = trajectory.replay.valid
    own = trajectory.logp.shape[-1]
    rows = torch.cat([valid.new_ones(valid.shape[:-1] + (own,)), valid], -1)
    return rows & ~trajectory.done[..., None]


def enemy_actions(trajectory: Trajectory) -> tuple[torch.Tensor, torch.Tensor]:
    """The enemy rows' joint actions at t (L, R, N - S) and the mask: alive at t."""
    own = trajectory.logp.shape[-1]
    return to_joint(trajectory.action)[..., own:], trajectory.alive[..., own:]


def density_targets(grid: HexGrid, trajectory: Trajectory, rows: torch.Tensor) -> torch.Tensor:
    """(L, R, 2C): the hex counts of the living ships at t, ally cells then enemy cells, in
    world coordinates relative to the arena centre and scaled by each environment's arena
    radius."""
    own = trajectory.logp.shape[-1]
    is_ally = (torch.arange(rows.shape[-2], device=rows.device) < own).expand(rows.shape[:-1])
    radius = trajectory.obs.globals[..., GLOBAL["arena_radius"]][..., None]
    return grid.counts(
        rows[..., SHIP["x"]], rows[..., SHIP["y"]], is_ally, rows[..., SHIP["alive"]] > 0.5, radius
    )
