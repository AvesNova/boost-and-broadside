"""Next-state, enemy-action and density labels (learner spec 5), built on the device per
micro-batch from the stored privileged rows and actions. Next state and enemy action are
row-aligned with the view, as the model's rows are.
"""

import torch

from bnb.contracts import Trajectory
from bnb.contracts.action import to_joint
from bnb.contracts.obs import GLOBAL, SHIP
from bnb.models.yemong.density import HexGrid
from bnb.models.yemong.inputs import believed
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


def rates(model, trajectory: Trajectory):
    """The extrapolation baseline's rates over the segment (`Yemong.segment_rates`), as the
    step read them; None for any other model."""
    method = getattr(model, "segment_rates", None)
    if method is None:
        return None
    reset = trajectory.obs.globals[..., GLOBAL["reset"]] > 0.5
    return method(trajectory.initial_carry, trajectory.obs, trajectory.replay, reset)


def next_state_targets(model, trajectory: Trajectory, rows: torch.Tensor) -> dict:
    """The next-state head's targets, (L, R, N, ...) per row: {"next_state": the code}, and
    with the delta head {"next_state_clipped": (L, R, N, components) bool,
    "next_state_slow": (L, R, N) bool}.

    Absolute (`YemongConfig.next_state`): the sharp state code of every row's true state at
    t + 1, in the model's frame (the true position plus the view's position offset at t).
    Delta: the code of every row's true change from t to t + 1 (`delta.DeltaCode.targets`),
    which changes were clamped to the code's padded range, and whether the frame was the
    heading (slower than `delta.SLOW_SPEED`). Under the extrapolation baseline the change
    is from the baseline of the true state at t under the rates the model read (`rates`:
    exact for the rows seen at t; for a hidden row, a diagnostic, the belief's rates on the
    true state)."""
    following = next_rows(trajectory, rows)
    offset = trajectory.replay.offset
    after = shifted_truth(following, offset)
    if model.delta_code is None:
        return dict(
            next_state=model.state_code.sharp(after, following[..., SHIP["ship_type"]].long())
        )
    code, clipped, slow = model.delta_code.targets(
        shifted_truth(rows, offset),
        after,
        rows[..., SHIP["ship_type"]].long(),
        rates(model, trajectory),
    )
    return dict(next_state=code, next_state_clipped=clipped, next_state_slow=slow)


def rotary_targets(model, trajectory: Trajectory, rows: torch.Tensor) -> torch.Tensor:
    """(L, R, N, 2, pairs): the true (cos, sin) of every row's rotary angles at t (its true
    position in the model's frame, the view's offset added, and its true heading): the
    rotary read-out's target. Read where the row is alive at t (`trajectory.alive`)."""
    means = shifted_truth(rows, trajectory.replay.offset)
    return model.phasors(means[..., 0:2], means[..., 4])


def next_state_mask(trajectory: Trajectory, model=None) -> torch.Tensor:
    """(L, R, N): the row's belief is valid at t (own rows always), and t did not end an
    episode. With the delta head (model's `delta_code`) also: the ship is alive at t and at
    t + 1 (a death or a spawn is not a change of state)."""
    valid = trajectory.replay.valid
    own = trajectory.log_probability.shape[-1]
    rows = torch.cat([valid.new_ones(valid.shape[:-1] + (own,)), valid], -1)
    mask = rows & ~trajectory.done[..., None]
    if getattr(model, "delta_code", None) is None:
        return mask
    truth = privileged_rows(trajectory)
    following = next_rows(trajectory, truth)
    return mask & (truth[..., SHIP["alive"]] > 0.5) & (following[..., SHIP["alive"]] > 0.5)


def split_masks(trajectory: Trajectory, mask: torch.Tensor) -> dict:
    """With `YemongConfig.delta_split`, the next-state mask (L, R, N) split by sight at t:
    {"physics_mask": own rows and the enemies seen at t (the physics head's), "correction_mask":
    the enemies whose belief at t is a forecast (the error head's)}. Both are parts of the
    next-state mask, so its count stays both terms' denominator."""
    replay = trajectory.replay
    own = trajectory.log_probability.shape[-1]
    hidden = replay.age > 0  # a valid belief not seen at t
    hidden = torch.cat([hidden.new_zeros(hidden.shape[:-1] + (own,)), hidden], -1)
    return dict(physics_mask=mask & ~hidden, correction_mask=mask & hidden)


def correction_targets(
    model, trajectory: Trajectory, next_state_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The error head's targets on every row (L, R, N, ...): (the code of e = truth(t + 1)
    - forecast(t + 1), its clipped flags, e itself (`correction.CorrectionCode.errors`)),
    with forecast(t + 1) the physics forecast (`delta.DeltaCode.advance`) of the row as the
    model read it at t (`inputs.believed`) and of the physics head's logits, detached: the
    error head's loss gives the physics head no gradient (O7); nothing here carries a
    gradient. Read on the correction mask."""
    rows = privileged_rows(trajectory)
    following = next_rows(trajectory, rows)
    offset = trajectory.replay.offset
    _, _, means, spreads = believed(trajectory.obs.ships, trajectory.replay)
    ship_type = rows[..., SHIP["ship_type"]].long()
    forecast, _ = model.delta_code.advance(
        means.float(),
        spreads.float(),
        next_state_logits.detach().float(),
        ship_type,
        model.state_code,
        rates(model, trajectory),
    )
    code = model.correction_code
    errors = code.errors(forecast, shifted_truth(following, offset).float(), ship_type)
    target, clipped = code.encode(errors)
    return target, clipped, errors


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
