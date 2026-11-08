"""Next-state, enemy-action and density labels (learner spec 5), built on the device per
micro-batch from the stored privileged rows and actions. Next state and enemy action are
row-aligned with the view, as the model's rows are.
"""

import torch

from bnb.contracts import Trajectory
from bnb.contracts.action import to_joint
from bnb.contracts.obs import GLOBAL, SHIP
from bnb.models.yemong.delta import RATE_FIELDS, Rates
from bnb.models.yemong.density import HexGrid
from bnb.models.yemong.inputs import Accelerations, believed
from bnb.models.yemong.inputs import truth as shifted_truth
from bnb.models.yemong.model import HELD


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


def _hidden(trajectory: Trajectory) -> torch.Tensor:
    """(L, R, N): the rows whose belief at t is a forecast (a valid belief not seen at t)."""
    own = trajectory.log_probability.shape[-1]
    hidden = trajectory.replay.age > 0
    return torch.cat([hidden.new_zeros(hidden.shape[:-1] + (own,)), hidden], -1)


def _hidden_decay(model) -> float | None:
    config = getattr(model, "config", None)
    return getattr(config, "physics_hidden_decay", None)


def true_rates(model, trajectory: Trajectory, rows: torch.Tensor) -> Rates:
    """The extrapolation baseline's rates of every row (L, R, N) read off the truth rows
    (`privileged_rows`): each row as a row seen at t - 1 and t reads them (`Yemong.rates`
    of the truth, every row visible: the true accelerations and their validity, the change
    against the truth at t - 1, the true gradient and shot flag). The segment's first tick
    has no truth before it and takes the carry's rows there, so its hidden rows' rates are
    not the truth's (`split_masks` leaves them out)."""
    offset = trajectory.replay.offset
    means = shifted_truth(rows, offset)
    initial = trajectory.initial_carry.previous[None].to(means.dtype)
    previous = torch.cat([initial, shifted_truth(rows[:-1], offset[:-1])], 0)
    valid = rows[..., SHIP["acceleration_valid"]] > 0.5
    world = torch.stack(
        [
            rows[..., SHIP["acceleration_x"]],
            rows[..., SHIP["acceleration_y"]],
            rows[..., SHIP["nose_acceleration"]],
        ],
        -1,
    )
    world = torch.where(valid[..., None], world, 0.0)
    # Yemong.rates reads the world change and its validity only
    accelerations = Accelerations(
        values=world, valid=valid, slow=torch.zeros_like(valid), world=world
    )
    held = torch.zeros_like(means[..., :1]).expand(means.shape[:-1] + (HELD,))
    out, _, _ = model.rates(rows, means, previous, accelerations, held, torch.zeros_like(valid))
    return out


def _merged(hidden: torch.Tensor, truth: Rates, read: Rates) -> Rates:
    """The rates of truth on the hidden rows (L, R, N) and of read elsewhere."""

    def pick(name: str) -> torch.Tensor:
        a, b = getattr(truth, name), getattr(read, name)
        where = hidden[..., None] if a.dim() > hidden.dim() else hidden
        return torch.where(where, a, b.to(a.dtype))

    return Rates(**{name: pick(name) for name in RATE_FIELDS})


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
    true state). With `YemongConfig.physics_hidden_decay` a hidden row trains the physics
    head, so its rates are the truth's (`true_rates`): every row's target is then truth to
    truth, the one a seen row of that ship would have."""
    following = next_rows(trajectory, rows)
    offset = trajectory.replay.offset
    after = shifted_truth(following, offset)
    if model.delta_code is None:
        return dict(
            next_state=model.state_code.sharp(after, following[..., SHIP["ship_type"]].long())
        )
    read = rates(model, trajectory)
    if read is not None and _hidden_decay(model) is not None:
        read = _merged(_hidden(trajectory), true_rates(model, trajectory, rows), read)
    code, clipped, slow = model.delta_code.targets(
        shifted_truth(rows, offset), after, rows[..., SHIP["ship_type"]].long(), read
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


def split_masks(trajectory: Trajectory, mask: torch.Tensor, model=None) -> dict:
    """With `YemongConfig.delta_split`, the next-state mask (L, R, N) split by sight at t:
    {"physics_mask": own rows and the enemies seen at t (the physics head's), "correction_mask":
    the enemies whose belief at t is a forecast (the error head's)}. Both are parts of the
    next-state mask, so its count stays both terms' denominator. With the model's
    `YemongConfig.physics_hidden_decay` tau also "physics_weight" (L, R, N) float32, the
    physics head's weight per row: 1 on the physics mask, exp(-age / tau) on the correction
    mask (age the seconds since the ship was seen, `Replay.age` ticks), past the segment's
    first tick (whose truth rates lack the tick before, `true_rates`), else 0."""
    hidden = _hidden(trajectory)
    out = dict(physics_mask=mask & ~hidden, correction_mask=mask & hidden)
    tau = _hidden_decay(model)
    if tau is not None:
        age = trajectory.replay.age.float() * model.tick_length  # s, the enemy rows'
        own = hidden.shape[-1] - age.shape[-1]
        age = torch.cat([age.new_zeros(age.shape[:-1] + (own,)), age], -1)
        decayed = torch.exp(-age / tau)
        later = torch.arange(hidden.shape[0], device=hidden.device)[:, None, None] > 0
        out["physics_weight"] = torch.where(
            out["physics_mask"], 1.0, torch.where(out["correction_mask"] & later, decayed, 0.0)
        )
    return out


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
