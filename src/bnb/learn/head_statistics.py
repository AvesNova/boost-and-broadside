"""The auxiliary heads' diagnostic series (diagnostics audit, groups 7 to 11): what each
head predicts against its labels and against the cheap baselines it must beat.

`sums` runs inside the micro-batch loss, on the detached outputs, and gives raw totals
(tensors, some of them vectors); `add` sums them over micro-batches and `series` turns the
update's totals into named series on the device. `belief_sums` reads the stored forecasts
(the behaviour policy's, one update stale) against the stored truth and needs no model
output. None of them changes the loss or its gradient.

Slices, quantity first:
- next state, by population: own ships (`ally`), enemies seen at t (`enemy_visible`) and
  enemies hidden at t (`enemy_hidden`). The gain is the "nothing changes" baseline's
  cross-entropy less the head's, in nats per token; hidden ships have no input state, so
  they have no gain, only the KL (diagnostics audit, group 7).
- enemy action, by the enemy's sight at t (`visible`, `hidden`). The baselines come from
  the batch: repeat the last action with the measured repeat rate r (else uniform over the
  other 41), cross-entropy -r ln r - (1 - r) ln((1 - r) / 41); and always predict the
  batch's action frequencies, cross-entropy the histogram's entropy.
- belief, by hidden age (`AGE_BUCKETS`), over hidden living enemies that hold a belief.
- rotary read-out (with `YemongConfig.rotary_readout`), its mean squared pair error
  (`bnb.learn.losses.phasor_square_error`, the mean over the blocks' read-outs; per block
  too, over the living ships) by population over the living ships, and for
  living hidden enemies by hidden age (`AGE_BUCKETS`, then `never_seen`: no sighting since
  the spawn) the error and the mean length of its (cos, sin) pairs (1: certain; it shrinks
  toward 0 with the model's uncertainty).
"""

import math

import torch

from bnb.contracts import Trajectory
from bnb.contracts.action import NUM_JOINT, NUM_POWER, NUM_SHOOT, NUM_STICK
from bnb.contracts.obs import SHIP
from bnb.contracts.series import Mean
from bnb.models.yemong import codes
from bnb.models.yemong.density import poisson_terms
from bnb.models.yemong.inputs import truth as shifted_truth
from bnb.models.yemong.shots import HIT_ALLY, HIT_ENEMY
from bnb.models.yemong.state_code import (
    LOG_INDEX_SPAN,
    NOSE_RATE_LIMIT,
    POSITION_PERIOD,
    SCALARS,
    velocity_reach,
)

from . import losses

# The state code's runs, in order (`ShipStateCode.runs`).
STATE_PARTS = ("position", "velocity", "heading", "nose_rate", *SCALARS, "gun_cooldown")
POPULATIONS = ("ally", "enemy_visible", "enemy_hidden")
GAIN_POPULATIONS = ("ally", "enemy_visible")
SIGHT = ("visible", "hidden")
FACTORS = ("power", "stick", "shoot")
SHOT_PARTS = ("displacement", "direction", "speed", "refractive_index")
# The outcome code's columns of the three rates (`bnb.models.yemong.shots.outcome_code`):
# the hits by side, then the clear miss (the last column).
SHOT_RATES = (("enemy_hit", HIT_ENEMY - 1), ("ally_hit", HIT_ALLY - 1), ("clear_miss", -1))
TEAMS = ("ally", "enemy")
# Hidden age buckets in seconds (lower bound inclusive) and their slice names.
AGE_BUCKETS = (
    ("age_0_0.1_seconds", 0.0, 0.1),
    ("age_0.1_0.5_seconds", 0.1, 0.5),
    ("age_0.5_1_seconds", 0.5, 1.0),
    ("age_1_2_seconds", 1.0, 2.0),
    ("age_2_5_seconds", 2.0, 5.0),
    ("age_5_10_seconds", 5.0, 10.0),
    ("age_10_30_seconds", 10.0, 30.0),
    ("age_over_30_seconds", 30.0, math.inf),
)
# The belief's quantities: name and its columns in state_code.MEANS order; the error of a
# two-column quantity is the length of the difference.
BELIEF_QUANTITIES = (
    ("position_metres", (0, 1)),
    ("velocity_metres_per_second", (2, 3)),
    ("heading_radians", (4,)),
    ("nose_rate_radians_per_second", (5,)),
    ("angle_of_attack_radians", (6,)),
    ("shield_recharge_wait_seconds", (7,)),
    ("shield", (8,)),
    ("energy", (9,)),
    ("gun_cooldown_seconds", (10,)),
    ("log_refractive_index", (11,)),
)
# Each quantity's columns are a run, read as a slice (indexing by a list copies the index
# to the device, a synchronisation).
assert all(columns == tuple(range(columns[0], columns[-1] + 1)) for _, columns in BELIEF_QUANTITIES)
# The quantities with a sigma, and its column in state_code.SPREADS order (velocity: the
# packed covariance at 1..3; position: the per-axis RMS sigma).
SIGMA_COLUMNS = {
    "position_metres": 0,
    "velocity_metres_per_second": 1,
    "heading_radians": 4,
    "nose_rate_radians_per_second": 5,
    "angle_of_attack_radians": 6,
    "shield_recharge_wait_seconds": 7,
    "shield": 8,
    "energy": 9,
    "log_refractive_index": 10,
}
TINY_VARIANCE = 1e-12  # a sigma of 0 (a sharp forecast) scores as this variance


def add(total: dict, part: dict) -> dict:
    """The sums of two sets of totals, name by name (either may be empty)."""
    if not part:
        return total
    if not total:
        return dict(part)
    return {name: total[name] + part[name] for name in total}


def _masked(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The sum of values (..., [k]) over the tokens of mask (...): a scalar or (k,)."""
    weight = mask.to(values.dtype)
    if values.dim() > mask.dim():
        return (values * weight[..., None]).flatten(0, -2).sum(0)
    return (values * weight).sum()


def _run_sums(per_group: torch.Tensor, runs: codes.Runs) -> torch.Tensor:
    """(..., runs) per-group values (..., groups) summed within each run."""
    return torch.stack(
        [part.sum(-1) for part in per_group.split([groups for groups, _ in runs], -1)], -1
    )


def _grouped_entropy(target: torch.Tensor, runs: codes.Runs) -> torch.Tensor:
    """(..., runs) the entropy of a grouped code (..., width), summed within each run."""
    parts = [
        -torch.special.xlogy(part, part).unflatten(-1, (groups, size)).sum((-2, -1))
        for part, (groups, size) in zip(
            target.split([groups * size for groups, size in runs], -1), runs, strict=True
        )
    ]
    return torch.stack(parts, -1)


@torch.no_grad()
def sums(
    model, unrolled, trajectory: Trajectory, label: dict, shot_parts, shot_runs: codes.Runs
) -> dict:
    """The heads' raw totals over one micro-batch (see the module's docstring); shot_parts
    (L, R, N, ages, parts): the shot trajectory's cross-entropy per part, from the loss;
    shot_runs: the trajectory code's parts as (groups, size)."""
    own = trajectory.log_probability.shape[-1]
    ships = trajectory.obs.ships
    visible = ships[..., SHIP["visible"]] > 0.5  # (L, R, N)
    alive = trajectory.alive
    rows = torch.arange(visible.shape[-1], device=visible.device)
    is_own = (rows < own).expand(visible.shape)
    out = {}
    # Group 7: next state.
    runs = model.state_code.runs
    logits = unrolled.next_state_logits.float()
    target = label["next_state"]
    cross_entropy = _run_sums(codes.grouped_cross_entropy(logits, target, runs), runs)
    code = model.tokens(trajectory.obs, trajectory.replay).code.float()
    baseline = _run_sums(codes.grouped_cross_entropy(codes.baseline(code), target, runs), runs)
    entropy = _grouped_entropy(target, runs).sum(-1)
    mask = label["next_state_mask"]
    populations = dict(
        ally=is_own, enemy_visible=~is_own & visible, enemy_hidden=~is_own & ~visible
    )
    for name, members in populations.items():
        chosen = mask & members
        out[f"next_state/{name}/count"] = chosen.float().sum()
        out[f"next_state/{name}/cross_entropy"] = _masked(cross_entropy, chosen)
        out[f"next_state/{name}/entropy"] = _masked(entropy, chosen)
        if name in GAIN_POPULATIONS:
            out[f"next_state/{name}/baseline"] = _masked(baseline, chosen)
    out["next_state/alive"] = alive.float().sum()
    out["next_state/supervised"] = (mask & alive).float().sum()
    out.update(_enemy_action(unrolled, trajectory, label, own, visible))
    out.update(_density(unrolled, label))
    if shot_parts is not None:
        out.update(_shots(unrolled, label, shot_parts, shot_runs, own, alive))
    if getattr(unrolled, "rotary_readout", None) is not None:
        out.update(_rotary_readout(unrolled, trajectory, label, populations, model.tick_length))
    return out


def _rotary_readout(unrolled, trajectory: Trajectory, label: dict, populations: dict, tick):
    """The read-out's totals, each the mean over the blocks' read-outs: per population the
    living ships and their error; per hidden age slice of the living hidden enemies, their
    count, error and mean pair length. Per block, the living ships' error (blocks,)."""
    readout = unrolled.rotary_readout.float()  # (L, R, N, blocks, 2, pairs)
    per_block = losses.phasor_square_error(readout, label["rotary_readout"][..., None, :, :])
    error = per_block.mean(-1)
    length = readout.square().sum(-2).sqrt().mean(-1).mean(-1)
    alive = trajectory.alive
    out = {
        "rotary_readout/alive/count": alive.float().sum(),
        "rotary_readout/block/error": _masked(per_block, alive),
    }
    for name, members in populations.items():
        chosen = alive & members
        out[f"rotary_readout/{name}/count"] = chosen.float().sum()
        out[f"rotary_readout/{name}/error"] = _masked(error, chosen)
    own = trajectory.log_probability.shape[-1]
    hidden = (alive & populations["enemy_hidden"])[..., own:]
    replay = trajectory.replay
    age = replay.age.float() * tick
    slices = [(name, replay.valid & (age >= low) & (age < high)) for name, low, high in AGE_BUCKETS]
    for name, members in [*slices, ("never_seen", ~replay.valid)]:
        chosen = hidden & members
        out[f"rotary_readout/{name}/count"] = chosen.float().sum()
        out[f"rotary_readout/{name}/error"] = _masked(error[..., own:], chosen)
        out[f"rotary_readout/{name}/length"] = _masked(length[..., own:], chosen)
    return out


def _enemy_action(unrolled, trajectory: Trajectory, label: dict, own: int, visible) -> dict:
    """Group 9's totals per sight slice."""
    log_probability = torch.log_softmax(unrolled.enemy_action_logits[..., own:, :].float(), -1)
    action, mask = label["enemy_action"].long(), label["enemy_mask"]  # (L, R, E)
    cross_entropy = -log_probability.gather(-1, action[..., None])[..., 0]
    correct = (log_probability.argmax(-1) == action).float()
    entropy = -(log_probability.exp() * log_probability).sum(-1)
    joint = log_probability.unflatten(-1, (NUM_POWER, NUM_STICK, NUM_SHOOT))
    factor_log = (
        joint.logsumexp((-2, -1)),
        joint.logsumexp((-3, -1)),
        joint.logsumexp((-3, -2)),
    )
    factor_action = (
        action // (NUM_STICK * NUM_SHOOT),
        (action // NUM_SHOOT) % NUM_STICK,
        action % NUM_SHOOT,
    )
    factor_cross_entropy = torch.stack(
        [
            -part.gather(-1, index[..., None])[..., 0]
            for part, index in zip(factor_log, factor_action, strict=True)
        ],
        -1,
    )  # (L, R, E, factors)
    # a repeat pair: the ship holds a label at t - 1 and t, with no reset in between
    pairs = torch.zeros_like(mask)
    pairs[1:] = mask[1:] & mask[:-1] & ~trajectory.done[:-1, :, None]
    repeated = torch.zeros_like(mask)
    repeated[1:] = action[1:] == action[:-1]
    histogram = torch.nn.functional.one_hot(action, NUM_JOINT).float()
    seen = visible[..., own:]
    out = {"enemy_action/rows": mask.new_full((), mask.numel(), dtype=torch.float32)}
    for name, sight in zip(SIGHT, (seen, ~seen), strict=True):
        chosen, paired = mask & sight, pairs & sight
        prefix = f"enemy_action/{name}"
        out[f"{prefix}/count"] = chosen.float().sum()
        out[f"{prefix}/cross_entropy"] = _masked(cross_entropy, chosen)
        out[f"{prefix}/correct"] = _masked(correct, chosen)
        out[f"{prefix}/entropy"] = _masked(entropy, chosen)
        out[f"{prefix}/factor_cross_entropy"] = _masked(factor_cross_entropy, chosen)
        out[f"{prefix}/histogram"] = _masked(histogram, chosen)
        out[f"{prefix}/pairs"] = paired.float().sum()
        out[f"{prefix}/repeats"] = (paired & repeated).float().sum()
        out[f"{prefix}/pair_cross_entropy"] = _masked(cross_entropy, paired)
    return out


def _density(unrolled, label: dict) -> dict:
    """Group 10's totals per team: the model's excess over the saturated Poisson, and the
    per-cell sums of the counts and of y ln y for the batch-mean null."""
    log_rates, counts = unrolled.density_log_rates.float(), label["density"].float()
    per_cell, saturated = poisson_terms(log_rates, counts)
    excess = (per_cell - saturated).flatten(0, -2)  # (steps, 2C)
    counts = counts.flatten(0, -2)
    cells = counts.shape[-1] // 2
    out = {"density/steps": counts.new_full((), counts.shape[0])}
    for index, team in enumerate(TEAMS):
        part = slice(index * cells, (index + 1) * cells)
        out[f"density/{team}/excess"] = excess[:, part].sum()
        out[f"density/{team}/counts"] = counts[:, part].sum(0)
        out[f"density/{team}/count_log_count"] = torch.special.xlogy(
            counts[:, part], counts[:, part]
        ).sum(0)
    return out


def _shots(unrolled, label: dict, shot_parts, runs: codes.Runs, own: int, alive) -> dict:
    """Group 11's totals: the trajectory's KL per part over the valid ages of the labelled
    shots, and the outcome's three rates, realised and predicted, over own ships' labelled
    shots."""
    mask = label["shot_mask"]  # (L, R, N)
    ages = label["shot_ages"] & mask[..., None]  # (L, R, N, ages)
    target = label["shot_trajectory"]  # (L, R, N, ages, width)
    entropy = _grouped_entropy(target, runs)  # (..., ages, parts)
    out = {
        "shot/age_count": ages.float().sum(),
        "shot/trajectory_kl": _masked(shot_parts.float() - entropy, ages),
    }
    rows = torch.arange(mask.shape[-1], device=mask.device) < own
    own_shots = mask & rows
    probability = torch.softmax(unrolled.shot_outcome_logits.float(), -1)
    outcome = label["shot_outcome"].float()

    def rates(code: torch.Tensor) -> torch.Tensor:
        return torch.stack([code[..., column] for _, column in SHOT_RATES], -1)

    out["shot/own_count"] = own_shots.float().sum()
    out["shot/realised"] = _masked(rates(outcome), own_shots)
    out["shot/predicted"] = _masked(rates(probability), own_shots)
    out["shot/own_alive"] = (alive & rows).float().sum()
    return out


def _gain(baseline_total, cross_entropy_total, count):
    """Mean(baseline - cross-entropy) over count tokens."""
    return Mean(baseline_total - cross_entropy_total, count)


def series(total: dict, model) -> dict:
    """The named series of the update's totals (`sums`, `add`); empty without them."""
    if not total:
        return {}
    out = {}
    # next state
    counts = {name: total[f"next_state/{name}/count"] for name in POPULATIONS}
    population_total = sum(counts.values())
    for name in POPULATIONS:
        count = counts[name]
        cross_entropy = total[f"next_state/{name}/cross_entropy"]
        out[f"next_state/kl/{name}"] = Mean(
            cross_entropy.sum() - total[f"next_state/{name}/entropy"], count
        )
        out[f"next_state/token_share/{name}"] = Mean(count, population_total)
        if name in GAIN_POPULATIONS:
            gain = total[f"next_state/{name}/baseline"] - cross_entropy  # (parts,)
            for part, value in zip(STATE_PARTS, gain.unbind(), strict=True):
                out[f"next_state/gain/{name}/{part}"] = Mean(value, count)
            out[f"next_state/gain/{name}"] = Mean(gain.mean(), count)
    out["next_state/supervised_share"] = Mean(
        total["next_state/supervised"], total["next_state/alive"]
    )
    # enemy action
    labelled = 0.0
    for name in SIGHT:
        prefix = f"enemy_action/{name}"
        count = total[f"{prefix}/count"]
        labelled = labelled + count
        cross_entropy = total[f"{prefix}/cross_entropy"]
        out[f"enemy_action/cross_entropy/{name}"] = Mean(cross_entropy, count)
        out[f"enemy_action/accuracy/{name}"] = Mean(total[f"{prefix}/correct"], count)
        out[f"enemy_action/entropy/{name}"] = Mean(total[f"{prefix}/entropy"], count)
        for factor, value in zip(
            FACTORS, total[f"{prefix}/factor_cross_entropy"].unbind(), strict=True
        ):
            out[f"enemy_action/factor_cross_entropy/{factor}/{name}"] = Mean(value, count)
        frequency = total[f"{prefix}/histogram"] / count.clamp_min(1.0)
        marginal = -torch.special.xlogy(frequency, frequency).sum()
        out[f"enemy_action/gain/marginal/{name}"] = _gain(marginal * count, cross_entropy, count)
        pairs = total[f"{prefix}/pairs"]
        rate = total[f"{prefix}/repeats"] / pairs.clamp_min(1.0)
        repeat = -torch.special.xlogy(rate, rate) - torch.special.xlogy(
            1.0 - rate, (1.0 - rate) / (NUM_JOINT - 1)
        )
        out[f"enemy_action/gain/repeat/{name}"] = _gain(
            repeat * pairs, total[f"{prefix}/pair_cross_entropy"], pairs
        )
        out[f"enemy_action/repeat_rate/{name}"] = Mean(total[f"{prefix}/repeats"], pairs)
    out["enemy_action/supervised_share"] = Mean(labelled, total["enemy_action/rows"])
    # density
    steps = total["density/steps"]
    for team in TEAMS:
        excess = total[f"density/{team}/excess"]
        cells_total = total[f"density/{team}/counts"]
        null = (
            total[f"density/{team}/count_log_count"]
            - torch.special.xlogy(cells_total, cells_total / steps)
        ).sum()
        out[f"density/deviance/{team}"] = Mean(excess, steps * cells_total.numel())
        out[f"density/explained_deviance/{team}"] = 1.0 - excess / torch.where(
            null > 0, null, torch.nan
        )
    # shots
    for part, value in zip(SHOT_PARTS, total["shot/trajectory_kl"].unbind(), strict=True):
        out[f"shot/trajectory_kl/{part}"] = Mean(value, total["shot/age_count"])
    shots = total["shot/own_count"]
    for (name, _), realised, predicted in zip(
        SHOT_RATES, total["shot/realised"].unbind(), total["shot/predicted"].unbind(), strict=True
    ):
        out[f"shot/{name}_rate/realised"] = Mean(realised, shots)
        out[f"shot/{name}_rate/predicted"] = Mean(predicted, shots)
    out["shot/supervised_share"] = Mean(shots, total["shot/own_alive"])
    if "rotary_readout/ally/count" in total:
        for name in POPULATIONS:
            prefix = f"rotary_readout/{name}"
            out[f"rotary_readout/error/{name}"] = Mean(
                total[f"{prefix}/error"], total[f"{prefix}/count"]
            )
        for index, error in enumerate(total["rotary_readout/block/error"].unbind()):
            out[f"rotary_readout/error/block_{index}"] = Mean(
                error, total["rotary_readout/alive/count"]
            )
        for name in (*(bucket for bucket, _, _ in AGE_BUCKETS), "never_seen"):
            prefix, count = f"rotary_readout/{name}", total[f"rotary_readout/{name}/count"]
            out[f"rotary_readout/hidden_error/{name}"] = Mean(total[f"{prefix}/error"], count)
            out[f"rotary_readout/hidden_length/{name}"] = Mean(total[f"{prefix}/length"], count)
    return out


# ---------------------------------------------------------------------- belief (group 8)


def _wrapped(difference: torch.Tensor, period: float) -> torch.Tensor:
    """The minimum-image difference on a circle of `period`."""
    return torch.remainder(difference + 0.5 * period, period) - 0.5 * period


@torch.no_grad()
def belief_sums(trajectory: Trajectory, tick_length: float) -> dict:
    """Group 8's totals over a stored batch: per hidden-age bucket, the count of hidden
    living enemies holding a belief, and per quantity the error's sum, the normalised
    square error's sum ((error / sigma)^2 per dimension) and the counts within 1 and 2
    sigma. The belief is the replay record's (what the behaviour policy believed at t,
    after its sight); the truth is the stored privileged row at t."""
    replay = trajectory.replay
    own = trajectory.log_probability.shape[-1]
    enemy = trajectory.obs.ships[..., own:, :]
    hidden = (enemy[..., SHIP["visible"]] < 0.5) & (trajectory.truth[..., SHIP["alive"]] > 0.5)
    chosen = hidden & replay.valid  # (L, R, E)
    truth = shifted_truth(trajectory.truth, replay.offset)
    means, spreads = replay.means.float(), replay.spreads.float()
    difference = means - truth
    difference = torch.cat(
        [
            _wrapped(difference[..., 0:2], POSITION_PERIOD),
            difference[..., 2:4],
            _wrapped(difference[..., 4:5], 2.0 * math.pi),
            difference[..., 5:],
        ],
        -1,
    )
    errors, normalised = [], []
    for name, columns in BELIEF_QUANTITIES:
        part = difference[..., columns[0] : columns[-1] + 1]  # a slice: a list index syncs
        errors.append(part.square().sum(-1).sqrt())
        if name not in SIGMA_COLUMNS:
            continue
        column = SIGMA_COLUMNS[name]
        if name == "velocity_metres_per_second":
            xx, xy, yy = spreads[..., 1], spreads[..., 2], spreads[..., 3]
            determinant = (xx * yy - xy * xy).clamp_min(TINY_VARIANCE**2)
            x, y = part[..., 0], part[..., 1]
            square = (yy * x * x - 2.0 * xy * x * y + xx * y * y) / determinant
        else:
            variance = spreads[..., column].square().clamp_min(TINY_VARIANCE)
            square = part.square().sum(-1) / variance
        normalised.append(square / len(columns))
    errors, normalised = torch.stack(errors, -1), torch.stack(normalised, -1)
    age = replay.age.float() * tick_length
    living = trajectory.truth[..., SHIP["alive"]] > 0.5
    out = {
        # the never-seen invariant (diagnostics audit, group 13): a living hidden enemy the
        # view holds no belief of was not seen since it spawned
        "fog/living_enemies": living.float().sum(),
        "fog/never_seen": (hidden & ~replay.valid).float().sum(),
    }
    for bucket, low, high in AGE_BUCKETS:
        members = chosen & (age >= low) & (age < high)
        out[f"belief/{bucket}/count"] = members.float().sum()
        out[f"belief/{bucket}/error"] = _masked(errors, members)
        out[f"belief/{bucket}/normalised_square_error"] = _masked(normalised, members)
        out[f"belief/{bucket}/within_one_sigma"] = _masked((normalised <= 1.0).float(), members)
        out[f"belief/{bucket}/within_two_sigma"] = _masked((normalised <= 4.0).float(), members)
    return out


def belief_series(total: dict) -> dict:
    """The named series of the update's belief totals (`belief_sums`, `add`)."""
    if not total:
        return {}
    out = {"fog/never_seen_share": Mean(total["fog/never_seen"], total["fog/living_enemies"])}
    sigma_names = [name for name, _ in BELIEF_QUANTITIES if name in SIGMA_COLUMNS]
    for bucket, _, _ in AGE_BUCKETS:
        count = total[f"belief/{bucket}/count"]
        for (name, _), value in zip(
            BELIEF_QUANTITIES, total[f"belief/{bucket}/error"].unbind(), strict=True
        ):
            out[f"belief/{name}/{bucket}"] = Mean(value, count)
        for quantity in ("normalised_square_error", "within_one_sigma", "within_two_sigma"):
            for name, value in zip(
                sigma_names, total[f"belief/{bucket}/{quantity}"].unbind(), strict=True
            ):
                out[f"belief/{quantity}/{name}/{bucket}"] = Mean(value, count)
    return out


# ---------------------------------------------------------------------- inputs (C1)

# The ship features the state code bins, with the design envelope its bins span
# (`bnb.models.yemong.state_code`); positions wrap, so they have none.
INPUT_FEATURES = (
    "speed",
    "nose_rate",
    "angle_of_attack",
    "shield_recharge_wait",
    "shield",
    "energy",
    "gun_cooldown",
    "log_refractive_index",
)


@torch.no_grad()
def input_sums(state_code, trajectory: Trajectory) -> dict:
    """The raw observation's totals over a stored batch (diagnostics checklist C1): the
    non-finite entries of every part, and per INPUT_FEATURES the visible living ship rows
    whose value lies outside the state code's envelope (beyond its bins, where it folds
    into the last one)."""
    obs = trajectory.obs
    out = {
        "input/nonfinite": sum(
            (~torch.isfinite(part)).float().sum()
            for part in (obs.ships, obs.fields, obs.globals, obs.zones)
            if part is not None
        )
    }
    ships = obs.ships
    shown = (ships[..., SHIP["visible"]] > 0.5) & (ships[..., SHIP["alive"]] > 0.5)
    ship_type = ships[..., SHIP["ship_type"]].long()
    shield_max = state_code.max_shield[ship_type]
    energy_max = state_code.max_energy[ship_type]
    column = {name: ships[..., SHIP[name]] for name in SHIP}
    speed = torch.sqrt(column["velocity_x"] ** 2 + column["velocity_y"] ** 2)
    cooldown_limit = (state_code.cooldown_bins - 1) * state_code.tick_length
    outside = dict(
        speed=speed > velocity_reach(),
        nose_rate=column["nose_rate"].abs() > NOSE_RATE_LIMIT,
        angle_of_attack=column["angle_of_attack"].abs() > state_code.angle_of_attack_limit,
        shield_recharge_wait=(column["shield_recharge_wait"] < 0.0)
        | (column["shield_recharge_wait"] > state_code.shield_delay),
        shield=(column["shield"] < 0.0) | (column["shield"] > shield_max),
        energy=(column["energy"] < 0.0) | (column["energy"] > energy_max),
        gun_cooldown=(column["gun_cooldown"] < 0.0) | (column["gun_cooldown"] > cooldown_limit),
        log_refractive_index=column["log_refractive_index"].abs() > LOG_INDEX_SPAN,
    )
    out["input/shown"] = shown.float().sum()
    out["input/outside"] = torch.stack(
        [(outside[name] & shown).float().sum() for name in INPUT_FEATURES]
    )
    return out


def input_series(total: dict) -> dict:
    """The named series of the update's input totals (`input_sums`, `add`)."""
    if not total:
        return {}
    out = {"input/nonfinite_count": total["input/nonfinite"].int()}
    for name, value in zip(INPUT_FEATURES, total["input/outside"].unbind(), strict=True):
        out[f"input/envelope_violation_share/{name}"] = Mean(value, total["input/shown"])
    return out
