"""The enemy-action figure set: the head's distribution over each living enemy's next joint
action against the action it took, split by whether the view saw the enemy at the tick.

The distribution is the head's (`bnb.models.yemong.decode.action_distribution`), with its
factor marginals (power, stick, shoot); the action is the one the enemy's controller took
at the tick. The baseline is the capture's action frequencies per sight slice (the
training metric's marginal baseline).
"""

import math
import pathlib

import torch

from bnb.contracts.action import NUM_JOINT, NUM_POWER, NUM_SHOOT, NUM_STICK
from bnb.contracts.obs import SHIP

from . import output

SIGHT = ("seen", "hidden")
FACTORS = (("power", NUM_POWER), ("stick", NUM_STICK), ("shoot", NUM_SHOOT))
RELIABILITY_BINS = 10


def actions(artifact: dict) -> dict:
    """Per alive enemy row: joint (rows, NUM_JOINT) probabilities, taken (rows,), seen
    (rows,), and per factor its marginal (rows, size) and the taken value (rows,)."""
    meta, rollout = artifact["meta"], artifact["rollout"]
    own = meta["ships_per_team"]
    part = rollout["enemy_action"]
    alive = part["alive"]
    seen = (rollout["ships"][..., own:, SHIP["visible"]] > 0.5)[alive]
    joint, taken = part["joint"][alive], part["taken"][alive].long()
    factored = joint.unflatten(-1, (NUM_POWER, NUM_STICK, NUM_SHOOT))
    marginals = dict(
        power=factored.sum((-2, -1)), stick=factored.sum((-3, -1)), shoot=factored.sum((-3, -2))
    )
    values = dict(
        power=taken // (NUM_STICK * NUM_SHOOT),
        stick=(taken // NUM_SHOOT) % NUM_STICK,
        shoot=taken % NUM_SHOOT,
    )
    return dict(joint=joint, taken=taken, seen=seen, marginals=marginals, values=values)


def _entropy(frequency: torch.Tensor) -> float:
    return abs(float(-torch.special.xlogy(frequency, frequency).sum()))  # no -0.0


def measures(artifact: dict) -> dict:
    """Per sight slice: the rows, the head's cross-entropy (nats) and the marginal
    baseline's, the accuracy of its most likely action, its mean probability of the action
    taken, and per factor the cross-entropy of the head's marginal and of the baseline's."""
    part = actions(artifact)
    out = {}
    for name, members in zip(SIGHT, (part["seen"], ~part["seen"]), strict=True):
        if not members.any():
            continue
        joint, taken = part["joint"][members], part["taken"][members]
        probability = joint.gather(-1, taken[:, None])[:, 0]
        frequency = torch.bincount(taken, minlength=NUM_JOINT).float() / len(taken)
        factors = {}
        for factor, size in FACTORS:
            marginal = part["marginals"][factor][members]
            value = part["values"][factor][members]
            share = torch.bincount(value, minlength=size).float() / len(value)
            factors[factor] = dict(
                head=round(
                    float(-marginal.gather(-1, value[:, None]).clamp_min(1e-12).log().mean()), 4
                ),
                baseline=round(_entropy(share), 4),
            )
        out[name] = dict(
            rows=int(members.sum()),
            head_cross_entropy=round(float(-probability.clamp_min(1e-12).log().mean()), 4),
            marginal_cross_entropy=round(_entropy(frequency), 4),
            uniform_cross_entropy=round(math.log(NUM_JOINT), 4),
            accuracy=round(float((joint.argmax(-1) == taken).float().mean()), 4),
            probability_of_taken=round(float(probability.mean()), 4),
            factors=factors,
        )
    return out


def reliability(predicted: torch.Tensor, happened: torch.Tensor, bins: int = RELIABILITY_BINS):
    """(bin centres, mean predicted, observed share, count) of a probability against its
    event, over equal-width bins (empty bins dropped)."""
    index = (predicted * bins).long().clamp(0, bins - 1)
    count = torch.bincount(index, minlength=bins).float()
    mean = torch.bincount(index, predicted, minlength=bins) / count.clamp_min(1)
    observed = torch.bincount(index, happened.float(), minlength=bins) / count.clamp_min(1)
    kept = count > 0
    centres = (torch.arange(bins) + 0.5) / bins
    return centres[kept], mean[kept], observed[kept], count[kept]


def render(artifact: dict, directory: pathlib.Path) -> tuple[list[pathlib.Path], dict]:
    """The enemy-action figure (enemy_action): per factor the head's mean marginal against
    the taken frequencies, seen and hidden; and the reliability of P(shoot)."""
    meta = artifact["meta"]
    numbers = measures(artifact)
    part = actions(artifact)
    figure, axes = output.new_figure(1, 4, size=(4.0, 3.8))
    for axis, (factor, size) in zip(axes[0, :3], FACTORS, strict=True):
        places = torch.arange(size).float()
        for slot, (name, members) in enumerate(
            zip(SIGHT, (part["seen"], ~part["seen"]), strict=True)
        ):
            if not members.any():
                continue
            share = torch.bincount(part["values"][factor][members], minlength=size).float()
            share = share / share.sum()
            head = part["marginals"][factor][members].mean(0)
            offset = -0.2 + 0.4 * slot
            colour = output.PREDICTED if name == "seen" else output.SECOND
            axis.bar(places + offset, share, 0.38, color=colour, alpha=0.35, label=f"{name}: taken")
            axis.plot(places + offset, head, "o", color=colour, label=f"{name}: head's mean")
        axis.set_xticks(places.tolist())
        axis.set_xlabel(factor)
        axis.set_ylabel("share")
        detail = ", ".join(
            f"{name} {numbers[name]['factors'][factor]['head']:.2f} vs "
            f"{numbers[name]['factors'][factor]['baseline']:.2f}"
            for name in SIGHT
            if name in numbers
        )
        axis.set_title(f"{factor}: cross-entropy head vs marginal\n{detail}", fontsize=8)
        axis.legend(frameon=False, fontsize=6)
    axis = axes[0, 3]
    axis.plot([0, 1], [0, 1], color=output.MUTED, linewidth=1)
    for name, members, colour in (
        ("seen", part["seen"], output.PREDICTED),
        ("hidden", ~part["seen"], output.SECOND),
    ):
        if not members.any():
            continue
        _, mean, observed, _ = reliability(
            part["marginals"]["shoot"][members, 1], part["values"]["shoot"][members] == 1
        )
        axis.plot(mean, observed, "-o", color=colour, markersize=4, label=name)
    axis.set_xlabel("predicted P(shoot)")
    axis.set_ylabel("observed share shooting")
    axis.set_aspect("equal")
    axis.legend(frameon=False, fontsize=8)
    axis.set_title("reliability of P(shoot)", fontsize=9)
    summary = "; ".join(
        f"{name}: {numbers[name]['rows']} rows, joint cross-entropy "
        f"{numbers[name]['head_cross_entropy']:.2f} vs marginal "
        f"{numbers[name]['marginal_cross_entropy']:.2f} nats, "
        f"accuracy {numbers[name]['accuracy']:.2f}"
        for name in SIGHT
        if name in numbers
    )
    figure.suptitle(
        f"Enemy action: the head's distribution against the action taken\n{summary}",
        color=output.INK,
        fontsize=10,
    )
    figure.tight_layout()
    return [output.save(figure, directory, "enemy_action", meta, numbers)], numbers
