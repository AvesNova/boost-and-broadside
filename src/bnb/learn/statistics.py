"""The learner's diagnostic quantities (diagnostics audit, groups 4, 5 and 18): pure functions
of what the loss already computes, summed over tokens so that the update can divide once.

None of them changes the loss or its gradient: they read detached values.
"""

import math

import torch

from bnb.contracts.action import NUM_JOINT, NUM_POWER, NUM_SHOOT, NUM_STICK

# The entropy histogram's bins over [0, ln NUM_JOINT]: the quantiles are read from it, to
# within a bin (ln 42 / 64, about 0.06 nats), without a sort over the tokens.
ENTROPY_BINS = 64
MAXIMUM_ENTROPY = math.log(NUM_JOINT)
# A decision is near certain when its most likely joint action has at least this probability
# (diagnostics audit, group 1).
NEAR_CERTAIN = 0.9
# The policy's per-token sums, in order, after the clip fraction and the approximate KL.
POLICY_SUMS = (
    "far_ratio_share",
    "entropy/joint",
    "entropy/power",
    "entropy/stick",
    "entropy/shoot",
    "near_certain_share",
)
ENTROPY_QUANTILES = (0.1, 0.5, 0.9)


def marginal_entropies(log_probabilities: torch.Tensor) -> torch.Tensor:
    """(..., 3): the entropies of the joint policy's power, stick and shoot marginals, from
    its log-probabilities (..., NUM_JOINT) in `bnb.contracts.action.to_joint`'s order."""
    joint = log_probabilities.exp().unflatten(-1, (NUM_POWER, NUM_STICK, NUM_SHOOT))
    marginals = (joint.sum((-2, -1)), joint.sum((-3, -1)), joint.sum((-3, -2)))
    return torch.stack([-torch.special.xlogy(part, part).sum(-1) for part in marginals], -1)


def policy_sums(
    logits: torch.Tensor,
    ratio: torch.Tensor,
    entropy: torch.Tensor,
    clip: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    """(len(POLICY_SUMS) + ENTROPY_BINS,) sums over the tokens of `mask`: tokens whose ratio
    is beyond twice the clip, the joint entropy, the three marginal entropies, the tokens
    whose likeliest action has probability NEAR_CERTAIN or more, then the entropy histogram's
    counts. logits (..., NUM_JOINT); ratio, entropy and mask (...)."""
    log_probabilities = torch.log_softmax(logits.detach().float(), -1)
    weight = mask.float()
    entropy = entropy.detach()
    marginals = marginal_entropies(log_probabilities)
    near = log_probabilities.amax(-1) >= math.log(NEAR_CERTAIN)
    far = (ratio.detach() - 1.0).abs() > 2.0 * clip
    sums = torch.stack(
        [
            (far.float() * weight).sum(),
            (entropy * weight).sum(),
            *(marginals * weight[..., None]).flatten(0, -2).sum(0).unbind(),
            (near.float() * weight).sum(),
        ]
    )
    return torch.cat([sums, entropy_histogram(entropy, weight)])


def entropy_histogram(entropy: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """(ENTROPY_BINS,) the weighted counts of `entropy` in equal bins over [0, ln NUM_JOINT]
    (a value past either end counts in the end bin)."""
    position = (entropy / MAXIMUM_ENTROPY * ENTROPY_BINS).floor().clamp(0, ENTROPY_BINS - 1)
    bins = torch.nn.functional.one_hot(position.long(), ENTROPY_BINS).float()
    return (bins * weight[..., None]).flatten(0, -2).sum(0)


def histogram_quantiles(counts: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """(Q,) the quantiles (Q,) of the entropy histogram `counts` (ENTROPY_BINS,), linear
    within a bin, on the device; NaN for an empty histogram."""
    total = counts.sum()
    cumulative = counts.cumsum(0)
    targets = quantiles * total
    index = torch.searchsorted(cumulative, targets).clamp(max=counts.shape[0] - 1)
    below = cumulative[index] - counts[index]
    fraction = ((targets - below) / counts[index].clamp_min(1e-30)).clamp(0.0, 1.0)
    value = (index + fraction) * (MAXIMUM_ENTROPY / counts.shape[0])
    return torch.where(total > 0, value, torch.nan)


def target_entropy(target: torch.Tensor) -> torch.Tensor:
    """(...) the entropy of each categorical target (..., classes), 0 log 0 = 0."""
    return -torch.special.xlogy(target, target).sum(-1)


def parameter_groups(model: torch.nn.Module) -> dict[str, list[str]]:
    """The model's parameters by module, for the norms and update ratios: each top-level
    child, with a child's `blocks` (a ModuleList, the trunk's) split into one group per
    block ("trunk_block_0")."""
    groups: dict[str, list[str]] = {}
    for child, module in model.named_children():
        blocks = getattr(module, "blocks", None)
        for name, _ in module.named_parameters():
            key = child
            if isinstance(blocks, torch.nn.ModuleList) and name.startswith("blocks."):
                key = f"{child}_block_{name.split('.')[1]}"
            groups.setdefault(key, []).append(f"{child}.{name}")
    for name, _ in model.named_parameters(recurse=False):
        groups.setdefault(name, []).append(name)
    return groups


def group_norms(
    tensors: dict[str, torch.Tensor], groups: dict[str, list[str]]
) -> dict[str, torch.Tensor]:
    """The L2 norm over each group's tensors (by parameter name), on the device."""
    return {
        group: torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(tensors[name].float()) for name in names])
        )
        for group, names in groups.items()
    }
