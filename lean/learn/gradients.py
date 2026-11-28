"""The gradient decomposition (queue item 11, author 2026-11-28; ported from v2's
src/bnb/learn/gradients.py): each loss term's gradient over one minibatch, on a cadence
(LearnConfig.gradient_diagnostics_every, off by default), a diagnostic.

`decompose` takes the first minibatch of an update's first epoch (the views of its first Adam
step), before that step, and runs minibatch_loss eagerly (never compiled: the training graphs
are untouched) with its weighted terms (`parts`). Each basic term's gradient (its coefficient
applied, before the global clip) is accumulated separately over two halves of the minibatch's
micro-batches and the measures come from their Gram matrices, so no pair of gradients is ever
held twice. The terms: the policy's (surrogate in RL; in the BC stage its weight is 0), the
teacher cross-entropy (`bc` in the BC stage, `teacher` in RL), entropy (the bonus, with its
minus sign), value, next_state, hit, and with the geometry heads pair, field and edge; value is
also split by level (`value_<level>`), next_state by column (`next_state_<column>`) and pair
by target (`pair_<column>`), the parts summing to their term. A term whose coefficient is 0 has
a zero gradient (no backward is run for it).

The measures (keys of the returned dict), scopes `model` (every parameter) and `trunk`
(TRUNK_MODULES: the encoder and the blocks, the parameters every head shares; the heads, the
auxiliary heads and the pair head are outside it):
- `gradient/norm/{scope}/{term}`: the term's gradient norm;
- `gradient/share/{scope}/{term}`: its projection on the step's direction, <g_t, g> / |g|^2
  with g the sum of the top-level terms (the top-level shares sum to 1; a part's shares sum
  to its term's);
- `gradient/cosine/trunk/{a}/{b}`: every pair's cosine, in the trunk only (whole-model
  cosines are pulled to 0 by the heads' disjoint parameters); NaN for a zero gradient;
- `gradient/agreement/{scope}/{group}`: |sum g_i| / sum |g_i| over a group's terms (1 when
  they all point one way, near 0 when they cancel): `terms` (the top level), `value`,
  `next_state`, `pair`;
- `gradient/coherent_fraction/{term}`: <g_A, g_B> / ((|g_A|^2 + |g_B|^2) / 2) between the
  two halves, in the trunk (1: the halves agree, about 0: the term's gradient is noise);
- `gradient/coherent_share/{part}`: each next-state column's (and pair target's) share of its
  term's coherent part, sum_p <g_A, g_B>_p;
- `gradient/actor_share` with `gradient/actor_norm` and `gradient/critic_norm`: in the trunk,
  |g_actor| / (|g_actor| + |g_critic|), the actor the policy terms (surrogate, teacher or bc,
  entropy), the critic the value term.

Under Adam a gradient share is not an update share (docs/rewrite/literature-loss-balance.md
section 1.3). Cost: one eager forward per micro-batch of the minibatch and one backward per
basic term with a non-zero coefficient (about 20); memory 2 x terms x parameters float32.
"""

import itertools

import torch
from torch import Tensor

TRUNK_MODULES = ("encoder", "blocks")
# The pair draw's own generator, so the diagnostic draws nothing from the update's generator or
# torch's default ones (the training step's results do not change).
DIAGNOSTIC_SEED = 11


def trunk_first(policy: torch.nn.Module) -> tuple[list[torch.nn.Parameter], int]:
    """The policy's parameters with the trunk's (TRUNK_MODULES) first, and how many elements
    the trunk holds."""
    trunk, rest = [], []
    for name, parameter in policy.named_parameters():
        (trunk if name.split(".")[0] in TRUNK_MODULES else rest).append(parameter)
    return trunk + rest, sum(parameter.numel() for parameter in trunk)


def flat_gradient(output: Tensor, parameters: list[torch.nn.Parameter], size: int) -> Tensor:
    """The gradient of `output` over `parameters` as one float32 vector (size,), zero for a
    parameter it does not reach (all zero for an output without a graph)."""
    if not output.requires_grad:
        return torch.zeros(size, device=parameters[0].device)
    gradients = torch.autograd.grad(output, parameters, retain_graph=True, allow_unused=True)
    return torch.cat(
        [
            (torch.zeros_like(p) if g is None else g).float().flatten()
            for p, g in zip(parameters, gradients, strict=True)
        ]
    )


def grams(halves: Tensor, trunk_size: int) -> dict[str, Tensor]:
    """Per scope, the Gram matrix of the stacked basic gradients halves (2B, P), trunk first."""
    trunk = halves[:, :trunk_size]
    trunk_gram = trunk @ trunk.T
    rest = halves[:, trunk_size:]
    return {"model": trunk_gram + rest @ rest.T, "trunk": trunk_gram}


def decomposition(
    grams_: dict[str, Tensor],
    combination: Tensor,
    names: tuple[str, ...],
    top_level: tuple[str, ...],
    groups: dict[str, tuple[str, ...]],
    families: dict[str, tuple[str, ...]],
    actor: tuple[str, ...],
    critic: tuple[str, ...],
) -> dict[str, Tensor]:
    """The measures of the module's docstring from the Gram matrices of the basic gradients.

    grams_: per scope, (2B, 2B) the inner products of the B basic gradients of half A, then of
    half B. combination (T, B): each named term (`names`) as a sum of basic gradients.
    top_level: the terms whose sum is the step; groups: the agreement's groups of terms;
    families: per term, its parts whose coherent shares are taken; actor and critic: the
    split's terms."""
    out = {}
    zero = torch.zeros_like(combination)
    whole = torch.cat([combination, combination], 1)  # (T, 2B): g = g_A + g_B
    half_a, half_b = torch.cat([combination, zero], 1), torch.cat([zero, combination], 1)
    index = {name: i for i, name in enumerate(names)}

    def vector(terms: tuple[str, ...]) -> Tensor:
        weights = torch.zeros(len(names), device=combination.device)
        for term in terms:
            weights[index[term]] = 1.0
        return weights

    total = vector(top_level)
    for scope, gram in grams_.items():
        full = whole @ gram @ whole.T  # (T, T)
        norms = full.diagonal().clamp_min(0.0).sqrt()
        step = total @ full @ total
        projection = full @ total
        for name, i in index.items():
            out[f"gradient/norm/{scope}/{name}"] = norms[i]
            out[f"gradient/share/{scope}/{name}"] = projection[i] / step
        for group, members in groups.items():
            weights = vector(members)
            together = (weights @ full @ weights).clamp_min(0.0).sqrt()
            out[f"gradient/agreement/{scope}/{group}"] = together / (norms * weights).sum()
        if scope != "trunk":
            continue
        for a, b in itertools.combinations(names, 2):
            i, j = index[a], index[b]
            out[f"gradient/cosine/trunk/{a}/{b}"] = full[i, j] / (norms[i] * norms[j])
        cross = (half_a @ gram @ half_b.T).diagonal()
        own_a = (half_a @ gram @ half_a.T).diagonal()
        own_b = (half_b @ gram @ half_b.T).diagonal()
        coherent = cross / (0.5 * (own_a + own_b))
        for name, i in index.items():
            out[f"gradient/coherent_fraction/{name}"] = coherent[i]
        for parts in families.values():
            family_total = sum(cross[index[name]] for name in parts)
            for name in parts:
                out[f"gradient/coherent_share/{name}"] = cross[index[name]] / family_total
        actor_vector, critic_vector = vector(actor), vector(critic)
        actor_norm = (actor_vector @ full @ actor_vector).clamp_min(0.0).sqrt()
        critic_norm = (critic_vector @ full @ critic_vector).clamp_min(0.0).sqrt()
        out["gradient/actor_norm"] = actor_norm
        out["gradient/critic_norm"] = critic_norm
        out["gradient/actor_share"] = actor_norm / (actor_norm + critic_norm)
    return out


SPLIT = ("value", "next_state", "pair")  # the terms split into parts (`<term>_<part>`)


def top_level_of(name: str) -> str:
    """The top-level term a basic term belongs to."""
    for term in SPLIT:
        if name.startswith(term + "_"):
            return term
    return name


def coefficients(config, weights: tuple[float, float, float]) -> dict[str, float]:
    """Each top-level term's coefficient; weights = (surrogate, entropy, teacher) as `update`
    passes them."""
    surrogate, entropy, teacher = weights
    return {
        "surrogate": surrogate,
        "teacher": teacher,
        "entropy": entropy,
        "value": config.value_coefficient,
        "next_state": config.next_state_coefficient,
        "hit": config.hit_coefficient,
        "pair": config.pair_coefficient,
        "field": config.field_coefficient,
        "edge": config.edge_coefficient,
    }


def decompose(
    config,
    policy,
    store,
    index: Tensor,
    owners: list[int],
    counts: Tensor,
    loss_names: tuple[str, ...],
    blank_teacher: Tensor | None,
    weights: Tensor,
    support,
    cloning: bool,
) -> dict[str, float]:
    """The module's measures over the first minibatch (owner 0) of the micro-batches `index`
    (store.batch_index's, host), split into two halves of its micro-batches; `counts`
    (RolloutStore.counts), `loss_names`, `blank_teacher`, `weights` and `support` as `update`
    uses them. Leaves every .grad as it was; draws from its own generator only."""
    from lean.learn.training import loss_inputs, minibatch_loss

    device = store.device
    rows = [row for row, owner in enumerate(owners) if owner == 0]
    first = index[rows]
    members = first.flatten().to(device)
    denominators = counts.index_select(0, members).sum(0)
    halves = [first[: (len(rows) + 1) // 2], first[(len(rows) + 1) // 2 :]]
    parameters, trunk_size = trunk_first(policy)
    size = sum(parameter.numel() for parameter in parameters)
    generator = torch.Generator(device=device).manual_seed(DIAGNOSTIC_SEED)
    coefficient = coefficients(config, tuple(weights.tolist()))
    names, accumulated = None, None
    for half, half_index in enumerate(halves):
        if half_index.shape[0] == 0:
            continue
        for batch in store.batches(loss_names, half_index):
            _, _, terms = minibatch_loss(
                config,
                policy,
                *loss_inputs(config, policy, store, batch, blank_teacher, generator),
                weights,
                denominators,
                support,
                parts=True,
            )
            if accumulated is None:
                names = tuple(terms)
                accumulated = torch.zeros((2, len(names), size), device=device)
            for position, name in enumerate(names):
                if coefficient[top_level_of(name)] != 0.0:
                    accumulated[half, position] += flat_gradient(terms[name], parameters, size)
            del terms

    teacher = "bc" if cloning else "teacher"
    names = tuple(teacher if name == "teacher" else name for name in names)
    parts = {term: tuple(name for name in names if top_level_of(name) == term) for term in SPLIT}
    top_level = tuple(dict.fromkeys(top_level_of(name) for name in names))
    everything = top_level + tuple(name for name in names if name not in top_level)
    combination = torch.zeros((len(everything), len(names)), device=device)
    for row, name in enumerate(everything):
        for column, basic in enumerate(names):
            if basic == name or basic in parts.get(name, ()):
                combination[row, column] = 1.0
    groups = {"terms": top_level, **{term: members for term, members in parts.items() if members}}
    families = {term: parts[term] for term in ("next_state", "pair") if parts[term]}
    actor = tuple(dict.fromkeys(("surrogate", teacher, "entropy")))
    out = decomposition(
        grams(accumulated.flatten(0, 1), trunk_size),
        combination,
        everything,
        top_level,
        groups,
        families,
        actor=actor,
        critic=("value",),
    )
    out = dict(zip(out, torch.stack(list(out.values())).tolist(), strict=True))
    out["gradient/trunk_parameters"] = float(trunk_size)
    return out
