"""The cadence groups (diagnostics audit, group 19 and `repr/`): the gradient decomposition
and the representation's health, measured every `bnb.runtime.metrics.CADENCE` updates.

The decomposition takes each loss term's gradient (its coefficient applied) over the
first two minibatches, accumulated separately over each (the two halves), and compares
them through Gram matrices, so no pair of gradients is ever held twice:
- `gradient/norm/{scope}/{term}`: the term's gradient norm;
- `gradient/share/{scope}/{term}`: its projection on the step's direction, <g_t, g> / |g|^2
  with g the sum of the top-level terms (the top-level shares sum to 1);
- `gradient/cosine/trunk/{a}/{b}`: every pair's cosine, in the trunk only (whole-model
  cosines are pulled to 0 by the heads' disjoint parameters);
- `gradient/agreement/{scope}/{group}`: |sum g_i| / sum |g_i| over a group's terms (1 when
  they all point one way, near 0 when they cancel);
- `gradient/coherent_fraction/{term}`: <g_A, g_B> / ((|g_A|^2 + |g_B|^2) / 2) between the
  two halves, in the trunk (1: the halves agree, about 0: the term's gradient is noise);
- `gradient/coherent_share/{population}`: each next-state population's share of the
  next-state gradient's coherent part, sum_p <g_A, g_B>_p;
- `gradient/actor_share` with `gradient/actor_norm` and `gradient/critic_norm`: in the trunk,
  |g_actor| / (|g_actor| + |g_critic|), the actor the policy and entropy terms, the critic
  the value term.
The scopes: `model`, every parameter; `trunk`, the shared encoder and trunk.

The representation (`representation/{quantity}/{block}`, per trunk block, on about
REPRESENTATION_TOKENS tokens of one micro-batch), tapped at the block's last sublayer (the
temporal one, or with `yemong.order = temporal_first` the last spatial one): the block
output's RMS, its effective rank (the exponential of the entropy of its normalised singular
values, the tokens centred) and the share of that sublayer's feedforward hidden units that
are dormant (their mean |activation| at most DORMANT_THRESHOLD of the layer's mean, as
ReDo).
"""

import contextlib
import itertools

import torch

from bnb.contracts.series import Deferred

REPRESENTATION_TOKENS = 4096
DORMANT_THRESHOLD = 0.1
TRUNK_MODULES = ("encoder", "trunk")


def trunk_first(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], int]:
    """The model's parameters with the trunk scope's (TRUNK_MODULES) first, and how many
    elements the trunk scope holds."""
    trunk, rest = [], []
    for name, parameter in model.named_parameters():
        (trunk if name.split(".")[0] in TRUNK_MODULES else rest).append(parameter)
    return trunk + rest, sum(parameter.numel() for parameter in trunk)


def flat_gradient(
    output: torch.Tensor, parameters: list[torch.nn.Parameter], size: int
) -> torch.Tensor:
    """The gradient of `output` over `parameters` as one float32 vector (size,), zero for a
    parameter it does not reach (and all zero for an output without a graph)."""
    if not output.requires_grad:
        return torch.zeros(size, device=parameters[0].device)
    gradients = torch.autograd.grad(output, parameters, retain_graph=True, allow_unused=True)
    return torch.cat(
        [
            (torch.zeros_like(p) if g is None else g).float().flatten()
            for p, g in zip(parameters, gradients, strict=True)
        ]
    )


def decomposition(
    grams: dict[str, torch.Tensor],
    combination: torch.Tensor,
    names: tuple[str, ...],
    top_level: tuple[str, ...],
    groups: dict[str, tuple[str, ...]],
    populations: tuple[str, ...],
    actor: tuple[str, ...],
    critic: tuple[str, ...],
) -> dict:
    """The series of the module's docstring from the Gram matrices of the basic gradients.

    grams: per scope, (2B, 2B) the inner products of the B basic gradients of half A, then
    of half B. combination (T, B): each named term (`names`) as a sum of basic gradients.
    top_level: the terms whose sum is the step; groups: the agreement's groups of terms;
    populations: the next-state populations' terms; actor and critic: the split's terms."""
    out = {}
    zero = torch.zeros_like(combination)
    whole = torch.cat([combination, combination], 1)  # (T, 2B): g = g_A + g_B
    half_a, half_b = torch.cat([combination, zero], 1), torch.cat([zero, combination], 1)
    index = {name: i for i, name in enumerate(names)}

    def vector(terms: tuple[str, ...]) -> torch.Tensor:
        weights = torch.zeros(len(names), device=combination.device)
        for term in terms:
            weights[index[term]] = 1.0
        return weights

    total = vector(top_level)
    for scope, gram in grams.items():
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
        population_total = sum(cross[index[name]] for name in populations)
        for name in populations:
            out[f"gradient/coherent_share/{name}"] = cross[index[name]] / population_total
        actor_vector, critic_vector = vector(actor), vector(critic)
        actor_norm = (actor_vector @ full @ actor_vector).clamp_min(0.0).sqrt()
        critic_norm = (critic_vector @ full @ critic_vector).clamp_min(0.0).sqrt()
        out["gradient/actor_norm"] = actor_norm
        out["gradient/critic_norm"] = critic_norm
        out["gradient/actor_share"] = actor_norm / (actor_norm + critic_norm)
    return out


def grams(halves: torch.Tensor, trunk_size: int) -> dict[str, torch.Tensor]:
    """Per scope, the Gram matrix of the stacked basic gradients (2B, P), trunk first."""
    trunk = halves[:, :trunk_size]
    trunk_gram = trunk @ trunk.T
    rest = halves[:, trunk_size:]
    out = {"model": trunk_gram + rest @ rest.T}
    if trunk_size:
        out["trunk"] = trunk_gram
    return out


# ------------------------------------------------------------------ representation


def effective_rank(tokens: torch.Tensor) -> torch.Tensor:
    """exp(entropy of the normalised singular values) of tokens (n, D), centred."""
    centred = tokens - tokens.mean(0)
    singular = torch.linalg.svdvals(centred)
    share = singular / singular.sum().clamp_min(torch.finfo(singular.dtype).tiny)
    return torch.exp(-torch.special.xlogy(share, share).sum())


def dormant_share(hidden: torch.Tensor) -> torch.Tensor:
    """The share of the units of hidden (n, H) whose mean |activation| is at most
    DORMANT_THRESHOLD of the mean over units."""
    score = hidden.abs().mean(0)
    return (score <= DORMANT_THRESHOLD * score.mean()).float().mean()


def sampled(tokens: torch.Tensor, count: int = REPRESENTATION_TOKENS) -> torch.Tensor:
    """About `count` rows of tokens (..., D), evenly strided, as (n, D) float32."""
    flat = tokens.detach().flatten(0, -2).float()
    return flat[:: max(1, flat.shape[0] // count)][:count]


def last_sublayer(block: torch.nn.Module, order: str) -> torch.nn.Module | None:
    """The sublayer whose output is `block`'s output in the trunk's `order`: the temporal one
    when it runs last ("spatial_first", or a block without spatial sublayers), else the last
    spatial one ("temporal_first"). None for a block without either."""
    spatial = getattr(block, "spatial", None)
    if order == "temporal_first" and spatial:
        return spatial[-1]
    return getattr(block, "temporal", None)


@contextlib.contextmanager
def block_taps(model: torch.nn.Module):
    """Yields {block: {"output": tokens, "hidden": the feedforward's hidden units}}, filled by
    hooks on each trunk block's last sublayer (`last_sublayer`: the block's output in either
    trunk order) while the context lasts (empty for a model without `trunk.blocks`); the
    hooks are removed on exit."""
    taps: dict[str, dict[str, torch.Tensor]] = {}
    trunk = getattr(model, "trunk", None)
    blocks = getattr(trunk, "blocks", None)
    order = getattr(trunk, "order", "spatial_first")
    handles = []
    for index, block in enumerate(blocks or ()):
        last = last_sublayer(block, order)
        if last is None:
            continue
        tap = taps.setdefault(f"trunk_block_{index}", {})

        def residual(module, inputs, tap=tap):
            tap["residual"] = inputs[0]

        def output(module, inputs, result, tap=tap):
            tap["output"] = tap.pop("residual") + result

        def hidden(module, inputs, tap=tap):
            tap["hidden"] = inputs[0]

        handles += [
            last.feedforward_norm.register_forward_pre_hook(residual),
            last.feedforward.register_forward_hook(output),
            last.feedforward.down.register_forward_pre_hook(hidden),
        ]
    try:
        yield taps
    finally:
        for handle in handles:
            handle.remove()


def representation_series(taps: dict[str, dict[str, torch.Tensor]]) -> dict:
    """The representation's series of the taps `block_taps` filled."""
    out = {}
    for block, tap in taps.items():
        if "output" not in tap or "hidden" not in tap:
            continue
        tokens = sampled(tap["output"])
        out[f"representation/activation_rms/{block}"] = tokens.square().mean().sqrt()
        # deferred: the singular values synchronise on a GPU, so they are taken on the host
        out[f"representation/effective_rank/{block}"] = Deferred.of(effective_rank, tokens)
        out[f"representation/dormant_share/{block}"] = dormant_share(sampled(tap["hidden"]))
    return out
