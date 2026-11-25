"""How a network agent acts in the game, the same in training and in evaluation: the seat's
view, the policy's precision, one step of a network held as a weightless template plus
weights, the joint action's sample and the carry's reset at an episode start.

lean.learn's rollout (the live policy and the league's networks) and lean.eval's runner all
act through one agent tick (agent_step); the league's networks and the runner's seats share
the per-network routing (routed_step: each network runs only on its own rows). Nothing here
depends on the learner.
"""

import copy
import dataclasses
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.func import functional_call

from lean.contract import SEAT_AXIS, Obs, random_keywords, split_joint


def precision(device, bf16: bool) -> torch.autocast:
    """The autocast context of a policy's forward pass: bf16 on a GPU when `bf16` (the
    network's training setting, lean.learn.LearnConfig.bf16); float32 otherwise. The model
    keeps its position phases and recurrence in float32 inside it."""
    device = torch.device(device)
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda")


def select_environments(chosen: Tensor, when_chosen: Tensor, otherwise: Tensor) -> Tensor:
    """Per-environment choice between two tensors with a leading E; `chosen` is (E,) bool."""
    return torch.where(chosen.view(-1, *[1] * (otherwise.dim() - 1)), when_chosen, otherwise)


def seat_view(views: tuple[Obs, Obs], environment: Tensor, team: Tensor) -> Obs:
    """The view of team `team[s]` in environment `environment[s]`, merged from the two
    fixed-team views `views` = (team 0's, team 1's), each with a leading E."""
    first, second = views
    chosen = team == 0
    return Obs(
        **{
            field.name: select_environments(
                chosen,
                getattr(first, field.name).index_select(0, environment),
                getattr(second, field.name).index_select(0, environment),
            )
            for field in dataclasses.fields(first)
        }
    )


def reset_carry(carry: Tensor, done: Tensor) -> Tensor:
    """Zero the policy's recurrent carry (blocks, E, 1 + N, 4, width) of the environments
    (seats) in `done` (E,) bool."""
    return carry * (~done).view(1, -1, 1, 1, 1)


def sample_joint(logits: Tensor, generator) -> Tensor:
    """One sample per row from categorical logits (..., JOINT), by the Gumbel-max trick."""
    uniform = torch.rand(logits.shape, **random_keywords(generator, logits.device)).clamp(
        1e-10, 1.0 - 1e-7
    )
    return (logits - torch.log(-torch.log(uniform))).argmax(-1)


class Stepper(nn.Module):
    """A policy whose forward is its `step`, so that torch.func.functional_call can run
    the step with another set of weights."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs, carry, position_offset):
        return self.policy.step(obs, carry, position_offset)


_TEMPLATES: dict[str, Stepper] = {}


def template(policy) -> Stepper:
    """The weightless Stepper (on the meta device) of the policy's architecture, one per
    architecture for the process: a compiled graph guards on the module, so every network of
    one architecture shares it, and its weights are inputs (new weights are new data)."""
    key = repr(policy.config)
    if key not in _TEMPLATES:
        _TEMPLATES[key] = copy.deepcopy(Stepper(policy)).to("meta").eval()
    return _TEMPLATES[key]


def stepper_weights(policy, parameters: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
    """A network's weights: the policy's parameters (or `parameters`, by the policy's own
    names) and buffers, named as Stepper's, detached copies."""
    weights = {}
    for name, tensor in policy.named_parameters():
        source = tensor if parameters is None else parameters[name].to(tensor.dtype)
        weights[f"policy.{name}"] = source.detach().clone()
    for name, tensor in policy.named_buffers():
        weights[f"policy.{name}"] = tensor.detach().clone()
    return weights


def split_weights(
    template_: Stepper, weights: dict[str, Tensor]
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """`weights` (stepper_weights) as (parameters, buffers) by the template's names, as
    functional_call takes them."""
    names = {name for name, _ in template_.named_parameters()}
    return (
        {name: value for name, value in weights.items() if name in names},
        {name: value for name, value in weights.items() if name not in names},
    )


class Network(NamedTuple):
    """A network agent: the weightless template, its parameters and buffers, and its
    precision (`bf16`, the setting it was trained with). A named tuple, so a compiled graph
    takes it as an input like a plain tuple."""

    template: Stepper
    parameters: dict[str, Tensor]
    buffers: dict[str, Tensor]
    bf16: bool = True


def network(policy, bf16: bool = True) -> Network:
    """A policy as a Network (its weights copied)."""
    template_ = template(policy)
    return Network(template_, *split_weights(template_, stepper_weights(policy)), bf16)


class AgentStep(NamedTuple):
    """One tick of a network agent (agent_step): `joint` the sampled joint actions (E, N) long,
    `logits` their float32 logits (E, N, JOINT), `value_logits` the critic's logits as the
    forward returned them, and `carry` the new recurrent carry, in the old carry's dtype."""

    joint: Tensor
    logits: Tensor
    value_logits: Tensor
    carry: Tensor


def agent_step(
    forward, bf16: bool, view: Obs, carry: Tensor, offset: Tensor, generator
) -> AgentStep:
    """The one agent tick every caller shares (training's live policy, the league's networks,
    the evaluation runner's seats): `forward(view, carry, offset)` -> (logits, value_logits,
    carry), a Policy.step or network_forward, run in the network's precision (`bf16`), then
    the joint action sampled from the logits."""
    with precision(offset.device, bf16):
        logits, value_logits, new_carry = forward(view, carry, offset)
    logits = logits.float()
    return AgentStep(
        sample_joint(logits, generator), logits, value_logits, new_carry.to(carry.dtype)
    )


def network_forward(network_: Network):
    """The Network's step as a function of (view, carry, offset): its template run with its
    weights (torch.func.functional_call)."""

    def forward(view: Obs, carry: Tensor, offset: Tensor):
        return functional_call(
            network_.template, (network_.parameters, network_.buffers), (view, carry, offset)
        )

    return forward


def network_step(
    network_: Network, view: Obs, carry: Tensor, offset: Tensor, generator
) -> AgentStep:
    """One tick of a Network on `view` (rows E): agent_step with its weights and precision."""
    return agent_step(network_forward(network_), network_.bf16, view, carry, offset, generator)


def routed_step(
    network_: Network,
    mine: Tensor,
    positions: Tensor,
    view_of,
    carry: Tensor,
    offset: Tensor,
    generator,
    action: Tensor,
    carry_out: Tensor,
) -> tuple[Tensor, Tensor]:
    """One network's moves on its own rows of R (capacity routing, mixture-of-experts style):
    the rows where `mine` (R,) bool take the rows 0, 1, ... of a W-row batch in order (W =
    positions.shape[0], never fewer than the network's rows), the network runs once on those
    W rows (`view_of(rows)` the view of the R-rows `rows` (W,), `carry` (blocks, R, ...) and
    `offset` (R, 2) read at them), and each of its rows reads its sampled action (N, 3) and
    new carry back into `action` (R, N, 3) and `carry_out` (the carry's shape); the other
    rows keep theirs. Padding rows (fewer of the network's rows than W) run on row 0's input
    and are never read. No host read, and one compiled graph for every network, W and R.
    Returns (action, carry_out), the carry contiguous (a compiled graph is specialised on
    strides, so the carry keeps one layout from tick to tick)."""
    rows = mine.shape[0]
    width = positions.shape[0]
    rank = mine.long().cumsum(0) - 1  # each of the network's rows: its row in the batch
    slot = torch.where(mine, rank, width)  # others to a spare row past the batch
    source = torch.zeros(width + 1, dtype=torch.long, device=mine.device)
    source = source.scatter(0, slot, torch.arange(rows, device=mine.device))[:-1]
    view = view_of(source)
    rows_carry = carry.index_select(SEAT_AXIS, source)
    step = network_step(network_, view, rows_carry, offset.index_select(0, source), generator)
    read = rank.clamp(min=0, max=width - 1)
    sampled = split_joint(step.joint).index_select(0, read)
    new_action = torch.where(mine[:, None, None], sampled, action)
    new_carry = step.carry.index_select(SEAT_AXIS, read)
    chosen = mine.view(*[1] * SEAT_AXIS, -1, *[1] * (carry_out.dim() - SEAT_AXIS - 1))
    return new_action, torch.where(chosen, new_carry, carry_out).contiguous()
