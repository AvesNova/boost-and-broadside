"""The temporal sublayer (spec 6.4): per query token over time, a causal depthwise
convolution and an RG-LRU, gated, then a gated MLP.

Sequences are time-first, (L, ..., D). `starts (L, ...)` marks the steps where an episode
starts; `starts[0]` is ignored, because the carry a sequence begins from continues the
segment (the caller clears it at a reset before the first step). One decision is `step`;
a stored segment is `sequence`, which must equal a loop of `step` with the carry cleared at
every start.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GatedMLP, norm

TAPS = 4  # the convolution reads the current input and the 3 before it
DECAY_EXPONENT = 8.0
CONTRIBUTION_FLOOR = 1e-6


def _later(starts: torch.Tensor) -> torch.Tensor:
    """The starts after the first step (the first continues the carry)."""
    return torch.cat([torch.zeros_like(starts[:1]), starts[1:]])


def parallel_scan(decay: torch.Tensor, contribution: torch.Tensor,
                  initial: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """h_t = decay_t h_(t-1) + contribution_t over the leading time axis, from `initial`.

    A Hillis-Steele scan: ceil(log2 L) rounds of elementwise work, no sequential loop over
    time. Returns every h (L, ...) and the last.
    """
    length = decay.shape[0]
    state = torch.cat([(contribution[0] + decay[0] * initial)[None], contribution[1:]])
    product = decay
    offset = 1
    while offset < length:
        state = torch.cat([state[:offset], state[offset:] + product[offset:] * state[:-offset]])
        product = torch.cat([product[:offset], product[offset:] * product[:-offset]])
        offset *= 2
    return state, state[-1]


class RGLRU(nn.Module):
    """The real-gated linear recurrent unit. Its recurrence and input gates are one matrix
    (recurrence rows first), so they cost one matrix product."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.log_decay = nn.Parameter(torch.linspace(0.0, 4.0, width))
        self.gates = nn.Linear(width, 2 * width)
        with torch.no_grad():  # each half as nn.Linear(width, width) would start
            bound = width**-0.5
            for half in self.gates.weight.chunk(2):
                nn.init.kaiming_uniform_(half, a=5**0.5)
            self.gates.bias.uniform_(-bound, bound)

    def _terms(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(decay a, contribution) of each input."""
        recurrence, gate = torch.sigmoid(self.gates(inputs)).chunk(2, -1)
        decay = torch.exp(DECAY_EXPONENT * recurrence * F.logsigmoid(self.log_decay))
        scale = torch.sqrt(torch.clamp_min((1.0 - decay) * (1.0 + decay), CONTRIBUTION_FLOOR))
        return decay, scale * gate * inputs

    def step(self, inputs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        decay, contribution = self._terms(inputs)
        return decay * state + contribution

    def sequence(self, inputs: torch.Tensor, state: torch.Tensor,
                 starts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """inputs (L, ..., D); starts (L, ...) bool. Returns (outputs, final state)."""
        decay, contribution = self._terms(inputs)
        decay = torch.where(_later(starts)[..., None], 0.0, decay)
        return parallel_scan(decay, contribution, state)


class CausalConvolution(nn.Module):
    """Depthwise over time: weight (D, 4), column k applied to the input 3 - k steps back."""

    def __init__(self, width: int) -> None:
        super().__init__()
        bound = 1.0 / TAPS**0.5  # torch's Conv1d default for a depthwise kernel of 4
        self.weight = nn.Parameter(torch.empty(width, TAPS).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(width).uniform_(-bound, bound))

    def step(self, inputs: torch.Tensor, buffer: torch.Tensor) -> torch.Tensor:
        """inputs (..., D); buffer (..., 3, D) the previous inputs, oldest first."""
        window = torch.cat([buffer, inputs[..., None, :]], -2)
        return (window * self.weight.t()).sum(-2) + self.bias

    def sequence(self, inputs: torch.Tensor, buffer: torch.Tensor,
                 starts: torch.Tensor) -> torch.Tensor:
        """inputs (L, ..., D); buffer (..., 3, D). A tap never reaches back past a start."""
        history = TAPS - 1
        length = inputs.shape[0]
        padded = torch.cat([buffer.movedim(-2, 0), inputs])  # (L + 3, ..., D)
        cut = torch.cat([torch.zeros_like(starts[:history - 1]), _later(starts)])  # (L + 2, ...)
        out = self.weight[:, history] * inputs + self.bias
        blocked = torch.zeros_like(starts)
        for lag in range(1, TAPS):
            # The tap at this lag is cut if an episode starts at t - lag + 1 ... t.
            blocked = blocked | cut[history - lag:history - lag + length]
            tap = padded[history - lag:history - lag + length]
            out = out + torch.where(blocked[..., None], 0.0, self.weight[:, history - lag] * tap)
        return out


class TemporalBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.input_norm = norm(width)
        self.recurrent_input = nn.Linear(width, width, bias=False)
        self.convolution = CausalConvolution(width)
        self.recurrence = RGLRU(width)
        self.gate_branch = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.feedforward_norm = norm(width)
        self.feedforward = GatedMLP(width, 4 * width)

    def _finish(self, tokens: torch.Tensor, normed: torch.Tensor,
                recurrent: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.output(recurrent * F.gelu(self.gate_branch(normed)))
        return tokens + self.feedforward(self.feedforward_norm(tokens))

    def step(self, tokens: torch.Tensor, state: torch.Tensor, buffer: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step: tokens and state (..., D), buffer (..., 3, D). Returns the outputs, the
        new state and the new buffer."""
        normed = self.input_norm(tokens)
        projected = self.recurrent_input(normed)
        state = self.recurrence.step(self.convolution.step(projected, buffer), state)
        buffer = torch.cat([buffer[..., 1:, :], projected[..., None, :]], -2)
        return self._finish(tokens, normed, state), state, buffer

    def sequence(self, tokens: torch.Tensor, state: torch.Tensor, buffer: torch.Tensor,
                 starts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A segment: tokens (L, ..., D), state (..., D), buffer (..., 3, D), starts (L, ...).
        Returns the outputs and the final state and buffer (unmasked, as `step` leaves them)."""
        normed = self.input_norm(tokens)
        projected = self.recurrent_input(normed)
        convolved = self.convolution.sequence(projected, buffer, starts)
        recurrent, state = self.recurrence.sequence(convolved, state, starts)
        history = TAPS - 1
        buffer = torch.cat([buffer.movedim(-2, 0), projected])[-history:].movedim(0, -2)
        return self._finish(tokens, normed, recurrent), state, buffer
