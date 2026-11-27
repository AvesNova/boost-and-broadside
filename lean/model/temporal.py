"""The temporal sublayer: Griffin's recurrent block per token (De et al. 2024, section 2.3;
v2's TemporalBlock, src/bnb/models/yemong/temporal.py), then a gated MLP.

    normed = norm(x)
    convolved = causal depthwise convolution over 4 ticks of recurrent_input(normed)
    state = RG-LRU(convolved)
    x = x + output(state * gelu(gate_branch(normed)))
    x = x + MLP(norm(x))

Time is the leading axis. The carry of a token is (4, width): row 0 the RG-LRU state, rows
1..3 the last three projected inputs, oldest first (the convolution's buffer). Before a
tick with `fresh` 0 (an episode start) the whole carry is zeroed; after a tick with `keep`
0 (a row that may not be attended) it is zeroed too; a state starting from zero takes its
first input unscaled (no sqrt(1 - a^2)). `sequence` (T ticks) and `step` (T = 1) are the
same code: the convolution reads its taps through the same cuts, and the recurrence is one
linear scan in float32 (lean.model.scan).
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lean.model.norm import Norm
from lean.model.scan import linear_scan

TAPS = 4  # the convolution reads the current input and the 3 before it
CONTRIBUTION_FLOOR = 1e-6  # v2's floor under 1 - a_t^2


class Temporal(nn.Module):
    """Griffin's recurrent block and a gated MLP, each pre-normed and residual.

    Initialisation (decided 2026-11-22; docs/rewrite/provenance-check.md section 4): the
    decay as the paper's text, a^c uniform in [retention_low, retention_high], c =
    recurrence_exponent; the rest as the official recurrentgemma code: the gates LeCun
    normal (std 1/sqrt(width)) with zero biases; the two input branches N(0, 1/width); the
    convolution N(0, 0.01/4) (std 0.05) with zero bias; the output projection and the MLP's
    down projection N(0, s/fan_in) with s = 2 / residual_layers; the MLP's up projection
    N(0, 1/width).
    """

    def __init__(self, config, residual_layers: int):
        super().__init__()
        width, hidden = config.width, config.mlp_ratio * config.width
        self.exponent = config.recurrence_exponent
        low = config.low_precision_norm_outputs
        self.input_norm = Norm(width, config.norm_epsilon, low)
        self.recurrent_input = nn.Linear(width, width, bias=False)
        self.gate_branch = nn.Linear(width, width, bias=False)
        self.convolution_weight = nn.Parameter(
            torch.empty(TAPS, width)
        )  # row k: the input k ticks back
        self.convolution_bias = nn.Parameter(torch.zeros(width))
        self.gates = nn.Linear(width, 2 * width)  # recurrence gate rows, then input gate rows
        self.decay = nn.Parameter(torch.empty(width))
        self.output = nn.Linear(width, width, bias=False)
        self.mlp_norm = Norm(width, config.norm_epsilon, low)
        self.mlp_input = nn.Linear(width, 2 * hidden, bias=False)  # gate rows, then up rows
        self.mlp_output = nn.Linear(hidden, width, bias=False)

        final_scale = 2.0 / residual_layers
        with torch.no_grad():
            for layer in (self.recurrent_input, self.gate_branch, self.gates, self.mlp_input):
                layer.weight.normal_(0.0, math.sqrt(1.0 / width))
            self.gates.bias.zero_()
            self.convolution_weight.normal_(0.0, math.sqrt(0.01 / TAPS))
            self.output.weight.normal_(0.0, math.sqrt(final_scale / width))
            self.mlp_output.weight.normal_(0.0, math.sqrt(final_scale / hidden))
            retention = torch.empty(width).uniform_(config.retention_low, config.retention_high)
            base = retention ** (1.0 / self.exponent)
            self.decay.copy_(torch.log(base / (1.0 - base)))

    def retention_at_open_gate(self) -> Tensor:
        """a^c per channel, (width,): the per-tick retention when the recurrence gate is 1."""
        return torch.sigmoid(self.decay) ** self.exponent

    def forward(
        self, tokens: Tensor, carry: Tensor, fresh: Tensor, keep: Tensor
    ) -> tuple[Tensor, Tensor]:
        """tokens (T, E, M, width); carry (E, M, 4, width) float32, before tick 0; fresh
        (T, E, 1, 1) float32, 0 where the carry is zeroed before that tick; keep (T, E, M, 1)
        float32, 0 for rows whose carry is zeroed after that tick. Returns the tokens
        (T, E, M, width) and the carry after the last tick."""
        ticks = tokens.shape[0]
        normed = self.input_norm(tokens)
        # The convolution and the recurrence run in float32 even under bf16 autocast:
        # 1 - a_t^2 with a_t near 1 has no precision left in bf16.
        projected = self.recurrent_input(normed).float()
        inputs = torch.cat(
            [carry[:, :, 1:].movedim(2, 0), projected]
        )  # (T + 3, ...): j = -3 .. T-1
        # link_j: the input at tick j survives into tick j + 1 (kept after j, no reset at
        # j + 1). A virtual tick T with no reset gives the buffer that leaves the segment.
        history = TAPS - 1
        keep_all = torch.cat([keep.new_ones((history,) + keep.shape[1:]), keep])
        before, after = (
            fresh.new_ones((history - 1,) + fresh.shape[1:]),
            fresh.new_ones((1,) + fresh.shape[1:]),
        )
        fresh_all = torch.cat([before, fresh, after])
        link = keep_all * fresh_all  # (T + 3, E, M, 1)
        weight = self.convolution_weight.float()
        convolved = weight[0] * projected + self.convolution_bias.float()
        survive = torch.ones_like(link[:1])
        taps = []
        for lag in range(1, TAPS):  # ticks t = 0 .. T (T + 1 of them) at this lag
            survive = survive * link[history - lag : history - lag + ticks + 1]
            tap = inputs[history - lag : history - lag + ticks + 1] * survive
            convolved = convolved + weight[lag] * tap[:ticks]
            taps.append(tap[ticks])
        buffer = torch.stack(taps[::-1], -2)  # (E, M, 3, width), oldest first

        recurrence, input_gate = torch.sigmoid(self.gates(convolved).float()).chunk(2, -1)
        log_retention = self.exponent * recurrence * F.logsigmoid(self.decay.float())
        retention = torch.exp(log_retention)
        # sqrt(1 - a_t^2) normalises an input against a state already at its stationary
        # scale; a state starting from zero (a reset, the tick after a masked one, or a zero
        # carry before tick 0) takes its first input at 1 instead (the released recurrentgemma
        # code's rule at a segment's first position).
        carried = (carry[:, :, 0] != 0).any(-1, keepdim=True).float()  # (E, M, 1)
        continuing = fresh * torch.cat([carried[None], keep[:-1]])  # (T, E, M, 1)
        normalised = torch.sqrt(
            torch.clamp_min(-torch.expm1(2.0 * log_retention), CONTRIBUTION_FLOOR)
        )
        scale = torch.where(continuing > 0.0, normalised, 1.0)
        states, state = linear_scan(
            retention * fresh * keep, scale * input_gate * convolved * keep, carry[:, :, 0]
        )

        mixed = states.to(tokens.dtype) * F.gelu(self.gate_branch(normed))
        tokens = tokens + self.output(mixed)
        gate, up = self.mlp_input(self.mlp_norm(tokens)).chunk(2, -1)
        tokens = tokens + self.mlp_output(F.gelu(gate) * up)
        return tokens, torch.cat([state[:, :, None], buffer], -2)
