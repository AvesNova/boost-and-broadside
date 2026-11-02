"""Triton kernels for Yemong's hot pieces, each behind a switch.

Every kernel computes the same function as a piece of Yemong's plain PyTorch code, which
stays as its twin: the plain path runs on the CPU, in float64 (the reference fixtures) and
wherever a switch is off, and a test holds each kernel to it. A kernel runs only where it
applies (CUDA, and the dtypes it handles); elsewhere the plain path runs whatever the switch
says. Each is a `torch.library.triton_op` with autograd, so it compiles under
`fullgraph=True` and captures in a CUDA graph.

Each kernel autotunes its tiles and warps per shape over a few configurations
(`triton.autotune`, the first call of a shape costs about a second), so a GPU picks its
own; a configuration a GPU cannot run (too much shared memory) is skipped. Where
configurations differ in their last bits (attention, the temporal mix, the heads), the
choice is cached on disk with Triton's results and Inductor's, so a process on the same
machine repeats the bits of an earlier one; the scan's and the gated MLP's configurations
all give the same bits.

The switches are process-wide and read at trace time: set them before compiling. Speed
comparisons between model types run both plain or both with kernels.
"""

import contextlib
import dataclasses


@dataclasses.dataclass
class Switches:
    """One switch per kernel: on, the kernel runs where it applies; off, its plain twin."""

    scan: bool = True  # the RG-LRU's recurrence over a segment (temporal.parallel_scan)
    attention: bool = True  # a spatial sublayer's rotary and attention (SpatialBlock._attend)
    gated_mlp: bool = True  # the gated MLP's hidden layer, 16-bit (GatedMLP.forward)
    temporal_mix: bool = True  # a temporal sublayer's gates, scan and output gate (segments)
    heads: bool = True  # linear-norm-gelu-linear, 16-bit (heads, the encoder's projections)


switches = Switches()


def switch_all(on: bool) -> None:
    """Every kernel on or off."""
    for field in dataclasses.fields(Switches):
        setattr(switches, field.name, on)


@contextlib.contextmanager
def plain():
    """Every kernel off inside the block (the plain PyTorch path); the switches are restored
    after it."""
    saved = dataclasses.replace(switches)
    switch_all(False)
    try:
        yield
    finally:
        for field in dataclasses.fields(Switches):
            setattr(switches, field.name, getattr(saved, field.name))
