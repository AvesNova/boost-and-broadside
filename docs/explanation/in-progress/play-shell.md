---
covers: []
---

# The play shell (in progress)

<!-- generated: section-status -->

Today `bnb.runtime.realtime.Realtime` runs one
match on the CPU (see [The training loop](../training-loop.md#realtime)).

It will cover the interactive shell: a window, input, rendering from snapshots of the state,
replay and spectating, on the CPU. It uses the runtime and adds no rules. Models run in the
client through their ONNX export.
