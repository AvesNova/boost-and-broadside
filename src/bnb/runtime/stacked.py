"""One network forward over stacked weight sets (`torch.vmap` over
`torch.func.functional_call`): the match engine's lanes, the league's opponent slots and the
training lanes each run one network architecture with several sets of weights, each set on
its own rows.

Run as a Python loop, every set unrolled a whole copy of the forward into the compiled
stage and ran its own small forward each tick: 16 copies of Yemong in the match engine,
about 60,000 traced ops and 529 s of compile (cold), and the forwards' latency added up
(`docs/rewrite/sections/compile-overhaul-notes.md`). Under vmap the stage holds one copy,
batched over the sets: the compile no longer grows with them, and the sets run as one
forward. (Tried first and dropped: `torch.compiler.nested_compile_region` per set; Dynamo
traced the forward once, but the compile still grew with the sets.)

`call` takes the parameters stacked (sets, ...) and the inputs (sets, rows, ...); the
random draws differ per set (`randomness="different"`), as separate forwards' did.
`padded_rows` lays sets with unequal row counts out as one (sets, most) index.
"""

import torch
from torch.utils import _pytree


def call(module: torch.nn.Module, parameters: dict[str, torch.Tensor], inputs: tuple):
    """`module` called on each set: `torch.func.functional_call(module, parameters[s],
    inputs[s])` for every s, as one batched call. Every tensor in `inputs` has the sets on
    its leading axis; None leaves pass through."""
    in_dims = _pytree.tree_map(
        lambda x: 0 if isinstance(x, torch.Tensor) else None, inputs, is_leaf=lambda x: x is None
    )

    # vmap returns tensors only: the output's other leaves (a memoryless network's None
    # carry) are taken out inside and put back after
    structure = {}

    def one(own_parameters, own_inputs):
        output = torch.func.functional_call(module, own_parameters, own_inputs)
        leaves, structure["spec"] = _pytree.tree_flatten(output)
        structure["tensor"] = [isinstance(leaf, torch.Tensor) for leaf in leaves]
        structure["other"] = [leaf for leaf in leaves if not isinstance(leaf, torch.Tensor)]
        return tuple(leaf for leaf in leaves if isinstance(leaf, torch.Tensor))

    tensors = iter(
        torch.func.vmap(one, in_dims=(0, in_dims), randomness="different")(parameters, inputs)
    )
    others = iter(structure["other"])
    leaves = [next(tensors) if is_tensor else next(others) for is_tensor in structure["tensor"]]
    return _pytree.tree_unflatten(leaves, structure["spec"])


def padded_rows(rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Sets of row indices (each (n_s,)) as (sets, most) indices, a short set padded with
    its own first row (row 0 for an empty set), and (sets, most) bool, the real entries."""
    most = max(1, max(int(r.numel()) for r in rows))
    index = torch.stack(
        [
            torch.cat([r, (r[:1] if r.numel() else r.new_zeros(1)).expand(most - r.numel())])
            for r in rows
        ]
    )
    valid = torch.stack([torch.arange(most, device=r.device) < r.numel() for r in rows])
    return index, valid
