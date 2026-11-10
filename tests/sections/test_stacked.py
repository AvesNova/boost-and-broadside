"""One forward over stacked weight sets (`bnb.runtime.stacked`): each set's rows get what a
forward of their own with that set's weights gives, on Yemong's plain path in float32 and
under its bf16 autocast (the attention's cast made explicit for vmap)."""

import pytest
import torch
import yemong_cases as cases
from torch.utils import _pytree

from bnb.runtime import stacked


class _Step(torch.nn.Module):
    """The model's decision without the sampling: its policy logits and its carry."""

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, carry, obs, reset):
        decision = self.model.step(carry, obs, reset)
        return decision.policy_logits.float(), decision.carry


@pytest.mark.parametrize("bfloat16", [False, True])
def test_a_stacked_forward_is_each_sets_own_forward(global_generator, bfloat16):
    """The policy logits and the carry, within the matrix products' rounding: batched and
    separate products round differently (float32's error; bf16's against each tensor's
    scale)."""
    config = cases.SHORT_SQUAD
    nets = [cases.model(config, seed=seed, bfloat16_matmuls=bfloat16) for seed in (0, 1, 2)]
    _, trajectory = cases.rollout(nets[0], config, environments=12, length=4)
    obs = _pytree.tree_map_only(torch.Tensor, lambda x: x[2], trajectory.obs)  # (R, ...)
    rows = obs.ships.shape[0]
    module = _Step(nets[0])
    carry = cases.controller(nets[0], config).init_carry(rows, cases.CPU)
    # no resets: a reset draws the view's position offset, and vmap's draws (one batched
    # draw, `randomness="different"`) are not a separate forward's
    arguments = (carry, obs, torch.zeros(rows, dtype=torch.bool))
    sets = [{f"model.{name}": value for name, value in net.state_dict().items()} for net in nets]
    parameters = {name: torch.stack([s[name] for s in sets]) for name in sets[0]}
    per_set = rows // len(nets)
    inputs = _pytree.tree_map_only(
        torch.Tensor,
        lambda x: x[: per_set * len(nets)].unflatten(0, (len(nets), per_set)),
        arguments,
    )
    tolerance = 2e-2 if bfloat16 else 1e-4
    with torch.no_grad():
        together = stacked.call(module, parameters, inputs)
        for index, weights in enumerate(sets):
            own = _pytree.tree_map_only(torch.Tensor, lambda x: x[index], inputs)
            alone = torch.func.functional_call(module, weights, own)
            for got, want in zip(
                _pytree.tree_leaves(together), _pytree.tree_leaves(alone), strict=True
            ):
                if not isinstance(want, torch.Tensor):
                    continue
                if not want.is_floating_point():
                    assert torch.equal(got[index], want)
                elif bfloat16:
                    error = (got[index].float() - want.float()).abs().max()
                    assert error <= tolerance * want.float().abs().max().clamp_min(1.0)
                else:
                    torch.testing.assert_close(got[index], want, rtol=tolerance, atol=tolerance)
