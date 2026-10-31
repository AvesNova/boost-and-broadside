"""A model's single-step forward exports to ONNX and matches eager (client play, rule 9)."""

import numpy as np
import pytest
import torch

from bnb.contracts import GameConfig, Obs
from bnb.game import match
from bnb.game.perception.observe import perceive
from bnb.models.mlp import MLP

onnxruntime = pytest.importorskip("onnxruntime")


class Step(torch.nn.Module):
    """The exported surface: one team view in, logits and values out (sampling is the
    client's)."""

    def __init__(self, model: MLP) -> None:
        super().__init__()
        self.model = model

    def forward(self, ships, fields, globals):
        return self.model(Obs(ships=ships, bullets=None, fields=fields, globals=globals))


def test_mlp_exports_to_onnx(tmp_path):
    cfg = GameConfig(3, 4, 1500.0)
    p = match.preset(cfg)
    _, obs, _ = perceive(p, match.reset(p, torch.rand(1, p.num_uniforms)))
    model = MLP(cfg.ships_per_team, cfg.num_ships, cfg.num_fields, hidden=64).eval()
    torch.nn.init.normal_(model.policy.weight, std=0.1)  # non-trivial outputs
    args = (obs.ships[:, 0], obs.fields[:, 0], obs.globals[:, 0])
    path = tmp_path / "mlp.onnx"
    torch.onnx.export(Step(model).eval(), args, str(path), dynamo=True,
                      input_names=["ships", "fields", "globals"])
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = session.run(None, {k: v.numpy() for k, v in zip(("ships", "fields", "globals"), args)})
    with torch.no_grad():
        want = Step(model)(*args)
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w.numpy(), rtol=1e-5, atol=1e-5)
