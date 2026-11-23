"""lean.train end to end at smoke size: the BC stage hands its averaged weights to RL."""

import sys

import torch

from lean import model, train
from lean.learn import load_checkpoint, policy_weights


def run(monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", ["lean.train", "--smoke", "--device", "cpu", *arguments])
    train.main()


def test_bc_then_rl_from_the_bc_checkpoint(monkeypatch, tmp_path, capsys):
    run(monkeypatch, "--stage", "bc", "--out", str(tmp_path))
    bc = load_checkpoint(tmp_path / "bc.pt")
    assert bc["stage"] == "bc" and bc["updates"] == 2 and bc["averages"]["count"] > 0
    run(monkeypatch, "--stage", "rl", "--start", str(tmp_path / "bc.pt"), "--out", str(tmp_path))
    rl = load_checkpoint(tmp_path / "rl.pt")
    assert rl["stage"] == "rl" and rl["teacher"] == {"updates": 2, "off": False}
    lines = capsys.readouterr().out.splitlines()
    assert sum(line.startswith("bc update") and "teacher_weight=1 " in line for line in lines) == 2
    assert sum(line.startswith("rl update 0") and "teacher_weight=0.5 " in line for line in lines) == 1
    # RL started from the BC stage's EMA (the hand-off): its EMA began there.
    start = model.Policy(bc["model_config"])
    start.load_state_dict(policy_weights(bc, "ema"))
    name, parameter = next(iter(start.named_parameters()))
    live_rl = rl["live"][name]
    assert not torch.equal(live_rl, parameter)  # RL moved it
    assert torch.allclose(rl["averages"]["ema"][name], parameter, atol=0.1)
