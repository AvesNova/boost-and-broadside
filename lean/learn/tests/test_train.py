"""lean.train end to end at smoke size: the BC stage hands its averaged weights to RL."""

import re
import shutil
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


def without_timing(lines: list[str], numbers) -> list[str]:
    """The update lines of RL updates `numbers`, without the wall-clock rate."""
    return [
        re.sub(r" steps_per_second \d+", "", line)
        for line in lines if any(line.startswith(f"rl update {number} ") for number in numbers)
    ]


def test_checkpoints_every_period_and_a_resume_gives_the_same_next_updates(monkeypatch, tmp_path, capsys):
    """RL at smoke size (64 steps an update, a checkpoint every 128 steps, none deleted):
    resuming from the checkpoint after update 1 gives the same updates 2 and 3 as the run that
    never stopped: the same metrics, live weights, averages, league tallies and games (eagerly
    on the CPU, to rounding)."""
    straight, resumed = tmp_path / "straight", tmp_path / "resumed"
    run(monkeypatch, "--stage", "rl", "--updates", "4", "--out", str(straight))
    first = capsys.readouterr().out.splitlines()
    names = sorted(path.name for path in (straight / "checkpoints").iterdir())
    assert names == ["rl_000000000128.pt", "rl_000000000256.pt"]
    middle = load_checkpoint(straight / "checkpoints" / names[0])
    assert middle["updates"] == 2 and middle["steps"] == 128 and middle["adam_steps"] > 0
    assert middle["optimiser"]["state"] and middle["league"]["taken"] == 2
    assert {"generator", "torch"} <= set(middle["random"]) and middle["carry"] is not None

    # The copy's plan writes to its own directory.
    middle["plan"]["out"] = str(resumed)
    (resumed / "checkpoints").mkdir(parents=True)
    torch.save(middle, resumed / "checkpoints" / names[0])
    shutil.copytree(straight / "snapshots", resumed / "snapshots")  # the checkpoint names its snapshots
    monkeypatch.setattr(sys, "argv", ["lean.train", "--device", "cpu", "--resume", str(resumed)])
    train.main()
    second = capsys.readouterr().out.splitlines()

    assert len(without_timing(second, (2, 3))) == 2
    assert without_timing(first, (2, 3)) == without_timing(second, (2, 3))
    a, b = load_checkpoint(straight / "rl.pt"), load_checkpoint(resumed / "rl.pt")
    for name, value in a["live"].items():
        assert torch.allclose(value, b["live"][name], rtol=1e-5, atol=1e-6), name
    for which in ("ema", "uniform"):
        for name, value in a["averages"][which].items():
            assert torch.allclose(value, b["averages"][which][name], rtol=1e-5, atol=1e-6), name
    end_a = load_checkpoint(straight / "checkpoints" / names[1])
    end_b = load_checkpoint(resumed / "checkpoints" / names[1])
    assert torch.equal(end_a["carry"].member, end_b["carry"].member)
    assert torch.allclose(end_a["league"]["games"], end_b["league"]["games"])
    assert torch.allclose(end_a["game_state"].ships, end_b["game_state"].ships, rtol=1e-5, atol=1e-4)


def test_a_checkpoint_holds_no_snapshot_weights_and_resume_reads_them_from_the_snapshot_files(
    monkeypatch, tmp_path, capsys
):
    """Each snapshot is written once to DIR/snapshots/<steps>.pt; a checkpoint names the
    snapshots its league holds and carries no slot weights for them or for the averages'
    slots, and a resume gives the league slots back from the files and the averages."""
    run(monkeypatch, "--stage", "rl", "--updates", "4", "--out", str(tmp_path))
    files = sorted((tmp_path / "snapshots").iterdir())
    assert [file.name for file in files] == [f"{steps:012d}.pt" for steps in (64, 128, 192, 256)]
    checkpoint = load_checkpoint(tmp_path / "checkpoints" / "rl_000000000256.pt")
    league = checkpoint["league"]
    held = [snapshot for snapshot in league["slot_snapshot"] if snapshot is not None]
    assert sorted(held) == [64, 128, 192, 256]
    assert all(weights is None for weights in league["slot_weights"])
    size = (tmp_path / "checkpoints" / "rl_000000000256.pt").stat().st_size
    # live weights, two averages (float32) and Adam's two moments: about five snapshots, not the four snapshots and two slots again
    assert size < 5.5 * files[0].stat().st_size
    saved = torch.load(files[0], weights_only=True)
    assert all(tensor.dtype == checkpoint["live"][name.removeprefix("policy.")].dtype for name, tensor in saved.items()
               if name.removeprefix("policy.") in checkpoint["live"])

    from lean.learn import League, LearnConfig, WeightAverages

    config = LearnConfig(**checkpoint["learn_config"])
    policy = model.Policy(checkpoint["model_config"])
    policy.load_state_dict(checkpoint["live"])
    averages = WeightAverages(policy, config)
    averages.load(checkpoint["averages"])
    restored = League(config)
    restored.load(league, policy, averages, tmp_path / "snapshots")
    for slot, snapshot in enumerate(league["slot_snapshot"]):
        if snapshot is not None:
            for name, value in torch.load(tmp_path / "snapshots" / f"{snapshot:012d}.pt", weights_only=True).items():
                assert torch.equal(restored.slot_weights[slot][name], value)
    for slot, which in ((0, "ema"), (1, "uniform")):
        for name, value in getattr(averages, which).items():
            assert torch.equal(restored.slot_weights[slot][f"policy.{name}"], value)
