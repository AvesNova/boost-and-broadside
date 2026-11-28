"""lean.train end to end at smoke size: the BC stage hands its averaged weights to RL."""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from lean import model, train
from lean.learn import load_checkpoint, policy_weights
from lean.learn.pool import SnapshotSchedule

# The smoke run's sizes, from lean.train's own settings: 4 environments (lean.train's --smoke)
# x its segments x its ticks an update, and RL runs of UPDATES updates below.
SMOKE = train.smoke_config()
PER_UPDATE = 4 * SMOKE.segments_per_update * SMOKE.segment_ticks
UPDATES = 4
UPDATE_STEPS = [PER_UPDATE * (index + 1) for index in range(UPDATES)]
CHECKPOINT_STEPS = [steps for steps in UPDATE_STEPS if steps % SMOKE.checkpoint_every == 0]
# A snapshot at the first update past each of the schedule's points.
SNAPSHOT_STEPS = sorted(
    {
        -(-point // PER_UPDATE) * PER_UPDATE
        for point in SnapshotSchedule(
            SMOKE.league_snapshot_steps, SMOKE.league_snapshot_period
        ).points(UPDATE_STEPS[-1])
    }
)


def checkpoint_name(steps: int) -> str:
    return f"rl_{steps:012d}.pt"


def run(monkeypatch, *arguments):
    monkeypatch.setattr(
        sys, "argv", ["lean.train", "--smoke", "--device", "cpu", "--wandb", "off", *arguments]
    )
    train.main()


def test_bc_then_rl_from_the_bc_checkpoint(monkeypatch, tmp_path, capsys):
    smoke = SMOKE
    run(monkeypatch, "--stage", "bc", "--out", str(tmp_path))
    bc = load_checkpoint(tmp_path / "bc.pt")
    assert bc["stage"] == "bc" and bc["updates"] == smoke.bc_updates
    assert bc["averages"]["count"] > 0
    run(monkeypatch, "--stage", "rl", "--start", str(tmp_path / "bc.pt"), "--out", str(tmp_path))
    rl = load_checkpoint(tmp_path / "rl.pt")
    assert rl["stage"] == "rl" and rl["teacher"] == {"updates": smoke.rl_updates, "off": False}
    lines = capsys.readouterr().out.splitlines()
    # The update lines print teacher_weight with :.4g (lean.train): BC's is the BC coefficient,
    # RL's first the teacher term's starting weight.
    bc_weight = f"teacher_weight={smoke.bc_coefficient:.4g} "
    rl_weight = f"teacher_weight={smoke.teacher_coefficient:.4g} "
    assert sum(line.startswith("bc update") and bc_weight in line for line in lines) == (
        smoke.bc_updates
    )
    assert sum(line.startswith("rl update 0") and rl_weight in line for line in lines) == 1
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
        for line in lines
        if any(line.startswith(f"rl update {number} ") for number in numbers)
    ]


def test_checkpoints_every_period_and_a_resume_gives_the_same_next_updates(
    monkeypatch, tmp_path, capsys
):
    """RL at smoke size (lean.train.smoke_config: a checkpoint every checkpoint_every steps,
    none deleted): resuming from the first checkpoint gives the same later updates as the run
    that never stopped: the same metrics, live weights, averages, league tallies and games
    (eagerly on the CPU, to rounding)."""
    straight, resumed = tmp_path / "straight", tmp_path / "resumed"
    run(monkeypatch, "--stage", "rl", "--updates", str(UPDATES), "--out", str(straight))
    first = capsys.readouterr().out.splitlines()
    names = sorted(path.name for path in (straight / "checkpoints").iterdir())
    assert names == [checkpoint_name(steps) for steps in CHECKPOINT_STEPS]
    assert len(names) >= 2  # a resume from the first, a comparison at the last
    middle = load_checkpoint(straight / "checkpoints" / names[0])
    resumed_at = CHECKPOINT_STEPS[0] // PER_UPDATE
    assert middle["updates"] == resumed_at and middle["steps"] == CHECKPOINT_STEPS[0]
    assert middle["adam_steps"] > 0 and middle["optimiser"]["state"]
    assert middle["league"]["taken"] == sum(
        steps <= CHECKPOINT_STEPS[0] for steps in SNAPSHOT_STEPS
    )
    assert {"generator", "torch"} <= set(middle["random"]) and middle["carry"] is not None
    assert middle["carry"].next_episodes is None  # redrawn at each segment's start

    # The copy's plan writes to its own directory.
    middle["plan"]["out"] = str(resumed)
    (resumed / "checkpoints").mkdir(parents=True)
    torch.save(middle, resumed / "checkpoints" / names[0])
    shutil.copytree(
        straight / "snapshots", resumed / "snapshots"
    )  # the checkpoint names its snapshots
    monkeypatch.setattr(
        sys, "argv", ["lean.train", "--device", "cpu", "--wandb", "off", "--resume", str(resumed)]
    )
    train.main()
    second = capsys.readouterr().out.splitlines()

    after = tuple(range(resumed_at, UPDATES))
    assert len(without_timing(second, after)) == len(after) > 0
    assert without_timing(first, after) == without_timing(second, after)
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
    # The league's two levels resume where they stopped: the same slot loads and debts.
    for name in ("slot_state", "slot_holder", "slot_source", "stored"):
        assert end_a["league"][name] == end_b["league"][name], name
    assert end_a["league"]["debt"] == pytest.approx(end_b["league"]["debt"], abs=1e-6)
    assert torch.equal(end_a["carry"].slot, end_b["carry"].slot)
    assert torch.allclose(
        end_a["game_state"].ships, end_b["game_state"].ships, rtol=1e-5, atol=1e-4
    )


def test_a_checkpoint_holds_no_snapshot_weights_and_resume_reads_them_from_the_snapshot_files(
    monkeypatch, tmp_path, capsys
):
    """Each snapshot is written once to DIR/snapshots/<steps>.pt; a checkpoint names the
    snapshots its league's pool holds and carries no weights for them, nor any slot's
    weights, and a resume gives the pool and the slots back from the files and the
    averages."""
    run(monkeypatch, "--stage", "rl", "--updates", str(UPDATES), "--out", str(tmp_path))
    files = sorted((tmp_path / "snapshots").iterdir())
    assert [file.name for file in files] == [f"{steps:012d}.pt" for steps in SNAPSHOT_STEPS]
    last = tmp_path / "checkpoints" / checkpoint_name(CHECKPOINT_STEPS[-1])
    checkpoint = load_checkpoint(last)
    league = checkpoint["league"]
    assert sorted(league["stored"]) == SNAPSHOT_STEPS
    assert all(weights is None for weights in league["stored"].values())
    assert "slot_weights" not in league
    size = last.stat().st_size
    # live weights, two averages (float32) and Adam's two moments: about five snapshots, not the
    # four snapshots and the slots again
    assert size < 5.5 * files[0].stat().st_size
    saved = torch.load(files[0], weights_only=True)
    assert all(
        tensor.dtype == checkpoint["live"][name.removeprefix("policy.")].dtype
        for name, tensor in saved.items()
        if name.removeprefix("policy.") in checkpoint["live"]
    )

    from lean.learn import League, LearnConfig, WeightAverages

    config = LearnConfig(**checkpoint["learn_config"])
    policy = model.Policy(checkpoint["model_config"])
    policy.load_state_dict(checkpoint["live"])
    averages = WeightAverages(policy, config)
    averages.load(checkpoint["averages"])
    restored = League(config)
    restored.load(league, policy, averages, tmp_path / "snapshots")
    networks = 0
    for slot, source in enumerate(league["slot_source"]):
        if source is None or source[0] == "scripted":
            continue
        networks += 1
        if source[0] == "snapshot":
            for name, value in torch.load(
                tmp_path / "snapshots" / f"{source[1]:012d}.pt", weights_only=True
            ).items():
                assert torch.equal(restored.slot_weights[slot][name], value)
        else:
            for name, value in getattr(averages, source[0]).items():
                assert torch.equal(restored.slot_weights[slot][f"policy.{name}"], value)
    assert networks > 0


def test_a_checkpoint_write_that_fails_leaves_the_previous_file_whole(monkeypatch, tmp_path):
    """Review B4: a checkpoint is written to a temporary file and then replaces the old one
    (as snapshots are), so a write that dies part way (a full disk, a kill) never leaves a
    truncated checkpoint behind."""
    from lean.learn import LearnConfig, WeightAverages, save_checkpoint

    policy = model.Policy(model.ModelConfig())
    config = LearnConfig()
    path = tmp_path / "rl.pt"
    save_checkpoint(path, policy, WeightAverages(policy, config), config, updates=1)
    save = torch.save

    def dies_part_way(_, file):
        with open(file, "wb") as handle:
            handle.write(b"partial")
        raise OSError("no space left on device")

    monkeypatch.setattr(torch, "save", dies_part_way)
    try:
        save_checkpoint(path, policy, WeightAverages(policy, config), config, updates=2)
    except OSError:
        pass
    monkeypatch.setattr(torch, "save", save)
    assert load_checkpoint(path)["updates"] == 1


def test_start_without_the_rl_stage_is_refused(monkeypatch, tmp_path):
    """Review B6: --start names the BC checkpoint RL starts from; with another stage it would
    be ignored silently, so it is refused."""
    with pytest.raises(SystemExit):
        run(monkeypatch, "--stage", "bc", "--start", str(tmp_path / "bc.pt"))
    with pytest.raises(SystemExit):
        run(monkeypatch, "--start", str(tmp_path / "bc.pt"))


ROOT = Path(__file__).resolve().parents[3]
STOP_UPDATES = 6


def train_process(*arguments) -> subprocess.Popen:
    """lean.train at smoke size in its own process (eager, the CPU, 2 threads, no W&B)."""
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2")
    return subprocess.Popen(
        [sys.executable, "-m", "lean.train", "--device", "cpu", "--threads", "2", *arguments],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def signalled(process: subprocess.Popen, signals: int) -> list[str]:
    """The process's lines; `signals` SIGTERMs sent, the first once update 0 is printed, the
    second once the first is acknowledged (two pending SIGTERMs merge into one)."""
    lines = []
    for line in process.stdout:
        lines.append(line.rstrip("\n"))
        if signals > 0 and line.startswith("rl update 0 steps"):
            process.send_signal(signal.SIGTERM)
        if signals > 1 and line.startswith("SIGTERM: stopping"):
            process.send_signal(signal.SIGTERM)
    process.wait(timeout=600)
    return lines


def test_sigterm_finishes_the_update_writes_a_checkpoint_and_a_resume_continues_exactly(tmp_path):
    """The graceful stop (lean.train's docstring): SIGTERM during an update, the run finishes
    it, writes a checkpoint, says so and exits 0; resuming from the directory gives the same
    later updates as a run that never stopped. A second signal ends the process at once."""
    smoke = ("--smoke", "--stage", "rl", "--updates", str(STOP_UPDATES), "--wandb", "off")
    straight = train_process(*smoke, "--out", str(tmp_path / "straight"))
    first = signalled(straight, 0)
    assert straight.returncode == 0, first

    stopped = train_process(*smoke, "--out", str(tmp_path / "stopped"))
    lines = signalled(stopped, 1)
    assert stopped.returncode == 0, lines
    assert any(line.startswith("SIGTERM: stopping after the update") for line in lines)
    (stop_line,) = [line for line in lines if line.startswith("rl: stopped after update ")]
    last = int(stop_line.split()[4])
    assert last < STOP_UPDATES - 1  # it stopped early
    path = tmp_path / "stopped" / "checkpoints" / checkpoint_name((last + 1) * PER_UPDATE)
    assert path.exists() and str(path) in stop_line
    assert not (tmp_path / "stopped" / "rl.pt").exists()  # the stage did not end
    assert not any(line.startswith(f"rl update {last + 1} ") for line in lines)
    checkpoint = load_checkpoint(path)
    assert checkpoint["updates"] == last + 1 and checkpoint["steps"] == (last + 1) * PER_UPDATE

    resumed = train_process("--resume", str(tmp_path / "stopped"), "--wandb", "off")
    second = signalled(resumed, 0)
    assert resumed.returncode == 0, second
    after = tuple(range(last + 1, STOP_UPDATES))
    assert len(without_timing(second, after)) == len(after)
    assert without_timing(first, after) == without_timing(second, after)
    a = load_checkpoint(tmp_path / "straight" / "rl.pt")
    b = load_checkpoint(tmp_path / "stopped" / "rl.pt")
    for name, value in a["live"].items():
        assert torch.allclose(value, b["live"][name], rtol=1e-5, atol=1e-6), name

    twice = train_process(*smoke, "--out", str(tmp_path / "twice"))
    lines = signalled(twice, 2)
    assert twice.returncode == -signal.SIGTERM, lines
    assert any(line.startswith("SIGTERM again: exiting now") for line in lines)
    assert not any(line.startswith("rl: stopped") for line in lines)


def test_wandb_offline_keeps_one_run_id_that_a_resume_continues(monkeypatch, tmp_path):
    """W&B offline (lean.wandb_log): the run is named for the directory and stage, its id kept
    in DIR/wandb.json, and a resume logs to the same id."""
    monkeypatch.setenv("WANDB_SILENT", "true")
    out = tmp_path / "logged" / "rl"
    arguments = ["--stage", "rl", "--updates", "2", "--out", str(out)]
    monkeypatch.setattr(
        sys, "argv", ["lean.train", "--smoke", "--device", "cpu", "--wandb", "offline", *arguments]
    )
    train.main()
    ids = json.loads((out / "wandb.json").read_text())
    assert set(ids) == {"rl"}
    monkeypatch.setattr(
        sys,
        "argv",
        ["lean.train", "--device", "cpu", "--wandb", "offline", "--resume", str(out)],
    )
    train.main()
    assert json.loads((out / "wandb.json").read_text()) == ids
    runs = sorted((out / "wandb").glob("offline-run-*"))
    assert len(runs) == 2 and all(run.name.endswith(ids["rl"]) for run in runs)
    from lean import wandb_log

    assert wandb_log.run_name(out, "rl") == "logged-rl"
    assert wandb_log.run_name(tmp_path / "logged", "rl") == "logged-rl"
