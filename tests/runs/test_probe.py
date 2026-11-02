"""The memory probe (runs spec 7, acceptance 10): the first update's peaks go into run.json,
and a budget set too low stops the run with the peak, the budget and the sizes."""

import pytest
import runs_cases as cases
import torch

from bnb.runs import probe, session
from bnb.runs.probe import MemoryBudgetError
from bnb.runs.records import Run
from bnb.runtime.train import Trainer

pytestmark = pytest.mark.gpu
CUDA = torch.device("cuda")


def _run(tmp_path, updates: int = 1):
    config = cases.tiny_train_config(updates=updates, compiled=True)  # a GPU rollout compiles
    with torch.random.fork_rng(devices=[CUDA]):
        torch.manual_seed(0)
        trainer = Trainer(config, CUDA)
    run = Run.create(tmp_path, "a", config, ["bnb", "train"], None)
    run.start_segment(0)
    return run, trainer


def test_the_probe_records_both_peaks_in_the_run(tmp_path):
    run, trainer = _run(tmp_path)
    with torch.random.fork_rng(devices=[CUDA]):
        session.train(run, trainer, log=lambda row: None)
    peaks = Run.open(tmp_path, "a").record["probe"]
    total = torch.cuda.get_device_properties(CUDA).total_memory
    assert 0 < peaks["allocated"] <= peaks["reserved"] <= total
    assert peaks["budget"] == int(total - probe.MARGIN) and peaks["total"] == total


def test_a_run_over_its_budget_stops_after_the_first_update_naming_the_sizes(tmp_path):
    run, trainer = _run(tmp_path, updates=5)
    total = torch.cuda.get_device_properties(CUDA).total_memory
    with (
        torch.random.fork_rng(devices=[CUDA]),
        pytest.raises(
            MemoryBudgetError,
            match=r"the first update reserved \d+\.\d\d GB, over the budget "
            r"of 0\.00 GB .* at 16 environments, 64 micro-batch rows",
        ),
    ):
        session.train(run, trainer, log=lambda row: None, margin=total)
    record = Run.open(tmp_path, "a").record
    assert record["status"] == "failed" and record["probe"]["reserved"] > 0
    assert trainer.updates_done == 1
