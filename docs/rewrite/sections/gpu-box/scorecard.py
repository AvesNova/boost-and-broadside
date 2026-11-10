"""scorecard.py RUN OUT.json: measure a BC run's 120M policy variants (live, uniform average,
exponential average) on fresh games, without training: the run's own TrainConfig with the
learning rate and the warm-up's start at 0, one variant's weights copied into the trainer's model
(in place, so the compiled graphs keep reading them), one update to settle, then MEASURED updates
whose rows are averaged. The learner seat plays the variant, so every training-side metric (BC KL
and agreement to the teacher, critic explained variance, outcome Brier, next-state and error-head
errors, the live lane's score against the scripted agent) describes that variant."""

import dataclasses
import json
import os
import pathlib
import sys

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
import torch  # noqa: E402

from bnb.cli import set_precision  # noqa: E402
from bnb.league.lanes import LaneHooks  # noqa: E402
from bnb.runs import checkpoint  # noqa: E402
from bnb.runtime import cloning  # noqa: E402
from bnb.runtime.train import Trainer  # noqa: E402

MEASURED = 3
run = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
step = sorted((p for p in (run / "policies").glob("step_*")), key=lambda p: int(p.name[5:]))[-1]
newest = sorted((p for p in (run / "checkpoints").glob("step_*")), key=lambda p: int(p.name[5:]))[-1]
config, _ = checkpoint.training_config(newest)
learner = dataclasses.replace(config.learner, learning_rate=0.0, warmup_start=0.0)
config = dataclasses.replace(config, learner=learner)
device = torch.device("cuda")
torch.manual_seed(config.seed)
set_precision(config.precision)
trainer = Trainer(config, device)
LaneHooks(trainer)
results = {}
for variant in ("live", "uniform_average", "exponential_average"):
    policy = checkpoint.load_policy(step / variant, device)
    with torch.no_grad():
        trainer.model.load_state_dict(policy.state_dict())
    rows = []
    for index in range(1 + MEASURED):
        row, _ = cloning.update(trainer)
        if index:
            rows.append(row)
    keys = set.intersection(*(set(r) for r in rows))
    mean = {
        k: sum(r[k] for r in rows) / len(rows)
        for k in keys
        if all(isinstance(r[k], (int, float)) and not isinstance(r[k], bool) for r in rows)
    }
    results[variant] = dict(policy=str(step / variant), **mean)
    print(variant, "kl", round(mean.get("bc/kl", float("nan")), 4), flush=True)
    out.write_text(json.dumps(results))
print("done", out)
