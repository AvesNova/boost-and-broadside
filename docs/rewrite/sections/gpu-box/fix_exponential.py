"""fix_exponential.py RUN_DIR STEP SOURCE: rewrite RUN_DIR/policies/step_STEP/exponential_average
with the run's true exponential average at STEP. SOURCE is either a training checkpoint
(`.../training.safetensors`, whose `league.exponential_average.*` tensors are the average) or a
policy directory (BC's final `policy-exponential/`). The record is the live policy's at the same
step (same build, return scale, steps) with the weights and their SHA-256 replaced.

Why: before d62bbc1d every `policies/step_*/exponential_average` held the uniform average
(`save_update_policies` saved both after loading both into one scratch model)."""

import hashlib
import json
import pathlib
import shutil
import sys

import safetensors.torch
from safetensors import safe_open

run, step, source = pathlib.Path(sys.argv[1]), int(sys.argv[2]), pathlib.Path(sys.argv[3])
snapshot = run / "policies" / f"step_{step}"
live, target = snapshot / "live", snapshot / "exponential_average"
names = set(safetensors.torch.load_file(live / "weights.safetensors"))
if source.name == "training.safetensors":
    prefix = "league.exponential_average."
    with safe_open(source, "pt") as f:
        weights = {k[len(prefix) :]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
else:
    record = json.load(open(source / "checkpoint.json"))
    assert record["steps"]["environment_steps"] == step, record["steps"]
    weights = safetensors.torch.load_file(source / "weights.safetensors")
assert set(weights) == names, sorted(set(weights) ^ names)[:5]
uniform = safetensors.torch.load_file(snapshot / "uniform_average" / "weights.safetensors")
assert any(not weights[k].equal(uniform[k]) for k in weights), "the source is the uniform average"
scratch = snapshot / "exponential_average.tmp"
if scratch.exists():
    shutil.rmtree(scratch)
shutil.copytree(live, scratch)
safetensors.torch.save_file({k: v.contiguous() for k, v in weights.items()},
                            scratch / "weights.safetensors")
record = json.load(open(scratch / "checkpoint.json"))
record["files"]["weights.safetensors"] = hashlib.sha256(
    (scratch / "weights.safetensors").read_bytes()).hexdigest()
json.dump(record, open(scratch / "checkpoint.json", "w"), indent=2)
shutil.rmtree(target)
scratch.rename(target)
print(f"{run.name} step {step}: exponential_average rewritten from {source}, "
      f"sha {record['files']['weights.safetensors'][:8]}")
