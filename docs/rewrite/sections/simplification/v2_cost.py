"""v2: ops of the learner, per update phase and per micro-batch, by function, with the GPU's
Triton kernels counted as one op each (forward and backward).

usage: python v2_learn.py V2_CHECKOUT [ppo|bc] [length]
"""

import dataclasses
import functools
import importlib
import inspect
import pkgutil
import sys

sys.path.insert(0, sys.argv[1] + "/src")
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F

import regions as R

objective = sys.argv[2] if len(sys.argv) > 2 else "ppo"
length = int(sys.argv[3]) if len(sys.argv) > 3 else 16
torch.set_num_threads(4)

from bnb.models.yemong import spatial, temporal  # noqa: E402
from bnb.models.yemong.kernels import attention as k_attention  # noqa: E402
from bnb.models.yemong.kernels import gated as k_gated  # noqa: E402
from bnb.models.yemong.kernels import heads as k_heads  # noqa: E402
from bnb.models.yemong.kernels import scan as k_scan  # noqa: E402
from bnb.models.yemong.kernels import temporal as k_temporal  # noqa: E402

# --- the GPU path: every kernel applies and counts as one op forward, one backward ---
for module in (k_attention, k_gated, k_heads, k_scan, k_temporal):
    module.applies = lambda *a, **k: True
k_heads.staged_applies = lambda *a, **k: True


def _gated(inputs, gate_up):
    gate, up = F.linear(inputs, gate_up).chunk(2, -1)
    return F.gelu(gate) * up


def _head(inputs, first, norm, last):
    hidden = first(inputs) if isinstance(first, torch.nn.Module) else F.linear(inputs, *first)
    return last(F.gelu(norm(hidden)))


def _staged(latent, age, ages, input_norm, first, norm, last):
    hidden = F.gelu(input_norm(latent[..., None, :] + age[ages]))
    return last(F.gelu(norm(first(hidden))))


def _scan(decay, contribution, initial):
    return temporal.parallel_scan(decay, contribution, initial)[0]


def _mix(pre, convolved, branch, starts, log_decay, initial, exponent, floor):
    recurrence, gate = torch.sigmoid(pre).chunk(2, -1)
    decay = torch.exp(exponent * recurrence * F.logsigmoid(log_decay))
    scale = torch.sqrt(torch.clamp_min((1.0 - decay) * (1.0 + decay), floor))
    contribution = scale * gate * convolved
    decay = torch.where(starts[..., None], 0.0, decay)
    states = temporal.parallel_scan(decay, contribution, initial)[0]
    return states, states * F.gelu(branch)


k_attention.attend = R.fused_autograd("attention", spatial.SpatialBlock._attend)
k_gated.gated_hidden = R.fused_autograd("gated MLP hidden", _gated)
k_heads.head = R.fused_autograd("head", _head)
k_heads.staged_head = R.fused_autograd("staged shot head", _staged)
k_scan.linear_scan = R.fused_autograd("scan", _scan)
k_temporal.temporal_mix = R.fused_autograd("temporal mix", _mix)

from bnb.learn import bc  # noqa: E402
from bnb.runtime import train as T  # noqa: E402
from bnb.runtime.lanes import LaneConfig  # noqa: E402

lineup = T.LINEUP if objective == "ppo" else importlib.import_module("bnb.runtime.cloning").LINEUP
game, pools = T.mixed(lineup, T.TRAINING_MIXES["real_scale_all_modes"])
learner_config = bc.config() if objective == "bc" else T.LearnerConfig()
config = T.TrainConfig(
    game=game,
    pools=pools,
    environments=256,
    length=length,
    shards=1,
    lookahead=length,
    lanes=LaneConfig(environments=128),
    host_store=False,
    compiled=False,
    objective=objective,
    learner=learner_config,
    lineup=("Yongma", "Yongma", "Jujak", "Jujak", "Jujak"),
)
trainer = T.Trainer(config, torch.device("cpu"))
learner = trainer.ppo
print("learner:", type(learner).__name__, "auxiliary:", learner.auxiliary)

# --- every function and method of the model and the learner is a region (exclusive) ---
prefixes = ("bnb.models.yemong", "bnb.learn", "bnb.models.heads")
skip = ("bnb.models.yemong.kernels",)
for info in list(pkgutil.walk_packages(importlib.import_module("bnb.models").__path__, "bnb.models.")) + list(
    pkgutil.walk_packages(importlib.import_module("bnb.learn").__path__, "bnb.learn.")
):
    try:
        importlib.import_module(info.name)
    except Exception:
        pass
wrapped = {}


def wrap(fn, label):
    if fn in wrapped:
        return wrapped[fn]

    @functools.wraps(fn)
    def w(*a, **k):
        R.STACK.append(label)
        R.CALLS[label] += 1
        try:
            return fn(*a, **k)
        finally:
            R.STACK.pop()

    wrapped[fn] = w
    return w


for m in [m for m in list(sys.modules.values()) if getattr(m, "__name__", "").startswith("bnb.")]:
    for attr, obj in list(vars(m).items()):
        mod = getattr(obj, "__module__", "") or ""
        if not mod.startswith(prefixes) or mod.startswith(skip):
            continue
        if inspect.isfunction(obj) and not hasattr(obj, "__wrapped__"):
            setattr(m, attr, wrap(obj, f"{mod.replace('bnb.', '')}.{obj.__qualname__}"))
        elif inspect.isclass(obj) and mod == m.__name__:
            for an, av in list(vars(obj).items()):
                if inspect.isfunction(av) and not (an.startswith("__") and an != "__call__"):
                    if issubclass(obj, torch.autograd.Function):
                        continue
                    setattr(obj, an, wrap(av, f"{mod.replace('bnb.', '')}.{obj.__qualname__}.{an}"))

# the learner object's bound stage functions were made at construction: rewrap them
for name in ("loss", "codes", "head_sums", "terms", "shot_tick"):
    fn = getattr(learner, name, None)
    if fn is not None and callable(fn):
        setattr(learner, name, wrap(fn, f"learner stage {name}"))


class Done(Exception):
    pass


PHASES = {}


BY_PHASE = {}


def phase(label, fn, *a, **k):
    before = {key: value[0] for key, value in R.EXCLUSIVE.items()}
    calls_before = dict(R.CALLS)
    R.STACK.append(label)
    R.CALLS[label] += 1
    try:
        return fn(*a, **k)
    finally:
        R.STACK.pop()
        BY_PHASE[label] = {
            key: (value[0] - before.get(key, 0), R.CALLS[key] - calls_before.get(key, 0))
            for key, value in R.EXCLUSIVE.items()
            if value[0] - before.get(key, 0) > 0
        }


def measured_update(segments, bootstrap, environment_steps=0, cadence=False, heads=True):
    with R.Counter():
        prepared = phase("phase: prepare (once per update)", learner.prepare, segments, bootstrap)
        views = segments.views
        order = torch.randperm(views)
        chosen = order.tensor_split(learner.config.minibatches)[0]
        sizes = learner.minibatch_sizes(len(prepared["counts"]))
        micro = learner._micro_batches(chosen, sizes if len(chosen) in sizes else ())
        derived = prepared["derived"]
        total = prepared["counts"][learner._on_device(chosen)].sum(0).clamp_min(1.0)
        denominator = dict(zip(type(learner).__mro__[0].__dict__.get("COUNTS", ()) or __import__("bnb.learn.ppo", fromlist=["COUNTS"]).COUNTS, total.unbind()))
        learner.optimizer.zero_grad(set_to_none=True)
        trajectory, shot_labels = next(iter(learner.staged(segments, micro[:1])))
        v = learner._on_device(micro[0])
        label = phase("phase: label codes (per micro-batch)", learner.labels, trajectory, shot_labels)
        part = {name: value.index_select(1, v) for name, value in derived.items()}
        from torch.utils.flop_counter import FlopCounterMode
        fc = FlopCounterMode(display=False)
        with fc:
            out = phase("phase: forward and loss (per micro-batch)", learner.loss, trajectory, label, part, denominator, prepared["scale"])
            forward_flops = fc.get_total_flops()
            phase("phase: backward (per micro-batch)", out[0].backward)
        PHASES["flops_forward"] = forward_flops
        PHASES["flops_total"] = fc.get_total_flops()
        PHASES["flops_by_module"] = {k: sum(v.values()) for k, v in fc.get_flop_counts().items()}
        PHASES["view_ticks"] = len(micro[0]) * trajectory.done.shape[0]
        PHASES["views_micro"] = len(micro[0])
        if out[5] is not None:
            with torch.no_grad():
                phase("phase: heads' sums (cadence only)", learner.head_sums, out[5], trajectory, label)

        def step():
            norm = torch.nn.utils.clip_grad_norm_(learner.model.parameters(), learner.config.max_grad_norm)
            learner._step(torch.isfinite(norm))

        phase("phase: clip and optimizer (per minibatch)", step)
        PHASES["micro_batches_in_first_minibatch"] = len(micro)
        PHASES["views"] = views
    raise Done


learner.update = measured_update
try:
    trainer.update()
except Done:
    pass

vt = PHASES["view_ticks"]
print(f"\nlearner {objective}, L {length}, micro-batch views {PHASES['views_micro']}, view-ticks {vt}")
print(f"FLOPs per view-tick: forward {PHASES['flops_forward']/vt/1e6:.1f} M, forward+backward {PHASES['flops_total']/vt/1e6:.1f} M")
mods = PHASES["flops_by_module"]
print("FLOPs per view-tick by module (top, M):")
for k, v in sorted(mods.items(), key=lambda r: -r[1])[:25]:
    print(f"   {v/vt/1e6:9.2f}  {k}")
print("\nBytes moved per view-tick, by phase (eager, unfused, KB):")
for label in sorted(k for k in R.BYTES if k.startswith("phase:")):
    print(f"   {R.BYTES[label]/vt/1e3:10.1f}  {label}")
fwd = "phase: forward and loss (per micro-batch)"
print("\nTop functions by bytes moved (exclusive, all phases, KB per view-tick):")
for k, v in sorted(R.BYTES_EX.items(), key=lambda r: -r[1])[:30]:
    print(f"   {v/vt/1e3:10.1f}  {k}")
