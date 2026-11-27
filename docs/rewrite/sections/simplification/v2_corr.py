"""v2: ops per training rollout tick, by section, with the GPU's fused kernels counted as one op.

usage: python v2_ops.py V2_CHECKOUT [ppo|bc] [environments] [lane_environments]
"""
# ruff: noqa: E402  (a design-record script, kept as written)

import sys

sys.path.insert(0, sys.argv[1] + "/src")
sys.path.insert(0, ".")
import regions as R
import torch
import torch.nn.functional as F

objective = sys.argv[2] if len(sys.argv) > 2 else "ppo"
environments = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
lane_environments = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
torch.set_num_threads(4)

from bnb.models.yemong import spatial  # noqa: E402
from bnb.models.yemong.kernels import attention as k_attention  # noqa: E402
from bnb.models.yemong.kernels import gated as k_gated  # noqa: E402
from bnb.models.yemong.kernels import heads as k_heads  # noqa: E402

# --- the GPU path: each kernel applies, and counts as one op (its plain twin computes) ---
k_attention.applies = lambda *a, **k: True
k_gated.applies = lambda *a, **k: True
k_heads.applies = lambda *a, **k: True
k_heads.staged_applies = lambda *a, **k: False  # the learner's shot query only
k_attention.attend = R.fused("attention", spatial.SpatialBlock._attend)


def _gated(inputs, gate_up):
    gate, up = F.linear(inputs, gate_up).chunk(2, -1)
    return F.gelu(gate) * up


def _head(inputs, first, norm, last):
    hidden = first(inputs) if isinstance(first, torch.nn.Module) else F.linear(inputs, *first)
    return last(F.gelu(norm(hidden)))


k_gated.gated_hidden = R.fused("gated MLP hidden", _gated)
k_heads.head = R.fused("head (linear-norm-gelu-linear)", _head)

from bnb.controllers import yemong as ycontroller  # noqa: E402
from bnb.game import api  # noqa: E402
from bnb.learn import bc  # noqa: E402
from bnb.models.yemong import belief, model, network, temporal  # noqa: E402
from bnb.models.yemong.heads import HeadGroup  # noqa: E402
from bnb.runtime import rollout as rollout_module  # noqa: E402
from bnb.runtime import train as T  # noqa: E402
from bnb.runtime.lanes import LaneConfig  # noqa: E402

# --- the launch's config (docs/rewrite/sections/launch/launch.toml) ---
game, pools = T.mixed(
    T.LINEUP
    if objective == "ppo"
    else __import__("bnb.runtime.cloning", fromlist=["LINEUP"]).LINEUP,
    T.TRAINING_MIXES["real_scale_all_modes"],
)
config = T.TrainConfig(
    game=game,
    pools=pools,
    environments=environments,
    length=8,
    shards=1,
    lookahead=0,
    lanes=LaneConfig(environments=lane_environments)
    if lane_environments
    else LaneConfig(environments=0),
    host_store=False,
    compiled=False,
    objective=objective,
    learner=bc.config() if objective == "bc" else T.LearnerConfig(),
    lineup=("Yongma", "Yongma", "Jujak", "Jujak", "Jujak"),
)
trainer = T.Trainer(config, torch.device("cpu"))
rollout = trainer.rollout
print("controllers:", {name: type(c).__name__ for name, c in rollout.controllers.items()})


# --- regions ---
def _staged(function, label):
    def run(carry):
        R.STACK.append(label)
        R.CALLS[label] += 1
        try:
            return function(carry)
        finally:
            R.STACK.pop()

    return run


rollout.stages = tuple(_staged(f, f"stage {f.__name__}") for f in rollout.stages)
for c in {type(c) for c in rollout.controllers.values()}:
    R.region(c, "step", f"controller {c.__name__} (outside the model)")
import importlib
import inspect
import pkgutil

import bnb.controllers as _ctrl

for _m in pkgutil.iter_modules(_ctrl.__path__):
    _mod = importlib.import_module(f"bnb.controllers.{_m.name}")
    for _n, _c in inspect.getmembers(_mod, inspect.isclass):
        if (
            _c.__module__ == _mod.__name__
            and "step" in _c.__dict__
            and _c is not ycontroller.YemongController
        ):
            R.region(_c, "step", f"scripted/other controller: {_n}")
R.region(api, "move", "game: move (flight physics, bullets)")
R.region(api, "launch_capture", "game: shot capture for labels")
R.region(api, "resolve", "game: resolve (hits, damage, deaths, mode rules)")
R.region(api, "perceive", "game: perceive (observation, fog)")
R.region(rollout_module.rewards, "reward_step_events", "rewards")
R.region(rollout_module.analytics, "resolve_tick", "analytics (resolve)")
R.region(rollout_module.analytics, "perceive_tick", "analytics (perceive)")
Y = model.Yemong
R.region(Y, "_step", "model: other (step glue)")
R.region(Y, "_cleared", "model: carry reset on new episode")
R.region(belief, "observe", "model: belief update from sightings")
R.region(Y, "tokens", "model: input features and codes (tokens)")
R.region(Y, "_encode", "model: input encoder (codes -> embeddings)")
R.region(network.Trunk, "step", "model: trunk (other)")
R.region(spatial.SpatialBlock, "forward", "model: trunk spatial attention blocks")
R.region(temporal.TemporalBlock, "step", "model: trunk temporal (RG-LRU) blocks")
R.region(HeadGroup, "forward", "model: output heads (MLPs)")
R.region(Y, "_outcome_logits", "model: output heads (MLPs)")
R.region(Y, "_values", "model: value decode")
R.region(Y, "_next_state_logits", "model: next-state decode")
R.region(Y, "rates", "model: next-state decode")
R.region(Y, "forecast", "model: belief forecast")
R.region(belief, "advance", "model: belief forecast")
from bnb.models.yemong import correction as ycorrection
from bnb.models.yemong import delta as ydelta

R.region(ydelta.DeltaCode, "advance", "model: forecast: delta-code decode (physics head)")
R.region(
    ycorrection.CorrectionCode, "correct", "model: forecast: correction-code decode (error head)"
)
R.region(Y, "_faded", "model: forecast: hidden-row fade")
from bnb.models.yemong import codes as ycodes

C = ycorrection.CorrectionCode
R.region(C, "probabilities", "corr: softmax over each code's bins")
R.region(ycodes, "symlog_median", "corr: median per component")
R.region(ycodes, "symlog_spread", "corr: quantile spread per component")
R.region(C, "decode", "corr: decode loop (other)")
R.region(C, "moments", "corr: combine axes, confidence, step caps")
R.region(C, "_cooldown_code", "corr: cooldown code")
R.region(C, "correct", "corr: apply step, covariance, clamps")
R.region(ycontroller.policy, "sample", "model: action sampling")
R.region(ycontroller, "action_uniforms", "model: action sampling")

with torch.no_grad():
    rollout._tick()
    R.reset()
    with R.Counter():
        rollout._tick()
calls = R.CALLS["corr: apply step, covariance, clamps"]
print(
    "correct() calls:",
    calls,
    " components per decode:",
    len(
        next(
            iter([c for c in rollout.controllers.values() if hasattr(c, "model")])
        ).model.correction_code.components
    ),
)
for k in sorted(R.INCLUSIVE):
    if k.startswith("corr:"):
        print(
            f"  {k:<50} exclusive per call {R.EXCLUSIVE[k][0] / calls:>6.0f}   inclusive {R.INCLUSIVE[k][0] / calls:>6.0f}   entered {R.CALLS[k] / calls:.0f}x per call"
        )
for k in ("corr: median per component", "corr: quantile spread per component"):
    print(k, R.OPS[k].most_common(8))
