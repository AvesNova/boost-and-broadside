"""v1: ops of the learner per phase (precompute once per update; one micro-batch's forward and
loss; its backward; clip and optimizer), by function.

usage: python v1_learn.py V1_CHECKOUT [num_steps]
"""
import dataclasses, functools, importlib, inspect, sys, tempfile
sys.path.insert(0, sys.argv[1] + "/src"); sys.path.insert(0, ".")
import torch
import regions as R
num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 16
num_envs = 288
torch.set_num_threads(4)
from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config.resolve import LaunchOverrides, resolve_profile
from boost_and_broadside.profiles import PROFILES
from boost_and_broadside.train.rl.ppo import PPOTrainer
base = PROFILES["rl"]
tokens = num_envs * (base.num_ships + base.num_fields) * num_steps
profile = dataclasses.replace(base, checkpoint_dir=tempfile.mkdtemp(), num_steps=num_steps,
                              logical_batch_tokens=tokens, launch=dataclasses.replace(base.launch, rollout_tokens=tokens),
                              elo_eval=dataclasses.replace(base.elo_eval, envs_per_matchup=16))
minibatch = tokens // profile.num_minibatches
resolved = resolve_profile(profile, LaunchOverrides(num_envs=num_envs, microbatch_tokens=minibatch))
print("rollouts per update", resolved.train_config.rollouts_per_update, "minibatch tokens", minibatch)
trainer = PPOTrainer(train_config=resolved.train_config, model_config=resolved.model_config,
                     ship_config=resolved.ship_config, device="cpu", use_wandb=False,
                     scripted_agent=StochasticScriptedAgent(resolved.ship_config, StochasticAgentConfig()),
                     compile_mode=None)
# every function and method of the model and the PPO code is a region
wrapped = {}
def wrap(fn, label):
    if fn in wrapped: return wrapped[fn]
    @functools.wraps(fn)
    def w(*a, **k):
        R.STACK.append(label); R.CALLS[label] += 1
        try: return fn(*a, **k)
        finally: R.STACK.pop()
    wrapped[fn] = w; return w
prefixes = ("boost_and_broadside.models", "boost_and_broadside.train")
for m in [m for m in list(sys.modules.values()) if getattr(m, "__name__", "").startswith("boost_and_broadside.")]:
    for attr, obj in list(vars(m).items()):
        mod = getattr(obj, "__module__", "") or ""
        if not mod.startswith(prefixes): continue
        short = mod.replace("boost_and_broadside.", "")
        if inspect.isfunction(obj): setattr(m, attr, wrap(obj, f"{short}.{obj.__qualname__}"))
        elif inspect.isclass(obj) and mod == m.__name__ and obj is not PPOTrainer:
            for an, av in list(vars(obj).items()):
                if inspect.isfunction(av) and not an.startswith("__"):
                    setattr(obj, an, wrap(av, f"{short}.{obj.__qualname__}.{an}"))
class Done(Exception): pass
BY_PHASE = {}
INFO = {}
def phase(label, fn, *a, **k):
    before = {key: v[0] for key, v in R.EXCLUSIVE.items()}; cb = dict(R.CALLS)
    R.STACK.append(label); R.CALLS[label] += 1
    try: return fn(*a, **k)
    finally:
        R.STACK.pop()
        BY_PHASE[label] = {key: (v[0] - before.get(key, 0), R.CALLS[key] - cb.get(key, 0))
                           for key, v in R.EXCLUSIVE.items() if v[0] - before.get(key, 0) > 0}
original_loss = PPOTrainer._compute_minibatch_loss
state = {"started": False}
def counted_loss(self, *a, **k):
    BY_PHASE["phase: prepare (once per update)"] = {key: (v[0], R.CALLS[key]) for key, v in R.EXCLUSIVE.items() if v[0]}
    from torch.utils.flop_counter import FlopCounterMode
    fc = FlopCounterMode(display=False)
    with fc:
        loss, diag = phase("phase: forward and loss (per micro-batch)", original_loss, self, *a, **k)
        fwd = fc.get_total_flops()
        phase("phase: backward (per micro-batch)", loss.backward)
    chunk = a[0]
    INFO.update(flops_forward=fwd, flops_total=fc.get_total_flops(), envs=chunk.alive.shape[1], ticks=chunk.alive.shape[0],
                by_module={k: sum(v.values()) for k, v in fc.get_flop_counts().items()})
    def step():
        params = list(self._policy_module.parameters())
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        for p in params:
            if p.grad is not None: torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        self.optim.step()
    phase("phase: clip and optimizer (per minibatch)", step)
    raise Done
PPOTrainer._compute_minibatch_loss = counted_loss
runtime = trainer._initialize_rollout_runtime()
with torch.no_grad():
    terminated = trainer._collect_rollout(runtime, False)
    trainer._compute_rollout_gae(runtime, terminated)
R.reset()
try:
    with R.Counter():
        trainer._update_epochs(all_buffers=[trainer.buffer, *trainer.aux_buffers])
except Done:
    pass
vt = INFO["envs"] * INFO["ticks"]
print(f"\nv1 learner, num_steps {num_steps}, micro-batch envs {INFO['envs']}, env-ticks {vt}")
print(f"FLOPs per env-tick: forward {INFO['flops_forward']/vt/1e6:.1f} M, forward+backward {INFO['flops_total']/vt/1e6:.1f} M")
for k, v in sorted(INFO["by_module"].items(), key=lambda r: -r[1])[:12]:
    print(f"   {v/vt/1e6:9.2f}  {k}")
print("Bytes moved per env-tick by phase (eager, KB):")
for label in sorted(k for k in R.BYTES if k.startswith("phase:")):
    print(f"   {R.BYTES[label]/vt/1e3:10.1f}  {label}")
print("Prepare (once per update), whole: %.1f KB per env-tick of the update" % 0)
