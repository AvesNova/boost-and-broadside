"""Exp 4: BC vs next-state trunk gradient relationship at the checkpoint."""

import sys, json, torch
import os as _os
S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, S)
from harness import _NullSnap, CKPT
from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config.diagnostics import GradientDiagnosticsConfig
from boost_and_broadside.launch import resolve_training_launch
from boost_and_broadside.train.rl.ppo import PPOTrainer

launch = resolve_training_launch(profile="bc", vram="off", device="cuda", seed=0,
                                 compile_mode=None, wandb=False, num_envs=128,
                                 microbatch_tokens=12288, allow_probe=False)
r = launch.resolved
trainer = PPOTrainer(train_config=r.train_config, model_config=r.model_config,
                     ship_config=r.ship_config, device="cuda", use_wandb=False,
                     scripted_agent=StochasticScriptedAgent(r.ship_config, StochasticAgentConfig()),
                     compile_mode=None,
                     gradient_diagnostics=GradientDiagnosticsConfig(level="top_level", interval=1,
                                                                    minibatches=4))
trainer.load_checkpoint(CKPT)
captured = []
orig = trainer._gradient_diagnostic_metrics
def patched(acc, seconds):
    out = orig(acc, seconds)
    captured.append(out)
    return out
trainer._gradient_diagnostic_metrics = patched

runtime = trainer._initialize_rollout_runtime()
runtime.elo_eval.step = lambda *a, **k: None
runtime.elo_eval.flush = lambda *a, **k: _NullSnap()
# burn in a few rollouts so the env distribution is the steady state, then diagnose
for i in range(3):
    term = trainer._collect_rollout(runtime, False)
trainer._compute_rollout_gae(runtime, term)
trainer._update_epochs(all_buffers=[trainer.buffer, *trainer.aux_buffers], update=1)
keys = ["grad_norm", "cos", "share"]
for i, rec in enumerate(captured):
    sel = {k: round(v, 5) for k, v in rec.items()
           if ("bc" in k or "next_state" in k) and ("trunk_top_level" in k or "top_level" in k)}
    print(f"--- diagnosed minibatch {i} ---")
    for k in sorted(sel):
        print(f"  {k:60s} {sel[k]}")
json.dump(captured, open(f"{S}/exp4_grad.json", "w"), indent=1)
