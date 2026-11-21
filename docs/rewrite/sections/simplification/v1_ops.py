"""v1: ops per training rollout step (one decision), by section.

usage: python v1_ops.py V1_CHECKOUT [num_envs] [elo_envs_per_matchup]
"""

import dataclasses
import sys
import tempfile

sys.path.insert(0, sys.argv[1] + "/src")
sys.path.insert(0, ".")
import torch

import regions as R

num_envs = int(sys.argv[2]) if len(sys.argv) > 2 else 256
elo_envs = int(sys.argv[3]) if len(sys.argv) > 3 else 32
torch.set_num_threads(4)

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig  # noqa: E402
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent  # noqa: E402
from boost_and_broadside.config.resolve import LaunchOverrides, resolve_profile  # noqa: E402
from boost_and_broadside.env import env as env_module  # noqa: E402
from boost_and_broadside.env import wrapper as wrapper_module  # noqa: E402
from boost_and_broadside.models.yemong import attention, encoder, griffin  # noqa: E402
from boost_and_broadside.models.yemong import policy as policy_module  # noqa: E402
from boost_and_broadside.profiles import PROFILES  # noqa: E402
from boost_and_broadside.train.rl import elo_eval, opponents  # noqa: E402
from boost_and_broadside.train.rl.ppo import PPOTrainer  # noqa: E402

base = PROFILES["rl"]
profile = dataclasses.replace(
    base,
    checkpoint_dir=tempfile.mkdtemp(),
    elo_eval=dataclasses.replace(base.elo_eval, envs_per_matchup=elo_envs),
)
resolved = resolve_profile(profile, LaunchOverrides(num_envs=num_envs, microbatch_tokens=4608))
print("profile rl:", resolved.env_config, "model", resolved.model_config)
trainer = PPOTrainer(
    train_config=resolved.train_config,
    model_config=resolved.model_config,
    ship_config=resolved.ship_config,
    device="cpu",
    use_wandb=False,
    scripted_agent=StochasticScriptedAgent(resolved.ship_config, StochasticAgentConfig()),
    compile_mode=None,
)
runtime = trainer._initialize_rollout_runtime()
slots = trainer._prepare_league_slots(runtime.num_recurrent)

# --- regions ---
P = policy_module.YemongPolicy
R.region(P, "get_action_and_value", "model: other (step glue)")
R.region(encoder.ShipEncoder, "forward", "model: input features and encoder (ships, fields)")
R.region(P, "_encode_bullets", "model: input features and encoder (bullets)")
R.region(griffin.YemongBlock, "step", "model: trunk (other)")
R.region(attention.TransformerBlock, "forward", "model: trunk spatial attention blocks")
R.region(griffin.GriffinTemporalBlock, "forward_sequence", "model: trunk temporal (RG-LRU) blocks")
R.region(griffin.GriffinTemporalBlock, "forward_nonrecurrent", "model: trunk temporal (RG-LRU) blocks")
R.region(policy_module, "_sample_action", "model: action sampling")
for head in ("action_head", "next_state_head", "value_head_local", "value_head_win", "team_pma"):
    pass  # instance modules: hooks below


def hook_heads(policy):
    module = getattr(policy, "_orig_mod", policy)
    for head in ("action_head", "next_state_head", "value_head_local", "value_head_win", "team_pma"):
        m = getattr(module, head, None)
        if m is None:
            continue
        label = "model: output heads (MLPs, team pooling)"
        m.register_forward_pre_hook(lambda *_a, label=label: (R.STACK.append(label), R.CALLS.__setitem__(label, R.CALLS[label] + 1)) and None)
        m.register_forward_hook(lambda *_a: R.STACK.pop() and None)


hooked = set()
for p in [trainer.policy, trainer.avg_policy] + [getattr(s, "policy", None) for s in (slots or [])]:
    if p is not None and id(p) not in hooked:
        hook_heads(p); hooked.add(id(p))

R.region(StochasticScriptedAgent, "get_actions_and_probs", "scripted agent")
R.region(StochasticScriptedAgent, "get_actions", "scripted agent")
R.region(wrapper_module.YemongEnvWrapper, "step", "env: wrapper step (other)")
R.region(wrapper_module.YemongEnvWrapper, "_physics_tick", "env: rewards and episode stats (per physics tick)")
R.region(env_module.TensorEnv, "tick", "env: tick (other)")
R.region(env_module, "update_ships", "game: ship physics (flight, fields)")
R.region(env_module, "advance_bullets", "game: bullets")
R.region(env_module, "resolve_collisions", "game: collisions, damage, deaths")
R.region(env_module.TensorEnv, "reset_envs", "env: resets")
R.region(wrapper_module.YemongEnvWrapper, "_get_obs", "game: observation")
R.region(wrapper_module.YemongEnvWrapper, "_refresh_field_obs", "game: observation")
R.region(elo_eval.EloEvaluator, "step", "elo evaluation games (other)")
R.region(type(trainer), "_collect_primary_step", "training step (other)")
R.region(type(trainer), "_collect_aux_steps", "aux games (other)")


def one_step():
    primary = trainer._collect_primary_step(
        obs=runtime.obs, hidden=runtime.hidden, hidden_t1=runtime.hidden_t1,
        action_buffer=runtime.action_buffer, num_envs=runtime.num_envs,
        num_ships=runtime.num_ships, num_recurrent=runtime.num_recurrent, slots=slots,
        env_stream=runtime.env_stream, net_stream=runtime.net_stream,
    )
    runtime.obs, runtime.hidden, runtime.hidden_t1, runtime.action_buffer, _ = primary
    trainer._collect_aux_steps(runtime.aux_obs, runtime.aux_hiddens, runtime.aux_hidden_t1s,
                               runtime.aux_action_buffers, runtime.aux_last_dones)
    runtime.elo_eval.step(0, True)


with torch.no_grad():
    one_step()
    R.reset()
    with R.Counter():
        one_step()

R.report(f"v1 rl rollout step, {num_envs} environments, elo {elo_envs} per matchup")
print("\nSections, inclusive, total in the step / calls / per call:")
for label in sorted(R.INCLUSIVE):
    n = max(R.CALLS[label], 1)
    print(f"  {label:<60} {R.INCLUSIVE[label][0]:>6} / {R.CALLS[label]:>3} = {R.INCLUSIVE[label][0] / n:>7.0f}")
for label in ("model: input features and encoder (ships, fields)", "game: collisions, damage, deaths", "env: rewards and episode stats (per physics tick)"):
    print(f"\nTop ops in {label}:")
    for name, n in R.OPS[label].most_common(8):
        print(f"   {n:>5} {name}")
