"""Exp 5: matched offline BC fitting with next_state_coef = 1 vs 0 (same frozen rollout)."""

import json
import sys
import time

import torch
import torch.nn.functional as F

import os as _os
S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, S)
from harness import _NullSnap, build  # noqa: E402

from boost_and_broadside.constants import POWER_SLICE, SHOOT_SLICE, TURN_SLICE  # noqa: E402
from boost_and_broadside.env.observation import ObsKey, YemongObservation  # noqa: E402

HEADS = {"power": POWER_SLICE, "turn": TURN_SLICE, "shoot": SHOOT_SLICE}


@torch.no_grad()
def eval_turn(trainer, buf, envs_per_chunk=4):
    B, N = buf.num_envs, buf.num_ships
    D = buf.initial_hidden.shape[-1]
    L = buf.initial_hidden.shape[0]
    hid_full = buf.initial_hidden.reshape(L, B, N, D)
    num = {k: 0.0 for k in HEADS}
    den = 0.0
    for s in range(0, B, envs_per_chunk):
        e = min(s + envs_per_chunk, B)
        T = buf.num_steps
        mb = YemongObservation(
            data={k: v[:, s:e] for k, v in buf.obs.items()},
            bullets=None
            if buf.bullet_obs is None
            else {k: v[:, s:e] for k, v in buf.bullet_obs.items()},
        )
        obs = mb.slice_time(0, T)
        hid = hid_full[:, s:e].reshape(L, (e - s) * N, D).contiguous()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, _, logits, _, _, _ = trainer.policy.evaluate_actions(
                obs=obs,
                actions=buf.actions[:, s:e].long(),
                initial_hidden=hid,
                alive_mask=obs[ObsKey.BELIEF_VALID].bool(),
                done_mask=buf.terminated[:, s:e],
            )
        logits = logits.float()
        exp = buf.expert_probs[:, s:e].float()
        m = ((exp.sum(-1) > 0) & buf.actor_masks[:, s:e] & buf.alive_mask[:, s:e]).float()
        for name, sl in HEADS.items():
            p = exp[..., sl].clamp(min=1e-8)
            logq = F.log_softmax(logits[..., sl], dim=-1)
            kl = -(exp[..., sl] * logq).sum(-1) + (p * p.log()).sum(-1)
            num[name] += float((kl * m).sum())
        den += float(m.sum())
    return {k: v / den for k, v in num.items()} | {"den": den}


CALLS = 3
out = {}
for ns in (1.0, 0.0):
    torch.manual_seed(0)
    trainer, _ = build(num_envs=128, microbatch_tokens=12288, seed=0)
    runtime = trainer._initialize_rollout_runtime()
    runtime.elo_eval.step = lambda *a, **k: None
    runtime.elo_eval.flush = lambda *a, **k: _NullSnap()
    for _ in range(2):
        term = trainer._collect_rollout(runtime, False)
    trainer._compute_rollout_gae(runtime, term)
    object.__setattr__(trainer.cfg, "next_state_coef", ns)
    curve = [round(eval_turn(trainer, trainer.buffer)["turn"], 4)]
    t0 = time.time()
    for e in range(CALLS):
        trainer._update_epochs(all_buffers=[trainer.buffer, *trainer.aux_buffers], update=e + 1)
        curve.append(round(eval_turn(trainer, trainer.buffer)["turn"], 4))
    trainer._collect_rollout(runtime, False)
    out[f"ns={ns}"] = dict(
        train_turn_kl=curve,
        fresh_rollout_turn_kl=round(eval_turn(trainer, trainer.buffer)["turn"], 4),
        epochs_per_call=trainer._schedule_state.num_epochs,
        seconds=round(time.time() - t0, 1),
    )
    print(json.dumps({f"ns={ns}": out[f"ns={ns}"]}), flush=True)
    del trainer
    torch.cuda.empty_cache()
json.dump(out, open(f"{S}/exp5_ns.json", "w"), indent=1)
