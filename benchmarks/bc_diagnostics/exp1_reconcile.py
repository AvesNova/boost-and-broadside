"""Exp 1: reconcile BC KL across evaluation paths on identical stored samples."""

import os as _os
import sys, time, json

S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
import torch
import torch.nn.functional as F

sys.path.insert(0, S)
from harness import build, CKPT
from boost_and_broadside.constants import POWER_SLICE, TURN_SLICE, SHOOT_SLICE
from boost_and_broadside.env.observation import ObsKey

HEADS = {"power": POWER_SLICE, "turn": TURN_SLICE, "shoot": SHOOT_SLICE}


def kl_terms(expert, logits):
    """Per-token per-head KL(teacher||policy), CE and teacher entropy. fp32 in."""
    out = {}
    for name, sl in HEADS.items():
        p = expert[..., sl].clamp(min=1e-8)
        logq = F.log_softmax(logits[..., sl].float(), dim=-1)
        ce = -(expert[..., sl] * logq).sum(-1)
        ent = -(p * p.log()).sum(-1)
        out[name] = (ce - ent, ce, ent)
    return out


@torch.no_grad()
def evaluate(trainer, buf, *, dtype: str, hidden_mode: str, num_minibatches: int, mbt: int):
    """Sum KL numerators and denominators over the whole rollout."""
    pol = trainer.policy
    acc = {k: 0.0 for k in ("num_power", "num_turn", "num_shoot", "ce", "ent", "den")}
    per_env = []
    for chunks in buf.get_minibatch_iterator(num_minibatches, mbt):
        for ch in chunks:
            T = ch.alive.shape[0]
            obs = ch.obs.slice_time(0, T)
            alive_full = obs[ObsKey.BELIEF_VALID].bool()
            hid = ch.hidden
            if hidden_mode == "zero":
                hid = torch.zeros_like(hid)
            ctx = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if dtype == "bf16"
                else torch.autocast("cuda", enabled=False)
            )
            with ctx:
                _, _, _, logits, _, _, _ = pol.evaluate_actions(
                    obs=obs,
                    actions=ch.actions.long(),
                    initial_hidden=hid,
                    alive_mask=alive_full,
                    done_mask=ch.terminated,
                )
            logits = logits.float()
            exp = ch.expert_probs.float()
            bc_valid = exp.sum(-1) > 0
            m = (bc_valid & ch.actor_mask & ch.alive).float()
            terms = kl_terms(exp, logits)
            for name in HEADS:
                acc[f"num_{name}"] += float((terms[name][0] * m).sum())
            acc["ce"] += float(sum((terms[n][1] * m).sum() for n in HEADS))
            acc["ent"] += float(sum((terms[n][2] * m).sum() for n in HEADS))
            acc["den"] += float(m.sum())
    return acc


def report(tag, acc):
    d = max(acc["den"], 1.0)
    tot = sum(acc[f"num_{n}"] for n in HEADS)
    print(
        f"{tag:34s} den={acc['den']:10.0f} kl_tot={tot/d:.4f} "
        f"power={acc['num_power']/d:.5f} turn={acc['num_turn']/d:.4f} "
        f"shoot={acc['num_shoot']/d:.5f} ce={acc['ce']/d:.4f} H_teacher={acc['ent']/d:.4f}"
    )
    return tot / d


def main():
    num_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    t0 = time.time()
    trainer, launch = build(num_envs=num_envs, microbatch_tokens=12288, seed=0)
    print(f"built in {time.time()-t0:.1f}s  num_envs={num_envs} "
          f"microbatch_tokens=12288 num_minibatches={trainer.cfg.num_minibatches}")
    runtime = trainer._initialize_rollout_runtime()
    runtime.elo_eval.step = lambda *a, **k: None
    from harness import _NullSnap
    runtime.elo_eval.flush = lambda *a, **k: _NullSnap()

    results = {}
    for r in range(3):
        t1 = time.time()
        term = trainer._collect_rollout(runtime, False)
        trainer._compute_rollout_gae(runtime, term) if hasattr(trainer, "_compute_rollout_gae") else None
        dt = time.time() - t1
        buf = trainer.buffer
        print(f"\n--- rollout {r}  ({dt:.1f}s collect) ---")
        a1 = evaluate(trainer, buf, dtype="bf16", hidden_mode="stored", num_minibatches=1, mbt=12288)
        results[f"r{r}_bf16_stored"] = report("bf16 / stored hidden", a1)
        a2 = evaluate(trainer, buf, dtype="fp32", hidden_mode="stored", num_minibatches=1, mbt=12288)
        results[f"r{r}_fp32_stored"] = report("fp32 / stored hidden", a2)
        a3 = evaluate(trainer, buf, dtype="fp32", hidden_mode="zero", num_minibatches=1, mbt=12288)
        results[f"r{r}_fp32_zero"] = report("fp32 / ZERO hidden", a3)
        # per-env-shard breakdown (8 shards) with correct num/den aggregation
        shard = []
        for chunks in buf.get_minibatch_iterator(8, None):
            acc = {k: 0.0 for k in ("num_power", "num_turn", "num_shoot", "ce", "ent", "den")}
            for ch in chunks:
                T = ch.alive.shape[0]
                obs = ch.obs.slice_time(0, T)
                with torch.no_grad(), torch.autocast("cuda", enabled=False):
                    _, _, _, logits, _, _, _ = trainer.policy.evaluate_actions(
                        obs=obs, actions=ch.actions.long(), initial_hidden=ch.hidden,
                        alive_mask=obs[ObsKey.BELIEF_VALID].bool(), done_mask=ch.terminated)
                exp = ch.expert_probs.float()
                m = ((exp.sum(-1) > 0) & ch.actor_mask & ch.alive).float()
                terms = kl_terms(exp, logits.float())
                for name in HEADS:
                    acc[f"num_{name}"] += float((terms[name][0] * m).sum())
                acc["den"] += float(m.sum())
            shard.append((acc["den"], sum(acc[f"num_{n}"] for n in HEADS) / max(acc["den"], 1),
                          acc["num_turn"] / max(acc["den"], 1)))
        print("  shards (den, kl_tot, kl_turn): " +
              " ".join(f"({d:.0f},{k:.3f},{t:.3f})" for d, k, t in shard))
    print(json.dumps(results, indent=1))


main()
