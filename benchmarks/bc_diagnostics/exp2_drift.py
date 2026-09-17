"""Exp 2: KL vs rollout index (distribution drift) + per-token dump for turn analysis."""

import sys, time, json
import torch
import torch.nn.functional as F

import os as _os
S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, S)
from harness import build, _NullSnap
from boost_and_broadside.constants import POWER_SLICE, TURN_SLICE, SHOOT_SLICE
from boost_and_broadside.env.observation import ObsKey

HEADS = {"power": POWER_SLICE, "turn": TURN_SLICE, "shoot": SHOOT_SLICE}


@torch.no_grad()
def eval_ordered(trainer, buf, envs_per_chunk=4, dump=False):
    """Evaluate the whole rollout in env order; sum KL numerators/denominators."""
    B = buf.num_envs
    N = buf.num_ships
    D = buf.initial_hidden.shape[-1]
    n_layers = buf.initial_hidden.shape[0]
    hid_full = buf.initial_hidden.reshape(n_layers, B, N, D)
    acc = {k: 0.0 for k in ("power", "turn", "shoot", "den", "ent", "ce")}
    turn_logp, keep = [], []
    for s in range(0, B, envs_per_chunk):
        e = min(s + envs_per_chunk, B)
        T = buf.num_steps
        obs_all = type(buf.obs)  # dict
        from boost_and_broadside.env.observation import YemongObservation
        mb_obs = YemongObservation(
            data={k: v[:, s:e] for k, v in buf.obs.items()},
            bullets=None if buf.bullet_obs is None else {k: v[:, s:e] for k, v in buf.bullet_obs.items()},
        )
        obs = mb_obs.slice_time(0, T)
        hid = hid_full[:, s:e].reshape(n_layers, (e - s) * N, D).contiguous()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, _, logits, _, _, _ = trainer.policy.evaluate_actions(
                obs=obs, actions=buf.actions[:, s:e].long(), initial_hidden=hid,
                alive_mask=obs[ObsKey.BELIEF_VALID].bool(), done_mask=buf.terminated[:, s:e])
        logits = logits.float()
        exp = buf.expert_probs[:, s:e].float()
        m = ((exp.sum(-1) > 0) & buf.actor_masks[:, s:e] & buf.alive_mask[:, s:e]).float()
        for name, sl in HEADS.items():
            p = exp[..., sl].clamp(min=1e-8)
            logq = F.log_softmax(logits[..., sl], dim=-1)
            ce = -(exp[..., sl] * logq).sum(-1)
            ent = -(p * p.log()).sum(-1)
            acc[name] += float(((ce - ent) * m).sum())
            acc["ce"] += float((ce * m).sum())
            acc["ent"] += float((ent * m).sum())
        acc["den"] += float(m.sum())
        if dump:
            turn_logp.append(F.log_softmax(logits[..., TURN_SLICE], dim=-1).cpu())
            keep.append(m.bool().cpu())
    if dump:
        return acc, torch.cat(turn_logp, 1), torch.cat(keep, 1)
    return acc, None, None


def main():
    n_roll = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    num_envs = 128
    trainer, launch = build(num_envs=num_envs, microbatch_tokens=12288, seed=0, recording=True)
    runtime = trainer._initialize_rollout_runtime()
    runtime.elo_eval.step = lambda *a, **k: None
    runtime.elo_eval.flush = lambda *a, **k: _NullSnap()
    rows = []
    for r in range(n_roll):
        last = r == n_roll - 1
        trainer.scripted_agent.records = []
        trainer.scripted_agent.recording = last or r == 0
        t0 = time.time()
        trainer._collect_rollout(runtime, False)
        tc = time.time() - t0
        acc, tlp, keep = eval_ordered(trainer, trainer.buffer, dump=last)
        d = acc["den"]
        row = dict(rollout=r, den=d, kl=(acc['power']+acc['turn']+acc['shoot'])/d,
                   turn=acc["turn"]/d, power=acc["power"]/d, shoot=acc["shoot"]/d,
                   H=acc["ent"]/d, collect_s=round(tc, 1))
        recs = trainer.scripted_agent.records
        if recs:
            cd = torch.stack([x["closest_dist"] for x in recs])
            row["mean_closest_dist"] = float(cd[cd < 1e6].mean())
            row["p_team"] = float(torch.stack([x["p_team"] for x in recs]).mean())
            row["alpha"] = float(torch.stack([x["alpha"] for x in recs]).mean())
        rows.append(row)
        print(json.dumps(row))
        if last:
            keys = list(recs[0].keys())
            stacked = {k: torch.stack([x[k] for x in recs]) for k in keys}
            torch.save(dict(
                expert=trainer.buffer.expert_probs.float().cpu(),
                turn_logp=tlp, keep=keep,
                alive=trainer.buffer.alive_mask.cpu(),
                actor=trainer.buffer.actor_masks.cpu(),
                terminated=trainer.buffer.terminated.cpu(),
                teacher=stacked,
            ), f"{S}/dump_rollout{r}.pt")
            print("dumped", f"{S}/dump_rollout{r}.pt")
    json.dump(rows, open(f"{S}/exp2_rows.json", "w"), indent=1)


main()
