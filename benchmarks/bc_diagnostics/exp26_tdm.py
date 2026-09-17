"""Exp 26: fit the legacy teacher in the legacy ENVIRONMENT, not just on legacy labels.

Exp 25 compared the two teachers on frontline states and found them equally hard
to fit -- but that comparison was contaminated: with zones present, ~43% of
tokens have no visible enemy, and the legacy dogfighter degenerates to "go
straight" there, which is why its entropy was 0.25 against the frontline
teacher's 0.47.

The reported history is that BC closed against the pre-zones TDM agent. That
agent is what `EnvConfig.frontline = None` selects, in the environment it was
written for: no zones, 18 entity tokens instead of 24, and
`get_actions_and_probs` taking its legacy branch. This runs the frozen
checkpoint there and fits the same turn head on the same frozen latent.

Caveat that cannot be removed cheaply: the checkpoint was *trained* in the
frontline environment, so its latent is out of distribution here. A low KL would
therefore be strong evidence (the target is easy despite an unsuited latent); a
high KL would be weak evidence, since it could be the distribution shift rather
than the target.
"""

import dataclasses
import json
import os as _os
import sys
import time

import torch

S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, S)
from exp8_oracle import fit_head, kl  # noqa: E402

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig  # noqa: E402
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent  # noqa: E402
from boost_and_broadside.constants import TURN_SLICE  # noqa: E402
from boost_and_broadside.env.observation import ObsKey, YemongObservation  # noqa: E402
from boost_and_broadside.launch import resolve_training_launch  # noqa: E402
from boost_and_broadside.train.rl.ppo import PPOTrainer  # noqa: E402

CKPT = "checkpoints/icy-energy-741/step_000167608320.pt"
TSTRIDE = 4


class _Null:
    live_elo = torch.tensor(0.0)
    avg_elo = torch.tensor(0.0)
    floating_games = 0
    match_counts = {}
    floating_label = None
    floating_elo = None
    ladder_counts = {}


def build_tdm(seed):
    torch.manual_seed(seed)
    launch = resolve_training_launch(profile="bc", vram="off", device="cuda", seed=seed,
                                     compile_mode=None, wandb=False, num_envs=128,
                                     microbatch_tokens=12288, allow_probe=False)
    r = launch.resolved
    tc = r.train_config
    scales = list(tc.scales)
    s0 = scales[0]
    scales[0] = dataclasses.replace(s0, env_config=dataclasses.replace(s0.env_config, frontline=None))
    tc = dataclasses.replace(tc, scales=tuple(scales))
    trainer = PPOTrainer(train_config=tc, model_config=r.model_config, ship_config=r.ship_config,
                         device="cuda", use_wandb=False,
                         scripted_agent=StochasticScriptedAgent(r.ship_config, StochasticAgentConfig()),
                         compile_mode=None, resolved_config_document=None)
    trainer.load_checkpoint(CKPT)
    return trainer


@torch.no_grad()
def latents(trainer, buf, envs_per_chunk=8):
    policy = trainer.policy
    block = policy.yemong_layers[-1]
    grab = []
    orig = block.sequence

    def wrapped(*a, **kw):
        out = orig(*a, **kw)
        grab.append(out[0].detach().to("cpu", torch.float16))
        return out

    block.sequence = wrapped
    B, N, T = buf.num_envs, buf.num_ships, buf.num_steps
    Dh = buf.initial_hidden.shape[-1]
    nl = buf.initial_hidden.shape[0]
    hid = buf.initial_hidden.reshape(nl, B, N, Dh)
    outs, tls = [], []
    for s in range(0, B, envs_per_chunk):
        e = min(s + envs_per_chunk, B)
        grab.clear()
        mb = YemongObservation(
            data={k: v[:, s:e] for k, v in buf.obs.items()},
            bullets=None if buf.bullet_obs is None else {k: v[:, s:e] for k, v in buf.bullet_obs.items()})
        obs = mb.slice_time(0, T)
        h = hid[:, s:e].reshape(nl, (e - s) * N, Dh).contiguous()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, _, logits, _, _, _ = trainer.policy.evaluate_actions(
                obs=obs, actions=buf.actions[:, s:e].long(), initial_hidden=h,
                alive_mask=obs[ObsKey.BELIEF_VALID].bool(), done_mask=buf.terminated[:, s:e])
        tls.append(torch.log_softmax(logits[..., TURN_SLICE].float(), -1)[::TSTRIDE].cpu())
        outs.append(grab[-1][::TSTRIDE, :, :N].contiguous())
    block.sequence = orig
    return torch.cat(outs, 1), torch.cat(tls, 1)


def collect(seed, n_roll=3, burn=3):
    trainer = build_tdm(seed)
    runtime = trainer._initialize_rollout_runtime()
    runtime.elo_eval.step = lambda *a, **k: None
    runtime.elo_eval.flush = lambda *a, **k: _Null()
    for _ in range(burn):
        trainer._collect_rollout(runtime, False)
    chunks = []
    for r in range(n_roll):
        t0 = time.time()
        trainer._collect_rollout(runtime, False)
        buf = trainer.buffer
        lat, tl = latents(trainer, buf)
        m = (buf.expert_probs.float().sum(-1) > 0) & buf.actor_masks & buf.alive_mask
        chunks.append(dict(lat=lat, logq=tl,
                           expert=buf.expert_probs[::TSTRIDE].float()[..., TURN_SLICE].cpu(),
                           keep=m[::TSTRIDE].cpu()))
        print(f"  seed{seed} {r}: {time.time()-t0:.0f}s tokens={tuple(lat.shape)}", flush=True)
    return {k: torch.cat([c[k] for c in chunks], 1) for k in chunks[0]}


def flat(d):
    k = d["keep"].reshape(-1)
    return dict(latent=d["lat"].reshape(-1, d["lat"].shape[-1])[k].float(),
                expert=d["expert"].reshape(-1, d["expert"].shape[-1])[k],
                logq=d["logq"].reshape(-1, d["logq"].shape[-1])[k])


def main():
    for tag, seed in (("train", 0), ("heldout", 1)):
        p = f"{S}/tdm_{tag}.pt"
        if not _os.path.exists(p):
            torch.save(collect(seed), p)
    tr, te = flat(torch.load(f"{S}/tdm_train.pt")), flat(torch.load(f"{S}/tdm_heldout.pt"))
    p = te["expert"].clamp_min(1e-8)
    rows = [
        dict(what="TDM tokens", n=len(te["latent"])),
        dict(what="teacher entropy", v=round(float(-(p * p.log()).sum(-1).mean()), 4)),
        dict(what="frozen policy head (OOD: frontline-trained)",
             turn_kl=round(float(kl(te["expert"], te["logq"])), 4)),
    ]
    for r in rows:
        print(json.dumps(r), flush=True)
    lq = fit_head(tr["latent"], tr["expert"], te["latent"])
    rows.append(dict(what="head refit on frozen latent -> legacy teacher, TDM env",
                     turn_kl=round(float(kl(te["expert"], lq)), 4)))
    print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(f"{S}/exp26_rows.json", "w"), indent=1)


if __name__ == "__main__":
    main()
