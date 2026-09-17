"""Exp 7a: collect layerwise trunk activations + teacher bearings for probing.

Writes probe_<tag>.pt holding, for every tap point in the trunk, the activation
of each ship token, alongside the teacher's own bearing intermediates and the
stored teacher/policy turn distributions. Train probes on one tag and evaluate
on another: the rollouts are independent draws, so a probe never sees its own
test states.
"""

import os as _os
import sys
import time

import torch

S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, S)
from harness import build, _NullSnap  # noqa: E402

from boost_and_broadside.constants import TURN_SLICE  # noqa: E402
from boost_and_broadside.env.observation import ObsKey, YemongObservation  # noqa: E402

# Raw observation channels kept alongside the activations so the representation
# audit and the latent probes are guaranteed to describe the same rollouts.
OBS_KEYS = [
    ObsKey.POS, ObsKey.ATT, ObsKey.VEL, ObsKey.HEALTH, ObsKey.TEAM_ID, ObsKey.ALIVE,
    ObsKey.BELIEF_VALID, ObsKey.OBJECT_TYPE, ObsKey.ZONE_ROLE, ObsKey.RADIUS,
]

TSTRIDE = 4  # keep every 4th timestep: consecutive steps are near-duplicates


def tap_points(policy):
    """(name, module, layout) for every trunk boundary, in forward order.

    Layout matters: spatial sublayers and the encoder emit (T*B, N+M, D), while
    a temporal sublayer in sequence mode emits (B*N, T, D). Getting this wrong
    silently shuffles states against their labels, which a probe would fit at
    chance level and report as "the layer carries no information".
    """
    taps = [("encoder", policy.encoder, "tb")]
    for b, layer in enumerate(policy.yemong_layers):
        for i, sub in enumerate(layer.spatial):
            taps.append((f"b{b}.spatial{i}", sub, "tb"))
        for i, sub in enumerate(layer.temporal):
            taps.append((f"b{b}.temporal{i}", sub, "bt"))
    return taps


@torch.no_grad()
def run(trainer, buf, envs_per_chunk=8):
    policy = trainer.policy
    taps = tap_points(policy)
    grab: dict[str, list] = {n: [] for n, _, _ in taps}
    grab["raw_input"] = []
    layouts = {n: lay for n, _, lay in taps}
    layouts["raw_input"] = "tb"

    def store(name, y):
        grab[name].append(y.detach().to("cpu", torch.float16))

    handles = []
    for name, mod, lay in taps:
        if lay == "tb":
            handles.append(
                mod.register_forward_hook(
                    lambda m, i, o, n=name: store(n, o[0] if isinstance(o, tuple) else o)
                )
            )
        else:
            # forward_sequence is called directly, so a forward hook never fires.
            orig = mod.forward_sequence

            def wrapped(*a, _o=orig, _n=name, **kw):
                out = _o(*a, **kw)
                store(_n, out[0])
                return out

            mod.forward_sequence = wrapped
            handles.append(("unwrap", mod, orig))
    handles.append(
        policy.encoder.feature_extractor.register_forward_pre_hook(
            lambda m, i: store("raw_input", i[0])
        )
    )

    B, N, T = buf.num_envs, buf.num_ships, buf.num_steps
    D = buf.initial_hidden.shape[-1]
    n_layers = buf.initial_hidden.shape[0]
    hid_full = buf.initial_hidden.reshape(n_layers, B, N, D)
    out_taps: dict[str, list] = {n: [] for n in grab}
    turn_logp = []
    for s in range(0, B, envs_per_chunk):
        e = min(s + envs_per_chunk, B)
        bs = e - s
        for n in grab:
            grab[n].clear()
        mb = YemongObservation(
            data={k: v[:, s:e] for k, v in buf.obs.items()},
            bullets=None
            if buf.bullet_obs is None
            else {k: v[:, s:e] for k, v in buf.bullet_obs.items()},
        )
        obs = mb.slice_time(0, T)
        hid = hid_full[:, s:e].reshape(n_layers, bs * N, D).contiguous()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, _, logits, _, _, _ = policy.evaluate_actions(
                obs=obs,
                actions=buf.actions[:, s:e].long(),
                initial_hidden=hid,
                alive_mask=obs[ObsKey.BELIEF_VALID].bool(),
                done_mask=buf.terminated[:, s:e],
            )
        turn_logp.append(torch.log_softmax(logits[..., TURN_SLICE].float(), -1)[::TSTRIDE].cpu())
        for n in grab:
            a = torch.cat(grab[n], 0)
            if layouts[n] == "tb":  # (T*B, N+M, D)
                a = a.reshape(T, bs, -1, a.shape[-1])[::TSTRIDE, :, :N]
            else:  # (B*N, T, D)
                a = a.reshape(bs, N, T, a.shape[-1]).permute(2, 0, 1, 3)[::TSTRIDE]
            out_taps[n].append(a.contiguous())
    for h in handles:
        if isinstance(h, tuple):
            h[1].forward_sequence = h[2]
        else:
            h.remove()
    return {n: torch.cat(v, 1) for n, v in out_taps.items()}, torch.cat(turn_logp, 1)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "train"
    n_roll = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    burn = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    num_envs = 128

    trainer, launch = build(num_envs=num_envs, microbatch_tokens=12288, seed=seed, recording=True)
    runtime = trainer._initialize_rollout_runtime()
    runtime.elo_eval.step = lambda *a, **k: None
    runtime.elo_eval.flush = lambda *a, **k: _NullSnap()
    # Burn in so the env distribution is the steady state, not the reset
    # transient that Exp 2 measured at ~0.05 KL lower.
    for _ in range(burn):
        trainer.scripted_agent.recording = False
        trainer._collect_rollout(runtime, False)

    chunks = []
    for r in range(n_roll):
        trainer.scripted_agent.records = []
        trainer.scripted_agent.recording = True
        t0 = time.time()
        trainer._collect_rollout(runtime, False)
        trainer.scripted_agent.recording = False
        recs = trainer.scripted_agent.records
        buf = trainer.buffer
        acts, tlp = run(trainer, buf)
        teacher = {k: torch.stack([x[k] for x in recs])[::TSTRIDE] for k in recs[0]}
        m = (buf.expert_probs.float().sum(-1) > 0) & buf.actor_masks & buf.alive_mask
        chunks.append(
            dict(
                acts=acts,
                turn_logp=tlp,
                teacher=teacher,
                expert=buf.expert_probs[::TSTRIDE].float()[..., TURN_SLICE].cpu(),
                keep=m[::TSTRIDE].cpu(),
                obs={str(k): buf.obs[k][::TSTRIDE][: m[::TSTRIDE].shape[0]].float().cpu()
                     for k in OBS_KEYS},
            )
        )
        print(f"rollout {r}: {time.time()-t0:.0f}s valid={int(chunks[-1]['keep'].sum())}", flush=True)

    merged = dict(
        acts={k: torch.cat([c["acts"][k] for c in chunks], 1) for k in chunks[0]["acts"]},
        turn_logp=torch.cat([c["turn_logp"] for c in chunks], 1),
        expert=torch.cat([c["expert"] for c in chunks], 1),
        keep=torch.cat([c["keep"] for c in chunks], 1),
        teacher={k: torch.cat([c["teacher"][k] for c in chunks], 1) for k in chunks[0]["teacher"]},
        obs={k: torch.cat([c["obs"][k] for c in chunks], 1) for k in chunks[0]["obs"]},
        world_size=launch.resolved.ship_config.world_size,
    )
    path = f"{S}/probe_{tag}.pt"
    torch.save(merged, path)
    print("wrote", path, {k: tuple(v.shape) for k, v in merged["acts"].items()})


main()
