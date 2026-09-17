"""Exp 15: does the belief/visibility mismatch cost turn KL?

`frontline_strategy` counts only *currently visible* enemies
(`~allied & visible & alive`). The trunk's spatial attention is masked by
`BELIEF_VALID` = "visible or previously observed" (`policy.py:374`), so every
remembered-but-currently-invisible enemy is a token competing for softmax mass
that the teacher ignores outright.

The policy can learn to gate on this -- `visible` and `time_since_observation`
are both input features -- but it has to learn it, and a stale token still
takes attention mass in the meantime. If turn KL rises with the number of
such ghost tokens, the mismatch is real and costed.

Also prints turn KL by visible-enemy count. Read that table with care: zero
visible enemies implies alpha=1, so the split is confounded with the alpha
effect. Conditioning on alpha>0.99 removes it (0.5709 vs 0.5633).
"""

import json
import os as _os

import torch

S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
NSHIPS = 8


def main():
    d = torch.load(f"{S}/probe2_heldout.pt")
    keep = d["keep"]  # (T,B,N)
    T, B, N = keep.shape
    o = {k: v[:T] for k, v in d["obs"].items()}
    team = o["team_id"][:, :, :NSHIPS]  # (T,B,N)
    belief = o["belief_valid"][:, :, :NSHIPS].bool()
    alive = o["alive"][:, :, :NSHIPS].bool()

    # Ghost = a token the trunk attends over but the teacher never counted.
    # Teacher's own visible-enemy count is recorded, so the difference needs no
    # re-derivation of the perception model.
    enemy_tok = (team[:, :, None, :] != team[:, :, :, None]) & belief[:, :, None, :] & alive[:, :, None, :]
    n_belief_enemy = enemy_tok.float().sum(-1)  # (T,B,N) per observer
    n_vis_enemy = d["internals"]["mag_n_visible_enemies"]
    ghosts = (n_belief_enemy - n_vis_enemy).clamp_min(0)

    exp = d["expert"].reshape(-1, d["expert"].shape[-1])
    logq = d["turn_logp"].reshape(-1, d["turn_logp"].shape[-1])
    k = keep.reshape(-1)
    kl = (exp.clamp_min(1e-8) * (exp.clamp_min(1e-8).log() - logq)).sum(-1)[k]
    g = ghosts.reshape(-1)[k]
    nv = n_vis_enemy.reshape(-1)[k]
    alpha = d["teacher"]["alpha"].reshape(-1)[k]

    rows = []
    print("turn KL by number of remembered-but-invisible enemy tokens:")
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 99)]:
        m = (g >= lo) & (g < hi)
        if int(m.sum()) < 200:
            continue
        rows.append(dict(ghosts=f"[{lo},{hi})", n=int(m.sum()),
                         turn_kl=round(float(kl[m].mean()), 4),
                         mean_visible_enemies=round(float(nv[m].mean()), 2),
                         mean_alpha=round(float(alpha[m].mean()), 3)))
        print(json.dumps(rows[-1]), flush=True)

    # Confounded with "enemies are far away", which also raises alpha. Match on
    # the visible-enemy count so the comparison is at equal teacher input.
    print("\nmatched on visible-enemy count:")
    matched = []
    for v in range(0, 5):
        mv = (nv >= v) & (nv < v + 1)
        if int(mv.sum()) < 500:
            continue
        a = mv & (g < 1)
        b = mv & (g >= 1)
        if int(a.sum()) < 200 or int(b.sum()) < 200:
            continue
        matched.append(dict(visible_enemies=v, n_no_ghost=int(a.sum()), n_ghost=int(b.sum()),
                            kl_no_ghost=round(float(kl[a].mean()), 4),
                            kl_ghost=round(float(kl[b].mean()), 4)))
        print(json.dumps(matched[-1]), flush=True)

    print("\nturn KL by visible-enemy count (fleet-size sensitivity):")
    byn = []
    for v in range(0, 6):
        m = (nv >= v) & (nv < v + 1)
        if int(m.sum()) < 300:
            continue
        byn.append(dict(visible_enemies=v, n=int(m.sum()), turn_kl=round(float(kl[m].mean()), 4)))
        print(json.dumps(byn[-1]), flush=True)

    json.dump(dict(by_ghosts=rows, matched=matched, by_visible=byn),
              open(f"{S}/exp15_rows.json", "w"), indent=1)


if __name__ == "__main__":
    main()
