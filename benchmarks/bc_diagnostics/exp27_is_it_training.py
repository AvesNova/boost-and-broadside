"""Exp 27: is this run still training at all, measured honestly?

Every KL number the run logs is taken *during* the update, on the rollout being
fitted (D3). That is the one measurement that cannot answer "are we making
progress on unseen states". This runs the production loop -- collect, score,
GAE, one `_update_epochs` call -- and records, per update:

  pre   turn KL on the freshly collected on-policy rollout, BEFORE the update
        touches it. This is the honest learning curve.
  post  turn KL on that same rollout after the update, i.e. how far the update
        fitted its own training data.
  next  `pre` of the following update -- how much of that fitting survived
        contact with fresh on-policy states.

If `pre` is flat across updates while `post` keeps dropping, the run is fitting
each rollout and generalising none of it, and the logged metric -- which lives
between `pre` and `post` -- will show improvement that is not there.
"""

import json
import os as _os
import sys
import time

import torch

S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, S)
from exp5_ns import eval_turn  # noqa: E402
from harness import _NullSnap, build  # noqa: E402


def main():
    n_updates = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    # Batch size is the obvious alternative explanation for a large
    # fit-then-lose gap, and the dev card forced num_envs=128 against the run's
    # 1280. Sweep what fits so the trend is visible rather than assumed.
    num_envs = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    torch.manual_seed(0)
    trainer, _ = build(num_envs=num_envs, microbatch_tokens=12288, seed=0)
    runtime = trainer._initialize_rollout_runtime()
    runtime.elo_eval.step = lambda *a, **k: None
    runtime.elo_eval.flush = lambda *a, **k: _NullSnap()

    # Burn in so the env distribution is steady state rather than post-reset.
    for _ in range(3):
        term = trainer._collect_rollout(runtime, False)

    rows = []
    for u in range(n_updates):
        t0 = time.time()
        term = trainer._collect_rollout(runtime, False)
        pre = eval_turn(trainer, trainer.buffer)["turn"]
        trainer._compute_rollout_gae(runtime, term)
        trainer._update_epochs(all_buffers=[trainer.buffer, *trainer.aux_buffers], update=u + 1)
        post = eval_turn(trainer, trainer.buffer)["turn"]
        rows.append(dict(update=u + 1, pre_update_fresh_kl=round(pre, 4),
                         post_update_same_rollout_kl=round(post, 4),
                         fitted=round(pre - post, 4), seconds=round(time.time() - t0, 1)))
        print(json.dumps(rows[-1]), flush=True)

    pres = [r["pre_update_fresh_kl"] for r in rows]
    first, last = sum(pres[:3]) / 3, sum(pres[-3:]) / 3
    summary = dict(
        num_envs=num_envs,
        epochs_per_update=trainer._schedule_state.num_epochs,
        mean_pre_first3=round(first, 4), mean_pre_last3=round(last, 4),
        progress_over_run=round(first - last, 4),
        mean_fitted_per_update=round(sum(r["fitted"] for r in rows) / len(rows), 4),
    )
    print(json.dumps(summary), flush=True)
    json.dump(dict(rows=rows, summary=summary), open(f"{S}/exp27_rows_{num_envs}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
