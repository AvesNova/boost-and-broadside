# Prompt for a fresh agent on the laptop

Paste the block below. It assumes native Ubuntu, the laptop's GPU, and the `core` branch.

```
You are continuing the lean-core rewrite of boost-and-broadside on this laptop (native
Ubuntu, one NVIDIA GPU). Check out the branch `core` and pull.

Read first, in this order: lean/README.md (the state of the first cut and the order of
work), each lean/*/NOTES.md, then docs/rewrite/core-brief.md sections 1, 3 and 4. The
audits behind the plan are docs/rewrite/model-audit.md, docs/rewrite/sections/game-audit.md
and docs/rewrite/sections/yardstick.md; read the part for an area when you work on it.

Do these in order, stopping to report after each:
1. Run `OMP_NUM_THREADS=1 PYTHONPATH=.:src python -m pytest lean -q` and
   `python -m lean.train --smoke` on the CPU, then the smoke run with `--device cuda`.
   Fix what breaks on the GPU. Report steps per second and peak memory.
2. Finish the framework benchmark: docs/rewrite/sections/framework-spike/run_laptop.sh
   (steady speed, the whole-loop rollout, latency, repeats 2 to 5). Set JAX's
   matrix-multiply precision to highest. Report the table and a recommendation for
   gates G1 and G2 in core-brief.md.
3. Take the first scoreboard numbers for lean/: operations per game step, per policy
   forward and per scripted call against the tripwires; compiled against uncompiled step
   time; compile time; what a new team size costs.
4. Extend the game's comparison with v2 to games that end, the clock rule and fields.

Rules: one GPU job at a time. Report failures plainly and never state a number you did
not measure. Do not start a long training run; propose it and wait. Commit small and push
to `core`. Questions for the author go in docs/questions.md.
```
