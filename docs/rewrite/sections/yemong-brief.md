# Brief: section 5, Yemong

You are implementing **Yemong**, the model of Boost and Broadside v2. It goes in
`bnb/models/yemong/`, with a controller over it. The goals are **speed and maintainability
together**:
- fast batched GPU rollouts and updates;
- the same code at batch size 1 on the CPU, and a single step that exports to ONNX for
  client play;
- code the next person can read and change.

## Where you work

- **Your worktree:** `/home/vizia/avesnova/bnb-yemong`, branch `section/yemong`. Work only
  here.
- **Scratch:** use `/tmp/bnb-scratch-yemong` for temporary files.
- **Do not read or modify** any other directory under `/home/vizia/avesnova`. In
  particular, stay out of the other worktrees, the v1 code, and the `physics-race*`
  repositories. The equivalence goldens are your only view of v1, by design.
- **Python:** set up with `uv sync` in your worktree, then always run through
  `uv run --no-sync ...` (never plain `python`).

## Read first

1. `docs/rewrite/sections/yemong.md`: the decisions, with the author's reasons.
2. `docs/rewrite/sections/yemong-spec.md`: **the spec you implement**.
3. `tests/reference/yemong/README.md`: the equivalence interface and its conventions.
4. `docs/rewrite/contracts.md` (the current-schema section) and `src/bnb/contracts/`.
5. `docs/rewrite/scaffold.md`: section 3 (design rules) and section 4 (layout and
   dependency rule).
6. `src/bnb/`, especially `controllers/`, `models/`, `runtime/`, and `tests/`.

## What to do

1. **Implement the spec:**
   - the model (`step`, `unroll`, the carry and belief, input composition, the network,
     the heads);
   - the codes and targets of sections 3, 8 and 9;
   - the controller;
   - the contract additions of section 10, exactly as written, with the runtime storing
     them.
2. **Write the equivalence adapter** `tests/reference/yemong/yemong_arm.py`. `make_arm()`
   returns an object with every method in the README, each a thin translation onto your
   code (fixture weight names to your parameters, the neutral carry layout to yours).
   - The adapter adapts; it does not reimplement. If a method needs more than renaming
     and reshaping, your module's interface is probably wrong.
   - The interface's `done` means "ended at this step". `unroll` takes the runtime's
     "reset before this observation"; convert in the adapter.
3. **Tests:**
   - **Everything must pass:** `uv run --no-sync pytest`, with the Yemong equivalence
     tests running, not skipped.
   - **Fixed:** the files under `tests/reference/` other than `yemong_arm.py`. Never
     change them.
   - **Add your own** under `tests/sections/` for acceptance items 2-6, and an ONNX
     export test beside `tests/contract/test_export.py`.
4. **Speed** (acceptance item 7):
   - Measure with ad-hoc scripts under the benchmark lock (`bench.lock.bench_lock()` in
     your script). Never wrap a lock-taking command in plain `flock`.
   - Report the general path. No size-gated algorithm branches unless you measure the
     code without them in the same run.
   - Keep benchmarking proportionate: a few runs, not a sweep.
5. **Measurements** asked for by the spec (marked **(measure)**): report them, and change
   a constant only when the measurement shows it is wrong.
6. **Commit and push often:**
   - commit to `section/yemong`, then `git push -u origin section/yemong`;
   - author Julian Kizanis <julianzia@gmail.com>;
   - end every commit message with
     `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Boundaries

- **You may change:**
  - `src/bnb/models/**` and `src/bnb/controllers/**`;
  - `src/bnb/contracts/**`, but only for the additions of spec section 10;
  - `src/bnb/runtime/**`, only to store those additions and to run a stateful
    controller. List each change in your report;
  - `tests/sections/**`, `tests/contract/**` (add only),
    `tests/reference/yemong/yemong_arm.py`;
  - your notes file.
- **You may not change:**
  - `src/bnb/game/**` and `src/bnb/learn/**` (PPO and labels are section 6);
  - `bench/`;
  - the fixed tests and goldens;
  - the docs, apart from your notes file.
- **If the spec, a contract or a fixture looks wrong:** do not work around it silently.
  Take the most literal reading, keep going, and write it up as a request in your notes.
- **Naming:** names in full words (scaffold rule 10). Spec symbols are not names.
- **No legacy:** no shims, dead code or compatibility paths.
- **Precision:** float32 by default; bf16 autocast only where measured faster and the
  spec allows it.
- **Smoothness and repeatability:** traded by judgement. Small costs for repeatability are
  worth paying; large ones are not.

## Hand back

Write `docs/rewrite/sections/yemong-notes.md` with:
- **Design:** the layout of `bnb.models.yemong`, the carry, the step and unroll paths,
  the compile and capture assumptions.
- **Decisions and deviations:** each with its reason.
- **Measurements:** the spec's (measure) items and the speed report.
- **Runtime and contract changes.**
- **Requests and open questions.**
- **Size:** line counts per module.

Your final message should summarize the same, in a page or less.
