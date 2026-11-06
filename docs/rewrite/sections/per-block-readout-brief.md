# Brief: a rotary read-out in every block (arm D), urgent

**Decided by the author (2026-11-07):** with `yemong.rotary_readout`, every Yemong block
reads its own rotary phasors for hidden ships from its own temporal sublayer, and that
block's spatial sublayers use them; visible ships use their true phasors in every block.
Per-block is the only form (not an option beside the single read-out): "every Yemong block
does the same thing, regardless if it's the first, the second, or the 99th."

**Deadline:** arm D starts training at about **04:25** (local). If this is merged and
tested before about 04:10, D runs with it. Commit and push as soon as it is correct; say
the time in your notes.

## What to change (branch from v2, which has the single read-out: `model.py` around lines
81-116, 343-345, 479-490; the learner's `rotary_readout` term)

- `Yemong.rotary_readout` becomes one `nn.Linear(width, 2 * rotary_pairs)` per block
  (a ModuleList), each reading that block's temporal sublayer output (the sublayer's
  output, as today for block 1), building that block's tables for hidden ships.
- The learner term: the mean over blocks of each block's squared phasor error, against
  the same true phasors, same coefficient (`learner.rotary_readout`, 0.007), same masks
  (every living ship). Log the error per block as well as the mean
  (`rotary_readout/error/block_<k>`), keep the existing series (as the mean).
- Still detached into the tables; visible ships, the global token and map tokens keep
  their exact tables in every block.
- Policy files from the single read-out need not load (no backward compatibility).

## Tests (CPU)

Adapt the existing tests (`tests/sections/test_yemong_order.py`,
`tests/sections/test_learner_rotary_readout.py`): each block's hidden tables equal that
block's read-out; `unroll` equals a loop of `step`; the term's mean over blocks; spatial_first
and read-out off unchanged bit for bit. Quick tier, ruff, docs (`yemong.md`, `learner.md`).

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-per-block`, branch `feat/per-block-readout`
  (from v2). Work only there. Scratch `/tmp/bnb-scratch-per-block`.
- **CPU only:** `CUDA_VISIBLE_DEVICES=""`, `nice -n 15`, at most 8 threads; arm C trains on
  the GPU. Never take the bench lock.
- **Python:** `uv sync --all-extras`, then `uv run --no-sync ...`.
- **Progress notes:** `/tmp/bnb-scratch-per-block/progress.md`, `## HH:MM` from
  `date +%H:%M`, 1-3 bullets; ETAs as clock times. Start with a plan note.

## Done

- Commit and push to `feat/per-block-readout`. Author Julian Kizanis <julianzia@gmail.com>;
  end each message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Final message: what changed, tests, the commit hash, the time.
