# Brief: temporal-first trunk and a rotary read-out (BC arms C and D)

**Decided by the author (2026-11-06; `decisions.md`, "Round four"):** two BC arms beside
the baseline (A: spatial, spatial, temporal with belief feedback) and rotary-off (R):

- **C:** each block temporal, spatial, spatial; belief feedback on (hidden ships' rows and
  rotary positions from the belief, as today).
- **D:** each block temporal, spatial, spatial; Griffin only (`yemong.forecast_feedback =
  false`), and hidden ships' rotary phasors **read from the first temporal layer**.

The point of D: Griffin-only memory without any forecast fed back between ticks, yet
hidden ships placed where memory thinks they are (not at their last sighting).

## What to build

1. **The order switch:** `YemongConfig.order: str = "spatial_first"` (today, bit for bit)
   or `"temporal_first"` (per block: temporal, then the spatial sublayers). `unroll` must
   still equal a loop of `step`. The carry and the scan keep working.
2. **The rotary read-out** (`YemongConfig.rotary_readout: bool = False`; valid with
   `temporal_first`):
   - A linear head on the first temporal layer's output, per ship token, giving a (cos, sin)
     pair for every rotary frequency: 2 x (2 x `rotary_octaves` position pairs +
     `len(heading_harmonics)` heading pairs), about 40 numbers. Not normalised: a pair's
     length below 1 says the model is unsure at that frequency (it shrinks toward (0, 0)).
   - **Visible ships** use their true phasors (from their observed position and heading:
     today's tables). **Hidden ships** use the read-out's phasors. Map tokens (fields,
     zones) keep their exact tables.
   - These tables drive every spatial sublayer after the first temporal layer (both
     blocks), in place of the belief-derived ones. Check the fused attention kernel accepts
     tables whose (cos, sin) are not unit length (it should: it multiplies), else use the
     plain twin there.
   - **Training:** a squared-error loss between the read-out and the **true** phasors (the
     truth's cos and sin of each frequency's angle), on **every** ship, visible and hidden,
     wherever the learner has the truth (the next-state labels' truth). Its optimum is the
     expected phasor under the model's belief, which is why the length shrinks with
     uncertainty. The gradient through the rotary into the read-out is allowed as well.
     A loss coefficient in the learner config (`rotary_readout`), set so its trunk
     gradient share is a few percent at the start (use the gradient diagnostics on a short
     CPU run, or reason from the other terms' units; say what you chose and why).
   - Log the read-out's error per population (ally, enemy visible, enemy hidden) and the
     mean pair length for hidden ships by time since seen, so the run shows it working.
3. **CLI:** `--set yemong.order=temporal_first --set yemong.rotary_readout=true` (and for
   D `--set yemong.forecast_feedback=false`); recorded in the build and config.json.

## Tests (CPU)

- `spatial_first` equals today exactly (a recorded step/unroll); `rotary_readout` off
  equals before.
- `temporal_first`: `unroll` equals a loop of `step`; the order is really temporal first.
- The read-out: visible ships' tables equal the true tables; hidden ships' equal the
  read-out's pairs; the loss is zero when the read-out equals the true phasors; the loss
  appears in the learner's terms (BC and PPO).
- A policy-file round trip keeps the settings.

## Docs

The Yemong page (the order, the read-out, what D is); the learner page (the new term).

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-temporal-first`, branch `feat/temporal-first`
  (from v2). Work only there. Scratch in `/tmp/bnb-scratch-temporal-first`. Don't read
  other directories under `/home/vizia/avesnova`.
- **CPU only:** `CUDA_VISIBLE_DEVICES=""`, `nice -n 15`; the GPU is running training.
  Never take the bench lock. No training runs (the coordinator launches the arms).
- **Python:** `uv sync --all-extras`, then always `uv run --no-sync ...`.
- **Progress notes:** append to `/tmp/bnb-scratch-temporal-first/progress.md`, `## HH:MM`
  heading from `date +%H:%M` (local), 1-3 bullets; ETAs as local clock times. Start with
  a plan note.
- **Tests:** CPU tests for what you add, the quick tier (`pytest -m "not slow and not
  gpu"`), ruff, docs.
- If something here is wrong, missing or ambiguous, say so in your notes and final message;
  make a sensible choice and keep going.

## Done

- Commit and push to `feat/temporal-first` in logical commits. Author Julian Kizanis
  <julianzia@gmail.com>; end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Final message, half a page:** the switches, the read-out and its loss coefficient
  (with the reasoning), the tests, the exact `--set` lines for C and D, anything open.
