# Replays and qualitative results

These clips complement the aggregate results with examples of tactics at different fleet
sizes. In every `vs_scripted` replay, the learned fleet is blue and the scripted fleet is
red.

## Primary replay: 8 learned vs 11 scripted

![Eight blue learned-policy ships versus eleven red scripted ships](results/replays/vs_scripted_8v11_seed03.gif)

Outnumbered 11 ships to 8, the learned fleet wins with three ships to spare. The matchup
is easy to follow and compact enough to serve as the README hero.

The quantitative sweep reports a 69.5% learned-team win rate at this 8-vs-11 matchup and
42.2% at 8-vs-12. See [zero-shot crossover](evaluation.md#zero-shot-crossover) for the raw
data and statistical caveats. The clip is one qualitative realization, not the source of
that aggregate rate.

## Large-scale transfer: 64 vs 80

![Sixty-four learned blue ships versus eighty scripted red ships](results/replays/vs_scripted_64v80_seed01.gif)

This clip shows the same variable-cardinality execution path at a much larger scale. The
stored crossover sweep reports a 91.1% learned-team win rate at 64-vs-80, while the
empirical boundary is between 87 and 88 scripted ships.

At about 5.3 MiB, it is intentionally kept off the root README. A future replay refresh
should add a lightweight poster image linked to the full GIF.

## Self-play

[View the 8-vs-8 self-play clip](results/replays/self_8v8_seed01.gif).

In self-play, both sides use the same weights. Team 1 receives a team-flipped observation
because the `ego_pass` policy is canonicalized to the team-0 perspective. This clip is
useful for understanding execution symmetry, but it is not independent benchmark evidence.
The implementation is in [`capture.py`](../src/boost_and_broadside/modes/capture.py).

Additional self-play clips at 2, 4, 16, 32, and 64 ships per side are preserved in
[`docs/results/replays/`](results/replays/). They should be promoted into the narrative only
when they demonstrate a distinct behavior rather than more visual scale alone.

## Other asymmetric clips

- [4 learned vs 5 scripted, seed 00](results/replays/vs_scripted_4v5_seed00.gif)
  and [seed 04](results/replays/vs_scripted_4v5_seed04.gif) are candidates for showing
  qualitative variability near the native training scale.
- [16 vs 22, seed 02](results/replays/vs_scripted_16v22_seed02.gif) provides an intermediate
  scale example.
- [32 vs 44, seed 04](results/replays/vs_scripted_32v44_seed04.gif) is another asymmetric
  result near, but below, that size's measured crossover.

Outcomes and behavior labels for these secondary clips have not all been preserved as
machine-readable metadata, so they remain linked rather than given interpretive captions.
A boundary loss/failure replay is intentionally still needed to balance the successful
examples.

## How capture works

[`capture.py`](../src/boost_and_broadside/modes/capture.py):

- loads the final `step_*.pt` checkpoint from the selected run;
- accepts symmetric or asymmetric team-size specifications;
- seeds PyTorch and the environment for each match;
- writes an MP4 by piping rendered RGB frames to `ffmpeg`;
- optionally creates a palette-optimized, downscaled GIF;
- records the terminal winner in console output and holds the terminal state on screen.

For self-play, team 1 uses a second recurrent state and team-flipped observation but shares
the policy weights. For `vs_scripted`, team 1 actions come from the stochastic scripted
controller.

Generate the selected 8-vs-11 scenario with:

```bash
uv run main.py --mode capture \
  --run resilient-resonance-682 \
  --scenarios vs_scripted \
  --sizes 8v11 \
  --seeds 3 \
  --gif
```

The default output directory is `gameplay_clips/`. Review outputs there before copying a
curated subset into `docs/results/replays/`.

## Provenance limitation and next format

The current GIFs encode frames but not the run, checkpoint step/hash, source commit,
capture arguments, or winner. Controller colors and the selected hero's terminal outcome
have been verified against code and the frames, but exact checkpoint provenance cannot be
reconstructed from the GIF alone.

Future captures selected for documentation should include a JSON sidecar with at least:

```json
{
  "run": "resilient-resonance-682",
  "checkpoint": "step_000999424000.pt",
  "checkpoint_sha256": "...",
  "source_commit": "...",
  "scenario": "vs_scripted",
  "team_sizes": [8, 11],
  "seed": 3,
  "winner": "team0",
  "survivors": [3, 0]
}
```

This schema is a recommendation, not metadata that already exists. Poster frames linked
to MP4/GIF files are also preferable for large clips. Both items are tracked in the
[deferred asset ledger](evidence.md#deferred-asset-and-analysis-ledger).
