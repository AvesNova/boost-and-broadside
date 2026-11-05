---
covers: [src/bnb/figures/, src/bnb/models/yemong/decode.py]
reviewed: 2026-11-06 14:00
---

# Figures

`bnb figures` draws what a policy's heads have learned, one figure set at a time. It works
in two steps, so a figure can be redrawn without the model:

1. **Capture.** Play a short self-play rollout with the policy's model (both seats, fog of
   war on, fresh games), run the model over it as the learner does
   (`bnb.models.yemong.model.Yemong.unroll`), decode every head with the decoder layer
   (`bnb.models.yemong.decode`) and store observations, truth and predictions as one
   artifact file (`bnb.figures.capture`). The capture also runs the scenario lab.
2. **Render.** Each figure set reads the artifact only, on the CPU, and writes PNGs.

<!-- not-runnable: minutes on the CPU, and needs a policy file -->
```bash
uv sync --extra tools                      # matplotlib
bnb figures shots scenarios --policy runs/RUN/policy
bnb figures all --capture runs/RUN/artifacts/figures/policy/captures/deathmatch_seed0.pt
bnb figures capture --policy RUN --environments 12   # capture only
```

The output goes to the run's `artifacts/figures/<policy directory>/` when the policy lies in
a run, else beside the capture read, else `figures/`; `--out` overrides it. The capture is
kept under the output's `captures/`. Every PNG carries its provenance in its PNG text
chunks: the policy, its environment steps and commit, the capture's commit and command line,
the render's commit and command line, and the set's numbers. Read them with any PNG viewer
that shows metadata, or with `PIL.Image.open(path).text`.

The capture runs on the CPU by default (`--device`); 12 environments of 256 ticks take
under a minute on a laptop CPU. The sets below are `bnb.figures.sets.FIGURE_SETS`.

## `shots`

**The question:** does the shot head's predicted path bend through the refractive fields
as real shots do, or does it only fly straight?

For every labelled shot of the capture (the learner's mask: shots of own ships and of
enemies seen at launch; a ship fires a counterfactual shot on every tick), three paths:

- the **true flight**: the learner's re-simulation (`bnb.learn.shots.resimulate`), the
  game's own launch and transport through the fields, at zero spread;
- the **head's path**: the decoded mean at each age, with its one-sigma radius
  (`bnb.models.yemong.decode.shot_path`);
- the **straight line**: launch position plus launch velocity times age.

A flight is **bent** when its true path ends more than `BEND_THRESHOLD` to the side of the
straight line. Bent flights split again by whether the shooter launched inside a field's
band, where its own row shows the model a nonzero `log_index_gradient_x` and
`log_index_gradient_y`, or met a band after launch.

- `shots_gallery`: the most bent shots and two straight ones, each over the field map (ln n,
  red above 0, blue below), with the true flight (orange), a fan of spread samples (thin
  orange), the head's mean per age (blue dots) with its one-sigma circles, and the straight
  line (grey, dashed).
- `shots_error`: the mean displacement error by age on straight and on bent flights, of the
  head and of the straight line, with the head's mean sigma; and, for bent flights at the
  last age, the head's offset to the side of the straight line against the true offset.

**How to read it.** On the scatter, points on the diagonal are flights the head bends
correctly; points on the horizontal zero line are flights it predicts straight. The
numbers (`bnb.figures.shots.measures`): the slope through the origin of the head's
lateral offset on the true one (1: it bends as shots do; 0: straight lines only) and its
correlation, per split; and the share of shots whose last age lies within the head's
one-sigma circle, which is `ONE_SIGMA_DISC` when the sigma is right. The straight line's
error on straight flights is drag: a bullet slows along its path.

## `scenarios`

**The question:** the same as `shots`, on controlled cases: one shot fired on maps that
differ only in one field.

The template is a real moment of the capture: an own ship of one view at the segment's last
tick, alive throughout and far from every other ship. The model sees that view's real
history. A scenario places one field relative to the launch point and the bullet's launch
direction (`bnb.figures.scenarios.SCENARIOS`, each named for its case), moves every other
field the flight could reach to the far side of the arena, and shows that map in every
observation of the segment, with each visible ship's `log_refractive_index` and gradient
recomputed for it. The ships' recorded motion is the real one, flown on the real map, so the
scenario fields are placed ahead of the shooter, where it has not been. The true flight is
the re-simulation of the one shot through the edited map, with no ship to hit, at zero
spread and over a fan of spread draws.

- `shots_scenarios`: one panel per scenario, drawn as in the gallery.

**How to read it.** Each panel's title gives the bend, the offset of the last position from
the open-field flight (`straight_open`), true and predicted. A head that reads the map
ahead bends with the true flight; one that does not draws the same line in every panel,
apart from what the launch point shows it (the index at the launch sets the bullet's
speed).

## `belief`

**The question:** where does the model believe each hidden enemy is, and is the sigma it
states honest?

The belief is the replay record's (what the policy believed at a tick, after its sight),
decoded to world coordinates; the truth is the privileged row at that tick. Only hidden,
living enemies that hold a belief count.

- `belief_ghosts`: frames with the most enemies hidden for a short while (0.2 to 3 s): own
  ships (green triangles), seen enemies (orange dots), each hidden enemy's true position
  (orange ring) joined to its ghost, the belief's mean (blue cross) with one- and two-sigma
  circles, labelled with the time since it was seen.
- `belief_coverage`: by time since seen (`AGE_BUCKETS`), the share of hidden enemies within
  one and two sigma, for position and velocity; and the mean position error against the
  mean sigma.

**How to read it.** Coverage uses the training metric's definition
(`bnb.learn.head_statistics.belief_sums`); an honest sigma covers `EXPECTED_COVERAGE`
(63% within one sigma, 98% within two), the dashed lines. Coverage under the line is an
overconfident belief. On the right, an honest belief's error tracks its sigma times the
square root of two; a flat sigma under a rising error is a belief that does not know how
long it has been blind.
