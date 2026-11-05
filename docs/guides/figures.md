---
covers: [src/bnb/figures/, src/bnb/models/yemong/decode.py, src/bnb/play/sides.py]
reviewed: 2026-11-06 16:00
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

## `next_state`

**The question:** how well does the next-state head forecast where each ship is one tick
later, for own ships, seen enemies and hidden ones?

The forecast at a tick, decoded to world coordinates, is scored against the privileged row
one tick later, on the rows the learner trains. Two baselines use only what the model was
given: the ship stays where it is (for a hidden enemy, where its belief puts it), and the
ship keeps its velocity for one tick.

- `next_state_track`: one enemy that is both seen and hidden in the segment: its true path
  with the forecasts (blue when it was seen at the tick, green when hidden); its forecast
  error over time with the hidden spans shaded, against the constant-velocity baseline and
  the head's sigma; and the mean error of each population against both baselines.

**How to read it.** For a seen ship, the forecast should beat keeping the velocity, which
already knows the ship's motion; the one-sigma share should be near
`bnb.figures.next_state.ONE_SIGMA`{.nolink}. For a hidden ship the forecast starts from the
belief, so its error is the belief's; the head adds what the belief cannot.

## `density`

**The question:** does the density head know where each side's ships are, beyond what the
view sees?

The true field is the learner's target: each living ship splits one unit over the three
cells of its grid triangle. The head's field is its expected count per cell. Two cheap
baselines: the constant map (each cell's mean count over the capture) and the visible ships
alone, counted the same way (for the own side, every ship).

- `density_maps`: for frames with many hidden enemies, the enemy field on the real hex
  cells: true, the head's expected count, and the head minus the truth; seen enemies as
  dots, hidden ones as rings.
- `density_error`: R^2 against the constant map, of the head and of the visible ships, per
  side; the enemy map's squared error per step by the number of hidden enemies; the head's
  squared error per cell by ring (0: the arena's centre).

**How to read it.** On the own side the visible ships are the truth, so the head's R^2
there is how sharply it places ships it sees. On the enemy side, a head that knows more
than it sees beats the visible ships, most of all when many enemies are hidden. The
training series `density/explained_deviance` is the same comparison in Poisson deviance
against the batch-mean map.

## `enemy_action`

**The question:** how well does the enemy-action head predict what each living enemy does
next, when the view sees it and when it does not?

- `enemy_action`: per factor of the joint action (power, stick, shoot), the share of each
  value the enemies took (bars) against the head's mean marginal (dots), seen and hidden;
  and the reliability of the head's P(shoot): its predicted probability, binned, against
  the share that shot.

**How to read it.** The titles give the cross-entropy (nats) of the head's marginal against
the marginal baseline, the capture's own action frequencies; the head knows something about
an enemy when it beats that baseline. The joint numbers (`bnb.figures.enemy_action.measures`)
add the accuracy of the most likely action and the mean probability of the action taken.
On the reliability plot a calibrated head lies on the diagonal.

## `critic`

**The question:** does the critic's value move with the game, and does the outcome head's
win probability come true?

It reads the capture's `games` record: the critic over the first segment and over
`--game-segments` more segments of the same games, so that games end inside the capture
(a deathmatch game can last minutes: 40 segments of 256 ticks is about 340 s).

- `critic_trace`: one view's game: the outcome head (P(win) - P(loss) as it gives it, the
  renormalised P(win), P(unresolved)) and each reward level's value (the own ships' mean),
  with every kill (orange) and loss (black) marked.
- `critic_outcome`: the reliability of the renormalised P(win) over the ticks of games that
  ended in the capture, and the Brier skill against the capture's result frequencies by the
  time left to the game's end.

**How to read it.** The outcome head is discounted: its targets put a share on
"unresolved" per tick, so its own P(win) is the discounted chance of a win. The figures
renormalise the resolved classes, P(win) / (P(win) + P(tie) + P(loss)), as its forecast of
the result. A skill above zero beats always forecasting the result frequencies; it should
rise as the end nears. The ticks of one game are strongly correlated, so a capture of few
games gives a noisy reliability curve: the counts beside each point are ticks, not games.

## Videos

`bnb figures video` films one game: the policy (team 0) against the scripted chaser
(team 1), fog on, drawn by the play shell offscreen and encoded to MP4 by ffmpeg
(`bnb.figures.video`). The frame is the shell's own, with its top bar (the score and the
clock) and kill feed; the fleets card and the minimap stay collapsed and the camera fits
the arena. A policy trained on a mixed batch records the batch's first game only
(`bnb.runtime.train.mixed`), so the video loads it for any mode of the preset asked for
(`bnb.play.sides.load_agent` with `any_mode`): the sizes and the ship card must still
match.

<!-- not-runnable: needs ffmpeg, an OpenGL context and a policy file -->
```bash
bnb figures video --policy runs/RUN/checkpoints/step_N --preset deathmatch
bnb figures video --policy RUN --video.seconds 10 --video.time-scale 8
```

The defaults (`bnb.figures.video.VideoConfig`): 25 s at 30 frames per second and 4x the
game's speed, so the first 100 s of the game, at 1280 x 720. The file is about 0.5 MB. The
agent runs on the CPU (`--device cpu` hides the GPU from torch); the drawing uses the GPU's
OpenGL over EGL, which is not CUDA, and about 100 MiB of its memory. One video takes about
50 s on a laptop CPU, nearly all of it the agent's decisions. It prints the game's tally:
kills, losses and captures by side, and the games that ended. The output goes to the run's
`artifacts/figures/<policy directory>/videos/` when the policy lies in a run, else
`figures/videos/`; `--out` names the file.

## Watching a training run

`bnb figures watch RUN` makes media of a run while it trains. It is a second process,
started by hand beside the trainer, that only reads the run (and writes its
`artifacts/figures/`): it polls `checkpoints/` every `--interval` seconds and, for each
complete checkpoint without media (`bnb.figures.watch`):

1. per mode (`--modes`, frontline and deathmatch by default), captures 4 environments of
   256 ticks on the CPU, with the same seed at every checkpoint, and draws the quick sets
   (`--sets`: `shots`, `belief`, `next_state`, `density`, `enemy_action`) into
   `artifacts/figures/<checkpoint>/<mode>/`; the capture is deleted unless
   `--keep-captures` (about 60 MB a mode);
2. per mode, films one game against the scripted agent into `<checkpoint>/videos/`;
3. writes `<checkpoint>/media.json` last: what was made, the sets' numbers, the videos'
   tallies, the seconds each part took and any error.

<!-- not-runnable: watches a live training run -->
```bash
nohup bnb figures watch RUN --wandb > /tmp/watch-RUN.log 2>&1 &
```

A part that fails (one set, one video) is recorded with its error and the rest goes on; a
checkpoint the retention deletes before it is read is skipped. The watcher hides the GPU
from torch, lowers its own priority (`--nice`) and keeps to `--threads` CPU threads, so
the trainer's work goes first; a checkpoint's media take about 2.5 minutes. It stops once
the run has stopped and every checkpoint has its media; a restarted watcher makes only
what is missing.

**W&B.** With `--wandb` each checkpoint's media are logged once, at its environment steps,
to a companion run `<run>-media` in the same project, grouped under the run's name, with
the run it watches in its notes. Two processes cannot write one W&B run, so the training
run's own is never touched. The companion run's id is the training run's W&B id followed
by `media`, so a restarted watcher resumes it, and its x-axis is the training run's
(`counters/environment_steps`). Its keys:

| Key | What |
|---|---|
| `video/<mode>` | the game's video |
| `game/<mode>/...` | its tally (`policy_kills`, `scripted_kills`, `policy_captures`, ...) |
| `figures/<mode>/<figure>` | each figure, such as `figures/deathmatch/belief_coverage` |
| `numbers/<mode>/<set>/...` | the numeric leaves of each set's numbers |
| `watcher/...` | the seconds each part took |
