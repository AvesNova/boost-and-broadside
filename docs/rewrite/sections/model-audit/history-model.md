# Design history: the Yemong model (inventory sections 1-6 and the model-side rows of 12)

Reconstructed from the repository's record on 2026-11-21. It reports what the notes say; it
does not judge the choices. Row numbers are the inventory's (`inventory.md`).

Paths are under `docs/rewrite/` unless they start with `docs/`, `src/` or `origin/main:`.
`sections/` is `docs/rewrite/sections/`. A citation `file:N` is a line number in the file as it
stands on the current branch.

## 0. What the record is, and its limits

- **Git cannot date the model.** The clone is shallow: the current branch holds 80 commits, the
  oldest 2026-11-14 (`1d5d8b3`..`77cc37a`), and every model file arrives in them already built.
  `git log -S` therefore dates nothing before 2026-11-14. All dates below come from the notes
  (which quote commit hashes that are not in this clone, e.g. `e334d032`, `6985a342`).
- **Two different things are called "v1".** The inventory's "v1 equivalents" describe
  `origin/main` (run `wild-universe-731`, trained 4v4): no belief, no categorical input codes,
  no rotary, no global or zone tokens, three independent action categoricals, dead entities
  masked out of attention. The v1 that the record reviews and that v2 was ported from is a
  later branch, `feat/counterfactual-shot-labels` / `feat/resim-shot-labels` (runs 734-764),
  which is **not in this clone** (`yemong-review.md:3-4`; `decisions.md:573, 1188-1195`). That
  later v1 already had the belief, the 469-wide categorical code, rotary attention on a 65,536 m
  torus, the joint action, the per-level categorical critic, and the density and shot heads
  (`yemong-review.md:10-49, 268-277`). So "INHERITED" below means inherited from that branch.
  Why that branch added those things over `origin/main` is **not in the available record**: its
  own docs (`docs/training.md`, `docs/internal/*`) are not here.
- **Rating noise the notes themselves report.** Two seeds of one configuration differ by about
  110 Elo, "systematic" by seed (`sections/gpu-box-log.md:591-594`); the round-2 bar for a win
  was +50 frontline / +40 on the two-mode mean (`sections/round2-plan.md:16-18`). The later
  independent review calls the 110 "an order of magnitude, not a measured spread": it is the gap
  between two seeds from a few pairs, one confounded (`review.md:147`).
- **No planned arm has run.** The 24 pairwise arms and 6 reference runs of 2026-11-15
  (`arm-review.md`) were being prepared when the simplification audit began; A1-A24 are all
  still plans. Every piece of strength evidence below is from the BC/RL rounds of 2026-11-06 to
  2026-11-14.

### Timeline (model-relevant)

| Date | Event | Source |
|---|---|---|
| before 2026-10 | Later-v1 branch: belief loop, 469-wide codes, rotary on a 65,536 m torus, joint 30-way action, per-level critic, density and shot heads. Run 734 failed from a delta next-state target. | `yemong-review.md:10-49, 489-492, 538-540` |
| 2026-10-16 | "Counts ... are fixed for a given run; zero-shot transfer across counts comes from the model, not from dynamic shapes." Framework then JAX. | `decisions.md:148-149, 77-79` |
| 2026-10-21 | **Decided: remove the belief loop** (Griffin-only memory, dual next-state head, damped rotary). On paper only. | `decisions.md:400-496` |
| 2026-10-22 | No wrap in gameplay; the random offset applied only when building observations. Code padding rule. | `decisions.md:718-722, 498-512` |
| 2026-10-31 | Framework is PyTorch. **v2 starts from v1's model "because it is known to train"**: the belief comes back; the 10-21 design is deferred as a head-to-head "later". No bullets. | `decisions.md:3-7`; `sections/yemong.md:5-18` |
| 2026-11-01 | Spec and first build; fix round adds the per-view offset (period 26,244 m) and corrects rotary to 8192..64 m. | `sections/yemong-notes.md:262-308` |
| 2026-11-02/03 | Five Triton kernels brought in; bf16 by default. | `sections/fast-yemong-brief.md:7-14` |
| 2026-11-06 | Inference ablations (rotary off collapses; forecast off moves the policy 0.004-0.007 nats). Griffin-only switch. Temporal-first arms C, D. | `decisions.md:1327-1343` |
| 2026-11-07 | Belief spread found overconfident (`spread_fit` "coarsest"). Delta heads decided. C-fixed champion. | `sections/belief-calibration-brief.md:3-16`; `sections/delta-heads-brief.md:3-9` |
| 2026-11-08 | Round 2 on the laptop: accelerations (no winner), persistence deltas, the error head and its stability fixes, extrapolation baseline, the quantile spread fix. | `sections/round2-log.md` |
| 2026-11-08 to 11-10 | Rented 3090: tournaments 1-10, hidden-physics, champion chosen at 60M BC, RL to 180M, nearest-MLP baseline, zero-shot fleet sizes, benched ships. | `sections/gpu-box-log.md` |
| 2026-11-11 | Rewards x10, value support 51 bins over +-2 -> 77 over +-3. | `sections/compile-overhaul-notes.md:640-663` |
| 2026-11-12 | Position codes: colour code 236,196 m, rotary ratio 3.405 with a 35 degree turn, arena code, geometry head; rotary read-out retired. Champion's head settings become a new run's defaults. | `sections/position-codes/progress.md:3-6, 71-87`; `sections/release-candidate/progress.md:33-45` |
| about 2026-11-13 | x2 game (v13): velocity code rescaled (99 bins), rotary finest 48 -> 64 m, belief decay constants halved. | `src/bnb/models/yemong/model.py:95-100`; `state_code.py:77-83`; `src/bnb/runtime/train.py:234` |
| 2026-11-14 | Attention shares; masked absent zones cost -325 Elo; fused key mask; gate option. Fog 100%. Seed isolation (keyed draws). | `sections/bc-v19/overnight-log.md:289-452`; `arms-todo.md:27, 34` |
| 2026-11-15 | Fixed-size maps (map-size sampling opt-in); absent fields; evaluation suite. | `arms-todo.md:93`; `sections/fixed-maps/progress.md` |
| 2026-11-21 | Simplification decisions: no belief, no error head, no shot trajectory, density probably out, outcome into value, no Triton kernels, simpler positions. | `simplification.md:118-130` |

---

## 1. Inputs and encodings

### 1a. Observation contract (rows 1.1-1.7)

**1.1 Ship rows; world coordinates; first 20 columns zero when hidden.**
- Added: design 2026-10-16 to 10-22 (`decisions.md:131-139`): a raw team-relative observation,
  "absolute world positions (models add rotary or other encodings themselves)". Hidden values as
  zeros plus a `visible` flag decided 2026-10-21 (`decisions.md:491-494`). Acceleration columns
  added 2026-11-08 (round-2 O1/O10, `sections/round2-plan.md:63-66`; merged
  `sections/round2-log.md:54-60`). `benched` 2026-11-10 (`sections/gpu-box-log.md:842-850`).
- Stated reason: "nothing model-specific (codes, belief composition, feature transforms) lives in
  the env" (`decisions.md:133-135`), because v1's env returned a `YemongObservation` and the
  observation depended on the policy's belief store (`decisions.md:57-60`). "Full truth plus
  masks" was rejected because it "leaks hidden information to any model that ignores or
  mishandles the mask, silently inflates results on a masking bug" (`yemong-review.md:128-131`).
  Relative/ego-centric inputs were rejected because they "conflict with count-agnostic models"
  (`yemong-review.md:104-105`); ego-centric frame later dropped as "xN compute"
  (`arms-todo.md:96`).
- Evidence: ARGUED.
- Still exists: STILL EXISTS for the contract. The acceleration columns' only reader today is the
  extrapolation baseline (the `acceleration_inputs` arm lost and is off): GONE with the belief
  unless a simple next-state head wants them. The model-exchange/leaderboard motive is now a
  non-goal (`simplification.md:81-82`).
- Size: none recorded.

**1.2 Columns observed but not read (`proper_speed`, `time_since_seen`).**
- `time_since_seen`: the first build read the observation column; the fix round switched to the
  belief's own age ("`time_since_seen` is `age x tick`, from the belief. `Replay.age` carries
  it", `sections/yemong-notes.md:119-122, 305`). The stated cause was that the spec's replay did
  not include the age. Evidence: ARGUED (a spec-consistency answer). Still exists: GONE with the
  belief; the observation column becomes the natural source again.
- `proper_speed`: NOT FOUND (looked in `contracts.md`, `sections/yemong-spec.md`, `sections/game-*`).

**1.3-1.5 Field rows, zone rows (Z = 5 in every mode), global row.**
- Added: fields and zones from v1; five zone slots in every mode and the `present` column with
  mixed batches (`sections/mixed-spec.md:33-45`; `sections/mixed-notes.md:330-338`). Global row:
  "Because one agent may play every mode, the observation carries the objective (zone roles and
  owners, capture progress, scores, time remaining) on the global token" (`decisions.md:160-161`).
  Fields' `present` column 2026-11-15 (`sections/absent-fields/progress.md`).
- Stated reason for fixed Z: "several games ... in one batch, one set of tensors and one compiled
  graph" (`sections/mixed-brief.md:3-7`).
- Evidence: ARGUED; cost MEASURED (+3.4% Yemong rollout, `sections/mixed-notes.md:337-338`).
- Still exists: PARTLY. Mixing modes stays (`simplification.md:143-146`), but the padding to one
  shape is listed as a band-aid of the static-shape rule ("5 zone slots everywhere (+3.4%
  rollout), absent fields as fake fields": `simplification.md:513`).

**1.6 No bullet observation.**
- Added: v1's shipped config already had bullet reading off (`yemong-review.md:23-24, 53-56`).
  Author, review part 1: "not observed by Yemong, but present in the raw observation"
  (`yemong-review.md:98-99`); superseded 2026-10-31 "no bullets in the observation"
  (`decisions.md:136`; `sections/yemong.md:14`).
- Stated reason: none recorded. The review asked "Was turning bullet reading off a deliberate
  gameplay/perf choice ... or a temporary setting?" (`yemong-review.md:86-87`); the outcome
  records the answer, not a reason.
- Evidence: INHERITED.
- Still exists: UNKNOWN. Consequence noted at the time: "Any dodging has to be inferred from enemy
  pose, `is_shooting` and the counterfactual shot training" (`yemong-review.md:53-56`), so the
  shot heads partly exist to compensate (chain C8). "Whether in-flight bullets reach the model"
  was parked (`sections/round2-plan.md:175`).

**1.7 Fog on in every training environment.**
- Added: 2026-11-14 (`arms-todo.md:34`): "100% fog in training, BC and the game. Full information
  is for debugging only." Earlier decision was a mix with a fog flag (`decisions.md:439-440`;
  `sections/yemong.md:44-46`).
- Evidence: NOT FOUND for the change (idea 48 "always on against a mix" is listed as settled
  "always on", `arm-review.md:302`, without a measurement).
- Still exists: STILL EXISTS (fog stays; only the belief goes).

### 1b. Ship-token feature groups (rows 1.8-1.17)

**1.8 / 1.20 The ship state code as input (542 wide, 23 softmax groups).**
- Added: INHERITED from later-v1 (469 wide, 22 groups: `yemong-review.md:37-40`); rebuilt in v2
  units with the angle of attack (490 wide, `sections/yemong-spec.md` section 3;
  `sections/yemong-notes.md:18`); 542 after the velocity code grew to 3 x 99 (Yemong v8,
  2026-11-14).
- Stated reason: none for the form itself. What the record gives: "Truth gives an exact code; a
  hidden enemy's belief gives a code smoothed by its spreads" (`yemong-review.md:39-40`), and the
  next-state head "predicts the same vector, so a forecast carries its own uncertainty"
  (`docs/explanation/yemong.md:123-126`). The author called it "one model's choice, not a
  contract" and was "unsure about the current belief system" (`yemong-review.md:110-112`).
- Evidence: INHERITED. The only input-encoding comparison: Fourier position/heading inputs in
  place of the colour runs lost by 212 Elo to their base at 30M BC, one seed (573 vs 785,
  `sections/gpu-box-log.md:439-446`); the planning notes discount it: "on an older model and game,
  and never against raw inputs" (`arm-review.md:166-173`).
- Still exists: PARTLY. Two of its three jobs go: presenting a belief's uncertainty as blur, and
  being the absolute next-state head's target format (that head is no longer the default). The
  audit's finding 4: "1,037 operations per forward against 148 for v1's whole encoder.
  Assumption: the model owns all feature encoding, as two-hot codes of everything. Open."
  (`simplification.md:301-303`).
- Doubts: arm A17 (codes against raw scalars) never ran. The 2026-10 review decided the codes
  should become HL-Gauss ("Decided: HL-Gauss codes go into the port", `yemong-review.md:628`),
  for shift invariance (`yemong-review.md:591-595`); the port kept v1's two-hot / moment-matched
  form ("v1, exact", `sections/yemong-spec.md` 3.3) and the padding rule for "the existing inputs
  and the critic" was parked (`sections/round2-plan.md:176`). So the documented defect of the
  two-hot input ("an observed ship looks different to the network depending on offset") is still
  in the code.
- Size: none.

**1.9-1.11 Team one-hot, ship-type one-hot, flags.**
- Added: team and type as input one-hots from v1/the review ("Teams: a categorical feature, two
  teams for now", `yemong-review.md:100-101`); ship types with the multi-type card; `benched`
  "since Yemong v3" (`sections/yemong-spec.md:233`).
- Evidence: INHERITED / ARGUED.
- Still exists: STILL EXISTS. N-team generality is now a non-goal (`simplification.md:79`).
- Size: `benched` is always 0 in training; `mask_benched` is untrained-on (12.3).

**1.12 Time since seen (symlog).**
- INHERITED (`yemong-review.md:41-42`). The review outcome said "Fourier-encoded like the other
  time features" (`yemong-review.md:122`); not implemented. Still exists: UNKNOWN (depends on
  what a hidden enemy's row is once it "leaves the observation").

**1.13 Respawn-wait features (16 s horizon, 4 harmonics).**
- INHERITED ("respawn time (Fourier)", `yemong-review.md:42`); horizon in
  `sections/yemong-spec.md:234-235`. Reason for 16 s and 4 harmonics: NOT FOUND. STILL EXISTS.

**1.14 Pending action (42 inputs; own previous action / the model's forecast of the enemy's).**
- Added: INHERITED. In later-v1 the pending command was the one physics consumes next and enemy
  commands "are private even when the ship is in sight", so enemy rows read the stored prediction
  "whether visible or not" (`yemong-review.md:421-425`). Ownership moved into the model's carry
  on 2026-10-31 (`sections/yemong.md:16`).
- Evidence: INHERITED. The author's 2026-11-08 choice for new information went the other way:
  "Physical accelerations as inputs ..., not the enemies' previous actions"
  (`sections/round2-plan.md:33`). Inference ablation `forecast` (which also makes every enemy
  action uniform) moved the best RL policy by 0.004-0.007 nats (`decisions.md:1330-1332`).
- Still exists: own rows STILL EXISTS; enemy rows GONE with the belief ("its only rollout use was
  the belief", `simplification.md:134-135`).
- Doubt: hidden assumption 4 of review 3b: "a soft distribution for enemies sits in the same
  channel as hard one-hots for allies; the model separates them by team" (`yemong-review.md:436-437`).

**1.15 Refractive-index gradient (scaled by 0.065 per m, zero when hidden).**
- INHERITED (v1 fed it as an identity feature); scale in `sections/yemong-spec.md:237-238`.
  Reason: NOT FOUND. STILL EXISTS.

**1.16 Local presence (radius 150 m).**
- INHERITED ("two local-presence scalars", `yemong-review.md:43-44`). Radius set by ratio: "about
  0.8 of the gun's reach, as v1's 330 px was of its 421 px" (`sections/yemong-spec.md:250-256`);
  measured 2026-11-01 at 0.75 of a median 197-213 m reach and kept (`sections/yemong-notes.md:169-175`).
  "39 presence radius | defaults kept" (`arm-review.md:289`).
- Stated reason for having it at all: NOT FOUND.
- Evidence: INHERITED; the constant MEASURED against a game that has since changed (real-scale
  ships, x2 speeds, different bullet life).
- Still exists: UNKNOWN.
- Size: a sum, not a mean, so it grows with local density (12.8). No note discusses this.

**1.17 A row with no belief sits at the map centre.**
- Added: fix round 2026-11-01: "A row with no belief sits at the map centre, not at the frame
  origin. Deviation: at the origin it would reveal the offset to rotary attention"
  (`sections/yemong-notes.md:283-284`).
- Evidence: ARGUED. Exists only because of the offset (1.18) and the belief (4.1).
- Still exists: GONE (both causes removed). The 2026-10-21 design assumed "there are no
  never-seen ships (every ship is revealed to both teams when it spawns)" (`decisions.md:466-468`);
  v2's game does have ships spawned unseen (inventory 4.2), so that invariant did not survive.
- Size: these rows are unmasked keys all at one point; their number scales with the fleet (12.6).

### 1c. Encodings (rows 1.18-1.31)

**1.18 Random position offset (uniform over 236,196 m per axis per match).**
- Added: v1 drew the map centre "uniformly on the torus per episode" (`decisions.md:723-725`).
  2026-10-22: physics in a map-local frame, "the per-episode random offset ... is applied only
  when building observations, so models still see uniform absolute positions"
  (`decisions.md:718-722`). The 2026-11-01 spec dropped it ("the centre is always the origin";
  the global token "drops v1's position code", `sections/yemong-spec.md:589`); the same day's fix
  round brought it back as "Design change: a per-view position offset", period 26,244 m
  (`sections/yemong-notes.md:268-287`). Period 236,196 m on 2026-11-12.
- Stated reason: "The torus exists for the network: translation invariance and uniform location
  codes" (`decisions.md:266-269`); "The categorical position codes then cannot memorize absolute
  map positions, while rotary attention and local presence read only differences, where the
  offset cancels" (`docs/explanation/yemong.md:79-81`). And: "Coarse levels are near-constant
  within one map, which is why map scale is randomized in training" (`yemong-review.md:64-66`).
- Evidence: INHERITED / ARGUED. No run compares with and without the offset.
- Still exists: GONE by decision: "the 236 km range and the random offset served varied map
  sizes, which are not needed now. The map is not toroidal (v1's was)" (`simplification.md:128`).
  Note the record's own stated reason was non-memorisation of absolute positions, not map size;
  the map-size link is through the code's range (1.19).
- Doubts: map-size randomisation, which the review said the absolute code needs, was off in
  every run that produced a rated agent ("Map size never varied in training",
  `sections/gpu-box-log.md:833`), and was made opt-in on 2026-11-15.

**1.19 Position colour code (9 levels of 3 colours per axis, 36 m to 236,196 m, blurred by sigma).**
- Added: INHERITED (9 levels on the 65,536 m torus, finest about 3.4 m: `yemong-review.md:64-65`).
  v2: 4 m to 26,244 m. 2026-11-12: 36 m to 236,196 m.
- Stated reason for the 2026-11-12 resize, the "sizing rule" (`sections/yemong-spec.md:273-284,
  306-309`; `src/bnb/models/yemong/state_code.py:59-73`): finest period above twice the largest
  one-tick move ("else the level's phase change per tick is ambiguous and the level reads as
  noise"; 10.7 m a tick, so above 21 m); coarsest above "the largest absolute span ... the largest
  evaluation maps are about 62 km across". The old defaults "break both rules".
- Evidence: ARGUED. No strength comparison of old against new codes. The 2026-11-09 champion
  (2161 Elo at RL 180M) trained on the old codes that "break both rules".
- Still exists: the 62 km evaluation maps came from the density rule (arena radius ~ sqrt(n/5)),
  which the author dropped for evaluation on 2026-11-10 ("no map-size deviation",
  `sections/gpu-box-log.md:877`) and for everything on 2026-11-15 (fixed maps). The blur exists
  for the belief. So GONE for the range and the blur; the Nyquist argument for the finest level
  STILL EXISTS for any periodic absolute code.
- Doubts: "if [variable maps] don't [survive], the colour codes and rotary frequencies are
  re-tuned for fixed sizes" (`arms-todo.md:89`): planned, never done. The audit describes the
  code as "an 8-level colour code down to 4 m" (`simplification.md:298`); the champion build has
  9 levels down to 36 m (inventory 1.19): the audit's sentence mixes the old finest level with
  the new range.

**1.21 Velocity code (3 axes x 99 bins, log-compressed, reach 1,059 m/s).**
- Added: INHERITED (3 projections x 81 bins). Ranges checked 2026-11-01: share of ship-steps above
  319 m/s at most 5.1e-5, "The ranges are kept" (`sections/yemong-notes.md:154-167`). Rescaled for
  the x2 game: "the x1 design (knee 50, spacing 2.5, 81 bins, reach 319 m/s) scaled exactly by 2"
  (`src/bnb/models/yemong/state_code.py:77-83`; `sections/bc-v19/overnight-log.md:71-72`).
- Evidence: INHERITED; range MEASURED. Still exists: PARTLY (the covariance blur serves the
  belief). A known limit: "the velocity code's bins ... are wider than any one-tick change ... so
  it cannot resolve accelerations" (`sections/acceleration-inputs-brief.md:9-12`).

**1.22-1.25 Heading, nose-rate, scalar and cooldown codes.** INHERITED; nose-rate range checked
(largest 7.4 of 8 rad/s, `sections/yemong-notes.md:166`). Shield and energy are coded as fractions
of the type's maximum (`sections/yemong-spec.md` "Deliberate changes"), although the review's
recorded outcome was "absolute units with global maximums for every channel, resources included
(not fractions of a type's maximum)" (`yemong-review.md:1472-1475`): the record contradicts
itself. Still exists: PARTLY (the spread halves go with the belief).

**1.26 Rotary position bands (8 bands, 64 m to 339.6 km, ratio 3.405, 35 degrees turn per band).**
- Added: INHERITED (base-2 harmonics of the torus, 10 per axis: `yemong-review.md:153-168`).
  v2 2026-11-01: 8 octaves; the first text gave periods 64 m down to 0.5 m by a sign error, caught
  by the implementer ("Ships fight at 100-1000 m, so the finest pairs alias heavily"), corrected
  to 8192..64 m (`sections/yemong-notes.md:146-147, 246-249, 304`). 2026-11-12: ratio 3.405 from
  48 m to 254.7 km with a turn. x2 game: finest 64 m ("ships close at up to ~900 m/s, 30 m a
  tick", `src/bnb/models/yemong/model.py:95-97`).
- Stated reason (2026-11-12): coarsest above "twice the largest pairwise distance. A relative
  phase is unique only within +-P/2 ... (the largest evaluation maps are about 62 km across)";
  ratio not a power of 2 because then "a one-cycle misread of a coarse band is invisible to every
  finer band"; the turn because the bands' alias lattices line up along the world axes
  (`sections/yemong-spec.md:285-305`).
- Evidence: MEASURED, but a geometric property, not play. Aliasing sweep
  (`sections/position-codes/progress.md:28-61`): the old ladder aliases exactly at 8192 m
  (similarity 1.000 at 16.4 km); ratio 3.405 on world axes has a far ghost at 64.6 km (0.90);
  turned sets have none within 130 km; 35 degrees chosen from a neighbourhood. The notes' own
  limit: "the similarity is the uniform-weight code (an untrained query)" (line 65-67).
  Reliance, not necessity: with rotary turned off at inference the best RL model "loses every
  game to the BC clone" (`decisions.md:1330-1331`). A BC arm trained without rotary (arm R,
  2026-11-06) was run and rated (`sections/arena-mixed-builds-brief.md:3-7`) but its result is
  NOT FOUND in the repository.
- Still exists: the need for relative geometry STILL EXISTS. The 340 km range is PARTLY expired:
  rotary reads differences, so the offset never forced its range; the 62 km evaluation maps did,
  and those are gone. In training the largest separation is about 15.6 km (inventory 12.38).
- Doubts: "rotary was never ablated; keeping both [rotary and the colour code] was decided by
  review" (`arm-review.md:175-182`, arm A18, never run). The launch file's comment still says
  48 m to 254.7 km and a docstring says 339,695 m where the product is 339,625 m (inventory,
  "Uncertain").

**1.27 Rotary heading harmonics (1, 2, 4).**
- INHERITED (4 attitude harmonics in v1; 1, 2, 4, 8 in the first v2 build). Reduced to 1, 2, 4 on
  2026-11-12 (`sections/position-codes/progress.md:4-5, 74-75`). Reason for dropping the 8th
  harmonic: NOT FOUND. Tokens without a heading get angle 0, so "map objects get an
  absolute-heading term rather than a relative one" (`yemong-review.md:164-166`): a noted quirk,
  never revisited.

**1.28 Arena code; 1.29 raw arena radius / 2600.**
- 1.29 from the spec ("`arena_radius / 2600 m`", `sections/yemong-spec.md` section 4); the density
  head's grid scales by this radius "which is on the global token" (`sections/yemong.md:25-26`).
- 1.28 added 2026-11-12: "so maps of every size read with the same resolution in proportion"
  (`docs/explanation/yemong.md:101-104`); range 0.25x to 8x (1,950 m to 62.4 km).
- Evidence: ARGUED. Still exists: PARTLY. The arena "enters as what it is (... its radius as an
  input)" in the new goals (`simplification.md:60-62`), but the 8x range served density-rule maps.
  Training shows the code two values (12.34). A NaN hunt cleared it: "log(0) at a closed boundary
  gives -inf, which clamps" (`sections/bc-v19/overnight-log.md:52-55`).

**1.30 Timer code.** Added with Frontline's timers (game version 8). Reason: "These are the
model's other countdowns of the same kind ..., so they share its scale"
(`docs/explanation/yemong.md:108-119`). ARGUED. STILL EXISTS. Size: 16 s horizon fixed (12.44).

**1.31 symlog scalars.** From the modes section (`sections/modes-notes.md:156, 238`). ARGUED.
STILL EXISTS. Size: the score target is 5 kills x ships per team (12.9), one value per mode in
training.

---

## 2. Tokens and embeddings (rows 2.1-2.10)

**2.1 One ship token per row, dead and unseen included; 3.4 no attention mask; 3.5 no neighbour limit.**
- Added: INHERITED from later-v1: "No attention masks at all: every ship is revealed on spawn and
  belief validity is sticky, so all tokens are always valid (keeps the flash kernel)"
  (`yemong-review.md:25-26`). (`origin/main` masked dead entities out of keys.) `mask_benched`
  2026-11-10 for unequal teams (`sections/gpu-box-log.md:842-850`); `mask_absent_fields`
  2026-11-15.
- Stated reason: the belief makes every row valid, and an unmasked call keeps the fast attention
  kernel.
- Evidence: INHERITED. Cost of masking MEASURED 2026-11-14: leaving the fused kernel cost +12.7%
  per learner micro-batch and +20.8% per policy step; a key mask built into the kernel then made
  masking free (`sections/bc-v19/overnight-log.md:421-452`).
- Still exists: GONE on both counts (no belief, no custom kernel). But see lesson L4: a trained
  model uses unmasked placeholder tokens as a place to put attention.
- Size: 12.5, 12.6. No note analyses how the single softmax behaves at 56 or 116 keys.
- Never tested: training with masked dead/unseen rows; "benched-ship training as a cheap arm"
  was "reopened" (`overnight-log.md:450`) and became arm A2, never run.

**2.2 Global token.** INHERITED ("super token"); rule and objective inputs per `decisions.md:160-161,
234-235`. Its position code returned with the offset (`sections/yemong-notes.md:276-277`): with
the map centre always at the origin the spec had dropped it. Outcome and density read it. Still
exists: STILL EXISTS as a place for global inputs; its 81-wide centre code GONE with the offset.

**2.3 / 2.7 Field tokens as key/value memory, 64 wide.** INHERITED (`map_read_mode="kv_memory"`,
`yemong-review.md:19-22`); the author kept "K/V-only memory; drop the full-attention variant"
(`yemong-review.md:238`). Reason: NOT FOUND (the review states the fact: "never queries, never
recurrent, never through the FFN"). One softmax over ships and map "(ships can trade attention
between a nearby field and a distant ally)" (`yemong-review.md:21-22`). Measured use: fields take
30-35% of an own ship's attention mass (`docs/explanation/yemong.md:666-675`). STILL EXISTS.

**2.4 Zone token.** INHERITED plus the lock timers (1.30). STILL EXISTS.

**2.5 Absent-zone token (one learned vector, placed at the map centre).**
- Added: mixed batches (`sections/mixed-spec.md:48-54`): "Absent zone tokens are a learned
  vector ... (selected with `where`; no mask)"; "an absent token takes the global token's
  rotation ... No rotation (the frame's origin) would leak the per-team position offset".
- Stated reason: one compiled graph at fixed shapes across modes.
- Evidence: ARGUED at adoption. MEASURED later (2026-11-14): the five identical slots take 13-14%
  of attention mass, a fifth of the last layer's; masking them in a trained RL policy costs
  -325 Elo in deathmatch (64-416 over 480 games), -158 in elimination; the BC policy -374
  (`sections/bc-v19/overnight-log.md:346-393`). "What looks like a sink is the zone readers with
  nothing to read" (`docs/explanation/yemong.md:677-692`).
- Still exists: the shape reason is GONE if modes run on their own slices; the placement at the
  centre is GONE with the offset. The functional role the model gave the token is a lesson (L4).
- Doubts: an explicit sink logit or the gate "stays an arm candidate" (A19), never run.
- Size: training never shows 1-4 zones (12.29).

**2.6 Kind projections, no learned type/slot embedding; 2.9 nothing indexed by slot.**
- INHERITED (`encoder_split`, `yemong-review.md:34-35`). "Separate encoders for allies and
  enemies ... Candidate ablation" (`sections/yemong.md:35-36`): never run.
- 2.9's reason: "zero-shot transfer across counts comes from the model" (`decisions.md:148-149`);
  "Nothing in the weights depends on N, M or the arena, so one set of weights plays at every
  size" (`sections/yemong-spec.md:35-36`). "Slot identity = memory identity: fine, with a test"
  (`yemong-review.md:239`).
- Evidence: ARGUED; that the weights do run at other sizes is MEASURED (section 7 below).
- STILL EXISTS.

**2.8 Token order and slicing by position.** From the observation's team ordering. At play only
enemy rows run the enemy-action/next-state heads, for speed (`sections/yemong-notes.md:334-335`):
that half exists for the belief. Still exists: PARTLY.

**2.10 Input-column initialisation N(0, 1/groups).** INHERITED (`sections/yemong-spec.md:311-313`).
Reason: NOT FOUND. GONE if the codes go.

---

## 3. Trunk (rows 3.1-3.14)

**3.1 Two blocks, width 128, temporal first.**
- Added: size INHERITED ("known to train: about 1500 Elo", `sections/yemong.md:7-9`). Order changed
  from spatial-first to temporal-first for BC arms C and D on 2026-11-06 (`decisions.md:1338-1343`).
- Stated reason: the order was introduced so arm D could have "Griffin-only memory without any
  forecast fed back between ticks, yet hidden ships placed where memory thinks they are"
  (`sections/temporal-first-brief.md:11-12`); generally "attention can use what the recurrence
  remembers" (`docs/explanation/yemong.md:697-699`).
- Evidence: arm C (temporal first, belief on) became champion as "C-fixed"; the only numbers in
  the repository are "30M: frontline -221 (C -266)" and "round 0: C and C-seed1 played the same"
  (`sections/round2-plan.md:16, 24`). The tournament of arms A, R, C, D (2026-11-07) is on the
  laptop only: NOT FOUND here. So: MEASURED per the notes, numbers not recoverable.
- Still exists: STILL EXISTS as a choice; its original purpose (arm D's read-out) is gone
  (read-out retired 2026-11-12).
- Never tested: A15 (2 Yemong blocks against 6 spatial blocks), A23 (width 192), `spatial_only`
  (exists as a switch, `docs/guides/training.md:160-167`; no result found).

**3.2 Spatial sublayer (2 heads of 64, one softmax over queries and map memory).** INHERITED.
Two wide heads because the rotation needs room: 38 of 64 dims are rotated; a config check refuses
more pairs than fit a head (`src/bnb/models/yemong/model.py`, `__post_init__`). A20 (2 against 4
heads) never ran. STILL EXISTS.

**3.3 Rotary attention.** See 1.26. No additive relative bias: v1's "relational bias" was "dropped
(off in v1's run)" (`sections/yemong-spec.md` section 12).

**3.6 Fused attention kernel (and the other four Triton kernels).**
- Added: 2026-11-02/03, from a second "blind race" implementation (`sections/fast-yemong-brief.md:1-14`).
- Evidence: MEASURED: training bf16 51.5 -> 37.7 ms per 160-row micro-batch (1.23x), attention
  9.0 ms of the gain (`simplification.md:331-340`).
- Still exists: GONE by decision 2026-11-21: "They treated the symptom (too many small
  operations) rather than its cause" (`simplification.md:130`).
- Size: its docstring: "fine up to a few dozen keys; at hundreds it needs a tiled kernel"
  (`src/bnb/models/yemong/kernels/attention.py:21-22`). The 40v40 evaluation (86 keys) did run
  through the rating engine (`sections/gpu-box-log.md:910-917`); which attention path it used is
  not recorded.

**3.7 Gated MLP.** INHERITED (GEGLU, 4x). Fusing gate and up measured no gain alone (88.4 vs
87.8 ms) and was reverted, then done in the fix round (`sections/yemong-notes.md:194-195, 331`).
"33 feed-forward width | defaults kept" (`arm-review.md:289`). STILL EXISTS.

**3.8 / 3.9 Temporal sublayer, RG-LRU.**
- INHERITED. Hidden assumption recorded in the review: Lambda = linspace(0, 4) gives "a time
  constant of ~6-7 decisions (~0.2 s at 30 Hz). Longer memory requires the gate to close ...
  Relevant if Griffin is to take over memory from the belief system later"
  (`yemong-review.md:196-201`). Author: "move to a long-memory initialization (Griffin-paper
  style)" (`yemong-review.md:237`; `decisions.md:436-438`).
- What the code has: linspace(0, 4) ("v1, exact", `sections/yemong-spec.md:430`; inventory 3.9).
  `references.md:112` says "the paper's decay initialisation was kept": the record contradicts
  the code. The long-memory initialisation was never implemented, because the Griffin-only
  design it belonged to was deferred on 2026-10-31.
- Evidence: INHERITED. A16 (Griffin against GRU) never ran.
- Still exists: STILL EXISTS, and matters more: with the belief removed the recurrence is the
  only memory of hidden enemies, which is exactly the case the review flagged.
- Related, also still true: "Truncated backprop at 128 steps (~4.3 s) ... What truncation blocks
  is learning to write a one-shot fact whose payoff comes more than one rollout later"
  (`yemong-review.md:204-205, 247-253`).

**3.10 Carry and resets.** INHERITED: episode end zeroes state and buffer; "Respawn does not reset
memory" (`yemong-review.md:187-192, 236`). STILL EXISTS.

**3.11 RMSNorm; 3.12 residual scheme; 3.14 default initialisation.** INHERITED. "34 RMSNorm |
defaults kept" (`arm-review.md:289`). The keyed, per-parameter-path draws of 3.14 are seed
isolation (2026-11-14, "Arms share random streams wherever they're the same", `arms-todo.md:27`):
GONE as a requirement (bit-exactness is a non-goal).

**3.13 Segment evaluation (`unroll`) from a stored replay.**
- Added 2026-11-01: "`step` returns what it read from its belief, and the runtime stores it.
  `unroll` then recomputes a segment in parallel without replaying the belief"
  (`sections/yemong.md:62-64`). Scan: Hillis-Steele replaced by a chunked scan (chunk 8: 12.9 ms
  and 448 MiB against 27.2 ms and 1152 MiB, `sections/yemong-notes.md:310-321`).
- Evidence: MEASURED (speed/memory, agreement 2e-6).
- Still exists: `unroll == scan(step)` STILL EXISTS; the replay record GONE with the belief.

---

## 4. Belief and forecast machinery (rows 4.1-4.9)

**4.1 / 4.2 / 4.7 The belief, sight assimilation, forecast feedback.**
- Added: INHERITED from later-v1. On 2026-10-21 it was decided to remove it: "One kind of memory:
  Griffin recurrence only ... Reasons: Griffin already keeps per-ship memory per token slot; the
  loop had no gradient (stored inputs) and could compound its own errors (run 734); it removes
  the decode, re-encode and round-trip corrections and most of the carry" (`decisions.md:402-409`).
  On 2026-10-31 it stayed after all: "Starting point: v1 Yemong ..., because it is known to
  train ... the Yemong-review design is added as an option once v1 works"
  (`sections/yemong.md:7-13`).
- Stated reason for keeping: a known-good starting point. No reason beyond that.
- Evidence gathered afterwards (all MEASURED, none a clean test of belief against no belief):
  - 2026-11-06, figures: "this loop is broken for positions: errors grow to kilometres while the
    decoded sigma stays near 1 m" (`sections/griffin-only-brief.md:13-15`).
  - 2026-11-06, inference ablation on the best RL model: "forecast feedback off moves the policy
    by 0.004-0.007 nats" (`decisions.md:1330-1332`). Reliance only.
  - 2026-11-07, BC arm C: hidden-ship position error 8 m / 35 m / 114 m / 545 m / 1.76 km at
    0.1-0.5 / 0.5-1 / 1-2 / 2-5 / 5-10 s, truth within one sigma 11% / 0.5% / 0.05% / 0% / 0%
    (`sections/belief-calibration-brief.md:7-13`). Fix: `spread_fit = "coarsest"`.
  - A Griffin-only BC arm (D: no feedback, hidden rotary from a read-out) was trained
    2026-11-06/07. Its rating is NOT FOUND; the read-out was retired on 2026-11-12 "in favour of
    the geometry head" (`docs/explanation/yemong.md:706-707`).
  - Every arm in tournaments 1-10 has the belief, including the "null" (absolute head). The
    comparisons are between ways of forecasting, see 4.3-4.6.
- Still exists: the problem (fogged enemies) STILL EXISTS; the mechanism is GONE by decision
  2026-11-21 ("Hidden enemies leave the observation; the Griffin layers carry memory",
  `simplification.md:122`).
- Doubts: arm A14 ("Belief input ... against the temporal layers alone ... Why an arm: never
  tested", `arm-review.md:139-146`) never ran. The 2026-10-21 note lists what is lost without a
  belief: "a hidden ship's position becomes implicit in the recurrent state instead of explicit;
  hidden ships lose geometry in attention (rotary encoding, relation bias); memory length must be
  learned" (`decisions.md:442-446`), and warns that "the identity rotation means 'at world
  (0, 0)', not 'no position'" (`decisions.md:447-449`).
- Size: none recorded.

**4.3 Extrapolation baseline; 5.4 delta (physics) head.**
- Added: delta head decided 2026-11-07 (`sections/delta-heads-brief.md:3-9`); extrapolation
  baseline as arm 4, 2026-11-08 (`sections/round2-plan.md:42-52, 71-83`).
- Stated reason: predict "each quantity's change from its physics baseline"; the baselines bench
  showed how little is left: extrapolation leaves a 1.9 mm median position residual against the
  rule's 6.6 mm, velocity 0.12 against 0.33 m/s; the nose rate overshoots, so it stays on
  persistence (`sections/round2-log.md:23-34`).
- Evidence: MEASURED.
  - Head accuracy: one-tick position error 0.023 m against the absolute head's 0.277 m, velocity
    0.245 against 0.95 m/s (`sections/round2-log.md:125-127`).
  - Play, BC 30M, correct base, seed-paired (`sections/gpu-box-log.md:587-594`, tournament 4,
    +-12-15): deadreckon-split 796 against absolute 657 at seed 0 (+139); 686 against 545 at
    seed 1 (+141). This is the bundle delta + extrapolation + split against the absolute head.
  - BC 60M, one seed: champion (bundle + hidden physics) 936 against absolute 872, "+64 on the
    mean ... past the +40 bar ... Caveat: seed 0 only" (`gpu-box-log.md:606-613`).
  - BC 120M: champion uniform 989 against absolute uniform 979, "now level (+10)"
    (`gpu-box-log.md:642`).
  - RL from each: +160 at 121.6M (2022 vs 1862, `gpu-box-log.md:733`); 2161 against 2010 at
    180.4M (`gpu-box-log.md:821-822`). One seed each.
  - The figure the audit quotes, "+27 Elo mean, under the +40/+50 bar" (`simplification.md:289-292`),
    is tournament 1 (`gpu-box-log.md:230-235`), which the log itself later invalidated: "no box
    run was C-fixed ... the box's tournament 1 compared arms all on the same wrong base"
    (`gpu-box-log.md:298-311`: the outcome level's discount and trace decay were the defaults,
    not the champion's). The corrected-base results above are larger.
- Still exists: the head becomes "a learner-only auxiliary loss, so a plain regression on
  next-tick truth may do" (`simplification.md:137-138`). The baseline, the 500-logit code, the
  marginal bias tables (about 510 lines) and the decode: GONE or "probably" (`simplification.md:543`).
- Doubts: (a) the reason later-v1 abandoned delta targets: "a delta target let belief error
  persist unseen; the docstring says that ended an earlier run" (`yemong-review.md:489-492`, run
  734). v2's delta head re-met it as "Open loop: ... the head learns the process noise Q and
  expected drift, never a correction toward the truth" (`sections/delta-heads-round2-notes.md:27-28`),
  and answered with the error head (4.6). (b) Suspected defect: the code's bounds "were last
  changed at game version 3 and were measured on the standard ship set at x1 speed. Training runs
  the real-scale set at x2 native speed (game v13), where a tick's move may pass the position
  bound" (`simplification.md:592-596`). (c) Confound C1/C2: with last-tick rates given, "held-control
  ticks leave ~0 to learn; the gradient lands on control changes ..., which is mostly predicting
  the others' policy" (`sections/round2-plan.md:144-145`). (d) "40 delta head, 41 belief decay |
  earlier v2 results" was taken as settled without an arm (`arm-review.md:300`).

**4.4 Rates held at the last sighting, decaying (0.25 s).**
- Added 2026-11-08 05:32-05:48 as a fix: "Arm 4's short run diverged by update 8 (velocity error
  1039 m/s at 2-5 s): for hidden rows the rates were the belief's own change, which includes the
  head's decoded change, so each tick's change became the next tick's acceleration and
  compounded. Agent's fix: hidden rows hold their last sighting's rates ... a decay of the held
  rates ... since a held thrust integrates without bound" (`sections/round2-log.md:103-104`).
- Evidence: MEASURED (the runaway and its cure). Tau: 0.2 s hurt (-124 against its base, 30M,
  laptop seed); 1.25 s and 10 s "+9 and +4 over 0.5": "Tau barely matters"
  (`gpu-box-log.md:439-446, 587-597`). Halved to 0.25 s for the x2 game by argument
  (`src/bnb/models/yemong/model.py`, `extrapolation_decay`).
- Still exists: GONE (exists only because of 4.3 inside the feedback loop).

**4.5 Physics forecast decode and covariance.**
- Added 2026-11-08 with arm 2. The caps are fixes: the Jacobian's "heading += nose-rate dt
  counted the turn twice ... and treated the damped nose rate as a random walk" (implied sigma
  1/3/12 rad against a real 0.18-0.88 rad); "F couples position to velocity only; heading
  variance capped at pi^2/3"; nose-rate variance capped likewise (`sections/round2-log.md:80-81`).
  The decode "needed a mean-inverse table: the plain decode was biased up to 0.37 bins at the
  padded ends" (`round2-log.md:86-87`).
- Evidence: MEASURED (calibration tables). Cost: 1,519 ops per rollout tick (inventory).
- Still exists: GONE.

**4.6 / 5.5 Error (correction) head and its step.**
- Added 2026-11-08 (arm 3). Purpose: "Its predicted variance is the belief's error variance,
  calibrated against the truth; it replaces any analytic covariance propagation"
  (`sections/round2-plan.md:100-102`), and the correction toward the truth that an open-loop
  delta head lacks.
- Every part of the step is a stability fix, in order (`sections/round2-log.md:99-102`):
  1. as briefed, the decoded mean was "tail-dominated (broad code to +-26 km), predicting
     40-400 m for every hidden row (realised 5 m ...), so the step saturated at r for 100% of
     rows" -> step toward the median;
  2. the median step "is unstable too ...: a closed loop, the replay one update stale, gain >> 1"
     -> confidence gain |m|^2 / (|m|^2 + sigma^2);
  3. plus a fixed rate eta = 0.1 "(time constant ~0.33 s; the author's 'rate-limited pseudo
     gradient descent')", then the cap r = the one-tick physical bound;
  4. the second-moment variance was "140x too wide at short hidden ages (... sigma 218 m ...
     against a realised mean square error of 2.6 m^2 ...) ... the confidence gain ... is about 0,
     so the correction never acts" -> the 16/50/84th-percentile spread
     (`sections/error-head-spread-fix-brief.md:3-24`). After the fix "The gain stays ~0.006 (the
     error is mostly unpredictable early, so the correction rightly barely acts)"
     (`round2-log.md:168-171`).
- Evidence for strength: the only pair that isolates the split is persist-split against
  persist-onehead in tournament 1 (wrong base): frontline 739 vs 682, deathmatch 704 vs 711,
  mean 722 vs 697 (+-17-18; `gpu-box-log.md:230-233`): +25, inside the notes' noise. Everything
  else is the bundle (4.3). Hidden physics on top of the split: +38 on the mean at seed 0 in
  tournament 3, then "mean +15 / +10" at the two seeds, trading deathmatch for frontline
  (`gpu-box-log.md:439-446, 594-595`).
- Calibration at the end of BC 30M: "Right at the shortest ages, 2-4x too narrow past 0.1 s
  (overconfident)" (`gpu-box-log.md:138-143`).
- Cost: correction decode 9,068 ops per rollout tick, 33.6% of the tick (inventory 4.6).
- Still exists: GONE by decision ("No correction (error) head. Follows from no belief",
  `simplification.md:123`).
- Size: the position bound stays 26,244 m, "a constant of that code"
  (`sections/position-codes/progress.md:87`), unrelated to the 236 km frame (12.39).

**4.8 Replay record; 4.9 resets.** The replay's dtypes are a bit-agreement fix: Inductor "fused
the fp32 -> bf16 -> fp32 round trip into the kernels that consume it, and by default drops the
rounding there", so the compiled step read unrounded spreads while the replay stored rounded
ones (1.4e-4 drift over 64 ticks); hence float32 spreads and int16 action forecasts, and the
rule "never store a rounded copy that a compiled stage also reads unrounded"
(`sections/yemong-notes.md:377-418`). The replay record GONE with the belief; the rule is lesson L7.

---

## 5. Output heads (rows 5.1-5.11)

**5.1 Policy head, 6.1 action space, 6.2 one joint 42-way categorical per ship.**
- Added: joint categorical INHERITED from later-v1 (30-way; `origin/main` had three independent
  categoricals). 42 from the stick-thirds decision: "Turn actions go from 5 to 7 ...; the joint
  command goes from 30 to 42" (`decisions.md:326-327`); levels later redefined as lift-coefficient
  fractions 0.4 / 0.75 / 1.0 (`decisions.md:310-318`).
- Stated reason for joint rather than factored: NOT FOUND. Review 3a only notes "joint categorical
  size grows multiplicatively with action factors (fine at 42)" (`yemong-review.md:375-376`).
- Per-ship ratio with shared weights "as in MAPPO" (`yemong-review.md:362-363, 393-394`).
- Evidence: INHERITED. STILL EXISTS. Size: none (12.23).

**5.2 Value head: per-ship levels, 77 symlog bins, two-hot.**
- Added: per-level categorical critic INHERITED ("DreamerV3 borrowings: two-hot categorical critic
  and percentile return normalization", `yemong-review.md:395-397`). Bins: 51 over +-5 -> +-2
  ("the largest measured return is 4.55 and every level's median is 0.02-0.13",
  `decisions.md:1208-1209`) -> 77 over +-3 on 2026-11-11.
- Stated reason for 77: returns "were about a tenth of the critic's units" (per-level 5th-95th
  percentiles of +-0.03 to +-0.6; return scale 1.12-1.22 "just above its floor of 1"), so "every
  reward weight times 10 ... and the value support widened to keep its spacing"
  (`sections/compile-overhaul-notes.md:640-663`).
- Evidence: MEASURED (the return distribution); the remedy ARGUED. The alternative "a per-level
  critic unit ... not done".
- Still exists: the audit files the whole thing under "categorical outputs over fixed, very wide
  ranges": "Goes: normalise advantages per batch and predict normalised returns"
  (`simplification.md:541`). The independent review objects that per-batch advantage
  normalisation "does not replace return scaling for the value loss across modes and team sizes
  with different return magnitudes" (`review.md:387-390`).
- Doubts: the review decided HL-Gauss with calibrated bin values for the critic
  (`yemong-review.md:660-661`; `references.md` says the critic uses it); the code is two-hot. A21
  (distributional against scalar) never ran. Why one head per reward level: "The critic's shape
  is the reward function's shape" (`yemong-review.md:289-292`); a single-value arm exists as a
  switch (`SUMMED_LEVEL`), no result found. "13 unified gamma/lambda | dropped; unified
  gamma/lambda failed on 2026-11-13" (`arm-review.md:292`).
- Size: the fixed +-19 support meets returns that grow with fleet size (12.16); nothing recorded.

**5.3 Outcome head (4 classes on the global token).**
- INHERITED (`yemong-review.md:274`). Its own discount and trace decay; BC needed gamma = lambda
  = 1 for it, which lived only in the laptop's queue commands and cost a day of mis-based runs
  (`gpu-box-log.md:298-311`).
- Evidence: INHERITED. Doubts on record: "removing the outcome level from the advantages until
  its critic works" (under discussion 2026-11-06, `decisions.md:1333-1334`); "Brier does not
  track Elo" (`gpu-box-log.md`, tournament 3 table). Decision 2026-11-21: "Outcome merges into
  value" (`simplification.md:126`).
- Size: P(win) - P(loss) x 10 is added to every own ship's value regardless of team size (12.14).

**5.6 Enemy-action head.**
- INHERITED; "Port as is: keep hidden enemies in the loss and keep feeding the prediction back as
  input. Both are candidates for later experiments" (`yemong-review.md:468-469`).
- Evidence: weight 0.11 solved against trunk-gradient shares; "halving it, run 763, did worse"
  (`decisions.md:1222`; later-v1, one run).
- Still exists: PARTLY: "keep or drop. Cheap ...; its only rollout use was the belief"
  (`simplification.md:134-135`).

**5.7 Density head (2 x 331 hex cells, log-rates clipped to [-20, 2]).**
- INHERITED. Made arena-relative so it "generalizes beyond the Frontline circle"
  (`yemong-review.md:331-332`; `sections/yemong.md:25-26`). The review decided a 367-cell disc
  with a 2-sigma buffer, HL-Gauss, a hard cut at the boundary and "No clamp on the log-rate"
  (`yemong-review.md:970-978`; `decisions.md:503-504, 1528-1529`); the port kept v1's 331 cells
  and the clamp ("v1, exact", `sections/yemong-spec.md:509`). The record contradicts the code.
- Stated reason for existing: NOT FOUND (the review describes it; no purpose is given).
- Evidence: none for strength. Its trunk gradient is the largest of the auxiliaries
  (`gradient/norm/trunk`: density 0.076-0.092, next_state 0.013-0.022, policy 0.029-0.072,
  `sections/bc-v19/overnight-log.md:21-24`); trunk share 0.12 in later-v1 (`decisions.md:1214`).
- Still exists: tentatively dropped (`simplification.md:125`).
- Size: bias initialised from the build's 10 ships; rates scale with fleet (12.10, 12.11). Cell
  width 780 m at 7.8 km.

**5.8 Shot-outcome head; 5.9 shot-trajectory head.**
- INHERITED, "tabled pending proper benchmarks ... Port the heads as they are in the meantime"
  (`yemong-review.md:1269-1271`). v2 changes: labels by learner-side re-simulation
  (`decisions.md:1466-1475`); the trajectory head got a second hidden layer and trains on every
  age (`sections/yemong.md:31-34`).
- Stated purpose: "If this ship fired now, where would the bullet go, and what would it hit?"
  (`yemong-review.md:1175-1176`); and, implicitly, to make up for bullets not being observed (1.6).
- Evidence: none for strength. "Shot-outcome gradient is mostly sampling noise"
  (`decisions.md:1233`). The benchmarks tabled item 15 asked for never ran.
- Cost (trajectory): re-simulation "about 26,700 compiled calls per update", about 46 MB per view
  tick, plus a 128-tick lookahead (`simplification.md:557-558, 664`).
- Still exists: trajectory head GONE; outcome head kept with labels from the real game
  (`simplification.md:124, 127, 152-163`).

**5.10 Geometry head (pairs; range, three angles, closure).**
- Added 2026-11-12, "replacing the rotary read-out" (`sections/position-codes/progress.md:5-6,
  81-86`). The read-out (2026-11-07 to 11-12) had been arm D's way to place hidden ships without
  a belief.
- Stated reason: "so the trunk learns to carry relative position, attitude and motion in its rows
  rather than only inside attention's rotation" (`docs/explanation/yemong.md:711-713`).
- Evidence: weight only. 0.007 was "set in commit e334d032, for the rotary read-out the head
  replaced", targeting a trunk gradient norm of about 0.02; measured 0.0205-0.0218
  (`docs/explanation/learner.md:204-205`; `sections/bc-v19/overnight-log.md:19-27, 115-134`). The
  A/B launched for it (BC 20M with and without, `position-codes/progress.md:93-103`) has no
  recorded result; "geometry x3 ... is a Stage A arm" (`sections/launch/progress.md:19`) never ran.
- Still exists: kept by decision 2026-11-21 (`simplification.md:127`).
- Size: 8 partners per anchor per tick, cycled (12.12); range bound 64 km (12.40).

**5.11 Heads at play.** Enemy-row heads run at play only for the belief: GONE.

---

## 6. Action space and sampling (rows 6.1-6.6)

- **6.1, 6.2**: see 5.1.
- **6.3 Inverse CDF with keyed uniforms.** The first build sampled with Gumbel-max from the default
  generator (`sections/yemong-notes.md:62-64, 92-93`). Changed by seed isolation, 2026-11-14:
  "Arms share random streams wherever they're the same and within rounding" (`arms-todo.md:27`).
  Evidence: ARGUED. Still exists: GONE (bit-exactness and common random numbers are non-goals,
  `simplification.md:75-76, 519`). The independent review notes the loss: "v2's notes concluded
  that 'pairwise arms must share seeds', but common random numbers are now a non-goal"
  (`review.md:223-226`).
- **6.4 No action masking.** INHERITED; masks are in the loss (`yemong-review.md:367-369`). The
  review found entropy and policy-gradient masks differed, "a bug", decided to align
  (`yemong-review.md:388-389`). STILL EXISTS.
- **6.5 Temperature 1.** "actions are sampled exactly as in training (temperature 1, never
  greedy)" (`evaluation-suite.md`, Players and play); a sweep exists
  (`sections/temperature-sweep.md`, 2026-11-11). Greedy evaluation "dropped" (`arms-todo.md:96`).
- **6.6 No action repeat.** "physics tick equals the decision tick (repeat 1): 30 Hz"
  (`decisions.md:526-529`), because the new flight model is "~100x more accurate per step".
  ARGUED. STILL EXISTS.

---

## 7. Size generalisation: what the record says

### 7.1 The stated goal and the design claims

- "Yemong's own rule (train at 5v5, evaluate at 50v50)" (`sections/yemong.md:53`).
- "Counts ... are fixed for a given run; zero-shot transfer across counts comes from the model,
  not from dynamic shapes" (`decisions.md:148-149`). "Nothing in the weights depends on N, M or
  the arena, so one set of weights plays at every size" (`sections/yemong-spec.md:35-36`).
- Original map plan: "At inference, map scale follows ship count (a target density). During
  training, map scale is randomized across the full range, or the coarse position levels go
  unused" (`decisions.md:266-269`). Neither half happened in a rated run: training never varied
  the map ("Map size never varied in training", `gpu-box-log.md:833`), and density-scaled
  evaluation maps were dropped (below).
- Losses: "A weight is an exchange rate between nats and does not move with fleet size"
  (`decisions.md:1198-1203`).
- Arm A2 names fleet-size generalisation "the paper's main claim" and says of the current agent:
  "rl-v20 is only even with Imugi at 2v2" (`arm-review.md:27-34`).

### 7.2 Every recorded evaluation away from the training size

**(a) Later-v1 / `origin/main`, run 731, trained 4v4, 4 fields** (`origin/main:docs/evaluation.md`;
a different model: no belief, no rotary, masked dead keys). Final live checkpoint against the
scripted controller, largest scripted team still beaten / first below 50%: 4 ships 7 / 8; 8: 13 /
14; 16: 25 / 26; 32: 46 / 47; 64: 76 / 77 (256 games a cell, fewer at the largest). Ratio "from
2.00x at one ship ... and 1.75x at the training size of four down to 1.19x at 64". Averaged
weights hold even against 1.89x at 4, 1.82x at 8, 1.71x at 16, 1.55x at 32 (4,096 games a point).
Elo stays "far above the scripted controller at every symmetric size from 1-vs-1 to 64-vs-64".
The doc's own caution: "It does not by itself establish a universal scale-invariance law."

**(b) v2, 2026-11-10, 3090 box** (`sections/gpu-box-log.md:830-917`). Champion = RL 180.4M
uniform average of deadreckon-split-hiddenphysics, trained 5v5 on fixed maps (the old 26,244 m /
8192 m position codes); null = the absolute-head line; nearest = a per-ship MLP over the nearest
3 allies and 4 enemies. Scripted = 1000.

| Condition | Champion | Null | Nearest MLP | Line |
|---|---|---|---|---|
| 5v5 rating, frontline / deathmatch | 1997 / 2404 | 1875 / 2235 | 1303 / 1386 | 861 |
| 10v10, training arena | 1875 / 2778 | 1680 / 2606 | 1240 / 1345 | 874 |
| 10v10, density arena (3677 m) | 1499 / 2317 | 1358 / 2113 | 958 / 1307 | 876 |
| 20v20, training arena, frontline | 1134 | | | 913-917 |
| 20v20, training arena, deathmatch | won 1024 of 1024, Elo a lower bound (~2551 +- 142) | | 859 +- 19 | 905-908 |
| 40v40, training arena, frontline | **526 +- 28 (below scripted)** | | 1041 +- 19 | 911-913 |
| 40v40, training arena, deathmatch | saturated (2445 +- 246) | | -440 +- 246 | 911-913 |
| 40v40 smoke, frontline at density | "every trained agent ~140-270 vs scripted 1000" | | | 836-838 |

Break-even ("5 network ships break even against m scripted", line 875): frontline champion 6.36
(x1.27), null 5.87 (x1.17), nearest 5.44 (x1.09); deathmatch champion 7.46 (x1.49), null 7.70
(x1.54, "tied"), nearest 5.67 (x1.13). At 10 ships, frontline, training arena: champion 11.19
(x1.12), null 11.15, nearest 10.79 (lines 881-882).

Fixed odds, deathmatch, training arena, 128 games a ratio (lines 902-910), champion win rate at
1:1 / 1.2:1 / 1.5:1 / 2:1 scripted-to-network:

| Network ships | 1:1 | 1.2:1 | 1.5:1 | 2:1 | Nearest at 1:1 |
|---|---|---|---|---|---|
| 10 | 1.00 | 0.95 | 0.73 | 0.43 | 0.89 |
| 20 | 0.97 | 0.91 | 0.69 | 0.22 | 0.31 |
| 40 | 0.97 | 0.53 | 0.03 | 0.00 | 0.00 |

The log's findings (lines 913-917): "deathmatch skill transfers zero-shot to 20 ships for Yemong
(~70% wins outnumbered 1.5:1 at 10 and 20 ships) and breaks at 40 in the 2600 m arena (1.2:1 a
coin flip, 1.5:1 3%); the nearest MLP fails as fleets grow even at even numbers (89% / 31% / 0%
at 10 / 20 / 40); crowded frontline erodes the champion's edge (1997 / 1875 / 1134 / 526 at 5 /
10 / 20 / 40) below the nearest MLP's at 40."

**(c) v2, rl-v20 (BC 85M + RL, real-scale, mixed modes, fixed maps), 2026-11-15**
(`evaluation-suite.md`, "Frontline break-even has no fit"; `arms-todo.md:166`). Head-to-head
against Imugi in frontline, rl-v20 won: 5v5 and 10v10 on the preset 7.8 km map, every game; 10v10
on the 11 km density map, 58 of 64; 14v14 on 13 km, 15 of 64 "though it outkilled Imugi (61 kills
against 52 per game)"; 20v20 on the preset map, 26 of 64 with 14 draws; 20v20 on 15.6 km, 0 of 64,
"about 8 kills against 27 deaths per game, and every game over in about 2 minutes". In deathmatch,
rl-dm "won 5 of 32 at 20 against 20". Pilot scores A - 1/2 for a close pair at 1v1, 2v2 and 10v10
are in the same table (frontline 10v10 -0.149, deathmatch 1v1 +0.219), on density maps.

**(d) Map scale.** "rl-v20, trained on fixed maps, collapses in frontline above 1.4x"
(`arm-review.md:19`). The measurement behind "1.4x" is NOT FOUND as a table; (c)'s 13 km and
15.6 km rows are consistent with it. bc-v21 with map-size sampling (sigma 0.35 over [0.5, 2]) was
launched 2026-11-14 and "stopped for the shutdown" (`sections/bc-v19/overnight-log.md:434`;
`ideas.md`, item 54). No agent has trained on variable maps.

**(e) Small teams.** "rl-v20 is only even with Imugi at 2v2" (`arm-review.md:31`).

**(f) Game side (not the model).** Ship-type balance itself shifts with fleet size under the
scripted agent; "20v20 fails on every card" in the balance work (`sections/balance-sizes/progress.md:388`).
Frontline with small teams captures slowly (2v2: 85% of games end on the clock, `arms-todo.md:154`).
Deathmatch tickets scale per ship.

### 7.3 What the notes say about why

- **Map size, not fleet size, is the measured cause of the frontline drop at 10v10**: "the bigger
  map costs frontline ~-375 (every trained agent; nearest below scripted) and deathmatch ~-460
  for Yemong, ~-40 for nearest: the frontline drop at 10v10 is the map size, not the fleet"
  (`gpu-box-log.md:876`). And for 20-a-side: "For an agent trained on one map size, frontline
  break-even there measured transfer to a map twice the size, not skill" (`evaluation-suite.md`).
- **Crowding**: "crowded frontline erodes the champion's edge" (`gpu-box-log.md:916-917`); 20 or
  40 ships a side on a 5v5 arena. No mechanism is identified.
- **The author's response each time was to remove the condition, not to diagnose the model**:
  "no map-size deviation" (2026-11-10, `gpu-box-log.md:877`); "The evaluation took longer than a
  training run; cut it" (line 884); fixed-size maps "until an agent trains successfully on
  variable-size maps" (2026-11-15, `arms-todo.md:93`); frontline break-even "left out of the
  score" (`arms-todo.md:91`).
- **Not examined anywhere in the record**: the attention softmax over 56-116 keys (12.5); unmasked
  dead and unseen rows piling up at the map centre (12.6); local presence as an unnormalised sum
  (12.8); the score-target input out of its training value (12.9); the value support against
  returns that grow with fleet size (12.16); whether the old 8192 m rotary ladder (which aliases
  exactly at 8192 m, and on which the 2026-11-10 champion was trained) contributed to the
  large-map collapse. The aliasing sweep was done two days after the fleet evaluation; the notes
  do not connect the two. The masked-zone result (-325 Elo) shows the trained model is sensitive
  to the set of keys it attends over, but no one tested the fleet-size analogue.
- **v1 against v2**: the record never compares them. The numbers are not comparable as they stand
  (different game, ship speeds, modes, map rule, opponent script); what can be said is that the
  `origin/main` model held a 1.44-1.55x crossover at 32 ships and 1.19x at 64, and the v2
  champion fell to a coin flip at 1.2:1 with 40 ships on the 5v5 arena.

### 7.4 Model-side rows of section 12

| Row | What the record says |
|---|---|
| 12.1 | Design intent since 2026-10-16 (7.1). Verified in practice by 7.2. |
| 12.2, 12.3 | Benched slots added 2026-11-10 for unequal-team evaluation only; `mask_benched` never trained with. "A benched slot costs what an active one does (the game, the scripted agent and the network run at fixed shapes over every slot)" (`sections/break-even/progress.md`). Training with benched ships = arm A2, never run. |
| 12.4 | NOT FOUND (a contract constant). |
| 12.5, 12.6 | Not discussed in terms of size. Attention shares measured at 5v5 only (2.5). |
| 12.7 | Docstring caveat only; kernels dropped. |
| 12.8, 12.43 | Radius set from gun reach (1.16); size effect not discussed. |
| 12.9 | Not discussed. |
| 12.10-12.12 | Density scaled by arena by design; bias from the build's ship count is v1's ("-4.19 at 5v5", `yemong-review.md:869-870`). Geometry coverage "in a larger fleet every ship within a few ticks, at a fixed cost per tick" (`docs/explanation/yemong.md:718-720`). |
| 12.22 | Static shapes per run (JAX-era rule, 2026-10-16); "Each shape is a cold compile" (`simplification.md:320-324`); evaluation at another size "spends longer compiling than playing" (`simplification.md:99-100`). |
| 12.23 | By design (per-ship ratio). |
| 12.25 | Fractions of the type's maximum; contradicts the review's "absolute units" outcome (1.22-1.25). |
| 12.27-12.29 | Five slots by the mixed-batch rule; partial zone sets never shown; the absent token is functionally used (2.5). |
| 12.30, 12.31 | Field-count sampling built 2026-11-15, off; arm A2. Scales fixed by the spec. |
| 12.33, 12.34, 12.35 | Fixed maps by the author's rule; arena code range from the density rule's 62 km. |
| 12.36-12.38 | Sizing rule (1.19, 1.26): ranges set for 62 km maps that are no longer played. |
| 12.39 | "The error head's position bound stays 26 244 m (a constant of that code)". |
| 12.40-12.42 | Geometry bound 64 km and hex grid by arena: ARGUED; shot codes from the card. |
| 12.44 | Timer horizon: 1.13, 1.30. |
| 12.48 | By construction. |

---

## 8. Chains of consequence

Each chain lists what exists only because of the item at its head. "->" reads "required".

**C1. The belief loop (4.1, 4.7)** — decided removed.
- -> the state code's spread half and blur (1.19 blur, 1.21 covariance blur, 1.23-1.25 blurred
  forms, 34 spread channels, moment matching) and the decode stack (padding bins, mean-inverse
  table, median window, probit quantiles: `simplification.md:534`);
- -> `spread_fit` "coarsest" (fix for metre-level sigma on kilometre errors);
- -> belief age as the source of time-since-seen (1.2, 1.12);
- -> the replay record and its dtypes (4.8), and with it the step/unroll bit-agreement machinery
  and the "drift" check (`simplification.md:521, 523`);
- -> enemy-action probabilities as pending-action input (1.14, enemy half) and int16 storage;
- -> enemy rows' heads evaluated at play (2.8, 5.11);
- -> rows with no belief placed at the map centre (1.17, jointly with C3);
- -> "no attention mask" as an invariant (2.1/3.4: "belief validity is sticky, so all tokens are
  always valid");
- -> the non-finite scrub and counter (a game NaN surfaced as an index assert in the belief
  decode: L6);
- -> the `sigma_miscalibrated` alert, hidden-row loss weighting, belief sums in `prepare`;
- -> the pinned-host segment store, in part (`simplification.md:565`).

**C2. The delta head inside the loop (4.3, 5.4)** — a delta target is open-loop, so:
- -> the error head (5.5) to pull the belief toward the truth and to supply a calibrated sigma;
  - -> median instead of mean (tail-dominated 26 km code);
  - -> confidence gain, rate eta 0.1, cap at the one-tick bound (closed loop with gain >> 1);
  - -> quantile spread (second moment 140x too wide);
  - -> three projection axes at 0/60/120 degrees (errors correlated 0.45-0.63 in every frame);
- -> the extrapolation baseline (4.3) -> rates in the carry (4.4) -> held rates at last sighting
  (runaway to 1,039 m/s) -> exponential decay of held rates (`extrapolation_decay`) -> its
  halving for the x2 game;
- -> acceleration columns in the observation (1.1) (the baseline needs last-tick rates);
- -> covariance propagation F P F^T + Q -> decoupled Jacobian and variance caps (double-counted
  turn, random-walk nose rate);
- -> hidden-physics training with `physics_hidden_decay` (the physics head ran on hidden inputs
  it never trained on: risk C5) -> `physics_hidden_fade` option (lost by 111);
- -> marginal bias tables (about 510 lines), per-quantity knees and bounds measured on an older
  game (suspected stale);
- -> two more loss coefficients and their trunk-share balancing.

**C3. The random position offset (1.18)** — decided removed.
- -> position codes on the global, field and zone tokens (the centre is no longer the origin);
- -> absent zones and no-belief rows placed at the shifted map centre ("would reveal the offset");
- -> float32 precision fixes: per-level remainder in the circular code, rotary via
  `2 pi (x mod p) / p` with exact periods (compiled and eager disagreed by 1e-3 in the code and
  2% in next-state logits: `sections/yemong-notes.md:106-111, 289-301`); `_nearest` unwrapping of
  decoded positions at the period seam;
- -> the colour code's coarsest period must cover the offset's range, not the map;
- -> tests that the offset reaches outputs only through the position codes.

**C4. Density-scaled evaluation maps up to about 62 km** — dropped 2026-11-10/15.
- -> colour-code period 236,196 m and finest 36 m (1.19);
- -> rotary coarsest 254.7 km, then 339.6 km (1.26), and with a long ladder the non-power-of-2
  ratio and the 35 degree turn (the aliasing they cure is between coarse bands);
- -> arena code over 0.25x-8x (1.28); geometry range bound 64 km (12.40);
- -> map-size sampling in the game (`arena_scale`) and its cascade of floors
  (`simplification.md:551`).

**C5. Static shapes, one compiled graph (JAX-era rule)** — no longer mandatory.
- -> five zone slots in every mode and the learned absent-zone token (2.5);
- -> benched slots for unequal teams and `mask_benched`; absent fields as inert fake fields and
  `mask_absent_fields`;
- -> dead and unseen ships kept as tokens rather than removed;
- -> one size per batch (12.46), a cold compile per size (12.22);
- -> constants as device tensors, "no host copy" rules, about 15 sites in the model.

**C6. The Triton attention kernel (3.6)** — dropped.
- -> "no mask" preference (masking through SDPA cost +12.7% / +20.8%) -> the fused key mask;
- -> power-of-two padding of queries and keys, single tile;
- -> plain twins and equality tests; autotune caches "kept to repeat the bits".

**C7. Returns a tenth of the critic's units (5.2)**
- -> every reward weight x10 -> value support 77 bins over +-3 -> the return-scale floor of 1 ->
  zero entropy coefficient at warm start (`simplification.md:541`).

**C8. Bullets not observed (1.6)**
- -> the `fired` flag as the only shooting cue; -> the counterfactual shot heads (5.8, 5.9) as
  the way aiming and dodging are taught -> shot capture in the rollout, re-simulation in the
  learner, part of the 128-tick lookahead.

**C9. Arm D's Griffin-only design (2026-11-06)**
- -> temporal-first order (3.1) -> rotary read-out -> per-block read-out -> retired, replaced by
  the geometry head (5.10), which inherited the read-out's loss weight 0.007.
  The order and the head remain; the design they served did not.

**C10. Bit-exact reproducibility** — non-goal.
- -> keyed inverse-CDF action sampling (6.3); keyed parameter initialisation (3.14); record
  defaults in `YemongConfig` different from a new run's (`TrainConfig.yemong`), with 20+ switches
  kept so old build records load (`simplification.md:587`); golden digests that fail on another
  machine (`gpu-box-log.md:145-160`).

---

## 9. Reasons that have expired

1. **1.18 random offset** — served translation invariance on v1's torus and "cannot memorize
   absolute map positions"; map is not toroidal, maps are fixed, decision made.
2. **1.19 / 1.26 / 1.28 ranges of 236 km, 340 km and 8x** — sized for 62 km density-rule
   evaluation maps, which were dropped on 2026-11-10 and 11-15.
3. **1.19-1.25 the spread/blur half of every code** — presents a belief's uncertainty; no belief.
4. **2.1 / 3.4 "no attention mask"** — rested on "belief validity is sticky" and on keeping the
   fast kernel; both go.
5. **1.17 and the centre placement of absent zones** — only there so the offset does not leak.
6. **4.4-4.6, 5.5 every clamp, gain, rate, cap, decay and quantile in the belief path** — each
   stops a runaway of the closed loop.
7. **4.8 replay record, int16 forecasts, float32 spreads, the drift check** — step/unroll bit
   agreement for a fed-back input.
8. **1.14 enemy rows of the pending action; 5.11 enemy heads at play** — the belief's only users.
9. **3.6 fused kernel, tile padding, plain twins** — treated the operation count; dropped.
10. **6.3 / 3.14 keyed draws** — common random numbers across arms; non-goal.
11. **2.5 / 1.4 five zone slots and 12.3 benched slots as padding** — one static shape.
12. **1.2 belief age instead of the observation's `time_since_seen`** — replay consistency.
13. **5.9 shot-trajectory head and its age query** — dropped; 5.3 separate outcome level with its
    own gamma/lambda — merged.
14. **1.1 acceleration columns** — fed an input arm that lost and a baseline that is going.
15. **Record-default switches in `YemongConfig`** — backward compatibility with old policy files.
16. **N-team generality in team inputs** — non-goal.

---

## 10. Reasons that still hold and are easy to forget

- **L1. A forecast fed back as input, with no gradient through the loop, compounds its own
  errors.** Run 734 (later-v1, delta target): `yemong-review.md:489-492, 538-540`. v2: velocity
  error 1,039 m/s within 8 updates (`sections/round2-log.md:103`); "a closed loop, the replay one
  update stale, gain >> 1" (`round2-log.md:101`). Applies to any future design that feeds a
  head's decoded output back.
- **L2. The mean and the second moment of a wide symlog categorical are dominated by slivers of
  tail mass.** Mean 40-400 m predicted where 5 m was realised; sigma 218 m against 1.6 m
  ("1e-3 in a 10 km bin is 1e5 m^2"): `round2-log.md:99`; `sections/error-head-spread-fix-brief.md:3-12`.
  Use a median and quantiles, or do not use such codes for quantities that must be decoded.
- **L3. The identity rotation is not "no position".** A token with zero rotary angles "would look
  like it sits at the torus origin"; it adds a spurious term (`yemong-review.md:125-127, 137-143`;
  `decisions.md:447-449`). With hidden enemies leaving the observation this is the live question:
  drop the token, or zero its rotated dimensions; never give it angle 0. The 2026-10-21 note also
  lists what a no-belief design loses (`decisions.md:442-446`).
- **L4. A trained model uses placeholder tokens as a "nothing to read" target; removing them at
  inference is costly.** Masking the absent-zone slots: -325 Elo (RL, deathmatch), -158
  (elimination), -374 (BC) (`sections/bc-v19/overnight-log.md:358-393`). Caveat in the same note:
  "this proves dependence, not that an explicit sink would beat the placeholder". Consequence for
  size: the key set at evaluation should look like the key set in training, or training must vary it.
- **L5. Rotary attention is load-bearing.** Turned off at inference the best RL model loses every
  game to the BC clone (`decisions.md:1330-1331`). (Reliance, not proof a model could not train
  without it.) And a power-of-2 ladder aliases exactly at its coarsest period: 8192 m reads as no
  shift (`sections/position-codes/progress.md:46`). Finest period above twice the largest relative
  move per tick (`sections/yemong-spec.md:282-284`).
- **L6. A NaN from the game surfaces far from its cause.** A physics 0/0 (`softmin(0, 0)`) became
  an out-of-bounds index in a table lookup inside the belief decode, 27M steps in; "a rollout-side
  non-finite detector ... would have turned a 30-minute hunt into a log line"
  (`overnight-log.md:48-67`). "The one real NaN came from the game ...; the learner never
  produced one" (`simplification.md:581`).
- **L7. A compiler may elide a precision round trip.** "never store a rounded copy that a compiled
  stage also reads unrounded" (`sections/yemong-notes.md:377-418`). Also: float32 loses the finest
  position level and the rotary angle at coordinates of 2e4-4e4 m unless reduced modulo the period
  first (`yemong-notes.md:289-301`): relevant to any frame much larger than the map.
- **L8. The RG-LRU's initial memory is about 0.2 s; the long-memory initialisation decided in
  2026-10 was never implemented.** `yemong-review.md:196-201, 237`; `sections/yemong-spec.md:430`.
  With the belief gone this is the only memory of hidden ships. Truncated backprop at 128 ticks
  (4.3 s) limits what can be learned to write (`yemong-review.md:247-253`).
- **L9. Head settings that live only in a launch command are not reproducible.** The outcome
  level's gamma = lambda = 1 for BC existed only in the laptop's queue lines; a day of box runs
  compared arms on a wrong base and a non-existent "regression" was chased
  (`gpu-box-log.md:298-311`). "rebuild a run's command from its `run.json`".
- **L10. Seed dominates single-seed arm gaps, and pairs by seed.** "Seed 0 is ~110 above seed 1
  for all three configs ...: systematic ..., so same-seed comparisons are paired and far tighter
  than 110; never compare across seeds" (`gpu-box-log.md:591-594`). Outcome Brier skill "does not
  track Elo".
- **L11. Bundled arms mislead.** Arm 2 lost 109 Elo in frontline; the cause was the ellipse spread
  inputs bundled with the delta head, found only by arm 2b (`round2-log.md:137-151, 193-199`).
  The delta bundle's gain has likewise never been split from the error head.
- **L12. Two-hot input codes are not shift-invariant.** "an observed ship looks different to the
  network depending on offset" (`yemong-review.md:591-595`); decided to fix, not fixed.
- **L13. Map size, not fleet size, is the measured killer so far**, and no agent has trained on
  more than one map size per mode (7.3).

---

## 11. Never tested

| Component | Status |
|---|---|
| Belief against recurrent memory alone (4.1) | A14 never ran; arm D's result not in the repository. |
| All auxiliary heads (5.4-5.10) | A13 never ran. "the gradient-share diagnostics show what the heads cost, not what they're worth" (`arm-review.md:130-137`). |
| Error head alone (5.5) | Only persist-split vs persist-onehead on a wrong base (+25). |
| Categorical input codes against raw scalars (1.8) | A17 never ran. |
| Rotary against colour code against both (1.19, 1.26) | A18 never ran; arm R's rating not found. |
| New position codes (236 km, ratio 3.405, turn) against old | No strength comparison. |
| Random offset on/off (1.18) | Never compared. |
| Geometry head (5.10) | BC A/B launched 2026-11-12, no result recorded; x3 arm never ran. |
| Density head (5.7), shot heads (5.8, 5.9) | No ablation; item-15 benchmarks never ran; A22 never ran. |
| Arena code (1.28), timers (1.30), local presence (1.16) | No comparison. |
| Joint against factored action (5.1) | No comparison in the available record. |
| Distributional against scalar value (5.2); HL-Gauss against two-hot critic | A21 never ran; HL-Gauss decided, not built. |
| Trunk: 2 blocks vs 6 spatial (A15), Griffin vs GRU (A16), heads 2 vs 4 (A20), width (A23), gate (A19), `spatial_only` | None ran / no result found. |
| Separate ally and enemy encoders (2.6) | "Candidate ablation", never run. |
| Long-memory RG-LRU initialisation (3.9) | Decided 2026-10-21, never built. |
| Training on variable maps (A1) | bc-v21 started, stopped. |
| Training on variable ship and field counts with masks (A2) | Never ran; switches exist, untrained-on. |
| Masking dead / unseen rows in training (3.4) | Never ran. |
| Mixed fog (1.7) | Replaced by 100% fog without a measurement. |
| Seed noise as a spread (A3) | Never ran; 110 is from a few pairs. |
| Size effects of softmax width, presence sum, score target, value support | Not examined. |

---

## 12. Could not find

No recorded reason, in the places listed:

- **Why the later-v1 branch added the belief, the categorical codes, rotary, the joint action,
  the density head and local presence over `origin/main`** — its docs are not in this clone
  (looked: `yemong-review.md`, `decisions.md`, `sections/yemong*.md`, `origin/main:docs/`).
- **1.2 `proper_speed`** in the observation and unread.
- **1.13** the 16 s horizon and 4 harmonics; **1.15** the gradient input and its scale.
- **1.16** why local presence exists (only its radius is justified).
- **1.27** why the 8th heading harmonic was dropped on 2026-11-12.
- **2.3** why fields are key/value-only (the review records the fact and the author's "keep").
- **2.10** the N(0, 1/groups) input initialisation.
- **5.1** why a joint categorical rather than factors.
- **5.7** what the density head is for.
- **12.4** `MAX_SHIPS = 255`, `MAX_FIELDS = 64`.
- **Round 0 ratings** (BC arms A, R, C, D, 2026-11-07, `runs-v2/ratings/2026-11-07-bc-arms`): the
  basis for temporal-first and for keeping rotary and the belief; only fragments in
  `sections/round2-plan.md:16, 24`.
- **The "collapses in frontline above 1.4x" measurement** behind arm A1.
- **The coordinator's message of 2026-11-12** that specified the new position codes; only its
  summary (`sections/position-codes/progress.md:3-6`) and the sizing rule survive.

---

## 13. Where the record contradicts itself or the code

1. **Belief**: decided removed 2026-10-21 (`decisions.md:402-409`, checked off at line 1561);
   kept 2026-10-31 (`sections/yemong.md:7-13`); `references.md:112` still says "Griffin-only
   memory ... became the port's design".
2. **RG-LRU initialisation**: decisions and `references.md` say the paper's long-memory
   initialisation; the spec and code have linspace(0, 4).
3. **Critic**: `references.md` and the review say HL-Gauss with calibrated bin values and 2
   padding bins; the code is two-hot, 77 bins, no padding.
4. **Input codes**: HL-Gauss "go into the port" (review); the port kept v1's two-hot and
   moment-matched forms. HL-Gauss is used only by the delta, error, arena and geometry codes.
5. **Density head**: review decided 367 cells and no clamp; code has 331 cells and a clamp.
6. **Resource channels**: review outcome "absolute units ... (not fractions of a type's maximum)";
   code uses fractions.
7. **"+27 Elo"**: quoted by the audit as the error head's evidence (`simplification.md:289-292`;
   repeated in `review-packet.md:122` and `review.md:385`). It is from tournament 1, which the
   log says ran on a wrong base; it compares the delta bundle with the absolute head, not the
   error head with its absence; and the corrected-base, seed-paired gaps are +139 and +141 at BC
   30M, +64 at 60M, +10 at 120M, about +150 after RL (one seed).
8. **Colour code in the audit**: "8-level colour code down to 4 m" for the 236 km frame; the
   champion build is 9 levels, 36 m finest.
9. **Rotary ladder**: launch file comment 48 m-254.7 km; code 64 m-339.6 km; docstring 339,695 m
   against 339,625 m computed (inventory).
10. **`physics_hidden_decay`**: 0.5 s in `docs/explanation/yemong.md:155` and the launch comment;
    0.25 s in `src/bnb/runtime/train.py:234` ("0.5 before the x2 game").
11. **"There are no never-seen ships"** (2026-10-21 invariant) against v2's belief, which is
    invalid for a ship that spawns unseen.
12. **Map-scale randomisation**: stated as required for the absolute code (`decisions.md:266-269`;
    `yemong-review.md:64-66, 104-105`); never on in a rated run; opt-in since 2026-11-15.
