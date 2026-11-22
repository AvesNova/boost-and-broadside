# Departure audit: lean against v1 and v2

2026-11-22. Read-only. Nothing was run. The audit compares `lean/` (branch `core`) with v1
and with v2, and checks each difference against the recorded decisions in
`docs/questions.md`, the lean `NOTES.md` files, `docs/rewrite/core-brief.md` (the plan) and
`docs/rewrite/model-audit.md` (the plan's source for the model and the learner).

**References.**
- `L:` is `lean/`. `V2:` is `src/bnb/`.
- `V1@731:` is the v1 tree at run 731's commit `3d1a2e55`, read with `git show` in
  `/home/vizia/avesnova/boost-and-broadside/main`.
- `YS:` is `docs/rewrite/sections/yardstick.md`, which records v1's recipe from `cfg731`
  and v1's code with its own references.
- `GA:` is `docs/rewrite/sections/game-audit.md` and `MA:` is `docs/rewrite/model-audit.md`.
- A reference marked † was read by a search subagent during this audit and reported with
  that file:line. I did not re-read it myself. All lean references, `V2:learn/ppo.py:143-180`,
  `V2:runtime/train.py:60-90` and `V2:game/modes/rules.py:20-82` I read directly.

**Status of a departure.**
- **unrecorded**: a builder's choice that no note or question states.
- **recorded**: a builder stated it in a `NOTES.md` or the README, and the author has not
  approved it.
- The kind `recorded decision` means the author or the plan decided the point. Where lean
  does not yet follow such a decision, the row says so.

---

## The 10 departures that matter most

1. **No self-play, and a fixed scripted half** (League, row 1). This is against the
   author's decision of 2026-11-22. Both v1 and v2 had 50% self-play with both sides
   trained. It costs half the training data, and it sets the opponent mix.
2. **The damage level is priced per shield point, and its value support is ±2** (Learner,
   row "damage unit"). One hit of about 20 points pays about 11 on the damage level against
   a win of 1. A ship's returns then sit far outside the ±2 categorical support and are
   clamped. That is about 100 times v2's damage-to-outcome ratio. The critic of that level
   cannot be right.
3. **The shaping rewards read the wrong state fields** (Learner, row "shaping").
   - "Facing" is the dot product of the angle-of-attack (cos, sin) with the direction to
     the enemy. It is not computed from the nose.
   - "Closing" uses the nose direction, not the velocity.
   - The notes say the opposite. Shaping is v1's guard against early passivity.
4. **Training batch about 30 times smaller than v1's or v2's** (Loop, row 1). The default
   is 256 environments × 128 ticks × 1 rollout, which is 32,768 steps per update, split
   into 4 minibatches. v1 used about 995k steps per update in 32 minibatches; v2 used about
   1.05M in 32. The warm-up, the snapshot cadence and the BC gate are all counted in these
   small updates.
5. **No weight averaging and no KL stop** (Learner, rows "averaged weights" and "KL").
   Both v1 and v2 had both. v1's averaged weights held 0.08 to 0.11 more break-even ratio
   at every size (`YS:598`).
6. **The outcome discount is 0.998 on a 180 s game** (Learner, row "discounts"). It was
   v1's value for v1's 17 s cap. Lean's games run tens of seconds after a roughly 9 s
   approach, and v2 used 0.9997 for the outcome. A win 45 s away is weighted by about 0.07.
7. **Fields: none by default, and one fixed set per run** (Game, row "fields"). The
   author's decision is fields drawn for every new environment. v2 had 10 random fields
   per environment and v1 had 4 per map. The model has no field tokens either (Model,
   row 1).
8. **Unshifted absolute position in the ship features** (Observation, row
   "absolute position"). x/R and y/R are inputs, and so are the outward direction and the
   boundary distance. The plan says "None absolute", and the random offset exists to stop
   memorised positions. This touches zero-shot transfer across sizes and maps.
9. **The trunk is thinner than both v1 and v2** (Model, rows "MLP", "temporal",
   "heads").
   - The gated MLP's hidden width is 1× the model width; v1 and v2 used 4×.
   - The temporal block is a bare RG-LRU, with no input projection, convolution, gate
     branch or MLP.
   - The policy and value heads are single linear layers; v1 and v2 used an MLP.
   - The model is 616,677 parameters, against v1's estimated ~2.0M (`YS:123`).
10. **The BC term goes to the scripted agent's sampled action, gated by a training-rollout
    win rate, inside PPO** (Learner, row "BC").
    - The plan (`core-brief.md` §4, `MA:368-369`) is a soft-label BC stage, then a
      decaying distillation term toward the clone.
    - v1 used soft labels, latched off after 3 updates.
    - v2 had a separate BC stage and no term in RL.

Close behind:
- the arena is 4,200 m, not v2's elimination arena of 7,800 m;
- one snapshot plays the whole league half for an update;
- the size-curve evaluation decides games at the clock by health, so v1's tie rule rarely
  applies;
- the global token lacks map size and mode.

---

## 1. Game

Matched, and not listed below (12 items):
- flight, lifted and checked to float32 rounding (`L:game/NOTES.md` "What was run");
- bullet motion and drag;
- the swept hit test with first contact (v2's rule);
- the damage form (kinetic × closeness × aspect);
- the quiet rule (coast, stick 0 to 2, no shot), `L:game/core.py:531-537`;
- friendly fire on;
- boundary damage of 40/s on a 300 m ramp;
- no respawn (the plan);
- the 30 Hz tick with a decision every tick (v2's; v1's decision rate);
- the side spawn rule;
- the edge-safe heading;
- K = 11 bullet slots.

| item | v1 | v2 | lean | kind | note |
|---|---|---|---|---|---|
| Fields | 4 refractive fields per map, regenerated from a 512-map cache and translated per reset (`YS:94-95`) | 10 random point-symmetric fields per environment, drawn at reset (`GA:104`; presets `V2:game/modes/rules.py:38-74`) | `GameConfig.fields = ()`: no fields, and one fixed tuple for the whole run (`L:game/core.py:49`, `:77`, `:153-155`); `reset_done` keeps the old fields (`:698-699`) | recorded decision (`docs/questions.md` #1; `L:game/NOTES.md` fix 11) | **lean does not yet follow the decision.** No run with fields was made. The scripted opponent assumes refractive index 1 (Scripted table) |
| Arena | 1024 × 1024 torus, no boundary (`YS:79`) | Elimination 7,800 m; deathmatch 4,200 m (`V2:game/modes/rules.py:43-44`, `:77-82`) | 4,200 m disc (`L:game/core.py:45`) | **recorded** (`L:game/NOTES.md` Q4; the `GA:1384-1389` recommendation, not the author's decision) | The README lists "what stage 1 reproduces" as still with the author. The 4,200 m arena is deathmatch's, applied to elimination |
| Clock and win rule | 1,024 physics ticks (17.1 s); at the cap the game is a tie and counts as a non-win (`YS:77-78`) | Elimination clock 180 s; health tie-break at the clock (†`V2:game/modes/phase.py:148-162`), but the shrink kills every ship before the clock (†`rules.py:242-248`) | 5,400 ticks; at the clock the larger sum of living shield fractions wins (`L:game/core.py:546-551`) | **recorded** (`L:game/NOTES.md` Q3; `GA:1377-1382` recommendation) | With no shrink, the health rule decides every clock game, which v2 never did in practice. Untested against v2: no game ended in the comparison |
| Spawn | Uniform over the torus, teams interleaved (`YS:84`) | Sides (`GA:179-181`) | "sides" by default, with a random axis per environment (`L:game/core.py:48`, `:624-633`); "mixed" is optional | v1 and v2 differ; lean chose v2 (recorded, Q4) | `YS:585` ranks this first among v1 ingredients for the size curve. The "sides" spawn is untested (`L:README.md`) |
| Spawn resources | Health and power uniform in 75% to 100% (`YS:86`) | not checked | Full shield; `spawn_energy` fixed per type (`L:game/core.py:657-658`) | unknown | I did not check v2's spawn store |
| Lineup | One ship type (`YS:87`) | Current default: Yongma or Jujak drawn per slot per episode (†`V2:runtime/train.py:255`); rl-v20 ran fixed Yongma ×2 + Jujak ×3 (†`docs/rewrite/sections/bc-v19/rl.sh:29`) | A fixed cycle by rank, Yongma then Jujak, the same for both teams (`L:game/core.py:50`, `:616-617`) | **recorded** (`L:game/NOTES.md` Q6: "Per-episode mirrored draws are not implemented") | `GA:1397-1400` recommended mirrored draws. With a fixed cycle, every 4v4 game is the same 2+2 lineup |
| Dead ship's speed and nose rate | n/a | Set to the speed floor and zero (`L:game/NOTES.md`) | Kept (`L:game/core.py:559-563`) | **recorded** (Known differences; agreed fix 3 allows it) | Harmless while masked |
| Result coding and killer tie-break | n/a | +1/−1/0; killer by tau, then miss distance, then index | 0/1/2/−1; killer by tau, then lowest index (`L:game/core.py:512-521`) | recorded | Cosmetic |
| Fog | None (`YS:96`) | On in every environment (†`V2:contracts/pools.py:49`) | None. `Obs` has no `visible` field (`L:contract.py:87-96`) | recorded decision (`docs/questions.md` #2, #5; `L:game/NOTES.md` fix 10) | The author: "built towards fog from the start". **lean does not follow this yet.** The plan's skeleton step 2 lists a `visible` row (`core-brief.md` §4) |

## 2. Observation and tokens

Matched (4 items):
- team flag and ship-type one-hot;
- dead ships masked out (v1's rule, and the author's fog answer #5 for elimination);
- `exists` masking;
- the random per-match offset drawn at reset and stored with the rollout
  (`L:learn/collect.py:192`, `:272-273`).

| item | v1 | v2 | lean | kind | note |
|---|---|---|---|---|---|
| Per-ship features | Position, velocity, attitude, angular velocity, health, power, cooldown, team, alive, pending action, and others (`YS:102`) | Categorical state codes (position colour code, two-hot velocity, binned heading and scalars), the pending action (42), local presence, fog columns (†`V2:models/yemong/inputs.py:436-451`, †`state_code.py:182-406`) | 20 linear scalars at fixed scales, 4 flags, type one-hot (`L:game/core.py:708-767`) | recorded decision (plan: about 20 scalars at fixed scales, `MA:345`) | The plan says "symlog where the range is wide". Lean uses no symlog. The energy scale is a fraction of the maximum |
| Absolute position | Absolute x, y as Fourier features (`YS:103`) | Absolute colour code, shifted by the per-match offset (†`inputs.py:173-194`) | x/R and y/R, the outward unit vector and the boundary distance, all unshifted (`L:game/core.py:754-760`), plus the offset Fourier code (`L:model/policy.py:339-349`) | **lean departs from both; unrecorded** | The plan says "Position: None absolute. Clipped distance and direction to the boundary" (`MA:347`). x/R and y/R reach the encoder with no offset, so the offset no longer prevents memorised positions. The boundary features also give absolute position, in polar form, in a disc arena |
| Pending or previous action | Every ship's pending action, opponents included (`YS:82`) | Own ships: the previous joint action; enemies: a forecast from the belief (†`inputs.py:412-417`) | None. Only a "fired last tick" flag inferred from the bullet ring (`L:game/core.py:737-739`) | lean departs from both; unrecorded | The plan's input list omits it. `YS:594` lists it as a v1 information edge. Fix 12 stores `fired` |
| Global token | None (fields and ships only) | Map-centre code, episode progress, arena radius and its code, fogged flag, mode flags, scores, front, wave timer, boundary timers (†`inputs.py:456-479`) | Tick/clock, own and enemy alive fractions (`L:game/core.py:773-775`) | lean departs from v2; unrecorded | No arena size and no mode. The author's answer #4 says mode and anything known at reset are given as inputs. Alive fractions are new (v2 had none) |
| Field and zone tokens | 4 field tokens (`YS:104`) | Field and zone tokens as key/value memory (†`inputs.py:481-538`) | None | recorded decision (`docs/questions.md` #3; `L:model/NOTES.md` "Token kinds") | **lean does not follow this yet** |
| Bullets | Bullet tokens, 10 per ship (`YS:105`) | Not tokens | Not tokens | recorded decision (`docs/questions.md` #3) | |
| Next-state target scales | n/a | n/a | Energy over 400 gives a mean square of about 1,470 (`L:learn/NOTES.md` "Found on the way") | recorded | The loss at coefficient 0.1 outweighs every other term |

## 3. Model

Matched (6 items):
- width 128;
- 2 blocks;
- 2 heads (v2);
- temporal first, then two spatial sublayers (v2's T S S);
- the joint 42-way action;
- rotary on heading with harmonics 1, 2, 4.

Two items follow the plan, not v1 or v2, and are not listed:
- the three-axis 60° Fourier ladder at 6 periods;
- per-ship categorical critic levels with HL-Gauss (`MA:349-355`).

| item | v1 | v2 | lean | kind | note |
|---|---|---|---|---|---|
| Encoder | 2-layer MLP shared by ships and fields (`YS:115`) | A per-kind first layer (ship, global, field, zone) to 256, then a shared second layer (†`V2:models/yemong/network.py:18-107`) | Two separate 2-layer encoders, one for ships and one for the global token, with nothing shared (`L:model/policy.py:271-272`) | recorded decision (`docs/questions.md` #3) | **lean does not follow this yet** |
| Map memory in attention | Fields as ordinary tokens | Fields and zones as key/value only, in one softmax (†`V2:models/yemong/spatial.py:83-106`) | None | recorded decision (#3) | **lean does not follow this yet** |
| Null key | None (`YS:117`) | None (†grep, the agent's report) | One learned null key, shared by every sublayer (`L:model/policy.py:273`, `:223`) | recorded decision (plan: `MA:351`, `core-brief.md` §4) | `L:model/NOTES.md` says to keep it only if still needed once fields are keys |
| Rotary for the global and null tokens | n/a | Map tokens sit at the map centre, heading 0 (†`inputs.py:500-507`) | "zero": the rotated dimensions are removed from their scores (`L:model/policy.py:355-359`) | **recorded** (`L:model/NOTES.md` choice 3: "a deviation from the letter of the task") | |
| Temporal block | Griffin block: conv 4, RG-LRU, gated MLP (`YS:119`) | Griffin recurrent block: input projection, causal conv 4, RG-LRU, GeLU gate branch, output projection, then a 4× gated MLP (†`V2:models/yemong/temporal.py:151-174`) | x + RG-LRU(norm(x)): no projection, no conv, no gate branch, no MLP (`L:model/policy.py:154-197`) | **lean departs from both; recorded** (`L:model/NOTES.md` choice 7 and the "Temporal block" section) | The plan says "a Griffin recurrent block per token" (`MA:353`) |
| MLP width | Gated MLP 4× (`YS:116`) | GatedMLP hidden 512 = 4× (†`spatial.py:41-42`) | `mlp_input` to 2×width, split into hidden and gate, so the hidden width is 1× = 128 (`L:model/policy.py:214-215`, `:236-237`) | **lean departs from both; unrecorded** | A quarter of the MLP capacity. Not mentioned in any note |
| Policy and value heads | Action MLP 128→256→12 (`YS:120`) | Heads are Linear→RMSNorm→GELU→Linear with hidden 256 (†`V2:models/yemong/kernels/heads.py:20-62`) | One linear layer each (`L:model/policy.py:276-278`) | lean departs from both; unrecorded | |
| Outcome critic | A team-pooled embedding for the win (`YS:121`) | A 4-class outcome head on the global token, team-pooled (†`V2:models/yemong/model.py:634-636`) | Per ship, like the other levels (`L:model/policy.py:381`) | recorded decision (plan `MA:355`: "per-ship ... including outcome") | |
| Value support | Scalar, Huber on normalised returns (`YS:145`) | 77 symlog bins to ±3 (±19 in reward units), two-hot (†`V2:models/yemong/critic.py:164-182`) | 77 uniform bins; the learner's support is ±1.5, ±2, ±2, ±4 (`L:learn/config.py:95-96`) | recorded (`L:learn/NOTES.md` "Second pass") | The damage support cannot hold the damage returns (see Learner) |
| Next-state head | Phase or additive deltas (`YS:122`) | Symlog HL-Gauss delta code over an extrapolation baseline, with a split error head (†`model.py:634`, `:1155`) | Linear regression of 7 scaled one-tick changes (`L:model/policy.py:61-69`, `:278`) | recorded (`L:model/NOTES.md` choice 11) | Scales are guesses, and the energy scale is wrong (Observation table) |
| Initialisation | not checked | Heads orthogonal, last layer gain 0.01; next-state bias set to the marginals (†`heads.py:16-42`, †`model.py:702-704`) | Policy weight × 0.01, value zero, PyTorch defaults elsewhere (`L:model/policy.py:280-284`) | recorded (choice 12) | Minor |
| Decay initialisation | not checked | Λ = linspace(0, 4, D) (†`temporal.py:80-82`) | a^c uniform in [0.9, 0.999] (`L:model/policy.py:169-171`) | recorded decision (plan: "the paper's decay initialisation") | |
| Parameter count | About 2.0M, inferred (`YS:123`) | not found | 616,677 (`L:model/NOTES.md`) | consequence of the rows above | |

## 4. Learner

Matched (6 items):
- 4 epochs;
- 128-tick segments;
- gradient clip 1.0;
- Adam with lr 3e-4 (v2) and eps 1e-5 (v1);
- a time-limit end treated as terminal (v2 `†V2:learn/returns.py:39`; v1 `done | truncated`, `V1@731:src/boost_and_broadside/train/rl/ppo.py:1156`);
- outcome paid to every ship that existed (v2, †`V2:learn/rewards.py:160-166`).

v1's discounts are also carried, but see the row.

| item | v1 | v2 | lean | kind | note |
|---|---|---|---|---|---|
| Clip | 0.15 (`YS:136`) | 0.2 (`V2:learn/ppo.py:150`) | 0.2 (`L:learn/config.py:79`) | v1 and v2 differ; lean chose v2 (recorded, `L:learn/NOTES.md` Departures) | The plan's `MA:361` says 0.2 |
| Minibatches | 32 (`YS:135`) | 32 (`V2:learn/ppo.py:176`) | 4 (`L:learn/config.py:81`) | **lean departs from both; recorded** ("Choices the plan left open") | With 256 environments that is 64 environments per minibatch |
| Learning-rate schedule | 1e-7 to 4.5e-4 over 5M steps, held to 100M, decayed to 1.5e-4 at 500M (`YS:139`) | 1e-7 to 3e-4 over 10M steps, then constant (`V2:learn/ppo.py:171-173`) | Linear warm-up over 10 updates (327k steps at the defaults), then constant (`L:learn/config.py:144-146`) | recorded (Departures) | The warm-up is about 30 times shorter than v2's in steps |
| Entropy | 0.005 on each of the 3 factors (`YS:138`) | −0.0081 per chain-rule factor, but 0 for a warm start such as rl-v20 (`V2:learn/ppo.py:155-157`; †`cli.py:931-941`) | 0.003 on the joint entropy (`L:learn/config.py:87`) | recorded (Departures) | `MA:319` recommends re-testing 0.001 to 0.003 |
| Value coefficient | 1.0, normalised Huber (`YS:146`) | 0.007 against a policy coefficient of 27 (`V2:learn/ppo.py:154`, `:158`) | 0.05 against a policy coefficient of 1 (`L:learn/config.py:88`) | recorded (Departures; plan: "value small") | |
| KL stop | 0.1, then 0.02 once the win rate passes 0.8 (`YS:137`) | target_kl 0.02, checked after each epoch (`V2:learn/ppo.py:178`; †`:1413-1420`) | None | **lean departs from both; recorded** (Departures) | |
| Averaged weights | A uniform running mean once BC ends (`YS:148`) | An EMA of 0.95 plus a uniform average after warm-up; both are pool entries and rated (†`V2:runtime/opponents.py:178-180`) | None | **lean departs from both; recorded** | `MA:370` says "at most one average, and only if measured". v1's averaged weights were better outnumbered (`YS:238`, `:598`) |
| GAE lambda | Per component, 0.64 to 0.941 (`YS:143`) | Per level, 0.81 to 0.97 (`V2:learn/ppo.py:146`) | One value, 0.95 (`L:learn/config.py:74`) | recorded decision (plan `MA:361`: one lambda) | v2's own comment, `ppo.py:143-144`: "One value for every level (tried 2026-11-13) let a run learn to stop fighting." That was about discounts; the plan's one lambda is untested |
| Discounts | Per decision, 0.998 / 0.990 / 0.982 / 0.951, for a 17 s cap (`YS:142`) | 0.982 damage, 0.990 kill, 0.9997 outcome (`V2:learn/ppo.py:145`) | v1's values (`L:learn/config.py:73`) | **v1 and v2 differ; lean chose v1; unrecorded consequence** | v1's 0.998 outcome discount was sized for a 17 s game. On lean's 180 s clock a win 1,350 ticks away is weighted by about 0.07 |
| Advantages | Per-component running RMS, the weights, then the buffer RMS (`YS:144`) | Plain sum, divided by an EMA percentile range floored at 1, no mean (†`V2:learn/critic.py:87-110`) | Plain sum, divided by the standard deviation over the update, no mean (`L:learn/returns.py:288-306`) | recorded decision (plan `MA:363`) | |
| Value targets | not checked | Fixed per update (†`V2:learn/ppo.py:1016-1052`) | Recomputed from the current critic at every epoch (`L:learn/update.py:392-394`) | recorded decision (plan `MA:363`: "recompute advantages once per pass") | Costs one extra forward pass per epoch |
| Damage unit | Raw health points (hit 10, health 100) times the weight, divided by the total ship count of 8 (`V1@731:src/boost_and_broadside/env/rewards.py:123-136`; normalisation `wrapper.py:382`) | 2.74 per 100 points against an outcome weight of 10 (†`V2:learn/rewards.py:58-64`) | 0.548 dealt and 0.274 taken per shield point; `damage_unit = 1.0` (`L:learn/config.py:103-106`; `L:learn/rewards.py:230-234`) | **recorded as unverified** (`L:learn/NOTES.md` "damage_unit 1.0 ... check"; "+-2 is unverified") | A 20-point hit is about 11 dealt and 5.5 taken against a win of 1. That is about 100 times v2's ratio and about twice v1's per hit (v1 divided all levels by 8). Damage returns far exceed the ±2 support |
| Shaping | Facing: max over enemies of a proximity weight × dot(nose, direction to enemy), clamped ≥ 0. Closing: velocity toward the nearest enemy over the maximum speed, clamped to [0, 1] (`V1@731:env/rewards.py:417-520`) | not part of v2's levels | Facing uses `angle_cos/sin` (the angle of attack) with no proximity weight or clamp. Closing uses `heading_cos/sin × speed` (the nose), clamped to [−1, 1] (`L:learn/rewards.py:183-184`, `:243`) | **lean departs from v1; unrecorded (a bug)** | `State.heading_*` is the nose and `angle_*` is the angle of attack (`L:contract.py:47-52`). The notes claim nose and velocity |
| Team sharing | Per-ship terms plus a kill assist; all rewards divided by the ship count (`YS:155-158`) | Individual (team_spirit 0) with an attribution ledger (`V2:learn/ppo.py:147`; †`rewards.py:169-236`) | 0.5 own + 0.5 team mean for kill and damage (`L:learn/config.py:107`; `L:learn/rewards.py:236-239`) | **lean departs from both; recorded** (Departures; README) | The scaling with team size is a recorded decision (`docs/questions.md` #7); the 0.5 share itself is a builder's choice |
| Kill assist, boundary credit | Kill assist +0.283; enemy field death and field damage pay the other team (`YS:155-156`) | A credit ledger; non-combat deaths pay the opposing team (†`rewards.py:152-195`) | Killer only; boundary deaths and damage credit nobody (`L:learn/rewards.py:222-228`) | **lean departs from both; recorded** | Boundary damage still charges the victim (`damage_taken` includes it, `L:game/core.py:530`) |
| BC | Cross-entropy to the scripted agent's **probabilities** inside PPO, 2.0 × max(0, 1 − w/0.45), latched off after 3 updates at 0.45; w from a 100-game evaluation window (`YS:163-168`) | A separate BC stage; no term in RL (†`V2:learn/ppo.py:49-60`) | Cross-entropy to the scripted agent's **sampled** action, same gate, no latch. w is an EMA (0.9) over training rollouts against the scripted agent, starting at 0 (`L:learn/update.py:411`; `L:learn/config.py:128-130`; `L:train.py:207-210`) | **v1 and v2 differ; lean chose v1, against the plan; recorded** | The plan is a soft-label BC stage, then distillation toward the clone (`core-brief.md` §4 skeleton step 3; `MA:367-369`). With no latch the term returns whenever w dips |

## 5. League and opponents

Matched (1 item): the learner's seat is drawn at random per episode
(`L:learn/collect.py:269`).

| item | v1 | v2 | lean | kind | note |
|---|---|---|---|---|---|
| Self-play share | 50% self-play, both sides trained (`YS:174`) | 50% self-play, both seats are the learner and recorded (`V2:runtime/train.py:82-83`) | 0%. Only one team per environment trains (`L:learn/collect.py:246-247`) | recorded decision (`docs/questions.md` #8; `L:learn/NOTES.md` "The league") | **lean does not follow this yet** |
| Scripted agent's role | A league member rated 1000; its share falls as the live rating rises (`YS:176-179`) | Imugi is one pool entry (†`V2:league/pool.py:52`) | A fixed first half of the environments, never redrawn (`L:learn/collect.py:216`). The whole batch is scripted while the pool is empty, which is the first 10 updates (`:217-218`; `L:train.py:201`) | recorded decision (#8) | **lean does not follow this yet** |
| Snapshot cadence and retention | Frozen at every 200 points of live rating (`YS:176`) | At set step counts (5M ... 160M, then every 30M); the newest 8 plus ages of 2^k (†`V2:runtime/snapshots.py:64-68`, †`league/pool.py:83-103`) | Every 10 updates (327k steps at the defaults), the last 8 kept (`L:learn/config.py:122-123`; `L:learn/pool.py:330-335`) | **recorded** (uniform pool in `L:learn/NOTES.md`) | The plan defers rating and PFSP to after M1 (`core-brief.md` §4). At the defaults the pool spans only the last 2.6M steps |
| Opponent sampling | exp(−\|Δrating\| / 200) (`YS:177`) | PFSP "hard", (1 − p)², 4 slots, one slot redrawn every 2 updates (†`V2:runtime/opponents.py:171-177`) | Uniform; **one snapshot for every league environment for a whole update** (`L:train.py:201`; `L:learn/pool.py:337-342`) | recorded (uniform) / unrecorded (one opponent per update) | v1 used 4 slots of 324 environments (`YS:175`) and v2 used 4 slots. Lean's league games each update are all against one past self |
| Ratings | Anchor-based live rating | Bradley-Terry with standard errors (†`V2:league/ratings.py:1-16`) | None | recorded decision (plan: after M1) | |

## 6. Scripted opponent

Matched to the plan's choice (`GA:1402-1406` Q7), not listed:
- Imugi's real-scale-elimination gate centres and scales, nearest target, drag-law lead,
  boundary steer;
- doctrines, the frontline layer and the switch-rate commitments dropped
  (`L:scripted/NOTES.md`).

| item | v1 | v2 | lean | kind | note |
|---|---|---|---|---|---|
| Which agent | A simple stochastic agent, about 170 or 338 operations (`YS:324-333`) | Imugi with doctrines, memory and frontline (†`V2:controllers/imugi.py:89-126`) | Imugi's combat core (`L:scripted/opponent.py:1-10`) | v1 and v2 differ; lean chose v2's core (recorded, the `GA` Q7 recommendation) | Break-even against it is a different number from v1's 1.75 (`YS:592`). The exchange rate against Imugi is not measured |
| Reads | The full state | `Obs`, the team's fogged view (`V2:controllers/imugi.py:89-126`, read) | The full `State` (`L:scripted/opponent.py:3`; `L:learn/collect.py:167`) | recorded decision (`docs/questions.md` #2) | Under fog it must read its team's view. There is no fog yet, so there is no cheating today, but the signature will have to change |
| Per-type characteristics | One type | Per-type characteristics | Gun range, corner speed and launch speed are Vizia's for every type (`L:scripted/opponent.py:86-95`) | recorded; agreed fix 13 | |
| Fields | Ignored (`YS:330`) | not checked | Velocity assumes index 1; bullet bending is ignored (`L:scripted/opponent.py:191-197`) | recorded | This matters once fields arrive |
| Arena radius | n/a | From `obs.globals` | Defaults to 4,200 m from the config (`L:scripted/opponent.py:77`) | recorded | It must follow `GameConfig` |
| Persona and memory in training | n/a | Imugi T = 1 in the pool | Spread 0 and no memory in training and as the BC teacher (`L:learn/config.py:124`; `L:learn/collect.py:167`) | unrecorded choice | `GA:1404` recommended a population for BC |

## 7. Training loop and evaluation

| item | v1 | v2 | lean | kind | note |
|---|---|---|---|---|---|
| Batch per update | 2,592 environments × 128 × 3 rollouts = 995,328 steps (`YS:132-133`) | 2,048 environments × 128 × 4 shards ≈ 1.05M steps (†`V2:runtime/train.py:206-210`) | 256 environments × 128 × 1 rollout = 32,768 steps (`L:train.py:169`, `:178`, `:202-206`) | **lean departs from both; unrecorded** | The README's laptop run used these defaults. Every count in updates (warm-up, snapshots, crossover cadence, BC gate momentum) inherits the 30× smaller update |
| Team sizes in training | 4v4 only (`YS:83`) | 5v5 only, equal teams required (†`V2:contracts/config.py:118`) | 4v4 by default; uniform draws and unequal teams supported (`L:learn/config.py:116-118`; `L:learn/collect.py:135-142`) | lean = v1; open with the author (README "Decisions still with the author") | The plan's step 4 says sizes drawn per environment, unequal included (`core-brief.md` §4; `MA:371`). Unequal teams have not run with the real game |
| Rollout storage | not checked | Device staging ring; pinned host store by default (rl-v20 set it off); micro-batched learner (†`V2:runtime/store.py:35-50`, †`learn/ppo.py:301-316`) | Everything on the device, no micro-batching. A minibatch is whole environments × full segment (`L:learn/update.py:395-402`) | lean departs from v2; unrecorded | Peak memory 3.6 GB at 256 environments 4v4 (`L:README.md`). Untested at v1/v2 batch sizes or 20v20 |
| Compilation | not checked | Rollout stages compiled with CUDA graphs; learner loss compiled (†`V2:runtime/rollout.py:196-199`) | Eager (`L:README.md`) | recorded (gate G1 pending) | |
| In-training evaluation | 5 slots of 512 environments that advance with the rollout; the scripted win rate over a 100-game window (`YS:188-192`) | 1,024 rating-lane environments in the same batched rollout (†`V2:runtime/lanes.py:121`) | A crossover at size 4 only, every 50 updates, 64 games, run as a separate loop (`L:train.py:171-179`, `:218-221`) | lean departs from both; unrecorded | The BC gate reads training games instead (Learner, row "BC") |
| Crossover games at the clock | 17 s cap; a tie is a non-win (`YS:272`, `:276`) | n/a | The 180 s health rule decides; only an exact tie is a draw (`L:eval/matches.py:72-74`; `L:game/core.py:546-551`) | consequence of the Game "clock" row; unrecorded for evaluation | "Ties as non-wins" nearly never applies, so ratios are not comparable with v1's |
| Crossover first size | Doubling, then bisection (`YS:273`) | n/a | A walk from parity (`L:eval/matches.py:129-137`) | recorded | |
| Games per cell | v1's formula (`YS:271`) | n/a | `games_for` exists, but `train.py` passes a fixed 64 (`L:train.py:172`, `:219`) | unrecorded | |
| Position offset in evaluation | n/a | Drawn per match (†`V2:models/yemong/model.py:865-870`) | None passed (`L:eval/matches.py:56`) | recorded (`L:README.md`) | Zero lies inside the training range, but it is always the same offset |

---

## v2 complexity lean correctly dropped (may look like departures)

- The belief path and the enemy-action, density, correction, shot and geometry heads:
  the plan keeps only next state (`core-brief.md` §4; `MA:250`, `:356`).
- Categorical state codes and two-hot inputs, replaced by scalars (`MA:345`).
- Local-presence counts, which scale with fleet size (`MA:154`).
- The outcome "unresolved" class and the 4-class outcome head (the plan's per-ship
  levels).
- The shrink, frontline, deathmatch and respawn: deferred to after M1 (`core-brief.md`
  §4).
- Cheese agents and PFSP or rating sampling: deferred (league refinements after M1).
- The policy coefficient of 27 and nine loss coefficients, collapsed to about three
  (`MA:202`).
- Host-pinned storage, CUDA-graph staging and AOTI: pending gate G1.

## Could not determine

- v2's spawn resources (store and shield spread at reset), and whether v2 resets cooldowns
  randomly.
- v1's value-target handling across epochs, and v1's initialisation.
- v2's parameter count, and v1's exact count (only an LFS-size estimate exists).
- Whether v2's death rule is exactly lean's ("hit while the shield is at zero"). Lean
  matched v2 on 169 deaths in the comparison test, but I did not read v2's `combat.py`.
- The real damage magnitudes per tick in lean's game. The ×11 per hit uses the card's
  `hit_damage` of about 19 to 23 before closeness and aspect, not a measured
  distribution.
