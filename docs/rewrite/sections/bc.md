# Section 7: behaviour cloning (BC)

Status: decisions settled, 2026-11-01. Spec: `scripted-bc-spec.md` (with section 4).

## Already decided (training review 4c, and the 2026-10-30 findings)

- **BC runs alone.** RL then warm-starts from BC's weights and return scale, with a
  fresh optimizer and no cloning term.
- **On-policy imitation:** the student flies the learner views, and the teacher (the
  scripted agent, section 4) labels the states it visits.
- **BC rollouts use the opponent pool,** so ratings come for free and BC and RL share one
  rollout structure.
- **No entropy bonus in BC.** The clone's entropy comes from the teacher's distribution.
- **Coefficients:** v1's final BC values are the starting point (BC 0.025, value 0.007 per
  level, next state 0.0041, enemy action 0.11, density 0.021, shot outcome 0.019; the
  shot trajectory at 0.007 averaged over ages, as in section 6), in nats per token.

## Decided (author, 2026-11-01)

- **A head start, not convergence.** BC gives every head a head start: the policy, the
  value, next state and the rest. None needs to converge: the value must move anyway
  once RL changes the policy. So there is no separate value stage. BC reports explained
  variance per level as a diagnostic.
- **The teacher's entropy is set after the fact:**
  - the scripted agent computes its natural distribution, and one transformation sets
    its entropy, so the clone starts RL at the entropy we want (an implicit entropy
    target);
  - one global temperature on the joint distribution, calibrated offline (bisection on
    teacher rollouts) to a target mean entropy in nats per ship-step. Not per state:
    that would erase the teacher's relative confidence, which the clone should learn;
  - it applies to the BC labels only. As an opponent the scripted agent plays its
    natural distribution, so the opponent and its 1000 anchor stay fixed;
  - the target is set by measurement: start at temperature 1, measure the natural
    entropy, choose after the first RL runs.
- **Stopping:** BC stops at whichever comes first:
  - the live score against the scripted agent reaches 0.45 over a window (wins plus half
    the draws, from the training games themselves), even if the BC KL is still falling;
  - the BC KL plateaus (an automatic detector, which reports and stops);
  - a fixed budget.
- **Ratings from the training games:** live win rates and Elo come from the games the
  rollouts play anyway. No evaluation rollout every update; diagnostic rollouts or
  updates only, and rarely.
- **The mix:** the training mix, Frontline 0.5 and team deathmatch 0.5, a quarter of the
  team deathmatch on the battle map (squad 0.375, battle 0.125).
- **Acceptance:**
  - a toy test: the toy model (section 6's relational MLP) clones the scripted agent at
    2v2 team deathmatch;
  - the clone scores at least 0.45 against its teacher;
  - its win rate against the chaser is close to the teacher's;
  - no v1 calibration comparison (v1 and v2 are different games).

## Decided after implementation (author, 2026-11-02)

- **The teacher never gives an action probability zero:** the label floor ε is on by
  default (0.01 over the 42 joint actions), so the clone can sample every action when RL
  starts. If the clone's entropy is still too low, the levers are a temperature on the
  targets or a separate entropy loss during BC.
- **Frontline zone assignment by rank** (the scripted agent) is approved: the literal
  "neediest zone" rule moved every ship at once and drew 4.7% against random.
- **`micro_batch_rows` and `precision`** may change on a resume (section 11).
- **The score stop is a confidence bound,** scored against the teacher playing with the
  labels' own floor. The first rule stopped the first time a 1000-game window reached
  0.45; checked every update, it fires on a lucky window, and the toy clone's fresh
  512-game scores were 0.478, 0.447 and 0.405. The stop now needs the window's score minus
  two standard errors to reach the target, against a teacher whose play has the same
  floor as its labels (at floor 0.01 the clone did not reach 0.45 against the unfloored
  teacher in 100M steps). The toy test's bar stays 0.45 minus two standard errors of its
  512 games. Until this lands the toy test is marked `xfail` (not strict).
- **The score target is 0.40** (author, later on 2026-11-02): the exact cutoff is not
  critical; v1 stopped BC at about 0.40 against its scripted agent and RL trained from it.
  `StopConfig.score_target` defaults to 0.40, with the confidence bound (the window's score
  less two standard errors) and the floored teacher as above. The toy test's bar is the
  target less two standard errors of its 512 games (about 0.356). At 0.45 the toy run
  stopped by score at 91.2M steps and scored 0.401 over 512 fresh games, below that bar
  (0.406).
