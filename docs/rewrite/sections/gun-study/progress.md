# Gun study: progress

Branch study/guns (from fix/spawn-heading). The question: do the real-scale cards kill in
the first head-on pass, so that who fires first decides every duel?

## 2026-11-12 11:40 KST: duels launched

- Tried: `bench/gun_lethality.py`, 1v1 duels (scripted agent both sides, real-scale
  deathmatch set, no fields, 3488 m arena), the first kill of each environment from a
  neutral spawn; 28 mirrors + 6 cross pairs, 512 duels each, four CPU shards
  (systemd units gunstudy-0..3, records duels-0..3.jsonl, one line per matchup, resumable;
  bound 14:06 KST).
- Pilot (P-51D mirror, 64 duels): 92% of kills in the first merge, 36% of duels a trade
  (both die on one tick), P(win | first hit) 0.88, measured time from first hit to kill
  0.70 s, hit fraction 0.38, 98% of damage from the front.
- Next: the full table; then the archetype draft.

## 2026-11-12 11:55 KST: duels done (34 matchups x 512)

- Worked: the hypothesis holds. In 25 of the 28 mirrors 96-99% of first kills fall in the
  first head-on merge, 20-43% of duels are trades (both die on one tick), P(win | first
  hit) 0.88-0.97, measured first hit to kill 0.5-1.0 s, 98-99% of damage from the front
  hemisphere. The exceptions are the three whose head-on kill takes longer than a pass:
  Hammerhead (5 Hz, shield 160: 25% first-merge), Anvil (2 Hz: 11%) and vizia (a third of
  the damage, head-on 0.2: 8%, median 21 s from first hit to kill). Wren is lethal but its
  P(win | first hit) is only 0.73 (it turns away from a losing pass).
- Cross pairs are head-on races decided by card: Zero beats Me 262 99%, P-47D beats
  Spitfire IX 92%, P-51D beats vizia 98%; table.md and gun_lethality.json here.
- Next: card variants of the P-51D mirror (damage, head-on fraction and angle, shield) to
  attribute the lethality to knobs and set a target; unit gunstudy-variants.

## 2026-11-12 12:30 KST: variants, target, archetype draft

- Worked: the P-51D mirror's variants (variants.jsonl, variants/table.md) show a switch
  between a head-on TTK of 1.0 and 1.5 head-on passes: damage x 1/2 (1.8 s, 1.0 pass)
  still 97% first-merge kills; damage x 1/3 (2.7 s, 1.5 passes) 4%, P(win | first hit)
  0.62. Damage, shield and head-on fraction act only through the head-on hits a kill
  needs (identical duels seed for seed); the head-on angle does nothing.
- Proposed target: head-on TTK (card, 100% hits) 1.5-2.5 head-on passes (2.7-4.5 s at
  P-51D numbers, 0.5-0.9 of the instantaneous 180-degree turn); tail TTK at most a fifth
  of that turn. vizia already sits inside.
- Draft: docs/explanation/game/ship-design/archetypes.md (Jujak turn, Yongma energy, as
  vizia handle deltas; both cards solve, variants_solved.json via variants_solve.py).
- Didn't: the threshold is one type's (the P-51D's speeds and ranges) and the scripted
  agent's; not checked on other pass times or with learned agents.
- Next (the author's call): pick the lethality target; implement the doctrine pieces in
  section 5 of the draft; run the capped loop on Jujak and Yongma.
- Docs quick tier: tests/docs has 8 failures, all also on the base commit (other benches'
  docstrings and other pages' references), none from this branch.

## 2026-11-12 12:48 KST: WWII archetypes, aspect statistics, duels launched

- Worked: the turn and energy fighters in numbers (`archetypes_ww2.py`, from the set's
  research): the energy fighter is 1.56 times as heavy on a wing 1.33 times as loaded,
  turns 1.29 times slower sustained (1.35 at 100 m/s), has 1.37 times the W/S over C_D0,
  1.4 times the weight of fire and twice the survivability; level top speed, ceiling, g
  limit and zoom are shared, and the turn fighters climb better (1.24) on more power per kg.
- Yongma and Jujak redone as vizia deltas scaled from those ratios (`archetypes_solve.py`:
  r^(+-s/2), s = 1 for flight, 1/2 for armour and guns); both solve; a speed group at s = 2
  solved as the alternative.
- Aspect statistics: Events gain `enemy_damage_by_aspect` and `kill_aspect` (head-on
  under 45 degrees, tail over 135); training series `combat/damage_share/{aspect}` and
  `combat/kill_share/{aspect}`; match statistics `damage_{aspect}`, `kills_{aspect}`;
  the gun-lethality bench records and reports both. Fingerprint unchanged.
- Launched: units aspect-p51d, aspect-vizia (mirrors), aspect-variant (P-51D with vizia's
  guns), aspect-cards-s1 and aspect-cards-s1-speed2 (Yongma and Jujak mirrors and cross
  pair, temporary cards), 512 duels each, records and logs in `aspect/`, one line per
  matchup, resumable; ETA 13:10 KST (worst 13:30), bound 14:48 (RuntimeMaxSec).

## 2026-11-12 13:05 KST: aspect shares measured

- Worked: all five units finished (12:57); `aspect/table.md` and `aspect/gun_lethality.json`.
  Damage head-on / flank / tail: P-51D mirror 0.99 / 0.01 / 0.00, P-51D with vizia's guns
  0.99 / 0.01 / 0.00, vizia 0.97 / 0.03 / 0.00, Yongma 0.95 / 0.05 / 0.00, Jujak
  0.98 / 0.02 / 0.00, Yongma v Jujak 0.83 / 0.17 / 0.00 (Jujak's own 0.76 / 0.24; kills
  0.60 / 0.40 / 0.00); Jujak wins the cross pair 92% (s = 1 and s = 1-speed-2 alike).
- No tail hit in 4,608 duels; an independent check (hit multipliers of 64 vizia duels,
  6,793 hits) finds none beyond ~72 degrees. The scripted agent's long fights are
  repeated head-on passes; the archetypes.md sentence that called them tail chases is
  corrected.
- vizia's mirror here: P(win | first hit) 0.82 ± 0.02 against 0.87 in the first run (the
  `--pairs` run seeds it differently); within about two standard errors.
- Next (the author's call): the doctrines of section 5 (the flank and tail strategies
  appear only with them or with learned play); the aspect series now track it in training.
