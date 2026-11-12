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

## 2026-11-12 12:00 KST: duels done (34 matchups x 512)

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
