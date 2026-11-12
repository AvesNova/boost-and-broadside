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
