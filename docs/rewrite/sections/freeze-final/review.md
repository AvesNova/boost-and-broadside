# Final freeze check: review (2026-11-15)

Game version 22, Imugi on both sides, fog and fields on, each preset's own map. The
numbers, their SEs and game counts are in [results.md](results.md). Done on this branch: the
shield fix (v22), vizia dropped, Imugi's doctrine lookup fixed, and a bench fix for
elimination results (the energy-against-turn elimination scores in the earlier partial
tables were too high by 4–40 points).

## The checks

| check | target | measured | verdict |
|---|---|---|---|
| deathmatch ended by tickets within 10 min | about 80% | 5v5 80.5 ± 2.5%, 10v10 83.6 ± 2.3% | pass |
| | | 2v2 43.4 ± 2.2%, 1v1 0.4 ± 0.4% | **fail** (problem b) |
| elimination 5v5 median | about 2:05 | 1:35 ± 3 s | **fail**: 30 s short |
| elimination 5v5 p80 | 3:10–3:20 | 2:18 ± 5 s | **fail**: about 55 s short |
| elimination 5v5 deaths by the boundary | 8–12% | 9.9 ± 1.1% | pass |
| elimination 5v5 last deaths by the boundary | 15–20% | 19.1 ± 2.5% | pass |
| Frontline front close to a random walk (P(reversal) 0.5) | 0.5 | 5v5 0.467 ± 0.019, 10v10 0.520 ± 0.020 | pass |
| | | 3v3 0.402 ± 0.020 | borderline: persistent |
| Yongma team against Jujak team | within 55/45 (soft) | 8–30% in every mode and size except deathmatch 10v10 (48.2 ± 1.8%); deathmatch 5v5 21.8 ± 1.8% | **fail** (problem a) |
| aspect mix | a good mix | 5v5: offset 11–16%, flank 48–53%, tail 28–36% | pass at 5v5 and up; tail above the earlier 10–14% hope |
| | | 1v1 and 2v2: tail 41–50% | borderline: small fights end from behind |
| mutual head-on | near zero | 3.0–4.3% of kills (1.5–3.1% at 1v1–2v2) | pass |
| head-on passes before a kill | 3–4 | PASSES_SUMMARY | **fail** |

Notes on the checks:

- **Elimination is short** at every size (1v1 median 2:33, 10v10 1:26), and no game reaches
  the clock. The boundary shares are where they should be, so the shrink is not the cause;
  the fights resolve sooner than the target assumed. If the 2:05 target matters, the knob
  is the shrink's start or speed, not the clock. I would accept it as is: a shorter
  elimination game gives more games per GPU-hour.
- **Head-on passes.** With the head-on fraction at 0 (physics v12), a hit straight on the
  nose does no damage, so the card's head-on time to kill is infinite and the 3–4 target no
  longer has a meaning on the card. In play, Imugi's kills come in or right after the first
  merge, as in every round since 2026-11-12 (0.4–0.6 then). I would retire the target, or
  restate it as merges per kill if repeated passes are still wanted.
- **Friendly fire** is a larger share than I expected: in deathmatch 10v10 energy against
  turn, 16% of the Jujak team's kills are its own ships (6% for Yongma). Not a target, but
  it is what holds that condition near even.

## (a) The Yongma team scores about 22% against the Jujak team

What the data say: Jujak wins at every size and in every mode, most at 2v2 (deathmatch 7.8 ±
1.6%, elimination 18.6 ± 1.7%) and in Frontline (12–18%). Only deathmatch 10v10 is near
even, and that is the Jujak team's friendly fire. The design gives Yongma 1.78 times the
shield and 1.2 times the damage against Jujak's 1.29 times the sustained turn. Under Imugi
the turn wins.

One reservation before any card change: this is Imugi's balance, not the game's. In round 12
Imugi's Yongma doctrine barely differed from its turn doctrine (KL 0.08 against Jujak's
2.6), so Imugi may not fly the slash-and-extend the archetype is built around. A card tuned
to even under Imugi could then be wrong for a trained agent. Also, training lineups are
mirrored (each slot's coin flip is the same for both teams), so the imbalance never decides
a training game; it changes which ship is the better pick, which matters for the player
and for the ship-cycle goal, not for the arms' fairness.

Options:

1. **A card handle or two** (the archetype procedure's step 4). Candidates, each moving one
   ratio: Jujak's `sustained_turn_rate` down (0.765 to about 0.70, a ratio of 1.18 instead of
   1.29); Yongma's `max_shield` or `hit_damage` up; Yongma's `low_speed_acceleration` up
   (8.8 against 13.5, the largest gap after the shield). Cost: a sweep of about six variants
   on the stored doctrines, energy against turn at deathmatch 2v2 and 5v5, elimination 5v5
   and Frontline 5v5 (512 games each, about 2 minutes each with compile): about 50 minutes
   of GPU, then the mixed-lineup mode checks again for the chosen card (about 30 minutes).
   Any card change bumps the game version (23), and the stored doctrines were searched on
   this card, so a doctrine re-search may follow (`bench/balance_round.py`, hours).
2. **A Yongma doctrine searched against Jujak** before touching the card: the archetype's
   own target says a disciplined Yongma should beat a Jujak that chases. If the search
   finds an energy doctrine that reaches 45%, the card is fine and Imugi was the gap. Cost:
   one doctrine search (`bench/balance_round.py` search stage, up to 40 generations; a
   generation took 25–30 minutes in the doctrines round, so up to most of a day unless it
   stops early) plus the 1,024-game check. No game version change.
3. **Accept it for the freeze** and record the imbalance: training is unaffected (mirrored
   lineups), and the trained agent's own energy-against-turn score can be measured after the
   first arm, which is the balance that matters for the release.

My recommendation: 3 for the arms, with 2 run when the GPU is free. A card change now would
cost a game version, the doctrines and the BC data, on evidence from a controller that may
not fly Yongma as designed. If 2 fails to reach 45%, then 1, after the arms' first results.

## (b) Small-team deathmatch rarely ends by tickets

What the data say. The premise that Imugi rarely finds the enemy holds only in part:

- **1v1**: a hit in 82% of games (median 40 s after the start), but 1.6 deaths in 10
  minutes, about a third by the boundary; 29% of games have no point at all and 41% end
  drawn. Every 1v1 in the mixed lineups is a same-type duel. The duel, not the search,
  is most of the problem.
- **2v2**: contact in every game (first hit 17 s), but a slow pace: 15.5 deaths a game
  against 10 tickets a team, 1.59 kills a minute (0.40 a ship-minute, against 0.84 at
  5v5). Spawn to first hit has a median of 30 s and a q80 of 63 s (15 s and 26 s at 5v5), so
  re-finding after a respawn is part of it.
- The map is 4200 m at every size (fixed maps, the author's rule of 2026-11-15), so a 2v2
  has two fifths of a 5v5's ships per area and a 1v1 one fifth.

Replaying the recorded games with other ticket counts (free, from the data already on
disk; mixed lineups, share ended by tickets within 10 min):

| tickets | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|
| 1v1 | 71% | 35% | 13% | 4% | 0% | 0% | 0% | 0% |
| 2v2 | 99% | 98% | 97% | 96% | 94% | 89% | 72% | 43% |

Options:

1. **Tickets at small sizes.** 2v2 at 7 tickets would end about 80% of games (6 gives 89%,
   8 gives 72%); a per-size table or a steeper exponent below 5v5. 1v1 reaches 80% at no
   ticket count; at 1 ticket it is a sudden-death duel (71%), which is elimination without
   the shrink. Cost: none to estimate (replay above); a rules change and a game version
   bump to adopt, confirmed by one 512-game run each at 1v1 and 2v2 (about 4 minutes).
2. **The density map for small games** (`arena_radius_for`: the radius times √(n/5),
   floored at 3000 m for deathmatch), which is already implemented for the map-size arm.
   It would put 1v1 and 2v2 on 3000 m instead of 4200 m: half the area. Cost: two runs
   with `--radius 3000` (about 4 minutes); it reverses the fixed-map rule for these sizes.
3. **Imugi's search after a respawn or a lost contact**: head for the last known enemy
   position or the map centre (the old `SEARCH = "centre"` met in the middle at once)
   instead of exploring the nearest unseen cell. Cost: about 4 minutes per variant at
   1v1 and 2v2; it changes the scripted opponent the league and BC data use, so it needs
   the 5v5 checks again (about 30 minutes). It would not fix the 1v1 duel itself.
4. **Accept a clock-decided small deathmatch**: the clock already gives a result by the
   kill lead. The cost is evaluation signal: 41% of 1v1 games are draws, which carry little
   information for the 1v1 deathmatch condition.

My recommendation: 1 for 2v2 (about 7 tickets; the cheapest and best supported by the
data), and for 1v1 either 1-ticket sudden death or dropping 1v1 deathmatch from the
evaluation conditions, since elimination 1v1 already measures the duel and ends every
game. Option 3 is worth a short test, but I would not change the scripted opponent for it
before the arms.
