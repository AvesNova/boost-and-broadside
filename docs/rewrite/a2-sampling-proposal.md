# A2 team-size and field-count sampling: proposal

For the author, 2026-11-16. The author's answers of the same day: A2 varies ship and field
counts together; 5v5 stays the most common size (at least half of the games); uneven sizes
(4v5, 3v5, 2v5, ..., 3v4) are included; the sampling equations are discussed before the
run. This page gives three candidate distributions, the rules around them that need a
choice, and a recommendation. The mechanism is built and takes any of them as a setting;
nothing here changes a run that does not set `team_sizes` (the default, None).

## The mechanism (built)

`TrainConfig.team_sizes` (`bnb.contracts.TeamSizes`) gives every training pool a table of
(team 0's ships, team 1's ships, fields, probability). Each episode draws one row by one
keyed uniform (seed, environment, episode), so the draw moves no other draw. A team's slots
past its size are benched and the fields past the count are absent, so the batch keeps its
static 5v5 shape with 10 field slots: nothing recompiles per size. Benching is the game's
own mechanism, the one the break-even measurement (`bnb crossover`) already uses for its
uneven lineups, so A2 and break-even play uneven games by the same rules.

A size pair is written larger against smaller, a against b, with S = 5 slots a side. Each
uneven pair is split evenly between the two orientations (5v3 and 3v5), and each side is
played by the learner half the time through the usual seat draw.

Settings (`--set 'team_sizes={...}'`; `{}` is the provisional default below):

| field | default | meaning |
|---|---|---|
| `rule` | `uniform_pairs` | the ship distribution: `uniform_pairs`, `independent_teams`, `near_full`, or `table` |
| `full_share` | 0.5 | the share of 5v5 games |
| `low` | 1 | the smallest team, before each mode's minimum |
| `uneven` | true | false keeps N against N only (not with `independent_teams`) |
| `decay` | 0.5 | `near_full` only |
| `pairs` | none | `table` only: (a, b, weight) rows |
| `fields` | `per_ship` | `per_ship` couples the field count to the ships; `game` leaves it to the game (every field, or `active_fields`) |
| `field_spread` | 0 | `per_ship`: draw the count evenly from this many playable counts either side |
| `series` | true | the training log's per-mode series of the drawn sizes |

## Three candidate ship distributions

Let m be the mode's smallest team (1 in deathmatch and elimination, 3 in Frontline) and P
the set of pairs (a, b) with m <= b <= a <= 5.

**A. Half 5v5, the rest even over pairs (`uniform_pairs`).**
p(5,5) = 1/2, and p(a,b) = (1/2) / (|P| - 1) for every other pair. |P| = 15 in deathmatch
and elimination (each other pair 3.6%), 6 in Frontline (each 10%).

**B. Independent teams (`independent_teams`).**
Each team draws its size on its own: q = sqrt(1/2) = 0.707 for 5, the rest (1 - q)/(5 - m)
each; p(a,b) = q_a q_b, both orders added. 5v5 is exactly 1/2.

**C. Near full (`near_full`).**
p(5,5) = 1/2; the rest in proportion to d^(10 - a - b) with d = 1/2: each missing ship
halves the weight.

### Deathmatch and elimination (m = 1): games / ship-steps, %

Ship-steps assume equal episode lengths; small deathmatch games run the clock more often
(the freeze measured 1v1 ending by tickets 0.4% of the time), so their real step share is
somewhat higher than shown, and very uneven games (5v1: 5 kills to win) shorter.

| pair | ships | fields | A | B | C |
|---|---|---|---|---|---|
| 5v5 | 10 | 10 | 50.0 / 63.6 | 50.0 / 58.6 | 50.0 / 56.6 |
| 5v4 | 9 | 10 | 3.6 / 4.1 | 10.4 / 10.9 | 16.2 / 16.5 |
| 5v3 | 8 | 8 | 3.6 / 3.6 | 10.4 / 9.7 | 8.1 / 7.3 |
| 5v2 | 7 | 8 | 3.6 / 3.2 | 10.4 / 8.5 | 4.1 / 3.2 |
| 5v1 | 6 | 6 | 3.6 / 2.7 | 10.4 / 7.3 | 2.0 / 1.4 |
| 4v4 | 8 | 8 | 3.6 / 3.6 | 0.5 / 0.5 | 8.1 / 7.3 |
| 4v3 | 7 | 8 | 3.6 / 3.2 | 1.1 / 0.9 | 4.1 / 3.2 |
| 4v2 | 6 | 6 | 3.6 / 2.7 | 1.1 / 0.8 | 2.0 / 1.4 |
| 4v1 | 5 | 6 | 3.6 / 2.3 | 1.1 / 0.6 | 1.0 / 0.6 |
| 3v3 | 6 | 6 | 3.6 / 2.7 | 0.5 / 0.4 | 2.0 / 1.4 |
| 3v2 | 5 | 6 | 3.6 / 2.3 | 1.1 / 0.6 | 1.0 / 0.6 |
| 3v1 | 4 | 4 | 3.6 / 1.8 | 1.1 / 0.5 | 0.5 / 0.2 |
| 2v2 | 4 | 4 | 3.6 / 1.8 | 0.5 / 0.3 | 0.5 / 0.2 |
| 2v1 | 3 | 4 | 3.6 / 1.4 | 1.1 / 0.4 | 0.3 / 0.1 |
| 1v1 | 2 | 2 | 3.6 / 0.9 | 0.5 / 0.1 | 0.1 / 0.0 |
| mean ships | | | 7.9 | 8.5 | 8.8 |
| mean fields | | | 8.1 | 8.8 | 9.1 |
| even games | | | 64% | 52% | 61% |

### Frontline (m = 3)

| pair | ships | fields | A | B | C |
|---|---|---|---|---|---|
| 5v5 | 10 | 10 | 50.0 / 56.8 | 50.0 / 54.8 | 50.0 / 54.9 |
| 5v4 | 9 | 10 | 10.0 / 10.2 | 20.7 / 20.4 | 21.1 / 20.8 |
| 5v3 | 8 | 8 | 10.0 / 9.1 | 20.7 / 18.2 | 10.5 / 9.2 |
| 4v4 | 8 | 8 | 10.0 / 9.1 | 2.1 / 1.9 | 10.5 / 9.2 |
| 4v3 | 7 | 8 | 10.0 / 8.0 | 4.3 / 3.3 | 5.3 / 4.0 |
| 3v3 | 6 | 6 | 10.0 / 6.8 | 2.1 / 1.4 | 2.6 / 1.7 |

### Pros and cons

- **A (uniform pairs).** One number (the 5v5 share) and every other pair alike: the easiest
  to state and to read in the per-size figures, and every pair gets a measurable share
  (3.6% of deathmatch games each, 10% of Frontline's; the games per pair in a run follow
  from the game lengths, to be read off the dry run's series). Cost: the smallest games
  (1v1, 2v1) take as many games as 5v4, so 5v5 keeps the most ship-steps of the three
  (64%) and the fewest learner ship-steps go to the near-5v5 sizes.
- **B (independent teams).** A natural story (each side's roster is drawn on its own) and
  the most uneven games, nearly all of them 5 against fewer. Cost: small even games almost
  vanish (2v2 0.5%, 1v1 0.5%), which is the regime the zero-shot fleet-size figures ask
  about, and 5vk dominates, a "beat the weaker team" curriculum.
- **C (near full).** Stays close to the evaluated 5v5 (5v4 and 4v4 take most of the
  other half), the gentlest change for an arm compared at 5v5. Cost: 1v1 to 2v2 get
  under 1% of games each, too few to measure per size; one more parameter (d).

A table rule (`rule: table`, `pairs`) gives any other distribution, for example one tuned
to equal ship-steps per pair.

## Field counts

Real-scale maps have 10 field slots in point-symmetric pairs, so an episode plays an even
count. Proposed: **one field per ship** (the 5v5 density, 10 fields for 10 ships):
fields = 10 (a + b) / 10, to the nearest even count, ties up (9 ships 10 fields, 7 ships 8,
3 ships 4). The tables above list it per pair. Alternatives, both settings: `field_spread
1` draws the count evenly from the even counts next to it (5v5 then plays 8 or 10), which
adds field variation at a fixed size; `fields: game` leaves the count to the game (all 10,
or the separate `active_fields` draw, uniform 0 to 10), which varies fields apart from the
ships. One field per ship is the provisional default because it keeps the map's density of
cover what 5v5 trains on; it never plays 0 fields.

## Rules that need a choice

1. **Tickets in uneven deathmatch.** Three readings:
   - *Each team's tickets from its own size* (the frozen table 4 N^(4/3) to the nearest 5,
     read as each team's lives): a team wins by killing the enemy's tickets. In 3v5 the
     3-ship team needs 35 kills, the 5-ship team 15. **This is the game's current rule**
     (`phase.deathmatch_targets`: the winning score counts the enemy's active ships), the
     one break-even uses; each ship carries about the same lives at any size.
   - *Both from the larger size*: both need 35 kills; the small team's ships die about
     12 times each, more clock-outs in small games.
   - *Each from its own size as a score target* (a handicap): the 3-ship team needs 15,
     the 5-ship team 35, which evens out the uneven game.
   Provisional default: the current rule, no game change. An uneven game then has a clear
   favourite, which is what teaches the learner to read the odds; the handicap would hide
   the size from the result.
2. **Frontline's 3-ship minimum** applies to each team: 5v3 and 4v3 play, 5v2 does not
   (the freeze: at 2v2 captures are rare and most games end on the clock). Break-even
   still goes down to 1 ship in every mode.
3. **Elimination** needs no rule: a team wins by wiping the enemy's active ships, and at
   the clock by the larger health sum, which favours the larger team.
4. **Arena size** stays the pool's (and the map-size arm's draw when that arm is on). The
   density rule (`arena_radius_for`: radius grows as sqrt(ships)) is not applied, so a 1v1
   plays on the 5v5 arena: more search, fewer encounters. Open: apply the rule per drawn
   size (the larger team's count, or the total), which would also scale the field
   positions with the arena.
5. **Clocks and the elimination shrink** stay the same at every size (the freeze).

## Recommendation

**A (uniform pairs) with half 5v5 and one field per ship, tickets as now** (the
provisional default, `--set 'team_sizes={}'`): it meets the 5v5 floor, covers every size
and orientation with enough games to measure per pair, and is one parameter to explain.
Choose C instead if A2's verdict is to be read mainly at 5v5 and the small sizes are only
a regulariser.

The training log records what was drawn, per mode and update: `team_size/larger/{mode}`,
`team_size/smaller/{mode}`, `team_size/fields/{mode}` (means over the episodes ended) and
`team_size/share/{a}v{b}/{mode}`, on by default with A2 (`series`).
