---
covers: [src/bnb/game/ship_design.py, bench/ship_design_search.py, bench/knob_independence.py]
---

# Designing ship types

> Since 2026-11-13 the real-scale set holds only Yongma, Jujak and vizia: a design's base
> is one of them, and the realism envelope is the 16 real propeller fighters as reference
> data ([The real planes](real-planes.md)). The searches below started from the
> researched types, then in the set.

The [balancing](real-scale-balance.md) evened the 27 real-scale types round by round and
found that the guns explained most of what was left. This page takes the other road: it
makes new types, built from the existing ones, whose head-to-head results are equal in
rating and form cycles (rock, paper, scissors), in the end seven types whose results form
the Paley tournament on seven vertices. Balance and cycles are the same search: a set of
types is balanced when no type and no pair of types dominates, and a perfect cycle is the
most even such set.

## What the set should be

The decisions behind the targets (2026-11-10):

- every value may move within reason, including the values every type shares today (the
  recharge, the head-on fraction, the half-lift drag, the shot cost, the damage per
  second, the muzzle speed); the only thing not allowed is a modifier between two types
  ("A does 50% more damage to B"): every result comes from the types' own values and the
  physics;
- fire rates are whole hertz dividing the 30 Hz tick; the damage per bullet fine-tunes the
  damage per second;
- trade-offs, not one dominant axis: many values should matter, and the types should feel
  different even at equal ratings, and all of them like WW2 fighters;
- the matchup should matter, and skill more.

### On the Elo scale

A score p between two players is a rating gap Δ = 400 log10(p / (1 - p)) Elo:

| Score | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 |
|---|---|---|---|---|---|---|
| Elo gap | 35 | 70 | 108 | 147 | 191 | 241 |

### What fighting games aim for

Fighting-game communities write a matchup as the games out of ten the first character
wins against an even opponent: 5-5 even, 6-4 a clear edge, 7-3 lopsided (a counter-pick),
8-2 and beyond a hard counter. A roster is thought healthy when nearly every matchup is
within 6-4 and a 7-3 is rare and treated as a fault; in Elo, an edge of 70, occasionally
up to about 150. The cycle band below (0.6 to 0.7) sits at the strong end of that
practice on purpose: a cycle is the matchups, so they have to be felt.

### Skill against the matchup

Split a game's log-odds into the players' skill gap, the types' ratings and the
matchup's cyclic part: logit P = (s_A - s_B) + (r_i - r_j) + c_ij (on the Elo scale). In a
designed set the ratings are equal and every matchup is a cyclic term of about ±70 to
±150 Elo. Whether skill matters more depends on how far apart the players are:

- **between players of a population** (a ladder's ratings spread with a standard deviation
  of a few hundred Elo), skill dominates: with players' skill spread σ = 300 Elo and
  matchup terms of ±108, the matchup is 108² / (108² + 2 x 300²) = 6% of the variance of the
  log-odds;
- **between matched players** (a gap of the order of 50 to 100 Elo), the matchup and the
  skill gap are of the same size, which is what "the matchup matters" means.

The operational test is the skill gap that overturns the worst matchup: the worst
matchup should be overturned by a skill gap smaller than one "class" of player (200 Elo,
the chess class interval; 76%). A 0.70 edge (147 Elo) passes; 0.80 (241) does not.

On the scripted agent's own scale, its variants give the skill steps (real_scale_deathmatch,
mixed lineups of every type, the round-1 champion at T = 1 against the variant, 1,024 games;
`ship-design/skill/`):

| The champion against | Score | Elo gap |
|---|---|---|
| itself at T = 1.5 | 0.568 ± 0.015 | 47 |
| itself at T = 2 | 0.697 ± 0.014 | 145 |
| itself at T = 3 | 0.857 ± 0.011 | 312 |
| its predecessor (balancing round 1, the adopt check) | 0.607 ± 0.015 | 75 |
| its predecessor (balancing round 2) | 0.566 ± 0.015 | 46 |
| a random perturbation of itself, sigma 0.25 (three draws; the round-2 champion, 512 games) | 0.516, 0.577, 0.897 | 11, 54, 377 |
| a random perturbation of itself, sigma 0.5 (three draws) | 0.682, 0.695, 0.950 | 132, 143, 512 |

(The perturbation is the search's own: every number times exp(sigma N(0, 1)).) A small
random change of the agent costs from nothing to several hundred Elo: the agent's skill is a
narrow ridge, and its steps are of the size of the designed edges.

So a 0.70 edge equals the gap between the agent and itself at twice the temperature, about
two of the balancing's search rounds; a 0.60 edge (70 Elo) is one round's gain. For the
scripted agent the designed matchups are as large as its skill steps by construction, which
is the "matters" half of the target; the "skill matters more" half is a statement about
players, whose skill spreads far wider than one agent's tuning steps.

### No dominant type, no dominant pair

Balduzzi et al. ("Re-evaluating Evaluation", 2018) evaluate a population by the Nash
equilibrium of the meta-game: here the game of picking a type against an opponent who
picks one, its payoff the score minus a half. A balanced set has a Nash equilibrium with
full support (every type is worth picking), and close to uniform; equivalently, every type
scores about 0.5 against the uniform field, which is the equal-rating condition. The
exploitability of the uniform pick (the best type's mean score against the field, less a
half) measures how far from it a set is.

"No pair of types dominant" has a classical form: a tournament has the property S_2 when
every pair of players is beaten by some third player. The Paley tournament on seven
vertices (type i beats i + 1, i + 2 and i + 4, mod 7) is the smallest tournament with S_2,
which is why seven types and that tournament are the target. Its matrix of log-odds is
three rotations (three cyclic planes), and in each plane the seven types sit on one circle
at angles 2πki/7 (the ship-cycles page on the branch feat/ship-cycles derives it):
every type has three wins and three losses, and uniform picking is its unique Nash
equilibrium. Because S_2 is about picks, the mixed-team test checks it
in play: a team of A and B together against a team of the D that beats both.

### Feeling different

Each type is a point in the knobs' normalised space (each knob's log in deviations of the
real propeller fighters; below). Two measures:

- the distance between any two types, against the real fighters' own spread: the 16
  propeller fighters are 0.95 to 6.5 deviations apart (median 3.4), so a design closer than 1
  to another is a near copy;
- each cyclic plane is built from its own group of knobs (flight; guns and armour; speed
  and reach), so no group carries the whole matchup.

### The targets

| Target | Measure | Value |
|---|---|---|
| Equal ratings | each type's Elo from the mean (round robin, 1,024 games a pair) | within ±35 (0.55) |
| The cycle | every designed edge's score | 0.60 to 0.70 (0.58 to 0.72 accepted) |
| No dominant type | exploitability of the uniform pick; Nash support | ≤ 0.02; every type |
| No dominant pair | the mixed-team test: D against A + B | D scores ≥ 0.55 in every case |
| Skill over matchup | the worst edge against the agent's skill steps | the worst edge under 147 Elo (0.70) |
| WW2 envelope | each design's Mahalanobis distance from the propeller fighters' mean | ≤ 3.5 (the real Bf 109K and Zero sit at 3.7) |
| Different | each pair of designs' distance in the knobs' space | ≥ 1.0 deviation |
| Trade-offs | the knob groups of the three planes | disjoint |

## The knobs

`bnb.game.ship_design.KNOBS`, each an existing handle or a ratio of them, chosen for a
reason and to be nearly independent of the others (below):

| Knob | Group | Why it is in the search |
|---|---|---|
| stall speed | flight | the wing loading: the corner speed and the slow fight |
| cruise speed | flight | the speed: who chooses the fight |
| sustained turn rate | flight | the turn fight |
| climb rate | flight | the store's charge, as a ratio to the set-wide level: the energy fight and the reset |
| store (`ceiling`) | flight | how long a ship can boost and fire |
| damage per second | combat | the weight of fire (the hit damage is it over the fire rate) |
| fire rate | combat | how the damage arrives: the hits a kill takes against a shield (10, 15 or 30 Hz native: the planes' 5, 7.5 and 15) |
| shield | combat | toughness (80 to 200) |
| muzzle speed | combat | the gun's reach and the lead a shot needs |
| dispersion | combat | the cone (the bullet spread over the muzzle speed) |

Every other handle is the design's base type's: the g limit, the low-speed acceleration,
the nose reversal, the dive and zoom accelerations, the stall angle, the zero-lift drag, the
half-lift drag, the hit radius, the recharge and the head-on fraction. The gun range follows
from the cruise and the muzzle speed, the climb time from the store and the climb rate, the
spawn energy is the base's share of the store, and a shot keeps the base's store cost per
second of fire. Knobs considered and left out until a search needs them: the recharge
rate and delay (the burst mechanism's second lever), the g limit, the zero-lift drag (it
and the cruise both set the engine's power), the low-speed acceleration (nearly redundant
with the sustained turn: the Hammerhead's solve had a condition number of 1e9), the hit
radius (the real span).

Two set-wide values are live as well (`engine_knee_scale` and `coast_exchange_scale` on
the card): the engine knee and the store's charge. The [pacing](real-scale-pacing.md) sets
them for the whole set; the design search holds them.

### The WW2 envelope

The envelope is the 16 real propeller fighters of the set; the Me 262 and the Me 163 are
left out (the author's decision: a jet's 228 m/s and a rocket's 81 m/s climb are outside what
the propeller types can be made to fight, and a design built to match them would be the
"Mach 2 with no damage" compensation the set should avoid), as are the invented designs. A
knob's bounds are the 16's range, except the damage per second (576 to 864 native, 288 to 432 at the planes' speed: the balancing
had set every type near 360) and the shield (80 to 200, the cap raised), and the sustained
turn, ±10% past the range (no turn rate is measured). Within the bounds, a design's realism
is its Mahalanobis distance from the 16's mean in the knobs' logs, under their correlation
shrunk half way to the identity, so a combination the real fighters do not show costs more
than its parts. The real fighters sit at 1.4 (La-7) to 3.8 (Bf 109K); the invented designs
at 2.3 (Shrike) to 7.8 (Hammerhead); the search allows 3.5 and penalises the square of the
excess.

### Independence

`bench/knob_independence.py` moves each knob by a quarter of its deviation either way at
eight bases and measures what the agent and the fight depend on: the agent's
characteristics (cruise, top and corner speeds, the turn rates at three sticks, the climb
power, the gun range, the bullet speed, the store) and closed forms of the fight (the time to
recover 40% of the store, the times to kill shields of 80 to 200, the interval between shots,
the hits a P-51D's gun needs, the bullet's time to 400 m, the dispersion, the seconds of fire
a store pays for). The Jacobian (log quantities per deviation) gives the singular values and
the aligned pairs.

| Base | Singular values of the ten knobs (largest, smallest) | Condition | With unit columns | Aligned pairs (\|cos\| > 0.5) |
|---|---|---|---|---|
| P-51D | 0.44, 0.066 | 6.7 | 1.55 | none |
| Spitfire IX | 0.51, 0.066 | 7.8 | 1.58 | none |
| Fw 190A | 0.51, 0.066 | 7.8 | 1.66 | none |
| P-47D | 0.44, 0.067 | 6.6 | 1.59 | none |
| Zero | 0.51, 0.066 | 7.8 | 1.61 | none |
| Ki-84 | 0.44, 0.066 | 6.7 | 1.61 | none |
| La-7 | 0.51, 0.066 | 7.7 | 1.63 | none |
| Zerstörer | 0.51, 0.064 | 8.1 | 1.95 | none |
| Albatross | 0.76, 0.061 | 12.5 | 6.49 | stall speed and turn +0.93, cruise and turn +0.88, stall and cruise +0.87 |

(`ship-design/independence/`.) At every real base the ten knobs are close to orthogonal:
with each column scaled to unit length the condition number is 1.6 to 2.0, and the raw
condition number (6.6 to 8.1) is the knobs' different leverage per deviation, not
redundancy. The invented Albatross is the exception: its stall speed, cruise and turn move the
same quantities (its long, low-drag wing makes the turn a matter of the speed it holds), so it
is not used as a base.

The suspects named before the analysis:

- **A global charge against each type's climb**: the same column (cos +1.00 at every base),
  as they must be: both scale the coast exchange and the engine's charging power. So the
  global level is the pacing's (set once for the set, below) and each design's climb knob is a
  ratio to it; the search holds the global.
- **The engine knee against the store and the climb**: the knee moves none of the agent's
  characteristics but the top speed (0.3% for a doubling) and none of the fight's closed
  forms, which all live below half the store; it is aligned with nothing, and does nothing to a
  fight below the knee.
- **Damage per bullet x fire rate against the shot cost and the shield**: with the hit damage
  as a knob, the damage and the rate would both move the weight of fire; the knobs are the
  damage per second and the rate instead, the shot keeps its store cost per second of fire,
  and the two are not aligned (the rate moves the interval between shots and how the kill
  times against different shields split). Measured on one kill time alone they were
  collinear (cos +1.00, an earlier run of the analysis): one quantity cannot tell them apart,
  four shields and the interval can.
- **The Hammerhead's sustained turn against its low-speed acceleration** (condition 1e9 in its
  solve): the low-speed acceleration is not a knob, and the Hammerhead is not a base.

The search runs in these coordinates: each knob's log in envelope deviations from its start.

## Making live what is tuned

A compiled game takes 20 minutes or more to build, so nothing the search changes may be a
compiled constant. The card is a tensor in the rollout's carry, and every per-type value
lives there; the arena writes a new card in place (`bnb.league.matches.Arena.set_card`),
together with the scripted agent's characteristics, which the agent reads as tensors (its
set-wide lengths are the card's maxima as tensors, not host numbers). The shared flight
constants are Python numbers compiled into the graph; of them only the engine knee and the
coast exchange are tuned, and both are card fields (`engine_knee_scale`, new, and
`coast_exchange_scale`), 1 on every solved card. The number of types is fixed (36 slots in
the design arena), as are the bullet slots (the arena is built with a 15 Hz type, the
fastest a design may fire). `tests/sections/test_ship_design.py` plays a card, writes one
with another knee, coast exchange, fire rate and shields no gun can break, and checks that
no new graph is compiled and that the new card is the one played.

## The search

`bench/ship_design_search.py search --spec SPEC.json --out DIR`: CMA-ES (Hansen's
(μ/μ_w, λ)) over the designs' searched knobs, each a log in envelope deviations from its
start. Every generation plays the optimiser's mean and its samples, each candidate's edges
64 games or more (half with each design on each side, every ship of a team one design, the
scripted agent on both sides), on one arena: a candidate's designs fill three of the 36
slots, so twelve candidates of a three-type cycle play in one batch. A flight knob's card is
the instant estimate (`bnb.game.handling.estimate.Estimate`, anchored on the base's solved
card); the winner is solved exactly before it is checked. The loss is the edges' squared
log-odds outside the band, plus the realism and distinctness penalties.

**Staging.** Three disjoint three-type cycles first, one per mechanism, each on one common
base so that the knobs of the other mechanisms are equal across its three types:

1. flight (stall speed, turn, climb, store): the energy fighter, the turner, the climber;
2. guns and armour (fire rate, damage per second, shield): the sprayer, the cannon, the armour;
3. speed and reach (cruise, muzzle speed, dispersion): the slasher, the sniper, the brawler.

A cycle that forms from one group of knobs alone is a mechanism, and three disjoint ones
are three planes. The seven then sit on all three: type i at angle 2πki/7 in plane k, its
knobs in each group the mixture of that cycle's three corners at its angle; a seven-type
search starts there. Building the three first measures each mechanism's strength alone and
tests that each is a cycle at all, which the seven-type search could not tell apart.

**What the case studies changed** ([Case studies](case-studies.md)). The scripted agent
fights on the turn alone (store, climb and cruise are worth nothing to it), so the flight
cycle cannot form from flight values with today's doctrine: its energy leg needs the energy
fight, flown by that type only, which needs per-type doctrine in the arena (each player its
own parameter set; the arena's scripted seat plays one set today). The guns are a ladder on
one airframe (no cycle among fifteen guns), the muzzle speed and the shield are the steepest
values (about 1,400 and 1,200 Elo a doubling), and the dispersion and the recharge are nearly
free (feel, not balance). So the order is: the guns-and-armour cycle first (the thresholds
of the shield against the damage per hit, with the turn and the target size, are what the
balancing's Mosquito-Zerstörer cycle is made of), the speed-and-reach cycle second (the muzzle
speed's steepness against the cruise), and the flight cycle once per-type doctrine exists.

`check` plays the chosen candidate at 1,024 games a pair (a round robin) and the
mixed-team test, in deathmatch and then in Frontline.

### The first search: the seed cycle

The balancing's round-2 cycle (Mosquito over Zerstörer over the light fighters over the
Mosquito, [Case studies](case-studies.md)) as three designs on the paced set, each on its own
base, searching the shield, the damage per second, the fire rate and the turn
(`ship-design/stage-1/seed.json`): Heron on the Mosquito, Bastion on the Zerstörer, Kestrel on
the Spitfire IX. 30 generations of 12 candidates, 64 games an edge, 33 s a generation
(19 minutes).

At the start, on the paced set, the round-2 cycle was gone (the mean's edges 0.45, 0.69 and
0.35 the way the search chose round: Bastion over Heron, Kestrel over Bastion, Heron over
Kestrel). The search's best mean (generation 28):

| Design | Shield | Fire rate | Damage a second | Turn deg/s | Realism |
|---|---|---|---|---|---|
| Heron (Mosquito: 110, 10 Hz, 369, 15.0) | 109 | 10 | 379 | 15.3 | 3.11 |
| Bastion (Zerstörer: 120, 10 Hz, 371, 11.8) | 117 | 10 | 410 | 13.0 | 3.39 |
| Kestrel (Spitfire IX: 100, 10 Hz, 361, 19.5) | 89 | 15 | 388 | 17.4 | 2.52 |

Checked at 1,024 games a pair (instant cards): Bastion over Heron 0.519 ± 0.016, Kestrel over
Bastion 0.639 ± 0.015, Heron over Kestrel 0.503 ± 0.016. The ratings are within 32 Elo of
their mean (the equal-rating target met) but only one edge is in the band: the search found
a balanced triple, not a cycle. Kestrel is the best pick against the uniform field (it scores
0.545 against it: exploitability 0.045).

What it says about the method:

- **64 games an edge is too few for a band 0.1 wide.** An edge's standard error at 64 games is
  0.06; the mean's edges wandered between 0.34 and 0.72 from generation to generation while
  its knobs moved little. The search now plays `games_late` (256) once its step size has
  halved.
- **Equalising is easy, cycling is not.** Every move that evens the three (Bastion's damage,
  Kestrel's shield and turn) also shrinks what made the round-2 cycle: the Zerstörer's shield
  over the light guns' threshold went from 130 to 117 and the light type's damage per hit fell
  with 15 Hz, so the thresholds no longer split them. The loss now has a curl term (`curl_weight`): the
  triangle's curl (the sum of its three log-odds, which no rating difference can make) must
  reach three times the band's lower log-odds, so that the search cannot buy equal ratings
  with the cycle.

**The exact cards** (`export`: each design solved, every handle met within 4e-5; the ship set
`design-seed`), 1,024 games a pair, `ship-design/stage-1/seed/`:

| Edge | Deathmatch | Frontline |
|---|---|---|
| Bastion over Heron | 0.527 ± 0.016 | 0.248 ± 0.013 |
| Kestrel over Bastion | 0.555 ± 0.016 | 0.645 ± 0.015 |
| Heron over Kestrel | 0.559 ± 0.016 | 0.645 ± 0.015 |

In deathmatch the exact cards go round, weakly: all three edges the cycle's way, two of them
beyond three errors; the ratings are within 7 Elo of their mean, the uniform pick's
exploitability is 0.011 and the Nash equilibrium has full support (0.39, 0.42, 0.19). Every
balance target is met; the band is not (a curl of 0.57 against the 1.22 a 0.60 cycle needs).
The instant cards had put Kestrel over Bastion at 0.639; the exact solve moved it to 0.555,
the estimate's error on a 10% turn change being worth 60 Elo here, so a result is accepted
only on exact cards. In Frontline the cycle does not hold: Heron (the Mosquito's airframe)
beats both others (0.75 and 0.64), the ratings spread 99 Elo.

**The continuation** (`stage-1/seed-2.json`: from the best mean, the exact cards' direction,
the curl term, 128 games an edge, a smaller step; 24 generations, 23 minutes) did not find
more: its best mean (generation 21) checked at 1,024 games a pair (instant cards) gives
0.535, 0.537 and 0.485 round the cycle, ratings within 12 Elo, its realism rising to 3.8 as it
pushed the shield and the turn. Within the envelope, the shield, the damage, the fire rate and
the turn give these three a curl of about 0.5 at equal ratings; a cycle in the band needs
about twice that, so it needs a second mechanism on top (the gun-on-airframe dependence of
the case studies, or the energy fight once the doctrine is per type).

**Does it survive the agent?** (`robust`, decision 6's test: each design flies its own random
perturbation of the agent's set, a different draw per design, six draws at each size, 256
games a pair; `stage-1/seed/robust-real_scale_deathmatch.jsonl`):

| Perturbation | Edges (Bastion > Heron, Kestrel > Bastion, Heron > Kestrel), per draw | Curl per draw | All three the cycle's way |
|---|---|---|---|
| sigma 0.1 | 0.55 0.50 0.57; 0.55 0.38 0.59; 0.61 0.47 0.57; 0.59 0.47 0.59; 0.56 0.64 0.48; 0.42 0.65 0.52 | 0.50, 0.08, 0.58, 0.63, 0.75, 0.37 | 1 of 6 |
| sigma 0.25 | 0.51 0.49 0.57; 0.74 0.10 0.84; 0.71 0.47 0.46; 0.50 0.54 0.55; 0.57 0.65 0.57; 0.30 0.82 0.45 | 0.29, 0.51, 0.66, 0.34, 1.22, 0.51 | 1 of 6 |

(an edge at 256 games has a standard error of 0.03). The curl keeps its sign in every draw
(0.08 to 1.22, about its unperturbed 0.57): the cyclic part belongs to the ships. Which edge
carries it does not: a tenth of the search's own step in each type's doctrine moves single
edges by up to 0.15, a quarter by up to 0.45, so a type's doctrine weighs more than its
design. A cycle at the band's 0.6 to 0.7 is about twice this curl; a cycle that survives the
agent needs that, or doctrine held fixed per type as part of the design.

## Running it

The runbook for an agent running these steps is
[ship-design/runbook.md](ship-design/runbook.md).
