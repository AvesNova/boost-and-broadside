---
covers: [src/bnb/league/cyclic.py, bench/scripted_search.py, bench/scripted_compare.py]
---

# Balancing the real-scale set

The [ship-type tournament](real-scale-tournament.md) rated the 27 real-scale types over a
2,900-point spread, and the guns explained most of it: the weight of fire and the shots a
type needs to kill a P-51D. This page balances the set in rounds, each of four steps:

1. tune the ships;
2. search the scripted agent's real-scale sets again, each mode against its current
   champion, on mixed lineups of the set's types;
3. rate the types in a tournament;
4. find which values explain what is left of the spread, and move one if it is loosely
   known.

The rounds stop when the ratings follow well-documented values, or when a change is within
the noise. Each round's data are in `real-scale-balance/`
(`round-N/`).

## The rules the rounds follow

**Equal damage per second.** Every type deals about the P-51D's 360 per second
(`bnb.game.real_scale.balanced_damage_per_second`):

    damage per second = 360 x (w / 3.68 kg/s)^0.05, kept within 5% of 360,

w being the type's weight of fire (for the heavy placeholder, the weight of fire its
placeholder damage would mean). The exponent keeps a trace of the real ordering: twice the
P-51D's weight of fire gives 3.5% more damage per second, and the band stops the Me 262's
13.6 kg/s at +5% (378) and the Wren's and Lantern's 1.36 kg/s at -5% (342). Each type keeps
its real fire rate, so a slower-firing type hits harder a shot: the Anvil's 2 Hz cannon
177 a shot, the 10 Hz cannon types about 36, the 15 Hz .50 batteries about 24. The shot cost
is unchanged (continuous fire costs every type the same multiple of its own climb charge).

**Leeway by confidence.** A value the research holds well (a "good" or "medium" datum in
`bnb.game.real_scale_aircraft`) stays where it is. A loosely known one (low, estimated,
invented or a placeholder) may move within its plausible range when a round's analysis
points at it. Every move is a `BALANCE` entry in `bnb.game.real_scale`, and the ship file's
note for the handle records the old value, the new and the reason; the equal-damage rule's
note does the same for the hit damage.

**The agent's searches.** `bench/scripted_search.py` searches each mode's set
(real_scale_deathmatch, real_scale_frontline) to win that mode, against that mode's current
champion (`--champion`), on mixed lineups (`--mixed`): every game draws each seat's type
from the set, and both teams fly the same draw seat for seat, so neither side is favoured
by its ships. Four types are held out of the draws (Spitfire XIV, Ki-84, Zerstörer,
Lantern) to check that a set generalises. Half of each candidate's 64 games a generation
are against the champion, the rest against other candidates; candidates play at a
temperature drawn per game from 1, 1.5, 2 and 3, the champion at 1. The ratings are
anchored on the champion. A champion that every candidate beats would hold the ratings
by the prior (the light retune's plateau rule stopped early for that reason), so when the
median candidate scores 0.75 against the champion, the generation's top candidate becomes
the champion and the plateau history starts again. The search stops when the median
rating has not risen 10 points in 5 generations, or at its generation bound.

A round's new champion is the best-rated survivor of the last generation (one of the
eight kept from the generation before, so rated on two sets of fresh games), checked
against the previous champion by `bench/scripted_compare.py`: 1,024 fresh games at T = 1
on mixed lineups of every type, and 512 on mixed lineups of the held-out types alone.

**The tournament.** `bench/ship_type_tournament.py play --allocation a_optimal`: the
league's tournament as designed, each batch's pairs chosen by A-optimal value, until every
type's standard error (against the P-51D, the anchor at 1000) is under 25 Elo. Twenty-five
is the league's own target: a quarter of the 100-point gap the rounds care about, and
within reach in about ten minutes.

**Rock, paper, scissors.** One Elo a type assumes the results are transitive. Three checks
say how far they are not (`bench/ship_type_tournament.py cyclic`, `bnb.league.cyclic`):

- the cycles by majority: the triples where i beats j, j beats k and k beats i, each in
  more than half their games (who wins, not by how much);
- the pairs whose observed score is more than three standard errors off a fit's
  prediction;
- held-out likelihood: each pair's games are dealt into five folds, and each model is
  fitted on four and scored on the fifth.

The models, from fewest assumptions added: Bradley–Terry (the logit link); the same
ratings with another link from rating gap to win probability, Thurstone's probit (lighter
tails) or a power link, sigmoid of about sign(z) |z|^gamma with gamma fitted (it stretches or
squeezes the large margins, so a margin that one Elo gets wrong without any reversal is
fitted without a cycle); and the multidimensional Elo mElo-2k (Balduzzi et al. 2018,
"Re-evaluating Evaluation"), which adds to each type a 2k-vector c, so the log-odds of i
beating j become r_i - r_j + c_i' W c_j, W the block-diagonal of k quarter-turn rotations:
k cyclic games on top of the rating, its vectors under an L2 penalty chosen with k by the
same folds. A cyclic part is kept only when it predicts held-out games better than the best
link without one, by more than twice the difference's standard error. The page then gives
both parts: the transitive rating, and each type's cyclic strength (the length of its
vector on the Elo scale; two types' cyclic term is the product of their strengths and the
sine of the angle between them).

"Cyclic" here is in the sense of the decomposition (a part of the log-odds no rating
difference can make), not necessarily a reversal of who wins: a pair whose margin is too
wide or too narrow for its two ratings, with the same winner, is cyclic in this sense when
no link explains it.

On the light retune's round robin (before any balancing, 44,928 games over all 351
pairs): one triple goes round by majority (Fw 190D over Yak-3 0.84, Yak-3 over Me 163 0.62,
Me 163 over Fw 190D 0.53). Bradley–Terry leaves 70 pairs more than three standard errors
off. Held out, the power link (gamma 0.61: the large margins squeezed) predicts 0.0047 nats
a game better than Bradley–Terry (standard error 0.0005), the probit 0.0058 worse; mElo
with k = 1 0.0166 better and k = 3 0.0210 (0.0010), leaving 6 pairs off. So the deviations
are in the margins, not in who wins, and the link explains about a quarter of them: the
rest is a margin pattern no single rating per type reproduces. Beside the 2,900-point
spread it is small: the cyclic part's root-mean-square term is 200 Elo against the
transitive part's 940 (4% of the summed squares).

