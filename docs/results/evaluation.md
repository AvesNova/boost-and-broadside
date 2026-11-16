---
covers: [src/bnb/league/crossover.py, src/bnb/league/evaluation.py, src/bnb/league/sequential.py, src/bnb/league/rating_pool.py]
---

# How it is measured

Every result on these pages comes from games played after training, between fixed
weights, on the frozen game. This page says how each kind of number is produced, and what
it does not show. The caveats of every result page are collected here.

## The opponent: Imugi

Imugi is a scripted agent with one parameter set per mode
([Imugi](../explanation/imugi.md)). It reads only the observation, the same one the network
reads, fog of war included, and nothing of the game's hidden state. It is the teacher of
behaviour cloning and the fixed point of every rating: Imugi is 1000 Elo in every mode.

## The crossover

`bnb crossover` (`bnb.league.crossover`) plays the policy against Imugi at many lineups
and finds the ratio of ships at which each side wins half the games
([the break-even lineup](../explanation/league.md#published-ratings)):

- **Lineups.** The weaker side fields 20 ships and the stronger side fewer; unused slots
  are benched, dead and out of play. Every game draws its lineup of ship types, each slot
  Yongma or Jujak with equal chance, both teams flying it slot for slot.
- **Fit.** One logistic regression of the policy's score (a win 1, a draw one half, a loss
  0) on the log ratio of ships, over every game. The break-even ratio is where the fitted
  curve crosses one half; its standard error comes from the fit's covariance.
- **Allocation and stop.** Games go to the lineups that most reduce the variance of the
  break-even estimate, and a mode stops at a target standard error of the log ratio or a
  budget of games.
- **The fan.** A line of the fan chart at 5 against 8 is the same fit read at the ratio
  5:8, at each training snapshot; the games played at that ratio are drawn as open points.

## Elo

`bnb pool` (`bnb.league.rating_pool`) rates every checkpoint of every run against Imugi,
per game preset, from all the games stored in one shared pool: one Bradley–Terry rating
per player, with Imugi fixed at 1000. 400 points is ten-to-one odds. By default a deathmatch
game is scored by who leads in kills at its end: a win, a draw or a loss.

## Arm decisions

`bnb evaluate` (`bnb.league.evaluation`) decides between two policies, normally an arm's
final checkpoint and the incumbent's, over a fixed set of conditions: head-to-head games in
each mode at several team sizes, and each policy's break-even curve against Imugi. A
group-sequential test (`bnb.league.sequential`) on the overall score stops at a planned
look with one side the winner or a tie, and a gate holds a winner that is clearly worse in
one mode to a tie. The number of looks, the thresholds and the indifference zone are fixed
before the first arm; the zone's width comes from pairs of runs that differ only in their
seed ([The decision suite](../explanation/league.md#the-decision-suite)).

## Caveats

- **One opponent.** The crossover and the Elo measure strength against Imugi and against
  the policy's own lineage. A policy can beat Imugi by a wide margin and still lose to a
  strategy Imugi never plays; the league's cheese agents cover some such strategies, not
  all.
- **The crossover is played at 20 ships a side.** The break-even games put 20 ships on the
  weaker side, a larger fleet than the training games. The ratio is a property of the
  policy at that fleet size, and the fit assumes the score depends on the ratio of ships
  alone.
- **The fan reads the fit.** A line at 5 against 8 is the logistic fit read at that ratio,
  not a tally of games at exactly 5 against 8. Where the outcomes are separated (every
  game at one side of a ratio won, every game at the other lost) the fit has no crossing,
  and the snapshot has no point on the line.
- **Draws.** The binomial standard errors treat a draw as half a win, which slightly
  overstates the uncertainty.
- **Seeds.** Most arms ran at one seed. The seed-to-seed spread, measured on two pairs of
  runs, is the scale against which an arm's difference is read.
- **The frozen game.** Every number is for one game version. A change to the game's rules,
  its ships or Imugi makes a new game, and the numbers do not carry over.

{{pending: evaluation_caveats | Any caveat the results themselves raise (for example a mode whose games mostly run the clock out, or a suspicious break-even flagged by the crossover), written after seeing them.}}
