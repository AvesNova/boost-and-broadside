# Decision rule for the pairwise arms (draft for the author)

How each arm in [arm-list.md](arm-list.md) is decided, and exactly what is measured at each step. Most of it is already decided and built; the evaluation suite ([evaluation-suite.md](evaluation-suite.md), `bnb evaluate`) is its reference. Items marked **(proposed)** need the author's approval. Every number marked **(provisional)** is replaced by a measurement from the first runs, at the point named.

## 1. What a pair is

- **The two arms:** the challenger and the incumbent, the champion's configuration when the challenger starts. They differ only in the arm's setting; every other setting is the champion's ("build on the winner").
- **Same seeds and streams:** both arms use the same seed and share every keyed random stream that they use the same way (seed isolation). Their games and rollouts start identical, and any difference comes from the setting.
- **Same length:** 100M RL environment steps each, after 100M BC. An RL fork starts from the incumbent's BC; an arm that changes BC trains its own. The incumbent's run is reused, never retrained, so a pair costs one new run.
- **Exceptions that are about steps:**
  - A24 (from scratch) trains for 300M;
  - A23 (width) is decided at equal steps and again at equal compute.
- **Several challengers against one incumbent** (A7's two premiums): each is decided against the incumbent. If more than one wins, the winners play each other with the same rule, and the winner of that becomes the champion.

## 2. The players that decide

- **Each arm's final checkpoint:** the exponential-average weights at exactly 100M RL steps, the weights we ship. The live weights don't decide.
- **Play:**
  - actions sampled at temperature 1, never greedy;
  - fog of war on;
  - each slot a 50/50 coin flip between Yongma and Jujak, the same for both teams;
  - seats alternating game by game;
  - each preset's fixed map.

## 3. The measurement: the evaluation suite

**Conditions (13):**
- 5v5 in all three modes;
- 1v1 and 2v2 in deathmatch and elimination;
- 10v10 in all three modes;
- break-even against 20 Imugi in all three modes.

**Per condition, a number in [−0.5, 0.5]:**
- **Head-to-head conditions:** A's mean score against B (win 1, draw ½, loss 0), minus 0.5. Its standard error is the games' sample standard deviation over √games.
- **Break-even:** each arm's win probability against Imugi as a logistic curve in log(its ships / Imugi's ships), with the weaker side at 20. A's score is its win probability in B's break-even lineup, minus 0.5, averaged with the mirror for B. The standard error comes from the two fits by the delta method.

**Overall score:** the three modes weighted a third each, and within a mode its conditions equally:
- frontline: 5v5, 10v10 and break-even;
- deathmatch: 5v5, 1v1, 2v2, 10v10 and break-even;
- elimination: 5v5, 1v1, 2v2, 10v10 and break-even.

The weights never change with the games played, so the score stays unbiased however games are allocated. The standard error is the square root of the weighted sum of the conditions' variances.

**Allocation:** every condition plays one batch first. After that, batches go where they most reduce the overall variance per engine-second, until the next look's information is reached.

## 4. The test

**A group-sequential test on the overall score:**
- **At each look:** Z = score / SE, and the information fraction t = SE⁻² / I_max.
- **Planned looks:** K = 4 at t = ¼, ½, ¾ and 1.
- **Error rates:** α = 0.05 two-sided, spent by an O'Brien–Fleming-type function. β = 0.2 under a true difference of ±δ.
- **Stopping:** at a look, A wins when Z ≥ u_k, B wins when Z ≤ −u_k, and the test stops with a tie when |Z| ≤ l_k. The last look always decides.
- **δ, the indifference zone (provisional 0.02,** about 52% against 50%, or 14 Elo). It is set after the seed pairs (§7).
- **Budget:** a cap in engine-hours; the last look comes when the budget is spent. **(provisional:** 2 hours on the 5090 per decision, about three times the pilot's expected cost.)

## 5. The no-regression gate

- **Rule:** a winner whose score in any one mode is at least 2 standard errors below zero doesn't win; the decision becomes a tie.
- **Measured:** each mode's score (the mean of its conditions) and its standard error from the same games as the overall score.
- **A 2-against-1 split between modes** is therefore a tie. If the tie rule still leaves it split, the incumbent stays.

## 6. Ties, and the late-climbing check (proposed)

**When it applies:** after every decision, not only after ties, if the losing arm (or either arm, after a tie) may still be climbing. This is the proposed extension of the decided tie rule.

**Measured:**
1. **The curve:** each arm's ladder snapshots from 50M to 100M RL (every 5M: 11 snapshots, exponential-average weights), rated at 5v5 in all three modes in the arm's shared pool, to about ±45 Elo each.
2. **The overall Elo per snapshot:** the mean of its three mode Elos, which mirrors the score's equal mode weights.
3. **The fit:** per arm, overall Elo = a + b · ln(steps) by weighted least squares, with the ratings' covariance. The slope b is in Elo per e-fold of steps; its standard error comes from the fit.
4. **The test:** the slope difference b_loser − b_winner and its standard error (for a tie, take each arm in turn as the "loser").

**What it decides:**
- **Extend:** if the loser's slope exceeds the winner's by more than 2 standard errors, both arms continue to 150M RL (50M more each, about 1.3 hours a run on the 5090), and the whole test (§3–5) runs again on the 150M finals. There is at most one extension per pair, and the 150M result stands.
- **Otherwise:** a win stands. A tie goes to the cheaper or simpler arm, which is normally the incumbent.

**Why log steps and the last 50M:** rl-v20's curve is close to linear in log(steps) from 25M to 90M (see [sections/league-schedule/results.md](sections/league-schedule/results.md)). Eleven points over the last 50M give a slope standard error about 3–4 times smaller than the five-snapshot version in the earlier tie rule.

## 7. Seed noise: setting δ (arm A3)

- **The pairs:** R0 at seeds 0 and 1, and the first arm that trains at seeds 0 and 1. Each pair plays the full suite against itself: seed 0 against seed 1 of the same configuration.
- **Measured:** each seed pair's overall score and standard error (§3).
- **δ is set to:** the root mean square of the two seed pairs' overall scores, floored at 0.02. A smaller true difference than seed noise isn't something a single-seed pair can decide, so differences inside it are treated as ties. **(proposed)**
- **The seed result is also a check:** a seed pair should show no significant mode-level differences under the gate (§5). If it does, the gate's 2 standard errors are too strict and get re-examined.

## 8. Outcomes

- **Challenger wins:** its setting becomes the default for every later arm, and its run is the new incumbent.
- **Incumbent wins, or a tie stands:** nothing changes. The challenger's run is kept for the paper's curves.
- **The arm fails to train** (its curve never rises above BC's level, or the run diverges): recorded as a failure with its curve. It gets no second seed and no extension.

## 9. What is recorded for every pair

- **The evaluation directory:**
  - settings;
  - every batch, with full outcomes per block, lineup and side counts;
  - every look, with score, SE, Z, the boundaries, the gate and the decision.
- **The late-climbing check's inputs and output:** snapshots, ratings, fits and the slope difference.
- **The ladder:**
  - every snapshot rated at 5v5 for every arm;
  - every condition for R0 and the final champion.
- **In [arm-list.md](arm-list.md):** the decision and its date.

## 10. Still open

- **(proposed)** §6, the late-climbing check after every decision, with its one 50M extension.
- **(proposed)** §7, the rule that sets δ from the seed pairs.
- **(provisional)** the 2-hour budget per decision, until the dry run times evaluation on the 5090.
