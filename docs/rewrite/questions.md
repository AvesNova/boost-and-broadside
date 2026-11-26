# Questions for the author (branch scripted-lookahead, overnight 2026-11-27)

Raised by the round-3 agent; each with the decision-safe choice made meanwhile.

1. **The cold compile is still over the 260 s budget.** The steady-flight table removed every
   flight-function evaluation from the agent's graph (`act` 7,864 -> 4,494 traced operations;
   cold 333 -> 312 s with another agent's CPU work running). What is left is the unrolled
   segment loop: per segment the medium (5 x about 90 operations), the nose spring in closed
   form (5 x 60), the lookup (8 x 40), plus the path arithmetic. Options: (a) the paths as one
   Triton kernel, like the pair stage (the speed order's last step; the largest cut, a day's
   work with its twin and test); (b) fewer path segments (4 instead of 5); (c) a larger budget
   for the agent's graph. Meanwhile: nothing changed beyond the table; the baseline is not
   updated.
2. **How much should the store be worth (`store_weight`)?** 0.25 is set (store mean 0.44-0.51,
   dive 23-25%, climb 22-28% of ship-ticks; beats round 2 68% in 1v1, 57% in 4v4). In 1v1 duels
   more keeps winning (0.4 beats 0.25 58%, 0.6 beats 0.4 62%) while 4v4 is even and the dive
   goes out of use (9% at 0.4, 1-5% at 0.6) with fewer kills per minute. Which matters more for
   the yardstick: 1v1 strength, or balanced use of the powers and a livelier 4v4? Progress.md
   round 3 step 3 has the tables.
3. **Weakening the preferred-speed term** (your suggestion) was measured: `speed_weight` 0.15
   with store weight 0.25 makes ships slow (0.61-0.68 corner) and loses to round 2 (44% 1v1,
   47% 4v4). Kept at 0.3.
