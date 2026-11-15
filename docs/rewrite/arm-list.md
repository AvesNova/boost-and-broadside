# Final arm list

One line is one training run. The author chooses which runs go in. Each challenger is compared with the incumbent of its time: the champion's configuration when the arm starts, so a losing change never carries into later arms. The entries are described, with their reasons and paper questions, in [arm-review.md](arm-review.md), and the costs come from [sections/arm-costs/numbers.md](sections/arm-costs/numbers.md).

**For every run:**
- **Game:** the frozen game (version 21, or 22 if the freeze check raises it).
- **Steps:** BC 100M (shared by RL forks), then RL 100M.
- **Defaults:** fixed maps, fixed 5v5, the split critic at 1:1, cheese agents in the league, mode mix 6/5/5, and the league snapshot schedule (every 5M to 50M, then doubling).

Costs are 5090-hours at $0.50 an hour, including the run's ladder and its decision.

**Classes:**
- **BC+RL:** its own BC, then RL.
- **RL fork:** RL only, from the shared BC of the arm named under "starts from".

## Required (decided)

| # | Run | Change from the defaults | Class | Starts from | Seed | Cost |
|---|---|---|---|---|---|---|
| 1 | R0 baseline | none; per-update policy files on (for the averaging suite) and the full ladder in every condition | BC+RL | — | 0 | 5.6 h ($2.80) |
| 2 | R0 baseline, second seed (A3) | as run 1, at seed 1 | BC+RL | — | 1 | 3.8 h ($1.90) |
| 3 | A1 variable maps | maps drawn at 0.5–2× scale (`arena_scale`) in BC data and RL | BC+RL | — | 0 | 3.9 h ($1.95) |
| 4 | A1 variable maps, second seed (A3) | as run 3, at seed 1 | BC+RL | — | 1 | 3.8 h ($1.90) |

Run 1 is the dry run's mainline run, and it counts as an arm if it's clean. Runs 1–4 are the box's first jobs; they measure seed noise for the sequential test's indifference zone. Subtotal: 17.1 h ($8.55).

## Candidates (not yet chosen)

| # | Run | Change from the incumbent | Class | Starts from | Cost |
|---|---|---|---|---|---|
| 5 | A2 variable counts | team sizes drawn per game (absent ships and fields masked); deathmatch and elimination down to 1v1, frontline to 3v3 | BC+RL | — | 3.7 h ($1.85) |
| 6 | A4 minibatches | 256 PPO minibatches per epoch instead of 32 | RL fork | incumbent's BC | 2.2 h ($1.10) |
| 7 | A5 training mix | modes 8/4/4 instead of 6/5/5 | RL fork | incumbent's BC | 2.0 h ($1.00) |
| 8 | A6 self-play one side | each self-play game trains one seat, with more environments and "latest" as a league member | RL fork | incumbent's BC | 2.1 h ($1.05) |
| 9 | A7 premium 2:1 | offence 2:1 over defence on damage and kill/death | RL fork | incumbent's BC | 2.0 h ($1.00) |
| 10 | A7 premium 2:1 annealed | 2:1 annealed slowly to 1:1 | RL fork | incumbent's BC | 2.0 h ($1.00) |
| 11 | A8 DPPO | DPPO's divergence mask instead of ratio clipping | RL fork | incumbent's BC | 2.0 h ($1.00) |
| 12 | A9 learning rate ×2 | twice the learning rate | RL fork | incumbent's BC | 2.0 h ($1.00) |
| 13 | A10 entropy | a stronger bonus or an entropy floor (one variant, to choose) | RL fork | incumbent's BC | 2.0 h ($1.00) |
| 14 | A11 steps per update | 4096 environments, or a 256-tick rollout (one variant, to choose) | RL fork | incumbent's BC | 2.0 h ($1.00) |
| 15 | A12 dense rewards decayed | damage and kill/death rewards fade to zero partway through RL | RL fork | incumbent's BC | 2.0 h ($1.00) |
| 16 | A13 no auxiliary heads | every auxiliary head removed | BC+RL | — | 3.1 h ($1.55) |
| 17 | A14 belief input | the temporal layers alone, without the next-step heads' outputs for hidden ships | BC+RL | — | 3.7 h ($1.85) |
| 18 | A15 layout | 6 spatial blocks instead of 2 Yemong blocks | BC+RL | — | 3.4 h ($1.70) |
| 19 | A16 temporal block | GRU instead of Griffin | BC+RL | — | 3.7 h ($1.85) |
| 20 | A17 input encoding | raw scalars instead of the colour/HL-Gauss codes | BC+RL | — | 3.7 h ($1.85) |
| 21 | A18 rotary only | rotary position encoding without the colour code | BC+RL | — | 3.7 h ($1.85) |
| 22 | A18 colour code only | the colour code without rotary | BC+RL | — | 3.7 h ($1.85) |
| 23 | A19 gated attention | Qwen elementwise output gate | BC+RL | — | 3.7 h ($1.85) |
| 24 | A20 attention heads | 4 heads of 32 instead of 2 of 64 | BC+RL | — | 3.7 h ($1.85) |
| 25 | A21 value output | symlog scalar instead of two-hot | BC+RL | — | 3.7 h ($1.85) |
| 26 | A22 shot head | "if I shot now" damage head instead of the full trajectory | BC+RL | — | 3.7 h ($1.85) |
| 27 | A23 width 192 | width 192 instead of 128, at equal steps; the equal-compute comparison and a learning-rate re-check are rows 28–29 | BC+RL | — | about 6.7 h ($3.35) |
| 28 | A23 width 128 extended | the incumbent continued to equal compute (about 270M RL) | RL continuation | incumbent | about 2.5 h ($1.25) |
| 29 | A23 learning-rate re-check | width 192 at the re-checked learning rate | RL fork | row 27's BC | about 2.5 h ($1.25) |
| 30 | A24 RL from scratch | no BC; RL for 300M | RL only | — | 5.1 h ($2.55) |
| 31 | R1 MLP baseline | an MLP instead of Yemong | BC+RL | — | 1.8 h ($0.90) |
| 32 | R2 final, second seed | the final configuration at the publication length (100M BC, 500M RL) | BC+RL | — | 10 h ($5.00) at width 128 |
| 33 | R2 final, third seed | as row 32 | BC+RL | — | 10 h ($5.00) at width 128 |
| 34 | R3 exploiter | trained only against the frozen final model | RL fork | final's BC | 1.6 h ($0.80) |
| 35 | R4 frontline specialist | frontline only | RL fork | final's BC | 1.8 h ($0.90) |
| 36 | R4 deathmatch specialist | deathmatch only | RL fork | final's BC | 1.8 h ($0.90) |
| 37 | R4 elimination specialist | elimination only | RL fork | final's BC | 1.8 h ($0.90) |
| 38 | R6 elimination-only fork | elimination only, for the mode-transfer measurement | RL fork | R0's BC | 1.8 h ($0.90) |
| 39 | R6 frontline-only fork | frontline only, for the mode-transfer measurement | RL fork | R0's BC | 1.8 h ($0.90) |

Rows 27–29 are split from A23's 11.7 h total in the arm review, and their split is approximate. R5, the averaging suite, needs no training: it builds averages from run 1's per-update files and rates them (0.5–2 h).

**Totals:**
- required: 17.1 h ($8.55);
- every candidate: about 113 h ($56.50);
- the 500M publication run is separate: 10–17 h.
