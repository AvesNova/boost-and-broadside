# Final arm list

One line is one training run. Each challenger is compared with the incumbent of its time: the champion's configuration when the arm starts, so a losing change never carries into later arms. The entries are described, with their reasons and paper questions, in [arm-review.md](arm-review.md), and the costs come from [sections/arm-costs/numbers.md](sections/arm-costs/numbers.md).

**For every run:**
- **Game:** the frozen game (version 21, or 22 if the freeze check raises it).
- **Steps:** BC 100M (shared by RL forks), then RL 100M.
- **Defaults:** fixed maps, fixed 5v5, the split critic at 1:1, cheese agents in the league, mode mix 6/5/5, and the league snapshot schedule (every 5M to 50M, then doubling).

Costs are 5090-hours at $0.50 an hour, including the run's ladder and its decision.

**Classes:**
- **BC+RL:** its own BC, then RL.
- **RL fork:** RL only, from the shared BC of the arm named under "starts from".

**Seeds (author, 2026-11-15):** R0 runs at two seeds. So does the first arm that trains, rather than A1 by default: A1 and A2 may fail to train at all, and a failed arm gets no second seed. The two pairs measure seed noise for the sequential test's indifference zone.

## Yes

| # | Run | Change from the incumbent | Class | Starts from | Seed | Cost |
|---|---|---|---|---|---|---|
| 1 | R0 baseline | none; per-update policy files on (for R5) and the full ladder in every condition. The dry run's mainline run, and an arm if it's clean | BC+RL | — | 0 | 5.6 h ($2.80) |
| 2 | R0 baseline, second seed | as run 1 | BC+RL | — | 1 | 3.8 h ($1.90) |
| 3 | A1 variable maps | maps drawn at 0.5–2× scale (`arena_scale`) in BC data and RL | BC+RL | — | 0 | 3.9 h ($1.95) |
| 4 | A2 variable counts | team sizes drawn per game (absent ships and fields masked); deathmatch and elimination down to 1v1, frontline to 3v3 | BC+RL | — | 0 | 3.7 h ($1.85) |
| 5 | Second seed of the first arm that trains | the first of runs 3, 4, 6… that trains, at seed 1 | as that arm | as that arm | 1 | about 3.8 h ($1.90) |
| 6 | A7 premium 2:1 | offence 2:1 over defence on damage and kill/death | RL fork | incumbent's BC | 0 | 2.0 h ($1.00) |
| 7 | A7 premium 2:1 annealed | 2:1 annealed slowly to 1:1 | RL fork | incumbent's BC | 0 | 2.0 h ($1.00) |
| 8 | A14 belief input | the temporal layers alone, without the next-step heads' outputs for hidden ships | BC+RL | — | 0 | 3.7 h ($1.85) |
| 9 | A13 fewer auxiliary heads | after A14: if the belief helps, every auxiliary head except next-state is removed; otherwise every one | BC+RL | — | 0 | 3.1 h ($1.55) |
| 10 | A15 layout | 6 spatial blocks instead of 2 Yemong blocks | BC+RL | — | 0 | 3.4 h ($1.70) |
| 11 | A16 temporal block | GRU instead of Griffin (built on `feat/gru-temporal`) | BC+RL | — | 0 | about 4.8 h ($2.40), provisional: the GRU sublayer is 8.6x Griffin's at the learner's 8-view micro-batch |
| 12 | A17 input encoding | raw scalars instead of the colour/HL-Gauss codes (`yemong.input_encoding=raw`) | BC+RL | — | 0 | 3.7 h ($1.85) |
| 13 | A24 RL from scratch | no BC; RL for 300M | RL only | — | 0 | 5.1 h ($2.55) |
| 14 | R1 MLP baseline | an MLP instead of Yemong | BC+RL | — | 0 | 1.8 h ($0.90) |

**R5, the averaging suite,** needs no training: it builds averages from run 1's per-update files and rates them (0.5–2 h, $0.25–1.00).

**Total for Yes:** about 52 h ($26) with R5, plus about 4 h ($2) for ties and checks.

## Maybe

| # | Run | Change from the incumbent | Class | Starts from | Cost |
|---|---|---|---|---|---|
| 15 | A4 minibatches | 256 PPO minibatches per epoch instead of 32 | RL fork | incumbent's BC | 2.2 h ($1.10) |
| 16 | A5 training mix | modes 8/4/4 instead of 6/5/5 | RL fork | incumbent's BC | 2.0 h ($1.00) |
| 17 | A6 self-play one side | each self-play game trains one seat, with more environments and "latest" as a league member | RL fork | incumbent's BC | 2.1 h ($1.05) |
| 18 | A8 DPPO | DPPO's divergence mask instead of ratio clipping | RL fork | incumbent's BC | 2.0 h ($1.00) |
| 19 | A9 learning rate ×2 | twice the learning rate | RL fork | incumbent's BC | 2.0 h ($1.00) |
| 20 | A10 entropy | a stronger bonus or an entropy floor (one variant, to choose) | RL fork | incumbent's BC | 2.0 h ($1.00) |
| 21 | A19 gated attention | Qwen elementwise output gate | BC+RL | — | 3.7 h ($1.85) |
| 22 | A22 shot head | "if I shot now" damage head instead of the full trajectory (`yemong.shot_head=damage`) | BC+RL | — | 3.7 h ($1.85) |
| 23 | A23 width 192 | width 192 instead of 128, at equal steps | BC+RL | — | about 6.7 h ($3.35) |
| 24 | A23 width 128 extended | the incumbent continued to equal compute (about 270M RL) | RL continuation | incumbent | about 2.5 h ($1.25) |
| 25 | A23 learning-rate re-check | width 192 at the re-checked learning rate | RL fork | row 23's BC | about 2.5 h ($1.25) |
| 26 | R3 exploiter | trained only against the frozen final model | RL fork | final's BC | 1.6 h ($0.80) |
| 27 | R4 frontline specialist | frontline only | RL fork | final's BC | 1.8 h ($0.90) |
| 28 | R4 deathmatch specialist | deathmatch only | RL fork | final's BC | 1.8 h ($0.90) |
| 29 | R4 elimination specialist | elimination only | RL fork | final's BC | 1.8 h ($0.90) |

Rows 23–25 split A23's 11.7 h total in the arm review, approximately. **Total for Maybe:** about 38 h ($19).

## No (author, 2026-11-15)

A11 steps per update, A12 dense rewards decayed, A18 position encoding, A20 attention heads, A21 value output, R2 final-model seeds, R6 single-mode forks for mode transfer.

**Not counted:** the 500M publication run, 10–17 h.

## Readiness: settings lines

Built behind a setting, each arm's change from a new run's build (defaults unchanged):

- **A17 input encoding** (built 2026-11-16, `feat/a17-a22-options`): `--set yemong.input_encoding=raw`. Raw scalars in place of every colour and HL-Gauss input code; targets unchanged ([yemong.md](../explanation/yemong.md#input-encoding-codes-or-raw)).
- **A22 shot head** (built 2026-11-16, `feat/a17-a22-options`): `--set yemong.shot_head=damage`. The "if I shot now" damage head (each own ship against every other ship) in place of the trajectory and outcome heads ([yemong.md](../explanation/yemong.md#the-shot-damage-head)).
