# Learner arm options (branch feat/learner-arm-options)

Four arm options as settings, defaults unchanged (2026-11-15).

## A10: entropy floor

- `learner.entropy_floor` (nats of joint entropy, None = off) and
  `learner.entropy_floor_coefficient` (0.1 by default). The term `entropy_floor` is the mean
  over own living ships of max(0, floor − H), H each ship-step's joint entropy
  (`losses.entropy_floor_penalty`). Logged as `policy/entropy_floor/penalty` when on; in
  the gradient decomposition's actor group.
- Per token, not on the minibatch mean: a mean-entropy hinge needs the whole minibatch's
  mean before its gradient, which breaks the micro-batch sum. A per-token floor is the
  stronger constraint: it also pushes up low-entropy states when the mean is above the floor.
- Both fields are in `checkpoint.OVERRIDABLE`. A BC config with a floor is refused (in a
  chain: `--rl-set`). A warm start keeps the floor; only the bonus becomes 0.
- The 0.1 default coefficient and any floor value are untested proposals for the author.
- Settings: `--rl-set learner.entropy_floor=<nats>` (chain) or `--set ...` (RL run or fork).

## R1: nearest MLP mode inputs

- `nearest.full_mode_inputs`: elimination flag, boundary fraction, symlog boundary seconds
  to zero, symlog next wave, symlog capture locks per zone role (own, enemy): 14 inputs
  after the first layout (196 → 210). The config default is False (stored builds load as
  recorded, with a FilledDefaults warning); `TrainConfig.nearest` sets True for new runs.
  298,166 parameters at a new run's defaults.
- Settings: `--model nearest` (nothing else).

## A24: chain without BC

- The chain now accepts `--bc-budget-steps` (its BC stage's budget; 0 = no BC stage). With 0
  the handover is recorded at step 0 (`stopped_by` "no_bc", policy null), no `-bc` run is
  made and the RL stage is a fresh run over the whole budget, with the warm start's entropy
  (0) like the chain's RL stage. `--print-config` prints `{"bc": null, "rl": ...}`.
- Settings: `--objective chain --chain-budget-steps 300000000 --bc-budget-steps 0`.

## R0: per-update policy files as a setting

- `TrainConfig.update_policies`: "off" (default), "rl", "bc" or "all"; a run writes every
  update's files when the setting names its stage (PPO = rl). The `--update-policies` CLI
  flag is gone; a resume changes it with `--set update_policies=...` (OVERRIDABLE). The
  run record's `update_policies` bool is still written (from the config), and the session
  still writes the starting weights before the first update.
- Settings: `--objective chain ... --set update_policies=rl` (BC stage keeps snapshots).

## Open for the author

- **A fork resets the entropy bonus to −0.0081.** `--from RUN` builds its config from the
  command line, so an RL fork of a chain's RL stage (A4–A7, A10) trains with LearnerConfig's
  −0.0081 bonus, not the RL stage's 0, unless `--set learner.entropy_power=0` (and stick,
  shoot) is given. Not changed here.
- Old section scripts (bc-v19, compile-split, steady-speed) still pass `--update-policies`;
  they now fail on the unknown flag (records, not changed). launch.toml now sets `update_policies=off`.
- A resumed run whose config predates `update_policies` resumes with "off".
