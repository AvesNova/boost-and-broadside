# League schedule and keep-every-member: progress

Decisions implemented (author, 2026-11-15): the league snapshot schedule (every 5M to 50M,
then the interval doubles each time the steps double) and keeping every league member.

## 2026-11-15, step 1: schedule, no pruning, no timed cadence

- **Tried / worked:** `SnapshotSchedule` gains `doubling_from`: after the listed steps and
  the period, from `doubling_from` the period doubles, and doubles again at each doubling of
  the steps. Validated: `doubling_from` must lie after the listed steps, on the period's
  grid, and be a multiple of twice the period, so every stage boundary is itself a member.
  `LEAGUE_SNAPSHOTS = SnapshotSchedule(every=5M, doubling_from=50M)`: 5 ... 50, 60 ... 100,
  120 ... 200, 240 ... 400, 480, 560, ... M (ten to 50M, then five per doubling).
  `count`, `due` and `points` walk the stages (a loop of log2 length). `RL_SNAPSHOTS`
  unchanged. The JSON record (`snapshot_schedule` in the league state) carries the new
  field; a record without it reads as no doubling.
- Removed: `bnb.league.pool.retained`, `RECENT_SNAPSHOTS`, `HISTORY_FROM`, the pruning loop
  in `League._snapshot` (with its directory deletion), `Gauge.remove` (only retention used
  it), and their tests.
- Removed the timed-cadence fallback: no training path sets `LeagueConfig.snapshots` to
  None (the CLI and `TrainConfig` take the default; only a test did), so
  `LeagueConfig.snapshots` is now `SnapshotSchedule` (not optional) and
  `League._measure_cadence`, `SNAPSHOT_SECONDS`, `started`/`durations`, the `snapshot_every`
  run-record field and the `League(run=...)` parameter (used only to record that period) go.
  `bnb.runs.session.CADENCE_UPDATES` stays: the training checkpoints' cadence uses it.
- Choices beyond the brief: `Entry.ordinal` and the league's `ordinal` counter existed only
  for retention, so they go too. Consequence: a league state saved before this change
  (entries with `ordinal`, or `snapshot_schedule: null`) does not resume on this code. No
  release run depends on one (the pilot is to be re-run).
  A resumed run keeps the schedule recorded in its state (unchanged behaviour), so a run
  started on the old schedule keeps it.
- Tests: schedule members and stage boundaries, count = len(points), due() across updates
  of 0.6M/0.75M/0.999999M steps to 1000M (every member taken by the update holding it),
  validation, JSON round trip; a league run of 31 snapshots (5M ... 960M) keeps every
  entry, gauge row and directory, and resumes with the same entries and schedule.
- **Next:** none; awaiting merge by the author.
