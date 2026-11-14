"""Snapshot schedules: the environment steps at which a run keeps a snapshot, a list of
steps and then a fixed period (`SnapshotSchedule`).

Two schedules use it (decided 2026-11-14): the policy files a run keeps
(`TrainConfig.policy_snapshots`, `RL_SNAPSHOTS`: dense early, where the Elo curve is
steep, then every 5M steps), and the league's members (`LeagueConfig.snapshots`,
`LEAGUE_SNAPSHOTS`: about 100-200 Elo apart on rl-v20's curve, then the last gap).
"""

import dataclasses

MILLION = 1_000_000


@dataclasses.dataclass(frozen=True)
class SnapshotSchedule:
    """The environment steps at which a snapshot is due: each of `steps` (ascending,
    positive), then every `every` steps after the last of them (None: the last gap between
    `steps`, or the one step itself when there is only one). With no `steps`, every
    `every` steps from 0."""

    steps: tuple[int, ...] = ()
    every: int | None = None

    def __post_init__(self) -> None:
        if not self.steps and self.every is None:
            raise ValueError("a snapshot schedule needs steps or a period")
        if any(step < 1 for step in self.steps) or any(
            later <= earlier for earlier, later in zip(self.steps, self.steps[1:])
        ):
            raise ValueError(f"snapshot steps {list(self.steps)} must be positive and ascending")
        if self.every is not None and self.every < 1:
            raise ValueError(f"the snapshot period {self.every} must be positive")

    @property
    def period(self) -> int:
        """The steps between snapshots after the last listed one."""
        if self.every is not None:
            return self.every
        if len(self.steps) == 1:
            return self.steps[0]
        return self.steps[-1] - self.steps[-2]

    def count(self, environment_steps: int) -> int:
        """The snapshots due at or before `environment_steps`."""
        listed = sum(step <= environment_steps for step in self.steps)
        last = self.steps[-1] if self.steps else 0
        return listed + max(0, environment_steps - last) // self.period

    def due(self, before: int, after: int) -> bool:
        """Whether a snapshot falls in (before, after]: an update that took the run from
        `before` to `after` environment steps keeps one."""
        return self.count(after) > self.count(before)

    def points(self, until: int) -> list[int]:
        """Every snapshot step at or before `until`, in order."""
        last = self.steps[-1] if self.steps else 0
        tail = range(last + self.period, until + 1, self.period)
        return [step for step in self.steps if step <= until] + list(tail)


# RL's policy files: 1, 2, 3, 4, 6, 8, 10, 12, 16 and 20M steps, then every 5M.
RL_SNAPSHOTS = SnapshotSchedule(
    tuple(step * MILLION for step in (1, 2, 3, 4, 6, 8, 10, 12, 16, 20)), 5 * MILLION
)
# The league's members: 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110, 130 and 160M
# steps, then the last gap (30M). Re-derived after the game freeze.
LEAGUE_SNAPSHOTS = SnapshotSchedule(
    tuple(step * MILLION for step in (5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110, 130, 160))
)
