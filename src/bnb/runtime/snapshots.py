"""Snapshot schedules: the environment steps at which a run keeps a snapshot, a list of
steps, then a fixed period, and optionally a period that doubles each time the steps double
(`SnapshotSchedule`).

Two schedules use it: the policy files a run keeps (`TrainConfig.policy_snapshots`,
`RL_SNAPSHOTS`, decided 2026-11-14: dense early, where the Elo curve is steep, then every
5M steps), and the league's members (`LeagueConfig.snapshots`, `LEAGUE_SNAPSHOTS`, decided
2026-11-15: every 5M steps to 50M, then an interval that doubles each time the steps
double).
"""

import dataclasses

MILLION = 1_000_000


@dataclasses.dataclass(frozen=True)
class SnapshotSchedule:
    """The environment steps at which a snapshot is due: each of `steps` (ascending,
    positive), then every `every` steps after the last of them (None: the last gap between
    `steps`, or the one step itself when there is only one). With no `steps`, every
    `every` steps from 0. With `doubling_from`, a snapshot step on the period's grid and a
    multiple of twice the period, the period doubles there and again each time the steps
    double: from `doubling_from` to twice it every 2 x period, to four times it every
    4 x period, and so on."""

    steps: tuple[int, ...] = ()
    every: int | None = None
    doubling_from: int | None = None

    def __post_init__(self) -> None:
        if not self.steps and self.every is None:
            raise ValueError("a snapshot schedule needs steps or a period")
        if any(step < 1 for step in self.steps) or any(
            later <= earlier for earlier, later in zip(self.steps, self.steps[1:])
        ):
            raise ValueError(f"snapshot steps {list(self.steps)} must be positive and ascending")
        if self.every is not None and self.every < 1:
            raise ValueError(f"the snapshot period {self.every} must be positive")
        if self.doubling_from is not None and (
            self.doubling_from <= self.last
            or (self.doubling_from - self.last) % self.period
            or self.doubling_from % (2 * self.period)
        ):
            raise ValueError(
                f"the snapshot doubling start {self.doubling_from} must be a later step on "
                f"the period's grid (after {self.last}, every {self.period}) and a multiple "
                f"of twice the period"
            )

    @property
    def last(self) -> int:
        """The last listed step (0 when none is listed)."""
        return self.steps[-1] if self.steps else 0

    @property
    def period(self) -> int:
        """The steps between snapshots after the last listed one (before any doubling)."""
        if self.every is not None:
            return self.every
        if len(self.steps) == 1:
            return self.steps[0]
        return self.steps[-1] - self.steps[-2]

    def _stages(self, until: int):
        """(start, end, interval) of each stretch of a fixed period after the listed steps
        that begins before `until`: snapshots at start + interval, ... up to end."""
        if self.doubling_from is None:
            yield self.last, until, self.period
            return
        yield self.last, self.doubling_from, self.period
        start, interval = self.doubling_from, 2 * self.period
        while start < until:
            yield start, 2 * start, interval
            start, interval = 2 * start, 2 * interval

    def count(self, environment_steps: int) -> int:
        """The snapshots due at or before `environment_steps`."""
        listed = sum(step <= environment_steps for step in self.steps)
        return listed + sum(
            max(0, min(environment_steps, end) - start) // interval
            for start, end, interval in self._stages(environment_steps)
        )

    def due(self, before: int, after: int) -> bool:
        """Whether a snapshot falls in (before, after]: an update that took the run from
        `before` to `after` environment steps keeps one."""
        return self.count(after) > self.count(before)

    def points(self, until: int) -> list[int]:
        """Every snapshot step at or before `until`, in order."""
        points = [step for step in self.steps if step <= until]
        for start, end, interval in self._stages(until):
            points += range(start + interval, min(until, end) + 1, interval)
        return points


# RL's policy files: 1, 2, 3, 4, 6, 8, 10, 12, 16 and 20M steps, then every 5M.
RL_SNAPSHOTS = SnapshotSchedule(
    tuple(step * MILLION for step in (1, 2, 3, 4, 6, 8, 10, 12, 16, 20)), 5 * MILLION
)
# The league's members: every 5M steps to 50M, then every 10M to 100M, every 20M to 200M,
# every 40M to 400M, and so on (5, 10, ..., 50, 60, ..., 100, 120, ..., 200, 240, ...).
LEAGUE_SNAPSHOTS = SnapshotSchedule(every=5 * MILLION, doubling_from=50 * MILLION)
