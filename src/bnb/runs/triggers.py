"""The trigger rules (diagnostics spec 3, the audit's "Final trigger rules"): checks on each
update's record, on the host, that flag a problem on the dashboard.

A rule measures its series (by name, or a pattern with `*` matching one or more name parts)
and says whether the measurement breaches it. Three kinds:
- invariant: a known correct range (`Above`, `Below`, `Outside`, `Rises`, `AboveShareOf`);
- anomaly: against the run's own history (`BelowRunningMean`, `DropFromPeak`,
  `ZScoreBelow`, `BelowShareOfPeak`, `BelowShareOfRunLevel`);
- trend: a slope over a window (`Rising`, `RisingWhileFlat`).
Against noise, every rule has a warm-up (no flag in the run's first updates), a hysteresis
(the breach must hold for consecutive updates) and a cooldown (a fired event is not repeated
within it). The thresholds are placeholders, to be tuned after the first runs.

Each update adds `alerts/{rule}` (1 while the rule's flag is up), `alerts/active` (the flags
up) and, when a rule fires, `alerts/events`: a table of (rule, severity, update, series,
value, threshold, panel). A series a run does not log leaves its rule idle. The rules' state
(the EMAs, peaks, windows, counts and cooldowns) is plain JSON, kept in the training
checkpoint so that a resume does not re-arm them. Only the skipped-step rule acts (the
session's checkpoint and batch dump); the rest report.
"""

import fnmatch
import math

ALERT, WARNING = "alert", "warning"
EVENT_COLUMNS = ("rule", "severity", "update", "series", "value", "threshold", "panel")


def matching(record: dict, pattern: str) -> list[str]:
    """The record's numeric series whose names match `pattern` (`*`: one or more parts)."""
    return sorted(
        name
        for name, value in record.items()
        if isinstance(value, int | float)
        and not isinstance(value, bool)
        and fnmatch.fnmatchcase(name, pattern)
    )


class Rule:
    """A rule's name, severity and the panel to open; its gate (warm-up updates, hysteresis
    updates, cooldown updates). Subclasses measure: `breaches(record, state)` returns the
    breaches of this update as (series, value, threshold) and may keep numbers in `state`."""

    def __init__(
        self,
        name: str,
        severity: str,
        panel: str,
        warmup: int = 0,
        hysteresis: int = 1,
        cooldown: int = 50,
    ) -> None:
        self.name, self.severity, self.panel = name, severity, panel
        self.warmup, self.hysteresis, self.cooldown = warmup, hysteresis, cooldown

    def breaches(self, record: dict, state: dict) -> list[tuple[str, float, float]]:
        """This update's breaches as (series, value, threshold)."""
        raise NotImplementedError


class Above(Rule):
    """Any series of `pattern` above `threshold`."""

    def __init__(self, name, severity, panel, pattern: str, threshold: float, **gate) -> None:
        super().__init__(name, severity, panel, **gate)
        self.pattern, self.threshold = pattern, threshold

    def breaches(self, record, state):
        """This update's breaches as (series, value, threshold)."""
        return [
            (series, record[series], self.threshold)
            for series in matching(record, self.pattern)
            if record[series] > self.threshold
        ]


class Below(Above):
    """Any series of `pattern` below `threshold`."""

    def breaches(self, record, state):
        """This update's breaches as (series, value, threshold)."""
        return [
            (series, record[series], self.threshold)
            for series in matching(record, self.pattern)
            if record[series] < self.threshold
        ]


class Outside(Rule):
    """Any series of `pattern` outside [low, high]."""

    def __init__(self, name, severity, panel, pattern: str, low: float, high: float, **gate):
        super().__init__(name, severity, panel, **gate)
        self.pattern, self.low, self.high = pattern, low, high

    def breaches(self, record, state):
        """This update's breaches as (series, value, threshold)."""
        out = []
        for series in matching(record, self.pattern):
            value = record[series]
            if value < self.low:
                out.append((series, value, self.low))
            elif value > self.high:
                out.append((series, value, self.high))
        return out


class Rises(Rule):
    """A counter `series` above its value at the last update (the first value seen is the
    baseline)."""

    def __init__(self, name, severity, panel, series: str, **gate) -> None:
        super().__init__(name, severity, panel, **gate)
        self.series = series

    def breaches(self, record, state):
        """This update's breaches as (series, value, threshold)."""
        value = record.get(self.series)
        if value is None:
            return []
        previous = state.get("previous")
        state["previous"] = value
        return [] if previous is None or value <= previous else [(self.series, value, previous)]


class AboveShareOf(Rule):
    """`numerator` above `share` of `denominator` (two series of the same update)."""

    def __init__(self, name, severity, panel, numerator: str, denominator: str, share, **gate):
        super().__init__(name, severity, panel, **gate)
        self.numerator, self.denominator, self.share = numerator, denominator, share

    def breaches(self, record, state):
        """This update's breaches as (series, value, threshold)."""
        if self.numerator not in record or self.denominator not in record:
            return []
        limit = self.share * record[self.denominator]
        value = record[self.numerator]
        return [(self.numerator, value, limit)] if value > limit else []


class BelowRunningMean(Rule):
    """`series` below `share` of its running mean (an EMA with `decay`, of the values before
    this one that did not breach)."""

    def __init__(self, name, severity, panel, series: str, share: float, decay=0.9, **gate):
        super().__init__(name, severity, panel, **gate)
        self.series, self.share, self.decay = series, share, decay

    def breaches(self, record, state):
        """This update's breaches as (series, value, threshold)."""
        value = record.get(self.series)
        if value is None or not math.isfinite(value):
            return []
        mean = state.get("mean")
        if mean is not None and value < self.share * mean:
            return [(self.series, value, self.share * mean)]  # nor does a breach here
        state["mean"] = value if mean is None else self.decay * mean + (1 - self.decay) * value
        return []


class DropFromPeak(Rule):
    """Any series of `pattern` more than `multiple` of its standard error (the series of the
    same name with `error` in place of `quantity`) below its running peak."""

    def __init__(
        self, name, severity, panel, pattern: str, quantity: str, error: str, multiple, **gate
    ):
        super().__init__(name, severity, panel, **gate)
        self.pattern, self.quantity, self.error, self.multiple = pattern, quantity, error, multiple

    def breaches(self, record, state):
        """This update's breaches as (series, value, threshold)."""
        out = []
        peaks = state.setdefault("peaks", {})
        for series in matching(record, self.pattern):
            value = record[series]
            stderr = record.get(series.replace(self.quantity, self.error, 1))
            if not math.isfinite(value):
                continue
            peak = peaks.get(series)
            peaks[series] = value if peak is None else max(peak, value)
            if peak is not None and stderr is not None and math.isfinite(stderr):
                limit = peak - self.multiple * stderr
                if value < limit:
                    out.append((series, value, limit))
        return out


class ZScoreBelow(Rule):
    """`series` more than `z` standard deviations below its running mean (EMAs of the mean and
    the variance with `decay`, of the values before this one that did not breach, so a
    lasting collapse stays flagged)."""

    def __init__(self, name, severity, panel, series: str, z: float, decay=0.95, **gate):
        super().__init__(name, severity, panel, **gate)
        self.series, self.z, self.decay = series, z, decay

    def breaches(self, record, state):
        """This update's breaches as (series, value, threshold)."""
        value = record.get(self.series)
        if value is None or not math.isfinite(value):
            return []
        mean, variance = state.get("mean"), state.get("variance", 0.0)
        if mean is None:
            state["mean"], state["variance"] = value, 0.0
            return []
        limit = mean - self.z * math.sqrt(variance)
        if variance > 0 and value < limit:
            return [(self.series, value, limit)]  # an outlier does not move the baseline
        difference = value - mean
        state["mean"] = mean + (1 - self.decay) * difference
        state["variance"] = self.decay * (variance + (1 - self.decay) * difference**2)
        return []


class BelowShareOfPeak(Rule):
    """Any series of `pattern` below `share` of its running peak."""

    def __init__(self, name, severity, panel, pattern: str, share: float, **gate) -> None:
        super().__init__(name, severity, panel, **gate)
        self.pattern, self.share = pattern, share

    def breaches(self, record, state):
        """This update's breaches as (series, value, threshold)."""
        out = []
        peaks = state.setdefault("peaks", {})
        for series in matching(record, self.pattern):
            value = record[series]
            peak = peaks.get(series)
            peaks[series] = value if peak is None else max(peak, value)
            if peak is not None and value < self.share * peak:
                out.append((series, value, self.share * peak))
        return out


class BelowShareOfRunLevel(Rule):
    """Any series of `pattern` below `share` of its run level (an EMA with `decay` of the
    values before this one)."""

    def __init__(self, name, severity, panel, pattern: str, share: float, decay=0.99, **gate):
        super().__init__(name, severity, panel, **gate)
        self.pattern, self.share, self.decay = pattern, share, decay

    def breaches(self, record, state):
        """This update's breaches as (series, value, threshold)."""
        out = []
        levels = state.setdefault("levels", {})
        for series in matching(record, self.pattern):
            value = record[series]
            level = levels.get(series)
            levels[series] = (
                value if level is None else self.decay * level + (1 - self.decay) * value
            )
            if level is not None and value < self.share * level:
                out.append((series, value, self.share * level))
        return out


def slope(values: list[float]) -> tuple[float, float]:
    """The least-squares slope of `values` against their index, and its standard error (inf
    for fewer than three values)."""
    count = len(values)
    if count < 3:
        return 0.0, math.inf
    mean_x, mean_y = (count - 1) / 2, sum(values) / count
    spread = sum((x - mean_x) ** 2 for x in range(count))
    gradient = sum((x - mean_x) * (y - mean_y) for x, y in enumerate(values)) / spread
    residual = sum((y - mean_y - gradient * (x - mean_x)) ** 2 for x, y in enumerate(values))
    return gradient, math.sqrt(residual / (count - 2) / spread)


class Rising(Rule):
    """Any series of `pattern` rising over its last `window` values: the slope more than
    `multiple` standard errors above 0."""

    def __init__(self, name, severity, panel, pattern: str, window=20, multiple=2.0, **gate):
        super().__init__(name, severity, panel, **gate)
        self.pattern, self.window, self.multiple = pattern, window, multiple

    def breaches(self, record, state):
        """This update's breaches as (series, value, threshold)."""
        out = []
        windows = state.setdefault("windows", {})
        for series in matching(record, self.pattern):
            values = (windows.get(series, []) + [record[series]])[-self.window :]
            windows[series] = values
            gradient, error = slope(values)
            if len(values) == self.window and gradient > self.multiple * error:
                out.append((series, gradient, self.multiple * error))
        return out


class RisingWhileFlat(Rule):
    """The sum of the `rising` series climbing (slope over `window` updates more than
    `multiple` standard errors above 0) while the mean of the `flat` series does not (slope
    within `multiple` standard errors of 0, or falling): reward up while the rating is
    not (the audit's G5)."""

    def __init__(
        self, name, severity, panel, rising: str, flat: str, window=50, multiple=2.0, **gate
    ):
        super().__init__(name, severity, panel, **gate)
        self.rising, self.flat, self.window, self.multiple = rising, flat, window, multiple

    def breaches(self, record, state):
        """This update's breaches as (series, value, threshold)."""
        rising = [record[name] for name in matching(record, self.rising)]
        flat = [record[name] for name in matching(record, self.flat)]
        flat = [value for value in flat if math.isfinite(value)]
        if not rising or not flat:
            return []
        for key, value in (("rising", sum(rising)), ("flat", sum(flat) / len(flat))):
            state[key] = (state.get(key, []) + [value])[-self.window :]
        if len(state["rising"]) < self.window or len(state["flat"]) < self.window:
            return []
        up, up_error = slope(state["rising"])
        level, level_error = slope(state["flat"])
        if up > self.multiple * up_error and level < self.multiple * level_error:
            return [(self.rising, up, self.multiple * up_error)]
        return []


class SideImbalance(Rule):
    """Self-play's team 0 and team 1 win shares (`episode/result/team_0`, `.../team_1`, counts
    of the update's games, accumulated over the run) more than `multiple` binomial standard
    errors apart."""

    def __init__(self, name, severity, panel, multiple: float = 3.0, **gate) -> None:
        super().__init__(name, severity, panel, **gate)
        self.multiple = multiple

    def breaches(self, record, state):
        """This update's breaches as (series, value, threshold)."""
        names = ("episode/result/team_0", "episode/result/team_1", "episode/result/draw")
        if any(name not in record for name in names):
            return []
        for name in names:
            state[name] = state.get(name, 0) + record[name]
        games = sum(state[name] for name in names)
        if games == 0:
            return []
        first, second = (state[name] / games for name in names[:2])
        error = math.sqrt(max(first + second - (first - second) ** 2, 0.0) / games)
        difference = abs(first - second)
        if error > 0 and difference > self.multiple * error:
            return [("episode/result", difference, self.multiple * error)]
        return []


class SkippedSteps(Rule):
    """`train/skipped_steps` above 0; each loss term with non-finite minibatches
    (`train/nonfinite_steps/{term}`) is a breach of its own, so the events name it."""

    def breaches(self, record, state):
        """This update's breaches as (series, value, threshold)."""
        skipped = record.get("train/skipped_steps", 0)
        if not skipped:
            return []
        terms = [
            (series, record[series], 0)
            for series in matching(record, "train/nonfinite_steps/*")
            if record[series] > 0
        ]
        return [("train/skipped_steps", skipped, 0), *terms]


def default_rules() -> tuple[Rule, ...]:
    """The audit's final trigger rules, with placeholder thresholds: 8 alerts, 13 warnings."""
    anomaly = dict(warmup=10, hysteresis=3)
    return (
        # alerts
        # every update with a skipped step fires, and acts
        SkippedSteps("skipped_step", ALERT, "train/skipped_steps", cooldown=1),
        Rises("recompile", ALERT, "counters/recompiles", "counters/recompiles", warmup=2),
        AboveShareOf(
            "memory",
            ALERT,
            "train/memory_peak_gigabytes",
            "train/memory_peak_gigabytes",
            "train/memory_budget_gigabytes",
            0.9,
        ),
        Above("behaviour_mismatch", ALERT, "policy/kl_first", "policy/kl_first", 1e-3),
        Above("nonfinite_input", ALERT, "input/nonfinite_count", "input/nonfinite_count", 0),
        Above("never_seen_ship", ALERT, "fog/never_seen_share", "fog/never_seen_share", 0),
        BelowRunningMean(
            "throughput",
            ALERT,
            "train/environment_steps_per_second",
            "train/environment_steps_per_second",
            0.5,
            warmup=3,
            hysteresis=2,
        ),
        DropFromPeak(
            "rating_drop",
            ALERT,
            "rating/live",
            "rating/live/*",
            "rating/live/",
            "rating/live_stderr/",
            2.0,
            **anomaly,
        ),
        # warnings
        ZScoreBelow(
            "entropy_collapse", WARNING, "policy/entropy", "policy/entropy/joint", 4.0, **anomaly
        ),
        Below(
            "critic_useless",
            WARNING,
            "critic/explained_variance",
            "critic/explained_variance/*",
            0.0,
            **anomaly,
        ),
        Above("support_clipping", WARNING, "critic/support_clip", "critic/support_clip/*", 0.01),
        Above(
            "envelope_violation",
            WARNING,
            "input/envelope_violation_share",
            "input/envelope_violation_share/*",
            0.01,
        ),
        BelowShareOfPeak(
            "representation_collapse",
            WARNING,
            "repr/effective_rank",
            "repr/effective_rank/*",
            0.7,
            **anomaly,
        ),
        Rising("dormant_units", WARNING, "repr/dormant_share", "repr/dormant_share/*", **anomaly),
        Outside(
            "update_ratio",
            WARNING,
            "train/update_ratio",
            "train/update_ratio/*",
            1e-6,
            1e-1,
            **anomaly,
        ),
        BelowShareOfRunLevel(
            "label_validity", WARNING, "supervised_share", "*/supervised_share*", 0.5, **anomaly
        ),
        Below(
            "sampling_collapse",
            WARNING,
            "pool/sampling_entropy",
            "pool/sampling_entropy/*",
            0.3,
            **anomaly,
        ),
        RisingWhileFlat(
            "reward_hacking", WARNING, "reward/rate", "reward/rate/*", "rating/live/*", **anomaly
        ),
        Below(
            "head_worse_than_trivial",
            WARNING,
            "auxiliary gains",
            "*/gain*",
            0.0,
            **anomaly,
        ),
        Outside(
            "sigma_miscalibrated",
            WARNING,
            "belief/normalised_square_error",
            "belief/normalised_square_error*",
            0.5,
            2.0,
            **anomaly,
        ),
        SideImbalance("side_imbalance", WARNING, "episode/result", warmup=10),
    )


class Triggers:
    """The rules and their state; `check` runs them on one update's record."""

    def __init__(self, rules: tuple[Rule, ...] | None = None, state: dict | None = None) -> None:
        self.rules = default_rules() if rules is None else rules
        self.state = {rule.name: dict(consecutive=0, fired=None, measure={}) for rule in self.rules}
        if state is not None:
            self.state.update({name: value for name, value in state.items() if name in self.state})

    def check(self, record: dict, update: int) -> dict:
        """This update's flags as series (`alerts/{rule}`, `alerts/active`) and, when a rule
        fires, the `alerts/events` table; `update` counts from 1."""
        out, events = {}, []
        for rule in self.rules:
            state = self.state[rule.name]
            breaches = rule.breaches(record, state["measure"])
            warm = update > rule.warmup
            state["consecutive"] = state["consecutive"] + 1 if breaches and warm else 0
            active = state["consecutive"] >= rule.hysteresis
            out[f"alerts/{rule.name}"] = int(active)
            fired = state["fired"]
            if active and (fired is None or update - fired >= rule.cooldown):
                state["fired"] = update
                for series, value, threshold in breaches:
                    events.append(
                        (rule.name, rule.severity, update, series, value, threshold, rule.panel)
                    )
        out["alerts/active"] = sum(out.values())
        if events:
            out["alerts/events"] = dict(columns=list(EVENT_COLUMNS), rows=[list(e) for e in events])
        return out

    def fired(self, record: dict, name: str) -> bool:
        """Whether rule `name` fired in `record` (the series `check` returned)."""
        events = record.get("alerts/events")
        return events is not None and any(row[0] == name for row in events["rows"])
