"""Per-interval throughput metrics (train/sps, train/ship_tokens_per_sec)."""

from boost_and_broadside.train.rl.logging import LoggingMixin


class _Probe(LoggingMixin):
    """Minimal carrier of the throughput state the helper touches."""

    def __init__(self) -> None:
        self._global_step = 0
        self._ship_steps = 0
        self._perf_mark_time = 0.0
        self._perf_mark_step = 0
        self._perf_mark_ship_steps = 0


def test_rates_measure_the_interval_not_the_run():
    probe = _Probe()

    # A slow first interval: 1000 steps in 10s.
    probe._global_step = 1000
    probe._ship_steps = 4000
    assert probe._throughput_since_last_log(10.0) == (100, 400, 10.0)

    # A fast second interval reads as fast, not as the run average (which
    # would still be ~366 sps here).
    probe._global_step = 5000
    probe._ship_steps = 20_000
    assert probe._throughput_since_last_log(12.0) == (2000, 8000, 2.0)


def test_zero_period_reports_zero_rather_than_dividing():
    probe = _Probe()
    probe._global_step = 100
    assert probe._throughput_since_last_log(0.0) == (0, 0, 0.0)
