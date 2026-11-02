"""One benchmark at a time on this machine. Children of a holder pass through (HELD_VARIABLE)."""

import contextlib
import fcntl
import os

LOCK_PATH = "/tmp/bnb-bench.lock"
HELD_VARIABLE = "BNB_BENCH_LOCK_HELD"


@contextlib.contextmanager
def bench_lock():
    """Hold the machine's benchmark lock for the block (passing through when a parent holds
    it), so GPU timings and GPU tests never share the device."""
    if os.environ.get(HELD_VARIABLE) == "1":
        yield
        return
    with open(LOCK_PATH, "a+") as lock_file:
        print(f"[bench] taking {LOCK_PATH}", flush=True)
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        os.environ[HELD_VARIABLE] = "1"
        try:
            yield
        finally:
            os.environ.pop(HELD_VARIABLE, None)
            fcntl.flock(lock_file, fcntl.LOCK_UN)
