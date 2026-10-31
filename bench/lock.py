"""One benchmark at a time on this machine. Children of a holder pass through (HELD_ENV)."""

import contextlib
import fcntl
import os

LOCK_PATH = "/tmp/bnb-bench.lock"
HELD_ENV = "BNB_BENCH_LOCK_HELD"


@contextlib.contextmanager
def bench_lock():
    if os.environ.get(HELD_ENV) == "1":
        yield
        return
    with open(LOCK_PATH, "a+") as fh:
        print(f"[bench] taking {LOCK_PATH}", flush=True)
        fcntl.flock(fh, fcntl.LOCK_EX)
        os.environ[HELD_ENV] = "1"
        try:
            yield
        finally:
            os.environ.pop(HELD_ENV, None)
            fcntl.flock(fh, fcntl.LOCK_UN)
