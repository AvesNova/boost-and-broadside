"""Kernel families of a saved trace per unit (micro-batch or tick): families.py TRACE UNITS."""
import sys

sys.path.insert(0, "docs/rewrite/sections/training-review-2")
import common  # noqa: E402

trace, units = sys.argv[1], float(sys.argv[2])
r = common.read_trace(trace)
total = r["kernel_ms"]
print(f"window {r['window_ms'] / units:.2f} ms/unit, busy {100 * r['busy_share']:.1f}%, kernels {r['kernels'] / units:.0f}/unit")
for k, v in r["families_ms"].items():
    print(f"{k:40s} {v / units:8.2f} ms/unit {100 * v / total:5.1f}%")
