"""Sample the memory of a process tree every `interval` seconds into a CSV, until the root exits.

Usage: tree_memory.py ROOT_PID OUT_CSV [CACHE_DIR] [INTERVAL]

Per sample: the processes in the tree (root + all descendants), how many are ptxas and how
many are Python compile workers; RSS and PSS of the root, of the Python workers, of ptxas
and of the whole tree (PSS splits shared pages, so the tree's PSS sum is what the tree
really holds, RSS sums overcount the shared libraries); and the size of CACHE_DIR (what a
cold compile writes: on a tmpfs /tmp it is RAM too). Prints the peaks at the end.
"""

import csv
import os
import sys
import time

import psutil

MB = 1 << 20


def cache_megabytes(path: str) -> float:
    total = 0
    for directory, _, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(directory, name)).st_size
            except OSError:
                pass
    return total / MB


def main() -> None:
    root = psutil.Process(int(sys.argv[1]))
    out = sys.argv[2]
    cache = sys.argv[3] if len(sys.argv) > 3 else ""
    interval = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    start = time.time()
    fields = [
        "clock", "seconds", "processes", "ptxas", "workers",
        "root_rss", "root_pss", "workers_pss", "ptxas_pss", "tree_rss", "tree_pss", "cache",
    ]
    peaks = dict.fromkeys(fields[2:], 0.0)
    cache_size, last_cache = 0.0, 0.0
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        while root.is_running() and root.status() != psutil.STATUS_ZOMBIE:
            try:
                processes = [root] + root.children(recursive=True)
            except psutil.NoSuchProcess:
                break
            row = dict.fromkeys(fields[2:], 0.0)
            for process in processes:
                try:
                    name = process.name()
                    memory = process.memory_full_info()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                row["processes"] += 1
                row["tree_rss"] += memory.rss / MB
                row["tree_pss"] += memory.pss / MB
                if process.pid == root.pid:
                    row["root_rss"] = memory.rss / MB
                    row["root_pss"] = memory.pss / MB
                elif name == "ptxas":
                    row["ptxas"] += 1
                    row["ptxas_pss"] += memory.pss / MB
                elif name.startswith("python"):
                    row["workers"] += 1
                    row["workers_pss"] += memory.pss / MB
            now = time.time()
            if cache and now - last_cache > 10:
                cache_size, last_cache = cache_megabytes(cache), now
            row["cache"] = cache_size
            for key, value in row.items():
                peaks[key] = max(peaks[key], value)
            writer.writerow(
                [time.strftime("%H:%M:%S"), round(now - start)]
                + [round(row[key]) for key in fields[2:]]
            )
            handle.flush()
            time.sleep(interval)
    print("peaks (MB):", {key: round(value) for key, value in peaks.items()})


if __name__ == "__main__":
    main()
