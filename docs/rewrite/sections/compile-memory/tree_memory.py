"""Sample the memory of a process tree every `interval` seconds into a CSV, until the root exits.

Usage: tree_memory.py ROOT_PID OUT_CSV [CACHE_DIR] [INTERVAL]

Reads /proc only (no psutil). Per sample: the processes in the tree (root + every
descendant), how many are ptxas and how many are Python compile workers; RSS and PSS of
the root, of the Python workers, of ptxas and of the whole tree (PSS splits shared pages,
so the tree's PSS sum is what the tree really holds; RSS sums count the shared libraries
and the forked workers' copy-on-write pages once per process); the tree's anonymous
memory (private, what a cgroup charges as anon); and the size of CACHE_DIR (what a cold
compile writes: on a tmpfs /tmp it is RAM too). Prints the peaks at the end.
"""

import csv
import os
import sys
import time

KB_PER_MB = 1024


def children_map() -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as handle:
                stat = handle.read()
        except OSError:
            continue
        parent = int(stat[stat.rindex(")") + 2 :].split()[1])
        children.setdefault(parent, []).append(int(entry))
    return children


def tree(root: int) -> list[int]:
    children = children_map()
    out, stack = [], [root]
    while stack:
        pid = stack.pop()
        out.append(pid)
        stack.extend(children.get(pid, []))
    return out


def rollup(pid: int) -> tuple[str, dict[str, float]] | None:
    try:
        with open(f"/proc/{pid}/comm") as handle:
            name = handle.read().strip()
        values = {}
        with open(f"/proc/{pid}/smaps_rollup") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) == 3 and parts[2] == "kB":
                    values[parts[0].rstrip(":")] = int(parts[1]) / KB_PER_MB
        return name, values
    except (OSError, ValueError):
        return None


def cache_megabytes(path: str) -> float:
    total = 0
    for directory, _, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(directory, name)).st_size
            except OSError:
                pass
    return total / (1 << 20)


def alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as handle:
            stat = handle.read()
        return stat[stat.rindex(")") + 2] != "Z"
    except OSError:
        return False


def main() -> None:
    root = int(sys.argv[1])
    out = sys.argv[2]
    cache = sys.argv[3] if len(sys.argv) > 3 else ""
    interval = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    start = time.time()
    fields = [
        "clock", "seconds", "processes", "ptxas", "workers", "root_rss", "root_pss",
        "root_anon", "workers_pss", "ptxas_pss", "tree_rss", "tree_pss", "tree_anon", "cache",
    ]
    peaks = dict.fromkeys(fields[2:], 0.0)
    cache_size, last_cache = 0.0, 0.0
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        while alive(root):
            row = dict.fromkeys(fields[2:], 0.0)
            for pid in tree(root):
                read = rollup(pid)
                if read is None or not read[1]:
                    continue
                name, values = read
                rss, pss = values.get("Rss", 0.0), values.get("Pss", 0.0)
                anon = values.get("Anonymous", 0.0)
                row["processes"] += 1
                row["tree_rss"] += rss
                row["tree_pss"] += pss
                row["tree_anon"] += anon
                if pid == root:
                    row["root_rss"], row["root_pss"], row["root_anon"] = rss, pss, anon
                elif name == "ptxas":
                    row["ptxas"] += 1
                    row["ptxas_pss"] += pss
                elif name.startswith("python"):
                    row["workers"] += 1
                    row["workers_pss"] += pss
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
