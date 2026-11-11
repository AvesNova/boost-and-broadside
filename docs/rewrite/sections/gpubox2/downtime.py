"""downtime.py HEARTBEAT_LOG [START_UTC]: the box's up and down periods from its heartbeat
(a line a minute while up; "boot" lines at each supervisor start). A gap over 90 s between
two lines is downtime. Prints each interruption (last line before, first line after, its
length, the up period before it) and the totals since START (default: the first line)."""

import datetime
import sys


def parse(path: str) -> list[tuple[datetime.datetime, bool]]:
    out = []
    for line in open(path):
        parts = line.split()
        if not parts:
            continue
        stamp = datetime.datetime.fromisoformat(parts[0].replace("Z", "+00:00"))
        out.append((stamp, len(parts) > 1 and parts[1] == "boot"))
    return out


def main() -> None:
    lines = parse(sys.argv[1])
    if len(sys.argv) > 2:
        start = datetime.datetime.fromisoformat(sys.argv[2].replace("Z", "+00:00"))
        lines = [entry for entry in lines if entry[0] >= start]
    kst = datetime.timezone(datetime.timedelta(hours=9))
    up_since, up, down, n = lines[0][0], datetime.timedelta(), datetime.timedelta(), 0
    for (before, _), (after, boot) in zip(lines, lines[1:]):
        gap = after - before
        if gap > datetime.timedelta(seconds=90):
            n += 1
            up += before - up_since
            down += gap
            print(f"interruption {n}: down {before.astimezone(kst):%m-%d %H:%M:%S} -> "
                  f"{after.astimezone(kst):%H:%M:%S} KST ({gap.total_seconds() / 60:.1f} min), "
                  f"after {(before - up_since).total_seconds() / 60:.1f} min up"
                  + ("" if boot else " (no boot line)"))
            up_since = after
    up += lines[-1][0] - up_since
    total = up + down
    print(f"{n} interruptions; up {up.total_seconds() / 3600:.2f} h, down "
          f"{down.total_seconds() / 3600:.2f} h of {total.total_seconds() / 3600:.2f} h "
          f"({100 * up / total:.0f}% up)")


if __name__ == "__main__":
    main()
