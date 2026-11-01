"""The coverage map: pages whose covered code changed after the page last did.

Each page declares in its front matter the paths it describes (`covers: [src/bnb/game/]`).
A page is **possibly stale** when a commit after the page's last commit touched one of
them, or when one of them has uncommitted changes and the page has none (so the warning
shows before the commit). This is a prompt to reread the page, not a failure.

    uv run --no-sync python -m tests.docs.staleness

prints the report. Outside git, or in a shallow clone (where history is cut), it prints why
it cannot tell and exits cleanly.
"""

import dataclasses
import subprocess

from .pages import ROOT, pages


@dataclasses.dataclass(frozen=True)
class Stale:
    page: str
    covers: tuple[str, ...]
    commits: tuple[str, ...]  # "hash subject", newest first; "uncommitted path" first


def _git(*args: str) -> str | None:
    """stdout of a git command in the repository, or None if git fails."""
    try:
        done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def unavailable() -> str | None:
    """Why the history cannot be read, or None when it can."""
    if _git("rev-parse", "--is-inside-work-tree") != "true":
        return "not inside a git work tree"
    if _git("rev-parse", "--is-shallow-repository") != "false":
        return "a shallow clone: the history is cut"
    return None


def uncommitted() -> set[str]:
    """Paths with changes not committed yet: modified, staged or untracked."""
    changed = (_git("diff", "--name-only", "HEAD") or "").splitlines()
    untracked = (_git("ls-files", "--others", "--exclude-standard") or "").splitlines()
    return set(changed) | set(untracked)


def _under(path: str, covered: str) -> bool:
    return path == covered.rstrip("/") or path.startswith(covered.rstrip("/") + "/")


def report() -> list[Stale]:
    """Every possibly stale page. Call `unavailable()` first."""
    out = []
    pending = uncommitted()
    for page in pages():
        if not page.on_site:
            continue
        covers = tuple(page.front_matter().get("covers", ()))
        if not covers:
            continue
        found = []
        if page.name not in pending:  # a page being edited is being brought up to date
            found += [f"uncommitted {path}" for path in sorted(pending)
                      if any(_under(path, covered) for covered in covers)]
        last = _git("log", "-1", "--format=%H", "--", page.name)
        if last:  # a page never committed was written against the code as it is
            commits = _git("log", "--format=%h %s", f"{last}..HEAD", "--", *covers)
            found += commits.splitlines() if commits else []
        if found:
            out.append(Stale(page.name, covers, tuple(found)))
    return out


def describe(stale: Stale) -> str:
    lines = [f"{stale.page} (covers {', '.join(stale.covers)}): "
             f"{len(stale.commits)} later change(s) to its code"]
    lines += [f"    {commit}" for commit in stale.commits[:10]]
    if len(stale.commits) > 10:
        lines.append(f"    ... and {len(stale.commits) - 10} more")
    return "\n".join(lines)


def main() -> None:
    reason = unavailable()
    if reason:
        print(f"staleness: skipped ({reason})")
        return
    found = report()
    if not found:
        print("staleness: every page is newer than the code it covers")
        return
    print(f"staleness: {len(found)} page(s) may be out of date; reread them against the code")
    for stale in found:
        print(describe(stale))


if __name__ == "__main__":
    main()
