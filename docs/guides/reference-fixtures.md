---
covers: [tests/reference/]
---

# Reference fixtures

`tests/reference` holds the equivalence tests: each checks a v2 section against **goldens**,
outputs computed in float64 by an independent reference. The tests and goldens are fixed.
A section passes them through an **adapter**, the one file it writes.

## What the goldens are

| Directory | Goldens from | Adapter |
|---|---|---|
| `tests/reference` (`golden/`) | the physics race's float64 oracle | `tests/reference/race_arm.py` |
| `tests/reference/rewards` | v1's code at the tag `v1-torch-final` | `tests/reference/rewards/rewards_arm.py` |
| `tests/reference/yemong` | v1's code at the tag `v1-torch-final` | `tests/reference/yemong/yemong_arm.py` |
| `tests/reference/learner` | v1's code at the tag `v1-torch-final` | `tests/reference/learner/learner_arm.py` |

- A golden is a `.npz` (or `.json`) file of inputs, weights where a module has them, and
  the reference's outputs.
- The generators live outside this tree. The tests speak an interface of their own; the
  adapter translates it to `bnb`.
  - The rewards, Yemong and learner sets describe their interface, conventions and
    tolerances in their directory's `README.md`{.plain}.
  - The race set (`tests/reference` itself) has no README. Its interface is the physics
    race's (numpy in, numpy out, the race's observation layout), implemented by `RaceArm`
    in `tests/reference/race_arm.py`; its tolerances are in `tests/reference/helpers.py`.
- Each set fails on deliberate mutants of its reference, so a passing set means the code
  computes what the reference computes, to the stated tolerance.

## Never edit them

The tests, helpers, conftests and goldens are fixed. Only the adapter changes. A golden is
evidence that the code matches an independent computation; regenerating it to match new
code would remove that evidence. When a spec changes the game on purpose, the goldens are
regenerated from the updated reference, before the section's agent starts, and the change
is recorded in the spec.

If a reference test fails, the code or the adapter is wrong, or the spec changed and the
fixtures have not caught up. Report it; do not change the fixture.

## Running them

<!-- not-runnable: runs a whole directory of tests -->
```bash
uv run --no-sync pytest tests/reference                  # all of them
uv run --no-sync pytest tests/reference/yemong           # one set
```

A set whose section is not built yet skips: its adapter's `make_arm()` returns `None`, and
the conftest skips with a message naming the section.

## Another implementation: `REFERENCE_ARM`

Each conftest reads the environment variable `REFERENCE_ARM`. When it is set to a directory
containing an `arm.py`{.plain} that defines `make_arm()`, the tests run against that implementation
instead of `bnb`.

<!-- not-runnable: needs an arm directory outside the repository -->
```bash
REFERENCE_ARM=/path/to/arm uv run --no-sync pytest tests/reference/rewards
```
