---
covers: [mkdocs.yml, docs/hooks/generated.py, tests/docs/]
reviewed: 2026-11-01
---

# Writing docs

The docs are a MkDocs Material site built from `docs/`, with the reference generated from
the code by mkdocstrings. `docs/rewrite/` holds the internal design records; it is outside
the site and these rules do not apply to it.

## Build and preview

<!-- not-runnable: changes the environment and starts a server -->
```bash
uv sync --extra docs
uv run --no-sync mkdocs serve                        # preview at http://127.0.0.1:8000
uv run --no-sync mkdocs build --strict --site-dir /tmp/bnb-site
```

The suite checks the docs on every run (`tests/docs`); see [Checks](#checks).

## Kinds of page

| Kind | Directory | For | Holds |
|---|---|---|---|
| Getting started | `docs/getting-started/` | a first visit | one path from nothing to a working setup, in order |
| Guides | `docs/guides/` | a task | the steps to do one thing, and what to check |
| Explanation | `docs/explanation/` | understanding | how a part works and why it is built that way |
| Reference | `docs/reference/` | looking something up | generated from docstrings and signatures; never written by hand |

- A **getting-started** page assumes nothing beyond the page before it.
- A **guide** assumes the setup is done and does one task. It links to explanation rather
  than repeating it.
- An **explanation** page describes what exists and the reasons it exists. It does not
  record how a design was argued; that belongs in `docs/rewrite/`.
- A **reference** page is one `:::` directive per package. Improve the reference by
  improving the docstrings in the code.
- A section that is not merged gets one short page under `docs/explanation/in-progress/`:
  that it is in progress, and what it will cover. Prose about a section waits until the
  section merges.

Every page appears in `nav` in `mkdocs.yml`. A page left out of the nav fails the build.

## Linking code

- **Names:** write the full dotted name in backticks, `bnb.game.api.reset`. The checks
  import every backticked name that starts with `bnb.` or `bench.`, so a renamed or deleted
  name fails the suite. A short name such as `Rules.pools` is not checked; use the full name
  where it matters.
- **Links into the reference:** `[reset][bnb.game.api.reset]` links to the generated
  entry. The strict build fails if the target is not in the reference.
- **Repository paths:** write them in backticks from the root, `src/bnb/game/rules.py` or
  `tests/sections/test_modes.py`. A test can be named as
  `tests/sections/test_modes.py::test_spawn_layout_does_not_reveal_the_seat`. The checks fail
  if the path, or the test, does not exist.
- **Pages:** link with relative paths to the `.md` file, `[Modes](../explanation/game/modes.md)`.

## Generated tables

A table of values that live in the code is generated at build time, never copied by hand.
Put the marker on a line of its own:

```markdown
<!-- generated-table: ship-columns -->
```

The build replaces it with the table, built by `docs/hooks/generated.py`. The names
available are the keys of `TABLES` there: the observation's columns (`ship-columns`,
`field-columns`, `zone-columns`, `global-columns`), the action layout (`action-layout`,
`stick-levels`, `power-modes`), the rewards (`reward-levels`, `reward-config`), the game
(`ship-card`, `game-constants`, `flight-constants`, `mode-clocks`, `mode-rules`,
`game-presets`, `game-config`, `pool-fields`), training (`training-mix`,
`training-lineup`, `mixed-pools`, `train-config`, `ppo-config`, `cli-options`), the model
(`yemong-config`), the benchmark (`bench-sizes`) and the `roadmap`.

To add a table, write a function that imports the value from `bnb` and returns Markdown,
and add it to `TABLES`. Where a value's unit or meaning is a comment in the source, the
dataclass tables read that comment; keep such comments short.

The roadmap's status comes from `docs/roadmap.yml`. When a section's status changes, edit
that file, not prose.

## Examples

Every fenced `python` or `bash` block says whether it runs, on the line before the fence:

```markdown
<!-- runnable -->
<!-- not-runnable: needs CUDA -->
```

- A **runnable** block runs in the suite, from the repository root, with CUDA hidden: a
  Python block in a fresh interpreter, a command block in bash. It must finish on the CPU
  in seconds and leave the repository as it was.
- A **not-runnable** block gives its reason: it needs a GPU, the benchmark lock, minutes,
  or it changes the environment. Its Python still has to parse, and its `bnb` imports have
  to resolve.

## What a page covers

Each page declares in its front matter the code it describes:

```markdown
---
covers: [src/bnb/game/physics/, src/bnb/game/rules.py]
---
```

A page that describes no particular code declares `covers: []`. The paths must exist. When
a commit after the page's last change touches a covered path, the suite warns that the page
may be stale. The report on its own:

<!-- runnable -->
```bash
uv run --no-sync python -m tests.docs.staleness
```

When you change covered code, reread the pages that cover it and fix what changed. If a
page is still correct, set `reviewed:` in its front matter to the date and commit it: any
commit to the page clears the warning. A page that is warned about on most commits covers
too much; narrow its `covers`.

## Voice

- Say what exists, why it exists and how it works.
- Neutral and precise. No self-praise and no evaluative adjectives: not "fast", "elegant"
  or "robust", but what was measured.
- Every number says where it was measured: the commit, the cell or the configuration, and
  the machine.
- Do not hardcode counts that the code defines (columns, levels, presets): point at the
  constant or embed the generated table.
- No design-debate history. Alternatives considered and rejected belong in `docs/rewrite/`.
- Short sentences and full words. Abbreviations only where universal (`cos`, `x`, `y`).
- Name things as the code names them.

## Checks

`tests/docs` runs with the rest of the suite:

| Check | Test | Fails when |
|---|---|---|
| strict build | `tests/docs/test_build.py` | a link, page, nav entry, anchor, mkdocstrings target or generated table is broken (needs the docs extra) |
| code references | `tests/docs/test_references.py` | a backticked name or path, a `:::` target or a Python block's import does not resolve; a page has no `covers`; a block has no run marker; a table name is unknown |
| examples | `tests/docs/test_examples.py` | a runnable example fails |
| coverage map | `tests/docs/test_staleness.py` | never; it warns |

The docstring report lists public modules, classes and functions whose docstring is
missing or one word:

<!-- runnable -->
```bash
uv run --no-sync python -m tests.docs.docstrings
```
