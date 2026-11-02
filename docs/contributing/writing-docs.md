---
covers: [docs/hooks/generated.py, tests/docs/]
reviewed: 2026-11-02
---

# Writing docs

The docs are a MkDocs Material site built from `docs/`, with the reference generated from
the code by mkdocstrings. `docs/rewrite/` holds the internal design records; it is outside
the site and these rules do not apply to it.

## Build and preview

<!-- not-runnable: starts a server -->
```bash
uv sync                                              # includes the docs group
uv run --no-sync mkdocs serve                        # preview at http://127.0.0.1:8000
uv run --no-sync mkdocs build --strict --site-dir /tmp/bnb-site
```

`mkdocs build --strict -q` exits 0 even when there are warnings: quiet mode hides the
warnings strict mode counts. Build without `-q` to check by hand. The suite checks the docs
on every run (`tests/docs`); see [Checks](#checks).

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

Every page appears in `nav`{.plain} in `mkdocs.yml`. A page left out of the nav fails the build.

## Linking code

- **Full names:** write the full dotted name in backticks, `bnb.game.api.reset`. The checks
  import every backticked name that starts with `bnb.` or `bench.`, so a renamed or deleted
  name fails the suite. The build turns each one into a link to its reference entry, and
  the strict build fails if the reference has no such entry.
- **Short names:** `Rules.pools`, `GameConfig` or `MAX_SHIPS` are resolved against an index
  of the code's classes, functions, members and constants in `src/bnb`, `bench`, `tests`
  and the docs hook (`tests/docs/names.py`). A short name must name exactly one thing;
  write an ambiguous one in full (`bnb.contracts.config.MODES`, not `MODES`{.plain}).
- **Lowercase words** in backticks (`shield_lost`, `fog_share`, `team_relative`) must be
  words the code uses: names, attributes, arguments or string constants such as column
  names.
- **Not code:** a backticked word that is not a Python name, such as a file format or a
  configuration key, opts out with `{.plain}` right after the span:
  `` `nav`{.plain} ``. The check skips it, and the page renders it as code.
- **Commands:** every `bnb ...` line, in a fence or in backticks, runnable or not, must
  name a command (`bnb.cli.COMMANDS`) and is parsed against that command's options
  (`bnb.cli.Train`, `bnb.cli.Runs`, `bnb.cli.ShipCard`), with known values for `--preset`
  and `--mode`.
- **No reference entry:** a full name the reference does not render, such as a constant
  assigned in a tuple, takes `{.nolink}`{.plain} after the span. It is still checked, but not
  linked.
- **Links into the reference** by hand: `[reset][bnb.game.api.reset]`.
- **Repository paths:** write them in backticks from the root, `src/bnb/game/rules.py` or
  `tests/sections/test_modes.py`. A test can be named as
  `tests/sections/test_modes.py::test_spawn_layout_does_not_reveal_the_seat`. The checks fail
  if the path, or the test, does not exist. A bare file name (`pyproject.toml`) must exist
  somewhere in the repository.
- **Pages:** link with relative paths to the `.md` file, `[Modes](../explanation/game/modes.md)`.
- **Terms:** link a term's first use on a page to the [glossary](../explanation/glossary.md).

`README.md` is checked like a page (names, paths, commands, fences and its links), but it is
read on GitHub without the hook: it cannot hold generated blocks, so it links to the pages
that do instead of repeating them.

## Generated blocks

Values that live in the code or in a data file are generated at build time, never copied
by hand. Put the marker on a line of its own:

```markdown
<!-- generated: ship-columns -->
```

The build replaces it with the block, built by `docs/hooks/generated.py`. The blocks:

<!-- generated: block-names -->

To add a block, write a function in the hook that imports the value from `bnb` (or reads
`pyproject.toml` or a data file) and returns Markdown, and add it to `BLOCKS`. A function
with a `page` argument receives the page's path, for links. Where a value's unit or meaning
is a comment in the source, the dataclass tables read that comment, without any bracketed
spec key; keep such comments short.

Two data files hold what the code does not:

- `docs/roadmap.yml`: each section's title, status, design record and page. Every status on
  the site comes from it; when a section's status changes, edit this file, not prose.
- `docs/packages.yml`: what each package holds, its reference page, or the roadmap section
  that will build it. The layers come from `LAYER` in `tests/contract/test_imports.py`.

## Fences and examples

Every fence names its language: `python`, `bash`, `markdown`, `text`, `yaml` and so on.
A `python`, `bash` or `text` block, or any block whose first word is `uv`, `bnb` or
`python`, says whether it runs, on the line before the fence:

```markdown
<!-- runnable -->
<!-- not-runnable: needs CUDA -->
```

- A **runnable** block runs in the suite, from the repository root, with CUDA hidden: a
  Python block in a fresh interpreter, a command block in bash. It must finish on the CPU
  in seconds and leave the repository as it was.
- A **not-runnable** block gives its reason: it needs a GPU, the benchmark lock, minutes,
  or it changes the environment. Its Python still has to parse, its `bnb` imports have to
  resolve, and its `bnb` commands have to use real options.

## What a page covers

Each page declares in its front matter the code it describes:

```markdown
---
covers: [src/bnb/game/physics/, src/bnb/game/rules.py]
---
```

A page that describes no particular code declares `covers: []`. The paths must exist. The
suite warns that a page may be stale when a covered path changed in a commit after the
page's last commit, or has uncommitted changes while the page has none, so the warning
shows before you commit. The report on its own:

<!-- runnable -->
```bash
uv run --no-sync python -m tests.docs.staleness
```

When you change covered code, reread the pages that cover it and fix what changed. If a
page is still correct, set `reviewed:` in its front matter to the date and commit the page.
Nothing reads `reviewed:`; it exists only so that confirming a page is a change to commit,
because the warning clears once the page has a commit newer than the code. A page that is
warned about on most commits covers too much; narrow its `covers`.

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

## Code rules

The code follows [Code style](code-style.md).

## Checks

`tests/docs` runs with the rest of the suite:

| Check | Test | Fails when |
|---|---|---|
| strict build | `tests/docs/test_build.py` | a link, anchor, page, nav entry, reference link (including every linked `bnb.` name) or generated block is broken |
| code references | `tests/docs/test_references.py` | a full name, short name, word, path, `:::` target, `bnb` command or Python import does not resolve; a fence has no language or no run marker; a page has no `covers`; a block name is unknown; a README link is broken |
| examples | `tests/docs/test_examples.py` | a runnable example fails |
| coverage map | `tests/docs/test_staleness.py` | never; it warns, for commits and uncommitted changes |

The docstring report lists public modules, classes and functions whose docstring is
missing or one word:

<!-- runnable -->
```bash
uv run --no-sync python -m tests.docs.docstrings
```
