# White paper

[`whitepaper.pdf`](whitepaper.pdf) is a seven-page summary of the v1 results: the
environment, the policy, training, the zero-shot fleet-size results for
`wild-universe-731`, how they were measured, and their limitations.

It is a hand-written snapshot, not a rendered page. Its numbers were typed in from
[`evaluation.md`](../evaluation.md) and the other docs and checked against them on
2026-11-23; they are not read from [`values.json`](../values.json), and
`scripts/render_docs.py --check` does not cover them. If the v1 results are ever
re-measured, this paper has to be updated by hand.

- `whitepaper.html` is the source and `assets/` holds the replay frames it uses; the
  charts are linked from where they already live in the repository.
- `build.js` renders the PDF with headless Chromium:

  ```bash
  node docs/portfolio/build.js
  ```

  The committed PDF has no email address. `CONTACT_EMAIL` adds one to the byline and
  `OUT` writes the result elsewhere, for a copy to send with an application.
