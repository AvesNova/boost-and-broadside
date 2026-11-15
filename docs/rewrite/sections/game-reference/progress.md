# Game reference

- 2026-11-15: the hashed `FINGERPRINT` is replaced by `tests/sections/game_reference.pt`
  (final state + observation leaves of each preset's 48-tick match, with the VERSION it was
  made at; 304 KB). Floats compare at rtol 1e-4 / atol 1e-6, integers and flags exactly; a
  VERSION mismatch fails. Regenerate: `uv run --no-sync python -m tests.sections.test_game_version --write`.
- Generated on the laptop; not yet checked on another machine.
