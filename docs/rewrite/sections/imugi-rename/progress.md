# Imugi rename: progress

2026-11-15, branch `feat/imugi-rename` from `integrate/next-runs`.

**Tried / worked**
- Modules: `controllers/scripted_<part>.py` is now `imugi_<part>.py` (combat, defaults, doctrine,
  frontline, memory, parameters and the three JSON files); the agent itself moved out of
  `controllers/scripted.py` into `controllers/imugi.py`. `scripted.py` keeps the chaser,
  random and the helpers (`floored_actions`, `stick_toward`, ...), which are not Imugi.
  `scripted_cheese.py` is `cheese.py`; `runtime/scripted_step.py` is `imugi_step.py`.
- Identifiers: `Scripted` is `Imugi`; `ScriptedCarry`, `ScriptedParameters`, `ScriptedPlay`,
  `ScriptedStep`, `ScriptedAgent` (play) take the `Imugi` prefix; `SCRIPTED_PLAYER`,
  `SCRIPTED_KIND`, `SCRIPTED_ENTRY` are `IMUGI_*`; the play controller kind `scripted_agent`
  is `imugi`.
- Command line: `imugi`, `imugi@T=<t>,floor=<f>`, `cheese:<name>`.
- Records: the recorded identity text is unchanged (`scripted`, `scripted@T=..`,
  `scripted:<name>`; seat kind and pool names `scripted`; random-stream salts
  `"scripted agent flags"` and `"scripted <name>"`; the cache directory for built steps is
  new, which only costs one rebuild). `bnb.runtime.opponents.stored_name` maps the command-line
  names to the identities in one place, `matches.player` applies it, and a record's own names
  still parse. Evaluation `settings.json` stores the opponent as the identity.
- Docs: `docs/explanation/imugi.md` (was `scripted-agent.md`); "the scripted agent" reads
  "Imugi" with the full phrase at first mention on a page. Dated logs and working notes under
  `docs/rewrite/sections/` keep their wording apart from module paths.

**Didn't / left**
- The chaser's play kind stays `scripted`; printed leaderboard rows show `imugi` through
  `shown_name`, other printouts show the recorded identity.
