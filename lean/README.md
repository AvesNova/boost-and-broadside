# Lean core

A lean rewrite of boost-and-broadside in PyTorch: the v2 behaviour without its bloat. One fixed map,
elimination with no shrink, a point-mass flight model, a scripted opponent, an attention + recurrent
policy trained by behaviour cloning and then PPO against a league, and a runner that rates the
results. Plain eager code that also compiles.

| Package | What it is | Notes |
|---|---|---|
| `contract.py` | The shared types (`State`, `StepOut`, `Obs`, `Constants`) and action helpers | `CONTRACT.md` |
| `game/` | The game step, the ship card, spawns, fog, the flight tuning analysis | `game/NOTES.md` |
| `scripted/` | The scripted opponent (yardstick, BC teacher, personas) and a weaker chaser | `scripted/NOTES.md` |
| `model/` | The policy: gated rotary attention, a Griffin recurrent block per token, categorical heads | `model/NOTES.md` |
| `learn/`, `train.py` | Rollout, the BC stage, PPO with a shrinking teacher term, the league, checkpoints | `learn/NOTES.md` |
| `eval/` | The evaluation runner, the game store, Elo, the crossover | `eval/NOTES.md` |
| `sandbox/`, `tool/` | The flight sandbox's world and the local web tool (tuning charts, fly the game) | `sandbox/NOTES.md`, `tool/NOTES.md` |

Where things are written down:

- Decisions (what lean is for, every deliberate change to v2): `docs/rewrite/core-decisions.md`.
- Planned work and status: `docs/rewrite/core-backlog.md`. What was done, when and how it was
  checked: `docs/rewrite/core-worklog.md`. Open items for the author: `docs/rewrite/author-review.md`
  and `docs/questions.md`.
- Measured speed, memory and compile time: `docs/rewrite/scoreboard.md`.

## Commands

Run from the repository root, each with `uv run --no-sync` in front and `PYTHONPATH=.:src`.

    pytest lean -m "not slow" -q                    # the fast tests, CPU; -m slow adds the compiled ones
    ruff format lean && ruff check lean             # format and lint at the project line length (100); run often
    python -m lean.train --smoke --device cpu       # BC then RL, a few tiny eager updates
    python -m lean.train --out DIR                  # BC (96 updates), then RL (1,000 updates)
    python -m lean.train --stage bc --out DIR       # BC only: DIR/bc.pt
    python -m lean.train --stage rl --start DIR/bc.pt --out DIR   # RL from the BC hand-off: DIR/rl.pt
    python -m lean.train --resume DIR [--updates N] # continue from DIR's latest checkpoint

    python -m lean.eval play STORE yardstick net:DIR/rl.pt:ema --games 512   # agents: lean/eval/NOTES.md
    python -m lean.eval report STORE yardstick net:DIR/rl.pt
    python -m lean.eval elo STORE --sizes 4 4
    python -m lean.eval crossover STORE DIR/rl.pt --sizes 4 8 --games 64

    uv sync --extra tools                           # once: fastapi, uvicorn, websockets
    python -m lean.tool [--port 8000]               # web tool at http://127.0.0.1:8000/

    PYTHONPATH=. python docs/rewrite/sections/scoreboard/compile_check.py   # the compile tripwire

Training and evaluation use a GPU when there is one, and training compiles; `--device cpu` and `--no-compile`
run eagerly. The compile tripwire needs the GPU (about 4 minutes on the laptop's 4070): run it at the
end of every step that touches `lean/`; it fails on a graph break, a recompile, or a function's
traced operations growing by more than 10%. Set `OMP_NUM_THREADS` to a few threads for small CPU runs.
A checkpoint plays in the evaluation as `net:CHECKPOINT[:live|ema|uniform]`.
