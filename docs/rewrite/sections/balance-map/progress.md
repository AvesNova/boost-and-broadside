# Balance map: progress (2026-11-16)

Branch `feat/balance-map`: c4ede6c8 (feat/lethality-sticky, game v23) + origin/feat/balance-eval
(5bcb20bb, merged clean), then **Imugi build 2, ede616c4** merged clean (coordinator, 18:40:
offset approach, lead turn, one-or-two-circle choice; the cards are unchanged, VERSION 23).
The search uses build 2. Code: `bench/balance_map.py`; tests `tests/sections/test_balance_map.py`.

## 18:10-18:50 set-up, search space, the solver's cost

- Built: worktree /home/vizia/avesnova/bnb-map (`.venv` linked), `ribs` 0.12.1 in the `tools`
  extra (the PyPI name is `ribs`; `pyribs` is a stub). On box5090 installed additively into
  /workspace/venv with `uv pip install ribs==0.12.1` (torch unchanged, 2.15.0+cu132); no
  `uv sync` there, since the main checkout's lock is another branch's.
- Search space: 13 handles, each a coordinate in [0, 1] of real-WWII-type bounds (log where
  ratios matter): sustained turn 0.50-0.85 rad/s, nose reversal 0.25-0.55 s, low-speed
  acceleration 8.8-20, climb time at the v23 store 200-500 s, cruise 260-360 m/s, stall 75-106
  m/s, g limit 6-9, shield 344-803, hit damage 10.3-24.0, gun range 400-700 m, store x0.5-2,
  shot cost x0.5-2, dive acceleration 38-50. Every other handle is v23 Jujak's; 30 Hz fire.
- Time to kill (lethality-sticky's definition): the reference's 0.6-1.4 s on the candidate
  sets the shield bounds, the candidate's on the reference the damage bounds, and the damage
  is clipped so the candidate's own mirror is inside the band too. **Not guaranteed between
  two candidates**: the band holds for a pair only if (shield spread) x (damage spread) <=
  1.4 / 0.6 = 2.33; checked when picks are chosen.
- Didn't: the handle solver on free handle sets. 22 of 30 uniform random cards fail, all on
  the low-speed acceleration (the engine's power, which cruise, zero-lift drag and climb set,
  over the stall speed caps it), and a solve is slow on the CPU: 39 s for Jujak (3
  iterations, 1 card), 627 s for 8 cards (16 iterations: a batch runs until its last card
  converges or fails).
- Worked: the cap's closed form (`low_speed_limits`, from `solver.first_guess`; Jujak 21.0
  against its 13.5, Yongma 12.7 against 8.8): the low-speed acceleration is clipped to 85% of
  it, and a card whose cap is under the real floor 8.8 is dropped before the solver (counted).
- Next: the solve's speed on the box (GPU and CPU), then the pilot.
