# Bullet life follows the gun range: progress

## 2026-11-12, part A (bullet life per type)

- Tried: the life is a card field (`bullet_life`, seconds, whole ticks), solved in
  `handling.weapons.combat_card` from the gun range: t = expm1(k R) / (k w0) rounded up to a
  tick (tolerance 1e-4 ticks for a range measured off a card), then the drag scale trims
  the range to R. Launch reads the shooter's life. `rules.bullet_life` is gone; the
  bullet drag is a live (log) slider in the Bullets group; `Rules.bullet_moves` became
  `Rules.longest_bullet_moves`, the card's longest life (static sizes: shot labels,
  lookahead, the model's ages). Cap: `LONGEST_BULLET_LIFE` 2 s (60 slots at 30 Hz in a
  tunable match).
- Worked: every type of every set (standard, design-seed, real-scale: 32 types) solves to
  exactly today's life (t* = 12.000 / 18.000 ticks to 4e-15), drag scale 1: no gameplay
  change. The game fingerprint at 45 slots equals the recorded one; it moves only with the
  slot count (60).
- Model/learner: Yemong's `bullet_moves` is the card's longest life (18 / 12 unchanged for
  every set), so the shot ages and their normalisation are unchanged. Shot labels now
  mask hits and closest passes past the shooter's own life (a no-op while every type has
  one life).
- Next: part B (bullet display options), then part C (head-on angle).

## 2026-11-12, part B (bullet display options)

- Done: `render.bullets.BulletLook` (style tracer or trail only, head radius, trail width
  at the head and the tail, trail seconds), defaults today's tracer; the session holds one,
  set from `bnb play --bullet-*` options and live in Settings > Bullets (style buttons and
  sliders; the head's size hides for trail only). Widths are screen pixels, not scaled by
  the interface scale, as every width in the scene (keeps today's look at any scale).
- Worked: render tests (trail only draws no head, takes its widths and length) and an
  interface test (the settings switch the style and set a width); an offscreen run with
  `--bullet-style trail` renders.
- Next: part C (head-on angle).
