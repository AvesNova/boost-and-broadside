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
