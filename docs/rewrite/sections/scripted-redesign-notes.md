# Scripted redesign: notes

Brief: `scripted-redesign-brief.md`. Evidence: `scripted-softness-notes.md`.

## The scale floors

Each logistic's smallest scale is 0.1 times the standard deviation of the natural logarithm
of its input, measured over the earlier (linear-ramp) agent's self-play on the CPU, 16
games per preset, 1500 decisions per seat (the softness audit's recorder; inputs below
their 1% quantile read as it, so an exact 0 does not dominate). The smallest over the three
presets, at the combat layer's site:

| Logistic (input) | std of ln(input) by preset (frontline, deathmatch, deathmatch_wide) | Floor |
|---|---|---|
| turn, level two, level three (angle error / stall angle) | 1.58, 2.46, 2.73 | 0.158 |
| boost slow (speed / corner speed) | 0.33, 0.31, 0.35 | 0.031 |
| boost far (store x distance / gun range) | 2.15 (1.43 at the Frontline layer), 1.83, 2.43 | 0.143 |
| reverse, shoot (distance / gun range) | 1.27, 1.35, 1.94 | 0.127 |
| aim (angle error / target size) | 2.05, 2.18, 2.31 | 0.205 |

The logistics are in the logarithm of their input: the inputs are positive ratios with
heavy upper tails (the aim error's 95% is 40 to 119 target sizes against a median of 0.5
to 12), so a floor of 0.1 x the linear standard deviation would have been 1.9 target sizes
for the aim, wider than the whole useful aiming window.
