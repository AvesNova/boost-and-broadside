// The model at a glance, for a paper column: inputs at the bottom, heads at the top.
#import "diagram.typ": *

#let width = 360
#let x = width / 2
#let card-width = 210
#let card-height = 44
#let gap = 34 // between a card and the next along the path

// The heads, a 2 x 2 grid in a group at the top.
#let heads-top = 4
#let head-width = 150
#let head-gap = 18
#let left-head = x - head-gap / 2 - head-width / 2
#let right-head = x + head-gap / 2 + head-width / 2
#let row-one = heads-top + 58
#let row-one-height = 40
#let row-two = row-one + row-one-height / 2 + 14 + 28
#let row-two-height = 66
#let heads-bottom = row-two + row-two-height / 2 + 12

// The block group, then the encoder and the inputs below it.
#let block-top = heads-bottom + gap
#let spatial = block-top + 50 + card-height / 2
#let temporal = spatial + card-height + gap + 2 * stack-offset
#let block-bottom = temporal + card-height / 2 + 14
#let encoder = block-bottom + gap + 2 * stack-offset + card-height / 2
#let tokens = encoder + card-height + gap + 2 * stack-offset
#let observation = tokens + card-height + gap
#let height = observation + card-height / 2 + 2

#show: canvas.with(width, height)

#group(4, heads-top, width - 4, heads-bottom, "Heads", style: "heads")
#card(left-head, row-one, head-width, row-one-height, "Action head",
  lines: ("joint-action logits",), stacked: true, badge: "× own ships")
#card(right-head, row-one, head-width, row-one-height, "Value head (critic)",
  lines: ("a categorical value per level",), stacked: true, badge: "× own ships")
#card(left-head, row-two, head-width, row-two-height, "Outcome head",
  lines: ("global token:", "win, tie, loss, unresolved"))
#card(right-head, row-two, head-width, row-two-height, "Auxiliary heads",
  lines: ("next state, forecast error,", "enemy action, density,", "shots, geometry"), kind: "auxiliary", dashed: true)

#group(x - 150, block-top, x + 150, block-bottom, "Yemong block × 2",
  subtitle: "temporal, then spatial × 2")
#card(x, temporal, card-width, card-height, "Temporal layer",
  lines: ("recurrence across ticks, per token",), kind: "temporal", stacked: true, badge: "× tokens")
#card(x, spatial, card-width, card-height, "Spatial layer × 2",
  lines: ("attention across ships, per tick",), kind: "attention")

#card(x, encoder, card-width, card-height, "Encoder",
  lines: ("per-kind projection, then shared",), stacked: true, badge: "× tokens")
#card(x, tokens, card-width, card-height, "Belief and tokens",
  lines: ("a token per ship, one global token",), kind: "input", stacked: true, badge: "× tokens")
#card(x, observation, card-width, card-height, "Observation",
  lines: ("ships, fields, zones, mode",), kind: "input")

#wire((x, observation - card-height / 2), (x, tokens + card-height / 2))
#wire((x, tokens - card-height / 2), (x, encoder + card-height / 2))
#wire((x, encoder - card-height / 2), (x, block-bottom))
#wire((x, block-bottom), (x, temporal + card-height / 2), head: false)
#wire((x, temporal - card-height / 2 - 2 * stack-offset), (x, spatial + card-height / 2))
#wire((x, spatial - card-height / 2), (x, heads-bottom))
