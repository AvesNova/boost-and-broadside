// The model at a glance, in one row for a web page: inputs on the left, heads on the right.
#import "diagram.typ": *

#let card-width = 128
#let card-height = 58
#let gap = 30 // between a card and the next along the path
#let stacked-room = 2 * stack-offset

// The heads, a column in a group on the right; the path runs along its middle.
#let head-width = 170
#let heads-top = 4
#let row-one = heads-top + 66
#let row-two = row-one + 74
#let row-three = row-two + 20 + 14 + 23
#let row-four = row-three + 23 + 14 + 28
#let heads-bottom = row-four + 28 + 12
#let y = (heads-top + heads-bottom) / 2

#let observation = 4 + card-width / 2
#let tokens = observation + card-width + gap
#let encoder = tokens + card-width + stacked-room + gap
#let block-left = encoder + card-width / 2 + stacked-room + gap
#let spatial = block-left + 14 + card-width / 2
#let temporal = spatial + card-width + gap
#let block-right = temporal + card-width / 2 + stacked-room + 14
#let heads-left = block-right + gap
#let head-x = heads-left + 12 + head-width / 2
#let heads-right = head-x + head-width / 2 + stacked-room + 12
#let width = heads-right + 4

#show: canvas.with(width, heads-bottom + 4)

#card(observation, y, card-width, card-height, "Observation",
  lines: ("ships, fields,", "zones, mode"), kind: "input")
#card(tokens, y, card-width, card-height, "Belief and tokens",
  lines: ("a token per ship,", "one global token"), kind: "input", stacked: true, badge: "× tokens")
#card(encoder, y, card-width, card-height, "Encoder",
  lines: ("per-kind projection,", "then shared"), stacked: true, badge: "× tokens")

#group(block-left, y - card-height / 2 - 62, block-right, y + card-height / 2 + 14,
  "Yemong block × L", subtitle: "spatial, then temporal")
#card(spatial, y, card-width, card-height, "Spatial layer",
  lines: ("attention across", "ships, per tick"), kind: "attention")
#card(temporal, y, card-width, card-height, "Temporal layer",
  lines: ("recurrence across", "ticks, per token"), kind: "temporal", stacked: true, badge: "× tokens")

#group(heads-left, heads-top, heads-right, heads-bottom, "Heads", style: "heads")
#card(head-x, row-one, head-width, 40, "Action head",
  lines: ("joint-action logits",), stacked: true, badge: "× own ships")
#card(head-x, row-two, head-width, 40, "Value head (critic)",
  lines: ("a categorical value per level",), stacked: true, badge: "× own ships")
#card(head-x, row-three, head-width, 46, "Outcome head",
  lines: ("global token: win, tie, loss",))
#card(head-x, row-four, head-width, 56, "Auxiliary heads",
  lines: ("next state, error, enemy action,", "density, shots, geometry"), kind: "auxiliary", dashed: true)

#wire((observation + card-width / 2, y), (tokens - card-width / 2, y))
#wire((tokens + card-width / 2 + stacked-room, y), (encoder - card-width / 2, y))
#wire((encoder + card-width / 2 + stacked-room, y), (spatial - card-width / 2, y))
#wire((spatial + card-width / 2, y), (temporal - card-width / 2, y))
#wire((temporal + card-width / 2 + stacked-room, y), (heads-left, y))
