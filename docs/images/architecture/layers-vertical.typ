// The two layers of a Yemong block, for a paper column: two separate panels, each read
// from the bottom up as a pre-norm residual pair (norm, mixer, sum; norm, MLP, sum).
#import "diagram.typ": *

#let side-width = 92 // the left column: the map memory, or the note on the mixer
#let main-width = 176
#let pad = 12
#let width = pad + side-width + 20 + main-width + 34 + pad
#let x = pad + side-width + 20 + main-width / 2 // the residual stream
#let side = pad + side-width / 2
#let lane = x + main-width / 2 + 18 // the skip lines
#let norm-height = 30
#let box-height = 48
#let title-room = 34 // above the last sum: the panel's title sits left of the stream

// A panel's height, from its mixer's.
#let panel-height(mixer-height) = (
  18 + 16 + 2 * norm-height + 2 * 22 + mixer-height + 2 * 22 + 4 * add-radius + 32
    + box-height + title-room
)

// One panel with its bottom edge at `bottom`; returns nothing, draws everything.
#let panel(bottom, title, subtitle, style, mixer, mixer-kind, mixer-height, left) = {
  let split-one = bottom - 18
  let norm-one = split-one - 16 - norm-height / 2
  let mixer-y = norm-one - norm-height / 2 - 22 - mixer-height / 2
  let add-one = mixer-y - mixer-height / 2 - 22 - add-radius
  let split-two = add-one - add-radius - 16
  let norm-two = split-two - 16 - norm-height / 2
  let mlp = norm-two - norm-height / 2 - 22 - box-height / 2
  let add-two = mlp - box-height / 2 - 22 - add-radius
  let top = add-two - add-radius - title-room

  group(4, top, width - 4, bottom, title, subtitle: subtitle, style: style)
  card(x, norm-one, main-width, norm-height, "RMSNorm", kind: "norm")
  card(x, mixer-y, main-width, mixer-height, mixer.at(0), lines: mixer.slice(1), kind: mixer-kind)
  add(x, add-one)
  card(x, norm-two, main-width, norm-height, "RMSNorm", kind: "norm")
  card(x, mlp, main-width, box-height, "Gated MLP", lines: ("GELU gate, 4 × width",))
  add(x, add-two)

  wire((x, bottom + 10), (x, norm-one + norm-height / 2))
  wire((x, norm-one - norm-height / 2), (x, mixer-y + mixer-height / 2))
  wire((x, mixer-y - mixer-height / 2), (x, add-one + add-radius))
  wire((x, add-one - add-radius), (x, norm-two + norm-height / 2))
  wire((x, norm-two - norm-height / 2), (x, mlp + box-height / 2))
  wire((x, mlp - box-height / 2), (x, add-two + add-radius))
  wire((x, add-two - add-radius), (x, top - 10))
  wire((x, split-one), (lane, split-one), (lane, add-one), (x + add-radius, add-one))
  wire((x, split-two), (lane, split-two), (lane, add-two), (x + add-radius, add-two))
  left(mixer-y, mixer-height)
}

#let gap = 44 // between the panels
#let temporal-bottom = 14 + panel-height(58)
#let spatial-bottom = temporal-bottom + gap + panel-height(box-height)

#show: canvas.with(width, spatial-bottom + 14)

#panel(
  temporal-bottom, "Temporal layer", "per token, across ticks", "temporal",
  ("Recurrent mixer", "linear, causal conv, RG-LRU;", "times a GELU gate, linear"),
  "temporal", 58,
  (y, h) => {
    // The recurrence: the state carries over to the next tick.
    let edge = x - main-width / 2
    wire((edge, y - 10), (edge - 14, y - 10), (edge - 14, y + 10), (edge, y + 10), radius: 5)
    let note = edge - 26
    label(note, y - 14, [*Griffin's* block], anchor: "right", colour: text-dark)
    label(note, y, [(RG-LRU) by default;], anchor: "right")
    label(note, y + 14, [*GRU* as an option], anchor: "right", colour: text-dark)
  },
)

#panel(
  spatial-bottom, "Spatial layer", "per tick, across tokens", "spatial",
  ("Multi-head attention", "rotary position and heading"),
  "attention", box-height,
  (y, h) => {
    card(side, y, side-width, 54, "Map memory", lines: ("fields and zones,", "keys and values"), kind: "input", size: 10)
    wire((side + side-width / 2, y), (x - main-width / 2, y))
  },
)
