// The two layers of a Yemong block, for a web page: two separate panels, each read from
// left to right as a pre-norm residual pair (norm, mixer, sum; norm, MLP, sum).
#import "diagram.typ": *

#let box-height = 58
#let norm-width = 78
#let mixer-width = 176
#let mlp-width = 130
#let title-room = 44 // the panel's title, above the skip lines
#let below = 82 // under the boxes: the map memory, or the note on the mixer
#let panel-height = title-room + 18 + box-height + below

#let left-edge = 14
#let split-one = left-edge + 20
#let norm-one = split-one + 16 + norm-width / 2
#let mixer = norm-one + norm-width / 2 + 22 + mixer-width / 2
#let add-one = mixer + mixer-width / 2 + 22 + add-radius
#let split-two = add-one + add-radius + 16
#let norm-two = split-two + 16 + norm-width / 2
#let mlp = norm-two + norm-width / 2 + 22 + mlp-width / 2
#let add-two = mlp + mlp-width / 2 + 22 + add-radius
#let right-edge = add-two + add-radius + 20
#let width = right-edge + 14

// One panel with its top edge at `top`.
#let panel(top, title, subtitle, style, mixer-lines, mixer-kind, under) = {
  let lane = top + title-room
  let y = lane + 18 + box-height / 2
  let bottom = top + panel-height

  group(left-edge, top, right-edge, bottom, title, subtitle: subtitle, style: style)
  card(norm-one, y, norm-width, box-height, "RMSNorm", kind: "norm")
  card(mixer, y, mixer-width, box-height, mixer-lines.at(0), lines: mixer-lines.slice(1), kind: mixer-kind)
  add(add-one, y)
  card(norm-two, y, norm-width, box-height, "RMSNorm", kind: "norm")
  card(mlp, y, mlp-width, box-height, "Gated MLP", lines: ("GELU gate,", "4 × width"))
  add(add-two, y)

  wire((left-edge - 10, y), (norm-one - norm-width / 2, y))
  wire((norm-one + norm-width / 2, y), (mixer - mixer-width / 2, y))
  wire((mixer + mixer-width / 2, y), (add-one - add-radius, y))
  wire((add-one + add-radius, y), (norm-two - norm-width / 2, y))
  wire((norm-two + norm-width / 2, y), (mlp - mlp-width / 2, y))
  wire((mlp + mlp-width / 2, y), (add-two - add-radius, y))
  wire((add-two + add-radius, y), (right-edge + 10, y))
  wire((split-one, y), (split-one, lane), (add-one, lane), (add-one, y - add-radius))
  wire((split-two, y), (split-two, lane), (add-two, lane), (add-two, y - add-radius))
  under(y + box-height / 2)
}

#let gap = 36 // between the panels
#show: canvas.with(width, 2 * panel-height + gap + 8)

#panel(
  4 + panel-height + gap, "Spatial layer", "per tick, across tokens", "spatial",
  ("Multi-head attention", "rotary position and heading"), "attention",
  edge => {
    card(mixer, edge + 30 + 22, 150, 44, "Map memory",
      lines: ("fields and zones: keys, values",), kind: "input", size: 10)
    wire((mixer, edge + 30), (mixer, edge))
  },
)

#panel(
  4, "Temporal layer", "per token, across ticks", "temporal",
  ("Recurrent mixer", "linear, causal conv, RG-LRU;", "times a GELU gate, linear"), "temporal",
  edge => {
    // The recurrence: the state carries over to the next tick.
    wire((mixer + 10, edge), (mixer + 10, edge + 14), (mixer - 10, edge + 14), (mixer - 10, edge), radius: 5)
    label(mixer, edge + 36, [*Griffin's* block (RG-LRU) by default;], colour: text-dark)
    label(mixer, edge + 50, [*GRU* as an option], colour: text-dark)
  },
)
