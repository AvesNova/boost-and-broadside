// Drawing helpers for the architecture diagrams (render.py compiles every other .typ here).
//
// Coordinates are numbers in points, x to the right and y down, from the canvas's top left;
// a card is placed by its centre. Colours follow v1's figure: inputs grey, projections and
// MLPs blue, norms yellow, attention salmon, the temporal mixer green, groups dashed.

#let ink = rgb("#374151")
#let text-dark = rgb("#111827")
#let text-soft = rgb("#4b5563")
#let badge-blue = rgb("#1d4ed8")

#let kinds = (
  input: (fill: rgb("#e5e7eb"), stroke: rgb("#9ca3af")),
  linear: (fill: rgb("#dbeafe"), stroke: rgb("#60a5fa")),
  norm: (fill: rgb("#fef3c7"), stroke: rgb("#f5b301")),
  attention: (fill: rgb("#fcc3b6"), stroke: rgb("#f26b4f")),
  temporal: (fill: rgb("#ccf5e1"), stroke: rgb("#2fbf84")),
  auxiliary: (fill: rgb("#f1f6ff"), stroke: rgb("#93c5fd")),
)

#let groups = (
  block: (fill: rgb("#f7f8ff"), stroke: rgb("#a5b4fc"), title: rgb("#1e3a8a")),
  heads: (fill: rgb("#fbfcfe"), stroke: rgb("#cbd5e1"), title: rgb("#334155")),
  spatial: (fill: rgb("#fff6f4"), stroke: rgb("#f26b4f"), title: rgb("#b4321a")),
  temporal: (fill: rgb("#f4fdf8"), stroke: rgb("#2fbf84"), title: rgb("#11734b")),
)

#let stack-offset = 5 // points between the stacked cards of a per-token card
#let line-width = 1.3pt

// The page: tight around a canvas of the given size, a sans-serif face throughout.
#let canvas(width, height, body) = {
  set page(width: width * 1pt + 8pt, height: height * 1pt + 8pt, margin: 4pt, fill: white)
  set text(font: ("Noto Sans", "DejaVu Sans"), size: 10pt, fill: text-dark)
  box(width: width * 1pt, height: height * 1pt, body)
}

#let at(x, y, body) = place(top + left, dx: x * 1pt, dy: y * 1pt, body)

// A card centred at (x, y): a title and optional lines below it. stacked draws two cards
// behind it (one per token or ship), badge labels the stack above its top right corner.
#let card(
  x, y, w, h, title,
  lines: (), kind: "linear", stacked: false, badge: none, dashed: false, size: 10.5,
) = {
  let colour = kinds.at(kind)
  let stroke = (paint: colour.stroke, thickness: line-width, dash: if dashed { "dashed" } else { none })
  let shape(body) = rect(
    width: w * 1pt, height: h * 1pt, radius: 4pt, fill: colour.fill, stroke: stroke, inset: 2pt, body,
  )
  let left-edge = x - w / 2
  let top-edge = y - h / 2
  if stacked {
    for depth in (2, 1) {
      at(left-edge + depth * stack-offset, top-edge - depth * stack-offset, shape([]))
    }
  }
  at(left-edge, top-edge, shape(align(center + horizon, {
    text(size: size * 1pt, weight: "semibold", title)
    for line in lines {
      linebreak()
      text(size: (size - 2) * 1pt, fill: text-soft, line)
    }
  })))
  if badge != none {
    let right-edge = x + w / 2 + (if stacked { 2 * stack-offset } else { 0 })
    let above = top-edge - (if stacked { 2 * stack-offset } else { 0 }) - 13
    at(right-edge - 200, above, box(width: 200pt, align(right, text(
      size: 8.5pt, weight: "bold", fill: badge-blue, badge,
    ))))
  }
}

// A dashed group box from (x0, y0) to (x1, y1), titled at its top left.
#let group(x0, y0, x1, y1, title, subtitle: none, style: "block", size: 12) = {
  let colour = groups.at(style)
  at(x0, y0, rect(
    width: (x1 - x0) * 1pt, height: (y1 - y0) * 1pt, radius: 7pt, fill: colour.fill,
    stroke: (paint: colour.stroke, thickness: 1.2pt, dash: (4pt, 3pt)),
  ))
  at(x0 + 10, y0 + 7, {
    text(size: size * 1pt, weight: "bold", fill: colour.title, title)
    if subtitle != none {
      linebreak()
      text(size: 8.5pt, fill: colour.title, subtitle)
    }
  })
}

// Plain text placed with its centre (anchor center) or its left edge (anchor left) at (x, y).
#let label(x, y, body, anchor: "center", size: 8.5, colour: text-soft, style: "normal") = {
  let width = 240
  let dx = if anchor == "center" { x - width / 2 } else if anchor == "left" { x } else { x - width }
  let alignment = if anchor == "center" { center } else if anchor == "left" { left } else { right }
  at(dx, y, box(width: width * 1pt, height: 0pt, align(alignment + horizon, text(
    size: size * 1pt, fill: colour, style: style, body,
  ))))
}

#let unit(dx, dy) = {
  let length = calc.sqrt(dx * dx + dy * dy)
  (dx / length, dy / length)
}

#let point(x, y) = (x * 1pt, y * 1pt)

// A polyline through the given points, its corners rounded, ending in an arrowhead.
#let wire(..points, head: true, radius: 6) = {
  let p = points.pos()
  let (lx, ly) = p.at(p.len() - 2)
  let (ex, ey) = p.at(p.len() - 1)
  let (ux, uy) = unit(ex - lx, ey - ly)
  let arrow = 7
  let pieces = (curve.move(point(..p.at(0))),)
  for index in range(1, p.len() - 1) {
    let (ax, ay) = p.at(index - 1)
    let (bx, by) = p.at(index)
    let (cx, cy) = p.at(index + 1)
    let (d1x, d1y) = unit(bx - ax, by - ay)
    let (d2x, d2y) = unit(cx - bx, cy - by)
    let room = calc.min(
      radius,
      calc.sqrt((bx - ax) * (bx - ax) + (by - ay) * (by - ay)) / 2,
      calc.sqrt((cx - bx) * (cx - bx) + (cy - by) * (cy - by)) / 2,
    )
    pieces.push(curve.line(point(bx - d1x * room, by - d1y * room)))
    pieces.push(curve.quad(point(bx, by), point(bx + d2x * room, by + d2y * room)))
  }
  let shorten = if head { arrow - 1 } else { 0 }
  pieces.push(curve.line(point(ex - ux * shorten, ey - uy * shorten)))
  place(top + left, curve(stroke: (paint: ink, thickness: line-width, join: "round"), ..pieces))
  if head {
    let (nx, ny) = (-uy, ux)
    place(top + left, polygon(
      fill: ink,
      stroke: none,
      point(ex, ey),
      point(ex - ux * arrow + nx * 3.6, ey - uy * arrow + ny * 3.6),
      point(ex - ux * arrow - nx * 3.6, ey - uy * arrow - ny * 3.6),
    ))
  }
}

#let add-radius = 8

// The residual sum: a circled plus centred at (x, y).
#let add(x, y) = {
  let r = add-radius
  at(x - r, y - r, circle(radius: r * 1pt, fill: white, stroke: (paint: ink, thickness: line-width)))
  place(top + left, line(start: point(x - r + 3.5, y), end: point(x + r - 3.5, y), stroke: line-width + ink))
  place(top + left, line(start: point(x, y - r + 3.5), end: point(x, y + r - 3.5), stroke: line-width + ink))
}
