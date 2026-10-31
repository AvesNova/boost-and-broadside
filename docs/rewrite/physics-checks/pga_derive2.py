"""Sandwich of a 2D PGA motor on a general bivector X = x01 e01 + x02 e02 + x12 e12, and
the reverse-motor sandwich (world -> body)."""
import pga_derive as d
M = d.mv(**{"1": "s", "e12": "p", "e01": "a", "e02": "b"})
X = d.mv(**{"e01": "x01", "e02": "x02", "e12": "x12"})
d.show("M X ~M:", d.gp(d.gp(M, X), d.rev(M)))
d.show("~M X M:", d.gp(d.gp(d.rev(M), X), M))
