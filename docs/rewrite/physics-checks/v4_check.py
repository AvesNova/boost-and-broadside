import os, sys, math
sys.path.insert(0, os.path.dirname(__file__))
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import ship_pga as G
# sandwich checks: position of a translator; heading rotation of a direction; reverse round trip
M = G.gp((1.0, 0.0, -0.5 * 3.0, -0.5 * -2.0), G.rotor(0.7))
print("position (3,-2):", [float(v) for v in G.position(M)])
d = G.from_point(*G.sandwich(M, *G.to_point(1.0, 0.0, 0.0)))
print("body x-axis in world (cos .7, sin .7):", [round(float(v), 6) for v in d], [round(math.cos(.7), 6), round(math.sin(.7), 6)])
b = G.from_point(*G.sandwich_rev(M, *G.to_point(*d, 0.0)))
print("round trip back to (1,0):", [round(float(v), 6) for v in b])
t = G.sandwich(G.rotor(0.5), 1.0, 0.0, 0.0)
print("rotor(0.5) on twist translation (1,0):", [round(float(v), 6) for v in t[:2]], "expect", [round(math.cos(.5), 6), round(math.sin(.5), 6)])
