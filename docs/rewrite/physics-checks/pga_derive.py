"""Derive expanded 2D PGA (R(2,0,1)) formulas with a tiny symbolic engine.

Basis bit 0 = e0 (squares to 0), bit 1 = e1, bit 2 = e2 (square to 1).
"""
from collections import defaultdict
from itertools import product

METRIC = [0, 1, 1]
BLADES = [0b000, 0b001, 0b010, 0b100, 0b011, 0b101, 0b110, 0b111]
NAMES = {0: "1", 1: "e0", 2: "e1", 4: "e2", 3: "e01", 5: "e02", 6: "e12", 7: "e012"}

def blade_mul(a, b):
    # sign from reordering
    s, x = 1, a >> 1
    while x:
        if bin(x & b).count("1") % 2: pass
        x >>= 1
    # canonical: count swaps
    swaps, aa = 0, a >> 1
    while aa:
        swaps += bin(aa & b).count("1"); aa >>= 1
    sign = -1 if swaps % 2 else 1
    common = a & b
    for i in range(3):
        if common >> i & 1:
            sign *= METRIC[i]
    return sign, a ^ b

class Poly(dict):
    def __add__(self, o):
        r = Poly(self)
        for k, v in o.items(): r[k] = r.get(k, 0) + v
        return Poly({k: v for k, v in r.items() if v})
    def __mul__(self, o):
        r = defaultdict(int)
        for (k1, v1), (k2, v2) in product(self.items(), o.items()):
            r[tuple(sorted(k1 + k2))] += v1 * v2
        return Poly({k: v for k, v in r.items() if v})
    def scale(self, c): return Poly({k: c * v for k, v in self.items()})
    def __str__(self):
        if not self: return "0"
        return " + ".join(f"{v}*{'*'.join(k) or '1'}" for k, v in sorted(self.items()))

def mv(**comps):  # name -> symbol
    return {b: Poly({(comps[NAMES[b]],): 1}) if NAMES[b] in comps else Poly() for b in BLADES}

def gp(A, B):
    R = {b: Poly() for b in BLADES}
    for a, b in product(BLADES, BLADES):
        if not A[a] or not B[b]: continue
        s, c = blade_mul(a, b)
        if s: R[c] = R[c] + (A[a] * B[b]).scale(s)
    return R

def rev(A):
    g = lambda b: bin(b).count("1")
    return {b: A[b].scale(-1 if g(b) in (2, 3) else 1) for b in BLADES}

def show(title, R):
    print(title)
    for b in BLADES:
        if R[b]: print(f"  {NAMES[b]:>5}: {R[b]}")

M = mv(**{"1": "s", "e12": "p", "e01": "a", "e02": "b"})        # motor
X = mv(**{"e01": "y", "e02": "mx", "e12": "w"})                # point x e20 + y e01 + w e12 (e20 = -e02)
show("motor sandwich on point (x stored as mx = -x):", gp(gp(M, X), rev(M)))
M2 = mv(**{"1": "t", "e12": "q", "e01": "c", "e02": "d"})
show("motor composition M*M2:", gp(M, M2))
