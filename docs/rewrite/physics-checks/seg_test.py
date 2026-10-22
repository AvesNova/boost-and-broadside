import sys, os, torch, math
sys.path.insert(0, os.path.dirname(__file__))
import integrators as I
torch.manual_seed(0)
n = 8
s0 = torch.rand(n) * 150 + 30; th0 = torch.rand(n) * 6
A = torch.randn(n) * 40; lat = torch.randn(n) * 10; CL = torch.randn(n) * 0.02; c = torch.rand(n) * 2e-3 + 5e-4
h = torch.full((n,), 1 / 30)
pos, s, th = I._segment(torch.zeros(n, dtype=torch.complex64), s0, th0, A, lat, CL, c, h)
# brute force
S, TH, X = s0.double(), th0.double(), torch.zeros(n, dtype=torch.complex128)
m = 100000; dt = (1 / 30) / m
for _ in range(m):
    X = X + S * torch.polar(torch.ones_like(S), TH) * dt
    dS = A.double() - c.double() * S * S
    dTH = CL.double() * S + lat.double() / S
    S, TH = S + dS * dt, TH + dTH * dt
print("speed err", (s - S).abs().max().item(), "theta err", (th - TH).abs().max().item(), "pos err", (pos - X).abs().max().item())
print(s[:3], S[:3]); print(th[:3], TH[:3])
