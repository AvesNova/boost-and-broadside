"""Reference dump from v2: hand-built edge cases, stepped on the CPU, leaves saved per tick."""
import sys, dataclasses, math
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/user/boost-and-broadside/src")
import torch, numpy as np
from bnb.game import api
from bnb.contracts import Action
R = dataclasses.replace
E = R(api.PRESETS["real_scale_elimination"], ships_per_team=1, num_fields=0)
types = api.ship_types(ship_set=E.ship_set)
rules = api.rules_of(E, types=types)
pool = rules.pools[0]
rules = R(rules, pools=(R(pool, mode=R(pool.mode, shrink_hold_seconds=1e9, shrink_end_lead_seconds=0.0)),))
B, N, K = 6, 2, rules.bullet_slots
g = torch.Generator().manual_seed(0)
s = api.reset(rules, types, api.uniforms(rules, B, generator=g), fogged=torch.zeros(B, dtype=torch.bool))
sh, bu = s.ships, s.bullets
def setv(t, env, ship, v):
    t = t.clone(); t[env, ship] = v; return t
x, y, hc, hs = sh.x.clone(), sh.y.clone(), sh.heading_cos.clone(), sh.heading_sin.clone()
speed, energy, shield, wait = sh.proper_speed.clone(), sh.energy.clone(), sh.shield.clone(), sh.shield_recharge_wait.clone()
cruise = 334.9
# park every ship far apart, flying +x at cruise, ship 1 far away
x[:, 0], y[:, 0], hc[:, 0], hs[:, 0] = 0.0, 0.0, 1.0, 0.0
x[:, 1], y[:, 1], hc[:, 1], hs[:, 1] = 0.0, 5000.0, -1.0, 0.0
speed[:] = cruise
# 0 stall: slow ship, full stick
speed[0, 0] = 40.0
# 1 boundary: just inside the edge, heading out
x[1, 0] = 7800.0 - 50.0
# 2 simultaneous kill: both shieldless, each with one bullet about to cross the other's flank
shield[2, :] = 0.0
bx, by, bdx, bdy, bsp, bml = bu.x.clone(), bu.y.clone(), bu.direction_x.clone(), bu.direction_y.clone(), bu.proper_speed.clone(), bu.moves_left.clone()
# ship 0 at (0,0) flying +x: after one tick it is at ~ (11.2, 0); ship 1's bullet comes down the y axis onto it
bx[2, 1, 0], by[2, 1, 0], bdx[2, 1, 0], bdy[2, 1, 0], bsp[2, 1, 0], bml[2, 1, 0] = 6.0, 30.0, 0.0, -1.0, 1800.0, 5
bx[2, 0, 0], by[2, 0, 0], bdx[2, 0, 0], bdy[2, 0, 0], bsp[2, 0, 0], bml[2, 0, 0] = -6.0, 5030.0, 0.0, -1.0, 1800.0, 5
# 3 shield recharge: damaged shield, wait nearly over
shield[3, 0], wait[3, 0] = 10.0, 0.05
# 4 energy depletion: three shots' worth, boost + shoot
energy[4, 0] = 3.5 * 575.0579
# 5 bullet at maximum life: target dead ahead, beam-on, at about the bullet's last move
x[5, 1], y[5, 1], hc[5, 1], hs[5, 1] = 742.0, 0.0, 0.0, 1.0
speed[5, 1] = 1e-3
ships = R(sh, x=x, y=y, heading_cos=hc, heading_sin=hs, proper_speed=speed, energy=energy, shield=shield, shield_recharge_wait=wait)
bullets = R(bu, x=bx, y=by, direction_x=bdx, direction_y=bdy, proper_speed=bsp, moves_left=bml)
s = R(s, ships=ships, bullets=bullets)
T = 24
power = torch.zeros(T, B, N, dtype=torch.int32); stick = torch.zeros_like(power); shoot = torch.zeros_like(power)
stick[:, 0, 0] = 6            # stall: full stick left
power[12:, 3, 0] = 1          # shield: quiet for 12 ticks, then boost
power[:, 4, 0] = 1; shoot[:, 4, 0] = 1
shoot[0, 5, 0] = 1            # one shot
def leaves(tree, prefix=""):
    out = {}
    for f in dataclasses.fields(tree):
        v = getattr(tree, f.name)
        if dataclasses.is_dataclass(v): out.update(leaves(v, prefix + f.name + "."))
        else: out[prefix + f.name] = v.numpy().copy()
    return out
dump = {"t0." + k: v for k, v in leaves(s).items()}
log = []
for t in range(T):
    a = Action(power=power[t], stick=stick[t], shoot=shoot[t])
    spread = torch.zeros(B, N, 2); uni = api.uniforms(rules, B, generator=g)
    s, ev = api.step(rules, types, s, a, spread, uni)
    for k, v in leaves(s).items(): dump[f"t{t+1}.{k}"] = v
    for k, v in leaves(ev).items(): dump[f"e{t+1}.{k}"] = v
    dump[f"a{t}.uniforms"] = uni.numpy()
    log.append((s.ships.proper_speed[0,0].item(), s.ships.angle_of_attack[0,0].item(), s.ships.energy[0,0].item(),
                ev.boundary_damage[1,0].item(), s.ships.shield[1,0].item(),
                ev.died[2].tolist(), ev.done[2].item(), ev.result[2].item(),
                s.ships.shield[3,0].item(), s.ships.shield_recharge_wait[3,0].item(),
                s.ships.energy[4,0].item(), ev.fired[4,0].item(),
                int(s.bullets.moves_left[5,0].max()), s.bullets.x[5,0,0].item(), ev.damage[5,0,1].item(), s.ships.shield[5,1].item()))
np.savez_compressed("ref_edge.npz", power=power.numpy(), stick=stick.numpy(), shoot=shoot.numpy(), **dump)
print("saved ref_edge.npz with", len(dump), "arrays")
print("tick | stall: speed aoa energy | boundary: dmg shield | simul: died done result | shield: value wait | energy: store fired | bullet: moves_left x dmg target_shield")
for t, r in enumerate(log, 1):
    if t in (1,2,3,4,5,6,10,11,12,13,14,24):
        print(t, "| %.2f %.4f %.0f | %.4f %.2f | %s %s %s | %.3f %.3f | %.1f %s | %d %.1f %.3f %.2f" % r)
