"""Quasi-static instantaneous envelope from the code's own functions: nose settled on the
full-stick target angle, rate = (C_L(angle) v^2 + T sin angle) / v, at a given store."""
import torch, math, json
from types import SimpleNamespace
from lean.game import core, card
DEG = 180 / math.pi
v = torch.linspace(10.0, 470.0, 47)
out = {}
for t in range(3):
    for energy_label, energy in (("spawn", card.CARD["spawn_energy"][t]), ("low", 0.01 * card.CARD["max_energy"][t])):
        for power in range(3):
            b = v.shape[0]
            p = core.per_type_constants(core.GameConfig(), torch.full((b,), t))
            control = SimpleNamespace()
            signed = p.stick_table[:, 6]
            control.stick_size, control.stick_sign = signed.abs(), torch.sign(signed)
            control.commanded = p.commanded_table[:, power]
            control.demand = p.demand_table[:, power]
            control.inverse_demand = 1 / control.demand
            e = torch.full((b,), float(energy))
            setting = core._setting(p, control, v, e)
            lateral, exchange, vertical, thrust, charge = setting
            reach = (lateral.abs() / (v * v) / p.lift).clamp(0, 1)
            angle = p.stall * (1 - torch.sqrt(1 - reach))
            cl = core._lift_coefficient(p, angle)
            aero = cl * v * v
            thrust_part = thrust * torch.sin(angle)
            rate = (aero + thrust_part) / v
            radius = v / rate
            speed_rate, energy_rate = core._rates(p, setting, torch.ones(b), v, cl, torch.cos(angle), 1 / (v * v))
            out[f"{t}_{energy_label}_{power}"] = dict(speed=v.tolist(), rate=(rate * DEG).tolist(), radius=radius.tolist(),
                                                     speed_rate=speed_rate.tolist(), energy_rate=energy_rate.tolist(),
                                                     thrust_share=(thrust_part / (aero + thrust_part)).tolist())
            i, j = int(rate.argmax()), int(radius.argmin())
            print(card.TYPE_NAMES[t], energy_label, power, "max rate %.1f at %.0f (R %.0f) | min R %.0f at %.0f (rate %.1f) | R@100 %.0f R@150 %.0f R@200 %.0f | dv/dt@corner %.0f dE/dt %.0fk" % (
                rate[i] * DEG, v[i], radius[i], radius[j], v[j], rate[j] * DEG, radius[9], radius[14], radius[19], speed_rate[i], energy_rate[i] / 1000))
json.dump(out, open("static.json", "w"))
