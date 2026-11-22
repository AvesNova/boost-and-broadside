import sys, dataclasses, time
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/user/boost-and-broadside/src")
import torch
torch.set_num_threads(4)
from bnb.game import api
from bnb.contracts import Action
from bnb.game.handling.characteristics import measure_characteristics
from bnb.controllers.imugi import Imugi
from bnb.controllers.scripted import Chaser
E = api.PRESETS["real_scale_elimination"]
types = api.ship_types(ship_set=E.ship_set)
rules = api.rules_of(E, types=types)
pool = rules.pools[0]
rules = dataclasses.replace(rules, pools=(dataclasses.replace(pool, mode=dataclasses.replace(pool.mode, shrink_hold_seconds=1e9, shrink_end_lead_seconds=0.0)),))
ch = measure_characteristics(types, rules)
S_ = rules.ships_per_team; N = 2*S_
def make(name):
    if name == "imugi": return Imugi(ch, S_)
    if name == "nodoctrine": return Imugi(ch, S_, doctrines=None)
    if name == "lite": return Imugi(ch, S_, doctrines=None, memory=False)
    if name == "chaser": return Chaser(ch, S_)
a_name, b_name, B, T = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
g = torch.Generator().manual_seed(11)
state = api.reset(rules, types, api.uniforms(rules, B, generator=g), fogged=torch.zeros(B, dtype=torch.bool))
state, obs, _ = api.perceive(rules, state)
agents = [make(a_name), make(b_name)]
# seat: agent A plays team 0 in the first half of the environments, team 1 in the second
half = B // 2
carries = [ag.init_carry(B, torch.device("cpu")) for ag in agents]
reset = torch.ones(B, dtype=torch.bool)
ended = torch.full((B,), -1); result = torch.zeros(B, dtype=torch.int32); left = torch.zeros(B, 2)
t0 = time.time()
seat_a = torch.cat([torch.zeros(half, dtype=torch.long), torch.ones(B - half, dtype=torch.long)])  # A's team per env
with torch.no_grad():
  for t in range(T):
    parts = []
    acts = {}
    for i, ag in enumerate(agents):
        team = seat_a if i == 0 else 1 - seat_a
        view = type(obs)(**{f.name: getattr(obs, f.name)[torch.arange(B), team] for f in dataclasses.fields(obs)})
        key = torch.randint(0, 2**62, (B,), generator=g)
        act, carries[i], _ = ag.step(carries[i], view, reset, key)
        acts[i] = (team, act)
    power = torch.zeros(B, 2, S_, dtype=torch.int32); stick = torch.zeros_like(power); shoot = torch.zeros_like(power)
    for i in (0, 1):
        team, act = acts[i]
        power[torch.arange(B), team] = act.power.to(torch.int32); stick[torch.arange(B), team] = act.stick.to(torch.int32); shoot[torch.arange(B), team] = act.shoot.to(torch.int32)
    reset = torch.zeros(B, dtype=torch.bool)
    a = Action(power=power.reshape(B, N), stick=stick.reshape(B, N), shoot=shoot.reshape(B, N))
    state, ev = api.step(rules, types, state, a, api.spread(rules, B, generator=g), api.uniforms(rules, B, generator=g))
    newly = (ended < 0) & ev.done
    ended = torch.where(newly, t+1, ended); result = torch.where(newly, ev.result, result)
    state, obs, _ = api.perceive(rules, state)
    if (ended >= 0).all(): break
    if t % 600 == 0: print("  tick", t, "ended", int((ended >= 0).sum()), round(time.time() - t0), "s", flush=True)
done = ended >= 0
res_a = torch.where(seat_a == 0, result, -result)[done]
w, d, l = int((res_a > 0).sum()), int((res_a == 0).sum()), int((res_a < 0).sum())
n = max(w + d + l, 1)
score = (w + 0.5 * d) / n
print(f"{a_name} vs {b_name}: {B} games, horizon {T*rules.tick_length:.0f} s, ended {int(done.sum())}; {a_name} W/D/L {w}/{d}/{l}, score {score:.3f} +- {(score*(1-score)/n)**0.5:.3f}; median length {float(ended[done].float().median())*rules.tick_length:.1f} s; wall {time.time()-t0:.0f} s", flush=True)
