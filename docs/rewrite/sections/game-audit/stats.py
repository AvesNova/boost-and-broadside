import sys, dataclasses, time
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/user/boost-and-broadside/src")
import torch
torch.set_num_threads(4)
from bnb.game import api
from bnb.contracts import Action
from bnb.game.handling.characteristics import measure_characteristics
from bnb.controllers.imugi import Imugi
E = api.PRESETS["real_scale_elimination"]
types = api.ship_types(ship_set=E.ship_set)
rules = api.rules_of(E, types=types)
pool = rules.pools[0]
mode = dataclasses.replace(pool.mode, shrink_hold_seconds=1e9, shrink_end_lead_seconds=0.0)
rules = dataclasses.replace(rules, pools=(dataclasses.replace(pool, mode=mode),))
ch = measure_characteristics(types, rules)
B, T = int(sys.argv[1]), int(sys.argv[2])
S_ = rules.ships_per_team; N = 2*S_
g = torch.Generator().manual_seed(7)
state = api.reset(rules, types, api.uniforms(rules, B, generator=g), fogged=torch.zeros(B, dtype=torch.bool))
state, obs, _ = api.perceive(rules, state)
agent = Imugi(ch, S_)
carry = agent.init_carry(2*B, torch.device("cpu"))
reset = torch.ones(2*B, dtype=torch.bool)
ended = torch.full((B,), -1); result = torch.zeros(B, dtype=torch.int32)
first_kill = torch.full((B,), -1)
shots = torch.zeros(B); dmg_enemy = torch.zeros(B); dmg_friendly = torch.zeros(B); hits_n = torch.zeros(B); bdmg = torch.zeros(B); kills = torch.zeros(B)
team = api.ship_team(S_)
enemy_pair = (team[:, None] != team[None, :])
t0 = time.time()
with torch.no_grad():
  for t in range(T):
    views = type(obs)(**{f.name: getattr(obs, f.name).flatten(0,1) for f in dataclasses.fields(obs)})
    key = torch.randint(0, 2**62, (2*B,), generator=g)
    act, carry, _ = agent.step(carry, views, reset, key)   # (2B, S) rows: env-major, team view
    reset = torch.zeros(2*B, dtype=torch.bool)
    def sim(v): return v.reshape(B, 2, S_).reshape(B, N)
    a = Action(power=sim(act.power), stick=sim(act.stick), shoot=sim(act.shoot))
    state, ev = api.step(rules, types, state, a, api.spread(rules, B, generator=g), api.uniforms(rules, B, generator=g))
    live = ended < 0
    shots += torch.where(live, ev.fired.sum(1).float(), 0.)
    dmg_enemy += torch.where(live, (ev.damage * enemy_pair).sum((1,2)), 0.)
    dmg_friendly += torch.where(live, (ev.damage * ~enemy_pair).sum((1,2)), 0.)
    hits_n += torch.where(live, (ev.damage > 0).sum((1,2)).float(), 0.)
    bdmg += torch.where(live, ev.boundary_damage.sum(1), 0.)
    kills += torch.where(live, ev.died.sum(1).float(), 0.)
    first_kill = torch.where((first_kill < 0) & live & ev.died.any(1), t+1, first_kill)
    newly = live & ev.done
    ended = torch.where(newly, t+1, ended); result = torch.where(newly, ev.result, result)
    state, obs, _ = api.perceive(rules, state)
    if (ended >= 0).all(): break
    if t % 300 == 0: print(t, "ended", int((ended>=0).sum()), round(time.time()-t0), "s", flush=True)
tick = rules.tick_length
done = ended >= 0
print(f"B={B} horizon {T} ticks ({T*tick:.0f} s); ended {int(done.sum())}/{B}")
e = ended[done].float()*tick
if done.any():
    print("game length s: median %.1f mean %.1f p10 %.1f p90 %.1f max %.1f" % (e.median(), e.mean(), e.quantile(0.1), e.quantile(0.9), e.max()))
fk = first_kill[first_kill>=0].float()*tick
print("first kill s: median %.1f p10 %.1f p90 %.1f (n=%d)" % (fk.median(), fk.quantile(0.1), fk.quantile(0.9), fk.numel()))
print("results team0 win/draw/loss:", int((result[done]>0).sum()), int((result[done]==0).sum()), int((result[done]<0).sum()))
print("per game: shots %.0f, pair-hits(tick,shooter,target) %.1f, kills %.2f, enemy dmg %.0f, friendly dmg %.0f, boundary dmg %.1f" % (shots.mean(), hits_n.mean(), kills.mean(), dmg_enemy.mean(), dmg_friendly.mean(), bdmg.mean()))
print("hit pairs per shot %.4f; friendly share of damage %.3f" % (hits_n.sum()/shots.sum(), dmg_friendly.sum()/(dmg_friendly.sum()+dmg_enemy.sum())))
print("wall", round(time.time()-t0), "s")
