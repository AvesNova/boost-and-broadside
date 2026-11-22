import sys, time, dataclasses
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/user/boost-and-broadside/src")
import torch
torch.set_num_threads(4)
from bnb.game import api
from bnb.contracts import Action
print("torch", torch.__version__)
game = api.PRESETS["real_scale_elimination"]
print(game)
types = api.ship_types(ship_set=game.ship_set)
rules = api.rules_of(game, types=types)
print("K", rules.bullet_slots, "tick", rules.tick_length, "U", rules.tick_uniform_count, "pool", rules.pools[0].episode_ticks, rules.pools[0].respawn_ticks, rules.pools[0].arena_radius)
print(rules.pools[0].mode)
B = 4
g = torch.Generator().manual_seed(0)
state = api.reset(rules, types, api.uniforms(rules, B, generator=g), fogged=torch.zeros(B, dtype=torch.bool))
print("x", state.ships.x[0]); print("speed", state.ships.proper_speed[0]); print("energy", state.ships.energy[0], "shield", state.ships.shield[0], "type", state.ships.ship_type[0])
print(Action)
N = rules.num_ships
t=time.time()
for i in range(60):
    a = Action(power=torch.randint(0,3,(B,N),generator=g), stick=torch.randint(0,7,(B,N),generator=g), shoot=torch.randint(0,2,(B,N),generator=g))
    state, events = api.step(rules, types, state, a, api.spread(rules, B, generator=g), api.uniforms(rules, B, generator=g))
print("60 ticks", time.time()-t, "s")
print("x", state.ships.x[0]); print("speed", state.ships.proper_speed[0]); print("energy", state.ships.energy[0]); print("alive", state.ships.alive[0], "tick", state.match.episode_tick)
print("bullets active", (state.bullets.moves_left>0).sum().item())
