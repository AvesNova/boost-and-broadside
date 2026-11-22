import sys, dataclasses, time, types as _t
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/user/boost-and-broadside/src"); sys.path.insert(0, "/home/user/boost-and-broadside/docs/rewrite/sections/simplification")
import torch
from lean_flight import *
from bnb.game.physics import combat
import os
print("cpu threads", torch.get_num_threads(), os.cpu_count())
E0 = api.PRESETS["real_scale_elimination"]
def med(f, n=30):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    ts.sort(); return 1e3 * ts[len(ts)//2]
for threads in (1, 4):
    torch.set_num_threads(threads)
    for S_ in (5, 50):
        E = dataclasses.replace(E0, ships_per_team=S_)
        types = api.ship_types(ship_set=E.ship_set); rules = api.rules_of(E, types=types); c, h = rules.flight, rules.tick_length
        B, N = 1, rules.num_ships
        g = torch.Generator().manual_seed(0)
        state = api.reset(rules, types, api.uniforms(rules, B, generator=g), fogged=torch.zeros(B, dtype=torch.bool))
        a = Action(power=torch.randint(0,3,(B,N),generator=g), stick=torch.randint(0,7,(B,N),generator=g), shoot=torch.ones((B,N),dtype=torch.long))
        sp, un = api.spread(rules, B, generator=g), api.uniforms(rules, B, generator=g)
        with torch.no_grad():
            for _ in range(12): state, _ = api.step(rules, types, state, a, sp, un)
            t_step = med(lambda: api.step(rules, types, state, a, sp, un))
            moved, sweep = api.move(rules, types, state, a, sp)
            t_move = med(lambda: api.move(rules, types, state, a, sp))
            t_hits = med(lambda: combat.hits(rules, types, sweep, moved.ships, moved.bullets, state.ships.alive))
            t_perc = med(lambda: api.perceive(rules, state))
            P = precompute(c, types, state.ships.ship_type)
            full = dict(x="x", y="y", hc="heading_cos", hs="heading_sin", speed="proper_speed", angle="angle_of_attack", cos="angle_of_attack_cos", sin="angle_of_attack_sin", nose_rate="nose_rate", energy="energy", logn="log_refractive_index", gx="log_index_gradient_x", gy="log_index_gradient_y")
            def lean():
                s = _t.SimpleNamespace(**{k: getattr(state.ships, v) for k, v in full.items()})
                flight_tick(c, P, h, s, a.stick, a.power, state.fields)
            t_lean = med(lean)
        print(f"threads {threads} B=1 {S_}v{S_} (N={N}, K={rules.bullet_slots}, pairs {N*N*rules.bullet_slots}): v2 step {t_step:.1f} ms (move {t_move:.1f}, hits {t_hits:.2f}), perceive(fog code) {t_perc:.1f} ms, lean flight tick {t_lean:.1f} ms; active bullets {(state.bullets.moves_left>0).sum().item()}")
