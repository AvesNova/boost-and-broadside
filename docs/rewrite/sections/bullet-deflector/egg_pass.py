"""Damage per pass, per engagement and lost to shield recharge, normal play and "always fire".

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/bullet-deflector/egg_pass.py \
        --out DIR [--envs-1v1 512] [--envs-4v4 256] [--ticks 1800] [--only CELL ...]

Cells: {1v1-egg, 1v1-off, 4v4-egg} x {normal, alwaysfire}. Same agent (look-ahead, both sides),
same hit models and setups as egg_play.py (its set_model, its passes: local minima of the 1v1
distance under the gun range, 1 s apart, events assigned to the nearest minimum of the same
game). "alwaysfire": after the agent decides every ship's fire action is 1 (the gun still waits
its cooldown) and GameConfig.fire_thrust_fraction = 1.0 (no thrust withheld on a firing tick,
core._fly). Damage is in units of a full shield (the card's max_shield); it is StepOut.hit_damage
(bullets, plus collisions credited to the other ship). Engagement: the passes of one game in a
row, ended by a kill or by the distance rising above the gun range between two passes.
Shield recharge: StepOut.shield_restored summed per ship life; a ship's damage taken over its
life = shield at the start + restored + overkill (the shield cannot go below 0).
"""

import argparse
import json
import time
from pathlib import Path

import egg_play
import numpy as np
import torch

from lean import game, scripted
from lean.contract import REAL, compile_dynamic
from lean.game import card, core


def summary(values) -> dict:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {}
    q = np.percentile(values, [25, 50, 75, 90])
    return dict(n=len(values), mean=float(values.mean()), p25=q[0], median=q[1], p75=q[2], p90=q[3])


def analyse_1v1(tr, gun_range, h, max_shield):
    """Passes (summed over both shooters and directed per shooter) and engagements."""
    dist, gid = tr["dist"], tr["gid"]  # (T, E)
    dmg, hits, shots, died = tr["dmg"], tr["hits"], tr["shots"], tr["died"]  # (T, E, 2) by shooter
    ticks, envs = dist.shape
    merge = int(round(1.0 / h))
    passes = []  # dicts per pass
    engagements = []
    stray = dict(hits=0.0, damage=0.0, kills=0.0)
    for e in range(envs):
        change = np.nonzero(np.diff(gid[:, e]))[0] + 1
        bounds = [0, *change.tolist(), ticks]
        for a, b in zip(bounds[:-1], bounds[1:]):
            d = dist[a:b, e]
            n = b - a
            minima = [
                t
                for t in range(1, n - 1)
                if d[t] <= d[t - 1] and d[t] < d[t + 1] and d[t] < gun_range
            ]
            kept = []
            for t in minima:
                if not kept or t - kept[-1] >= merge:
                    kept.append(t)
            seg = lambda x: x[a:b, e]  # noqa: E731
            # A game that ends on a kill inside the gun range ends the pass there (the distance
            # has no later tick to turn upward): the last tick is a pass unless one is within 1 s.
            if died[b - 1, e].sum() > 0 and d[n - 1] < gun_range:
                if not kept or n - 1 - kept[-1] >= merge:
                    kept.append(n - 1)
            if not kept:
                stray["hits"] += float(seg(hits).sum())
                stray["damage"] += float(seg(dmg).sum())
                stray["kills"] += float(seg(died).sum())
                continue
            kept_a = np.array(kept)
            per = np.zeros((len(kept), 4, 2))  # damage, hits, shots, kills by shooter
            for k, arr in enumerate((dmg, hits, shots)):
                s = seg(arr)
                ts = np.nonzero(s.sum(-1))[0]
                if len(ts):
                    near = np.abs(ts[:, None] - kept_a[None]).argmin(1)
                    np.add.at(per[:, k], near, s[ts])
            kill_ticks = np.nonzero(seg(died).sum(-1))[0]
            for t in kill_ticks:
                near = int(np.abs(t - kept_a).argmin())
                victim = int(np.argmax(seg(died)[t]))
                per[near, 3, 1 - victim] += 1  # shooter = the other ship
            for p in per:
                passes.append(p)
            # engagements: cut where the distance exceeds the gun range between passes
            start = 0
            for i in range(1, len(kept) + 1):
                cut = i == len(kept) or d[kept[i - 1] : kept[i] + 1].max() > gun_range
                if cut:
                    block = per[start:i]
                    engagements.append(
                        dict(
                            passes=i - start,
                            damage=block[:, 0].sum(),
                            kills=block[:, 3].sum(),
                            first_pass_kill=block[0, 3].sum(),
                        )
                    )
                    start = i
    P = np.array(passes) if passes else np.zeros((1, 4, 2))
    total = P[:, :, :].sum(-1)  # (n, 4) summed over shooters
    damage = total[:, 0] / max_shield
    kill = total[:, 3] > 0
    partial = (damage > 0) & ~kill
    out = dict(passes=int(len(P)), stray=stray)
    out["pass_hits"] = summary(total[:, 1])
    out["pass_damage_shields"] = summary(damage)
    out["pass_shots"] = summary(total[:, 2])
    out["share_no_hit"] = float((total[:, 1] == 0).mean())
    out["share_no_damage"] = float((damage == 0).mean())
    out["share_kill"] = float(kill.mean())
    out["share_partial"] = float(partial.mean())
    out["share_damage_ge_1_shield"] = float((damage >= 1.0).mean())
    out["pass_damage_given_kill"] = summary(damage[kill])
    # directed: one sample per (pass, shooter)
    dd = (P[:, 0, :] / max_shield).reshape(-1)
    dk = (P[:, 3, :] > 0).reshape(-1)
    dh = P[:, 1, :].reshape(-1)
    out["directed_hits"] = summary(dh)
    out["directed_damage_shields"] = summary(dd)
    out["directed_share_no_hit"] = float((dh == 0).mean())
    out["directed_share_kill"] = float(dk.mean())
    out["directed_share_partial"] = float(((dd > 0) & ~dk).mean())
    out["directed_share_damage_ge_1_shield"] = float((dd >= 1.0).mean())
    out["directed_damage_given_kill"] = summary(dd[dk])
    out["directed_damage_given_hit"] = summary(dd[dh > 0])
    E_ = engagements
    out["engagements"] = len(E_)
    if E_:
        eng_d = np.array([x["damage"] for x in E_]) / max_shield
        eng_p = np.array([x["passes"] for x in E_])
        eng_k = np.array([x["kills"] > 0 for x in E_])
        out["engagement_damage_shields"] = summary(eng_d)
        out["engagement_passes"] = summary(eng_p)
        out["engagement_share_kill"] = float(eng_k.mean())
        out["engagement_damage_given_kill"] = summary(eng_d[eng_k])
        out["passes_to_kill"] = summary(eng_p[eng_k])
        out["kills_in_one_pass_share"] = float((eng_p[eng_k] == 1).mean()) if eng_k.any() else None
        out["kills_on_first_pass_share"] = (
            float(np.mean([x["first_pass_kill"] > 0 for x in E_ if x["kills"] > 0]))
            if eng_k.any()
            else None
        )
    return out


@torch.no_grad()
def run_cell(name, config, size, envs, ticks, model, always, agent_call, step_call, device, seed):
    generator = torch.Generator(device).manual_seed(seed)
    sizes = torch.full((envs, 2), size, dtype=torch.long, device=device)
    state = game.reset(config, envs, sizes, generator, device)
    state = egg_play.set_model(config, state, model)
    slots = state.x.shape[1]
    c = state.constants
    first = lambda item: float(torch.as_tensor(item).flatten()[0])  # noqa: E731
    max_shield, gun_range = first(c["max_shield"]), first(c["gun_range"])
    h = config.tick_length
    trace = size == 1
    if trace:
        tr = dict(
            dist=np.zeros((ticks, envs), np.float32),
            gid=np.zeros((ticks, envs), np.int32),
            dmg=np.zeros((ticks, envs, 2), np.float32),
            hits=np.zeros((ticks, envs, 2), np.float32),
            shots=np.zeros((ticks, envs, 2), np.float32),
            died=np.zeros((ticks, envs, 2), np.float32),
        )
        game_id = torch.zeros(envs, dtype=torch.int32, device=device)
    life_damage = torch.zeros(envs, slots, device=device)
    life_restored = torch.zeros(envs, slots, device=device)
    start_shield = state.shield.clone()
    last_full = torch.zeros(envs, slots, device=device)
    acc = dict(
        living=0.0, kills=0.0, action_fire=0.0, shots=0.0, real_ended=0.0, real_enemy=0.0,
        real_mate=0.0, damage=0.0, restored=0.0,
        dead_damage=0.0, dead_restored=0.0, dead_start=0.0, dead_overkill=0.0, games=0.0,
    )  # fmt: skip
    ttk = []
    started = time.time()
    for t in range(ticks):
        living = state.alive & state.exists
        uniforms = scripted.draw_uniforms(envs, slots, generator, device)
        action, _ = agent_call(state, uniforms)
        if always:
            action = torch.cat([action[..., :2], torch.ones_like(action[..., 2:])], -1)
        acc["living"] += float(living.sum())
        acc["action_fire"] += float((action[..., 2] == 1)[living].sum())
        before = state
        if trace:
            gx, gy = state.x[:, 1] - state.x[:, 0], state.y[:, 1] - state.y[:, 0]
            tr["dist"][t] = torch.hypot(gx, gy).cpu().numpy()
            tr["gid"][t] = game_id.cpu().numpy()
        state, out = step_call(
            state, action, game.draw_noise(config, envs, slots, generator, device)
        )
        acc["shots"] += float(state.fired[living].sum())
        # Real shots among the bullets that reported: real before the tick, or written this
        # tick by a ship that fired (a hit makes a bullet CONTACTED).
        bullet_slots = before.bullet_state.shape[-1]
        written = (
            torch.arange(bullet_slots, device=device) == (before.tick % bullet_slots)[:, None, None]
        )
        real = torch.where(written, state.fired[..., None], before.bullet_state == REAL)
        ended = (out.shot_outcome >= 0) & real
        acc["real_ended"] += float(ended.sum())
        acc["real_enemy"] += float(((out.shot_outcome == 1) & real).sum())
        acc["real_mate"] += float(((out.shot_outcome == 2) & real).sum())
        taken = out.hit_damage.sum(1)  # (E, victim)
        acc["damage"] += float(taken.sum())
        acc["restored"] += float(out.shield_restored.sum())
        life_damage += taken
        life_restored += out.shield_restored
        if trace:
            tr["dmg"][t] = (
                torch.stack([out.hit_damage[:, 0, 1], out.hit_damage[:, 1, 0]], -1).cpu().numpy()
            )
            outcome_hit = ((out.shot_outcome >= 1) & real).sum(-1)
            tr["hits"][t] = outcome_hit.float().cpu().numpy()
            tr["shots"][t] = state.fired.float().cpu().numpy()
            tr["died"][t] = out.died.float().cpu().numpy()
        full = before.shield >= 0.99 * max_shield
        last_full = torch.where(full, float(t), last_full)
        died = out.died
        if bool(died.any()):
            acc["kills"] += float(died.sum())
            ttk.extend(((t + 1 - last_full[died]) * h).tolist())
            acc["dead_damage"] += float(life_damage[died].sum())
            acc["dead_restored"] += float(life_restored[died].sum())
            acc["dead_start"] += float(start_shield[died].sum())
            overkill = life_damage - life_restored - start_shield
            acc["dead_overkill"] += float(overkill[died].sum())
        if bool(out.done.any()):
            done = out.done
            acc["games"] += float(done.sum())
            fresh = game.deal(config, envs, slots, sizes, generator, device)
            state = game.take_deal(state, done, fresh)
            last_full = torch.where(done[:, None], float(t), last_full)
            life_damage = torch.where(done[:, None], 0.0, life_damage)
            life_restored = torch.where(done[:, None], 0.0, life_restored)
            start_shield = torch.where(done[:, None], state.shield, start_shield)
            if trace:
                game_id = game_id + done.int()
        if (t + 1) % 600 == 0 or t + 1 == ticks:
            elapsed = time.time() - started
            eta = time.strftime("%H:%M:%S", time.localtime(started + elapsed * ticks / (t + 1)))
            print(f"  {name}: tick {t + 1}/{ticks} kills {acc['kills']:.0f} ETA {eta}", flush=True)
    minutes = envs * ticks * h / 60.0
    living = max(acc["living"], 1.0)
    dead = max(acc["dead_damage"], 1e-9)
    record = dict(
        cell=name, size=size, envs=envs, ticks=ticks, always_fire=always,
        wall_s=round(time.time() - started, 1),
        kills=acc["kills"], games_ended=acc["games"],
        kills_per_minute=acc["kills"] / minutes,
        time_to_kill=summary(ttk),
        action_fire_share=acc["action_fire"] / living,
        shots_per_ship_tick=acc["shots"] / living,
        real_bullets_ended=acc["real_ended"],
        hit_chance_per_bullet=(acc["real_enemy"] + acc["real_mate"]) / max(acc["real_ended"], 1.0),
        shots_per_kill=acc["real_ended"] / max(acc["kills"], 1.0),
        total_damage_shields=acc["damage"] / max_shield,
        total_restored_shields=acc["restored"] / max_shield,
        restored_share_of_damage=acc["restored"] / max(acc["damage"], 1e-9),
        kill_lives=dict(
            damage_per_kill_shields=acc["dead_damage"] / max(acc["kills"], 1) / max_shield,
            restored_per_kill_shields=acc["dead_restored"] / max(acc["kills"], 1) / max_shield,
            overkill_per_kill_shields=acc["dead_overkill"] / max(acc["kills"], 1) / max_shield,
            start_shield_per_kill_shields=acc["dead_start"] / max(acc["kills"], 1) / max_shield,
            restored_share=acc["dead_restored"] / dead,
            overkill_share=acc["dead_overkill"] / dead,
            counted_share=acc["dead_start"] / dead,
        ),
    )  # fmt: skip
    if trace:
        record.update(analyse_1v1(tr, gun_range, h, max_shield))
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--envs-1v1", type=int, default=512)
    parser.add_argument("--envs-4v4", type=int, default=256)
    parser.add_argument("--ticks", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    device = torch.device("cuda")
    agent = scripted.ScriptedConfig()
    today = float(torch.as_tensor(core.ship_card(core.GameConfig())["spread_speed"]).flatten()[0])
    egg_turn = card.DEFLECTOR_PRESETS["egg"]["deflector_turn"]
    agent_call = compile_dynamic(lambda s, u: scripted.decide(agent, s, None, None, u))
    arguments.out.mkdir(parents=True, exist_ok=True)
    cells = []
    for always in (False, True):
        fraction = 1.0 if always else GAME_DEFAULT_FRACTION
        configs = {
            "1v1": game.GameConfig(field_count=0, fog=False, fire_thrust_fraction=fraction),
            "4v4": game.GameConfig(fire_thrust_fraction=fraction),
        }
        for setup, deflector in (("1v1", "egg"), ("1v1", "off"), ("4v4", "egg")):
            turn = egg_turn if deflector == "egg" else 0.0
            model = dict(deflector_turn=turn, bullet_spread=today)
            label = "alwaysfire" if always else "normal"
            cells.append((f"{setup}-{deflector}-{label}", setup, configs[setup], model, always))
    if arguments.only:
        cells = [cell for cell in cells if cell[0] in arguments.only]
    started = time.time()
    for index, (name, setup, config, model, always) in enumerate(cells):
        size = 1 if setup == "1v1" else 4
        envs = arguments.envs_1v1 if size == 1 else arguments.envs_4v4
        step = compile_dynamic(lambda s, a, n, config=config: game.step(config, s, a, n))
        record = run_cell(
            name, config, size, envs, arguments.ticks, model, always,
            agent_call, step, device, arguments.seed,
        )  # fmt: skip
        (arguments.out / f"{name}.json").write_text(json.dumps(record, indent=1))
        print(json.dumps(record), flush=True)
        elapsed = time.time() - started
        eta = time.strftime("%H:%M", time.localtime(started + elapsed * len(cells) / (index + 1)))
        print(f"cell {index + 1}/{len(cells)} done; all cells ETA {eta}", flush=True)


GAME_DEFAULT_FRACTION = game.GameConfig().fire_thrust_fraction

if __name__ == "__main__":
    main()
