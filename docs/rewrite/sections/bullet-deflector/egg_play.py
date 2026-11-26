"""The egg deflector in play: the look-ahead scripted agent on both teams, cells of (hit model,
setup), on the GPU, the agent and the step compiled once (every hit-model number is a runtime
row: a cell changes rows, never the graph).

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/bullet-deflector/egg_play.py \
        --plan grid --out DIR [--envs-1v1 512] [--envs-4v4 256] [--ticks 1800]

Setups: "1v1" (fog off, no fields), "4v4" (GameConfig(): fields and fog). A cell's hit model
is card.DEFLECTOR_PRESETS["egg"] with `deflector_turn` 0 for "off" (the egg's code with no turn:
the straight paths, hit widths 24 m) and the spread given in mrad (per axis, over the bullet
launch speed). The ship card is changed too, so every re-deal keeps the cell's model. Finished
games are re-dealt. God mode: every ship's shield refilled before each tick (nobody dies).

Per cell: kills per minute, draws, time to kill (a ship's last full shield to its death),
shots per kill, hit chance per real bullet (its flight ended on a ship), friendly-fire share
(real bullets ending on a team-mate), head-on kill share (the killer within 30 deg of the
victim's nose), store and power use, the agent's action entropy, firing share, stall share
(below 1.1 stall speed), hugging (a ship's centre inside an enemy's egg), close share (an enemy
within 30 m); 1v1 also: passes (local minima of the distance under the gun range, 1 s apart),
hits per pass, passes with no hit, hits in control (the shooter in range, nose within 30 deg of
the target, behind its beam), hits by aspect (head-on within 30 deg of the target's nose, rear
within 30 deg of its tail, side between), merge-turn time (after a pass with both noses within
60 deg of the other: until the heading changed by 90 deg, capped at 10 s). Each cell is written
to DIR/<cell>.json when done, with progress lines and clock-time ETAs.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from lean import game, scripted
from lean.contract import POWERS, compile_dynamic
from lean.game import card, core

HEAD_ON = math.cos(math.radians(30.0))
CONTROL_NOSE = math.cos(math.radians(30.0))
MERGE_NOSE = math.cos(math.radians(60.0))
FIRE_BIN, FIRE_BINS = 50.0, 30  # m: the nearest enemy's distance at a real shot, histogram


_ORIGINAL_CARDS: dict = {}  # by card key: the game's own card, before any cell's model


def set_model(config, state, model: dict):
    """The hit model on the state and on the cached ship card (so a re-deal keeps it), each
    cell from the game's own card (a cell's rows never carry into the next)."""
    core.ship_card(config)
    for key, value in list(core._SHIP_CARDS.items()):
        original = _ORIGINAL_CARDS.setdefault(key, value)
        core._SHIP_CARDS[key] = core.with_hit_model(original, **model)
    return state.replace(constants=core.ship_constants(config, state.type))


def inside_egg(state, egg: dict) -> torch.Tensor:
    """(E, N) a living ship's centre inside a living enemy's egg."""
    dx = state.x[:, :, None] - state.x[:, None, :]  # me minus him (E, me, him)
    dy = state.y[:, :, None] - state.y[:, None, :]
    cos, sin = state.heading_cos[:, None, :], state.heading_sin[:, None, :]
    along = dx * cos + dy * sin
    across = dy * cos - dx * sin
    semi = torch.where(along >= 0, egg["egg_front"], egg["egg_rear"])
    inside = (along / semi) ** 2 + (across / egg["egg_half_width"]) ** 2 < 1.0
    living = state.alive & state.exists
    enemy = state.team[:, :, None] != state.team[:, None, :]
    return (inside & enemy & living[:, None, :]).any(-1) & living


@torch.no_grad()
def run_cell(name, config, size, envs, ticks, model, egg, god, agent_call, step_call, device, seed):
    generator = torch.Generator(device).manual_seed(seed)
    sizes = torch.full((envs, 2), size, dtype=torch.long, device=device)
    state = game.reset(config, envs, sizes, generator, device)
    state = set_model(config, state, model)
    slots = state.x.shape[1]
    c = state.constants

    def first(item):
        return float(torch.as_tensor(item).flatten()[0])

    max_shield, gun_range = first(c["max_shield"]), first(c["gun_range"])
    stall = first(c["stall_speed"])
    h = config.tick_length
    acc = dict(
        living=0.0,
        kills=0.0,
        games=0.0,
        draws=0.0,
        real_ended=0.0,
        real_enemy=0.0,
        real_mate=0.0,
        fire=0.0,
        store=0.0,
        entropy=0.0,
        stall=0.0,
        hug=0.0,
        close=0.0,
        head_on_kills=0.0,
        collision_damage=0.0,
        bullet_damage=0.0,
    )
    power_counts = torch.zeros(POWERS, device=device)
    fire_distance = torch.zeros(FIRE_BINS, device=device)
    ttk = []
    last_full = torch.zeros(envs, slots, device=device)
    # 1v1 traces for passes, control and merges (CPU at the end).
    trace = size == 1
    if trace:
        distance_trace = torch.zeros(ticks, envs, device=device)
        hits_trace = torch.zeros(ticks, envs, 2, device=device)  # by shooter
        control_trace = torch.zeros(ticks, envs, 2, dtype=torch.bool, device=device)
        heading_trace = torch.zeros(ticks, envs, 2, device=device)
        merge_ok = torch.zeros(ticks, envs, dtype=torch.bool, device=device)
        aspect_hits = torch.zeros(3, device=device)  # head-on, side, rear
    started = time.time()
    for t in range(ticks):
        if god:
            state = state.replace(shield=torch.full_like(state.shield, max_shield))
        living = state.alive & state.exists
        uniforms = scripted.draw_uniforms(envs, slots, generator, device)
        action, choice = agent_call(state, uniforms)
        joint = choice.joint
        entropy = -(joint * joint.clamp_min(1e-30).log()).sum(-1)
        acc["entropy"] += float(entropy[living].sum())
        acc["living"] += float(living.sum())
        acc["fire"] += float((action[..., 2] == 1)[living].sum())
        acc["store"] += float((state.energy * state.constants["inverse_max_energy"])[living].sum())
        acc["stall"] += float((state.speed < 1.1 * stall)[living].sum())
        power_counts += torch.bincount(action[..., 1][living], minlength=POWERS)
        acc["hug"] += float(inside_egg(state, egg).sum())
        gap = torch.hypot(
            state.x[:, :, None] - state.x[:, None, :], state.y[:, :, None] - state.y[:, None, :]
        )
        enemy = (state.team[:, :, None] != state.team[:, None, :]) & living[:, None, :]
        nearest = torch.where(enemy, gap, torch.inf).amin(-1)
        acc["close"] += float(((nearest < 30.0) & living).sum())
        firing = (action[..., 2] == 1) & living & (state.cooldown <= 0)
        bins = (nearest[firing] / FIRE_BIN).clamp_max(FIRE_BINS - 1).long()
        fire_distance += torch.bincount(bins, minlength=FIRE_BINS).float()
        if trace:
            gx, gy = state.x[:, 1] - state.x[:, 0], state.y[:, 1] - state.y[:, 0]
            d = torch.hypot(gx, gy).clamp_min(1e-6)
            distance_trace[t] = d
            heading_trace[t] = torch.atan2(state.heading_sin, state.heading_cos)
            for s, sign in ((0, 1.0), (1, -1.0)):
                lx, ly = sign * gx / d, sign * gy / d
                aim = state.heading_cos[:, s] * lx + state.heading_sin[:, s] * ly
                away = state.heading_cos[:, 1 - s] * lx + state.heading_sin[:, 1 - s] * ly
                control_trace[t, :, s] = (d < gun_range) & (aim > CONTROL_NOSE) & (away > 0.0)
            # Both noses within 60 deg of the other (for merges).
            a0 = state.heading_cos[:, 0] * gx / d + state.heading_sin[:, 0] * gy / d
            a1 = -(state.heading_cos[:, 1] * gx / d + state.heading_sin[:, 1] * gy / d)
            merge_ok[t] = (a0 > MERGE_NOSE) & (a1 > MERGE_NOSE)
        before = state
        state, out = step_call(
            state, action, game.draw_noise(config, envs, slots, generator, device)
        )
        real = state.bullet_real
        ended = (out.shot_outcome >= 0) & real
        acc["real_ended"] += float(ended.sum())
        acc["real_enemy"] += float(((out.shot_outcome == 1) & real).sum())
        acc["real_mate"] += float(((out.shot_outcome == 2) & real).sum())
        bullet_hits = ((out.shot_outcome >= 1) & real).sum(-1)  # (E, N) by shooter
        damage = out.hit_damage
        acc["bullet_damage"] += float(damage.sum())
        # Hit aspects (1v1): the bullets' (turned) direction against the target's nose.
        if trace:
            hits_trace[t] = bullet_hits.float()
            nose_x = state.heading_cos.flip(1)[..., None]
            nose_y = state.heading_sin.flip(1)[..., None]
            hit_cos = -(state.bullet_dir_x * nose_x + state.bullet_dir_y * nose_y)
            hit = (out.shot_outcome == 1) & real
            aspect_hits[0] += float((hit & (hit_cos > HEAD_ON)).sum())
            aspect_hits[2] += float((hit & (hit_cos < -HEAD_ON)).sum())
            aspect_hits[1] += float((hit & (hit_cos.abs() <= HEAD_ON)).sum())
        # Deaths: time since the last full shield; the killer's aspect.
        full = before.shield >= 0.99 * max_shield
        last_full = torch.where(full, float(t), last_full)
        died = out.died
        if bool(died.any()):
            acc["kills"] += float(died.sum())
            ttk.extend(((t + 1 - last_full[died]) * h).tolist())
            killer = damage.argmax(1)  # (E, N victim) the attacker with the most damage
            kx = before.x.gather(1, killer) - before.x
            ky = before.y.gather(1, killer) - before.y
            kd = torch.hypot(kx, ky).clamp_min(1e-6)
            facing = (before.heading_cos * kx + before.heading_sin * ky) / kd
            acc["head_on_kills"] += float((died & (facing > HEAD_ON)).sum())
        done = out.done
        if bool(done.any()):
            acc["games"] += float(done.sum())
            acc["draws"] += float((done & (out.result == 2)).sum())
            fresh = game.deal(config, envs, slots, sizes, generator, device)
            state = game.take_deal(state, done, fresh)
            last_full = torch.where(done[:, None], float(t), last_full)
        if (t + 1) % 600 == 0 or t + 1 == ticks:
            elapsed = time.time() - started
            eta = time.strftime("%H:%M:%S", time.localtime(started + elapsed * ticks / (t + 1)))
            print(f"  {name}: tick {t + 1}/{ticks} kills {acc['kills']:.0f} ETA {eta}", flush=True)
    minutes = envs * ticks * h / 60.0
    living = max(acc["living"], 1.0)
    out = dict(
        cell=name,
        size=size,
        envs=envs,
        ticks=ticks,
        god=god,
        model={k: float(v) for k, v in model.items()},
        wall_s=round(time.time() - started, 1),
        kills_per_minute=acc["kills"] / minutes,
        games_ended=acc["games"],
        draws=acc["draws"],
        draw_share=acc["draws"] / max(acc["games"], 1.0),
        time_to_kill_median_s=float(np.median(ttk)) if ttk else None,
        time_to_kill_p25_s=float(np.percentile(ttk, 25)) if ttk else None,
        time_to_kill_p75_s=float(np.percentile(ttk, 75)) if ttk else None,
        real_bullets_ended=acc["real_ended"],
        shots_per_kill=acc["real_ended"] / max(acc["kills"], 1.0),
        hit_chance_per_bullet=(acc["real_enemy"] + acc["real_mate"]) / max(acc["real_ended"], 1.0),
        friendly_fire_share=acc["real_mate"] / max(acc["real_enemy"] + acc["real_mate"], 1.0),
        head_on_kill_share=acc["head_on_kills"] / max(acc["kills"], 1.0),
        fire_share=acc["fire"] / living,
        mean_store=acc["store"] / living,
        power_use=(power_counts / power_counts.sum().clamp_min(1)).tolist(),
        entropy=acc["entropy"] / living,
        stall_share=acc["stall"] / living,
        inside_enemy_egg_share=acc["hug"] / living,
        enemy_within_30m_share=acc["close"] / living,
        fire_distance_histogram_50m=fire_distance.tolist(),
        fire_distance_median_m=_median(fire_distance),
    )
    if trace:
        out.update(
            passes(distance_trace, hits_trace, control_trace, heading_trace, merge_ok, gun_range, h)
        )
        total = float(aspect_hits.sum())
        out["hits_by_aspect"] = dict(
            zip(("head_on", "side", "rear"), (aspect_hits / max(total, 1.0)).tolist())
        )
        out["damaging_hits"] = total
    return out


def _median(histogram: torch.Tensor) -> float | None:
    total = float(histogram.sum())
    if total == 0:
        return None
    cumulative = histogram.cumsum(0) / total
    index = int((cumulative >= 0.5).nonzero()[0])
    below = float(cumulative[index - 1]) if index else 0.0
    share = float(histogram[index]) / total
    return FIRE_BIN * (index + (0.5 - below) / max(share, 1e-9))


def passes(distance, hits, control, heading, merge_ok, gun_range, h):
    """Passes, hits per pass, control share of hits and merge-turn times from the 1v1 traces."""
    d = distance.cpu().numpy()  # (T, E)
    hits = hits.cpu().numpy()  # (T, E, 2)
    control = control.cpu().numpy()
    heading = heading.cpu().numpy()
    merge_ok = merge_ok.cpu().numpy()
    ticks, envs = d.shape
    merge = int(round(1.0 / h))
    per_pass, merges = [], []
    in_control = float((hits * control).sum())
    for e in range(envs):
        de = d[:, e]
        minima = [
            t
            for t in range(1, ticks - 1)
            if de[t] <= de[t - 1] and de[t] < de[t + 1] and de[t] < gun_range
        ]
        kept = []
        for t in minima:
            if not kept or t - kept[-1] >= merge:
                kept.append(t)
        if not kept:
            continue
        counts = np.zeros(len(kept))
        hit_ticks = np.nonzero(hits[:, e].sum(-1))[0]
        if len(hit_ticks):
            nearest = np.abs(hit_ticks[:, None] - np.array(kept)[None]).argmin(1)
            np.add.at(counts, nearest, hits[hit_ticks, e].sum(-1))
        per_pass.extend(counts.tolist())
        for t in kept:
            if not merge_ok[max(t - merge, 0), e]:
                continue
            for s in range(2):
                change = np.abs(np.angle(np.exp(1j * (heading[t:, e, s] - heading[t, e, s]))))
                over = np.nonzero(change > math.radians(90.0))[0]
                if len(over):
                    merges.append(min(over[0] * h, 10.0))
                elif ticks - t > 10.0 / h:
                    merges.append(10.0)
    per_pass = np.array(per_pass) if per_pass else np.zeros(1)
    total_hits = float(hits.sum())
    return dict(
        passes=int(len(per_pass)),
        hits_per_pass_mean=float(per_pass.mean()),
        passes_with_no_hit=float((per_pass == 0).mean()),
        hits_in_control_share=in_control / max(total_hits, 1.0),
        merge_turn_90_median_s=float(np.median(merges)) if merges else None,
        merges=len(merges),
    )


def spread_speed(mrad: float) -> float:
    """The per-axis spread (m/s) of `mrad` milliradians at the bullet launch speed."""
    constants = core.ship_card(core.GameConfig())
    launch = float(torch.as_tensor(constants["bullet_launch_speed"]).flatten()[0])
    return mrad * 1e-3 * launch


def plan(name: str, today: float):
    """[(cell name, setup, model overrides, god)]."""
    egg = card.DEFLECTOR_PRESETS["egg"]
    cells = []
    if name in ("godmode", "all"):
        for deflector in ("off", "egg"):
            turn = 0.0 if deflector == "off" else egg["deflector_turn"]
            model = dict(deflector_turn=turn, bullet_spread=today)
            cells.append((f"god-1v1-{deflector}", "1v1", model, True))
            cells.append((f"god-4v4-{deflector}", "4v4", model, True))
    if name in ("grid", "all"):
        for mrad in (None, 5.0, 10.0, 20.0):
            spread = today if mrad is None else spread_speed(mrad)
            label = "today" if mrad is None else f"{mrad:g}mrad"
            for deflector in ("off", "egg"):
                turn = 0.0 if deflector == "off" else egg["deflector_turn"]
                model = dict(deflector_turn=turn, bullet_spread=spread)
                for setup in ("1v1", "4v4"):
                    cells.append((f"{setup}-{deflector}-{label}", setup, model, False))
    if name in ("turns", "all"):
        for degrees in (30.0, 60.0):
            model = dict(deflector_turn=math.radians(degrees), bullet_spread=today)
            for setup in ("1v1", "4v4"):
                cells.append((f"{setup}-egg{degrees:g}-today", setup, model, False))
    if name in ("extra", "all"):
        for damage in (7.0, 14.0):
            model = dict(
                deflector_turn=egg["deflector_turn"], bullet_spread=today, hit_damage=damage
            )
            for setup in ("1v1", "4v4"):
                cells.append((f"{setup}-egg-today-damage{damage:g}", setup, model, False))
        model = dict(deflector_turn=0.0, bullet_spread=today, nose_cone=1.0)
        for setup in ("1v1", "4v4"):
            cells.append((f"{setup}-off-today-nosecone", setup, model, False))
        for front in (24.0, 80.0):
            model = dict(deflector_turn=egg["deflector_turn"], bullet_spread=today, egg_front=front)
            for setup in ("1v1", "4v4"):
                cells.append((f"{setup}-egg-today-front{front:g}", setup, model, False))
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="grid")
    parser.add_argument("--only", nargs="*", default=None, help="cell names to run")
    parser.add_argument("--envs-1v1", type=int, default=512)
    parser.add_argument("--envs-4v4", type=int, default=256)
    parser.add_argument("--ticks", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", nargs="*", default=[], help="look-ahead rows name=value")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    device = torch.device("cuda")
    agent = scripted.ScriptedConfig().replace(
        **{item.split("=")[0]: float(item.split("=")[1]) for item in arguments.set}
    )
    today = float(torch.as_tensor(core.ship_card(core.GameConfig())["spread_speed"]).flatten()[0])
    configs = {
        "1v1": game.GameConfig(field_count=0, fog=False),
        "4v4": game.GameConfig(),
    }
    steps = {
        setup: compile_dynamic(lambda s, a, n, config=config: game.step(config, s, a, n))
        for setup, config in configs.items()
    }

    def agent_call(state, uniforms):
        return scripted.decide(agent, state, None, None, uniforms)

    agent_call = compile_dynamic(agent_call)
    egg = {k: card.DEFLECTOR_PRESETS["egg"][k] for k in ("egg_front", "egg_rear", "egg_half_width")}
    arguments.out.mkdir(parents=True, exist_ok=True)
    cells = plan(arguments.plan, today)
    if arguments.only:
        cells = [cell for cell in cells if cell[0] in arguments.only]
    started = time.time()
    for index, (name, setup, model, god) in enumerate(cells):
        size = 1 if setup == "1v1" else 4
        envs = arguments.envs_1v1 if size == 1 else arguments.envs_4v4
        shape = {**egg, **{k: model[k] for k in ("egg_front",) if k in model}}
        record = run_cell(
            name,
            configs[setup],
            size,
            envs,
            arguments.ticks,
            model,
            shape,
            god,
            agent_call,
            steps[setup],
            device,
            arguments.seed,
        )
        record["agent_set"] = arguments.set
        (arguments.out / f"{name}.json").write_text(json.dumps(record, indent=1))
        elapsed = time.time() - started
        eta = time.strftime("%H:%M", time.localtime(started + elapsed * len(cells) / (index + 1)))
        print(json.dumps(record), flush=True)
        print(f"cell {index + 1}/{len(cells)} done; all cells ETA {eta}", flush=True)


if __name__ == "__main__":
    main()
