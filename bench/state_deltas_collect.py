"""Collect per-tick ship states in self-play for the state-delta analysis
(docs/rewrite/sections/state-deltas-brief.md).

One player in both seats of every game, fog on, staggered clocks, on the CPU. Each segment
of `--ticks` ticks of `--environments` games is written to its own file as soon as it is
played (nothing accumulates in memory):

    <out>/<player>/<preset>/segment_<k>.npz

with, per tick t and ship row (team 0's view order, all N ships):
    rows (L, B, N, C) float32: the privileged ship row at the start of tick t (SHIP_COLUMNS);
    power (L, B, N) int8: the power index acted on tick t (0 coast, 1 boost, 2 reverse);
    stick (L, B, N) int8;
    hit (L, B, N) bool: the ship took shot damage on tick t;
    boundary (L, B, N) bool: it took boundary damage on tick t;
    zone (L, B, N) bool: it is inside a Frontline zone at t (False in other modes);
    done (L, B) bool: the environment reset after tick t.

Run (CPU only, capped):
    systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 \
        nice -n 15 env CUDA_VISIBLE_DEVICES="" uv run --no-sync python \
        bench/state_deltas_collect.py --player scripted --preset deathmatch
"""

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from bnb.contracts import Obs
from bnb.contracts.obs import GLOBAL, SHIP_COLUMNS
from bnb.controllers.scripted import Scripted
from bnb.game import api
from bnb.game.handling.characteristics import characteristics
from bnb.runs import checkpoint
from bnb.runtime.train import model_controller

PARTS = ("ships", "fields", "globals", "zones")
CALM_FERN = "/home/vizia/avesnova/runs-v2/players-calm-fern/step_100663296-exponential_average"


def build(player: str, game, device):
    """(controller, ship card) of a player."""
    if player == "scripted":
        types = api.ship_types(device)
        return Scripted(characteristics(types, api.rules_of(game)), game.ships_per_team), types
    path = CALM_FERN if player == "calm-fern" else player
    record = checkpoint.read_policy(path).raw
    model = checkpoint.load_policy(path, device).eval()
    card = record.get("ship_card") or {}
    return model_controller(model, game), api.ship_types(device, **card)


@torch.no_grad()
def main() -> None:
    """Play the segments and write each one to disk."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--environments", type=int, default=64)
    parser.add_argument("--ticks", type=int, default=1024)
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", default="/tmp/bnb-scratch-deltas/segments")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    game = api.PRESETS[args.preset]
    rules = api.rules_of(game)
    controller, types = build(args.player, game, device)
    B, S, N, T = args.environments, game.ships_per_team, game.num_ships, api.TEAMS
    out = pathlib.Path(args.out) / pathlib.Path(args.player).name / args.preset
    out.mkdir(parents=True, exist_ok=True)

    state = api.reset(rules, types, api.uniforms(rules, B), torch.ones(B, dtype=torch.bool))
    clocks = api.episode_ticks(rules, B, device)
    state = api.start_clocks(state, torch.randint(0, 1 << 30, (B,)) % clocks)
    state, obs, privileged = api.perceive(rules, state)
    carry = controller.init_carry(B * T, device)
    order = api.view_order(S, device)[0]  # sim index of each row of team 0's view
    meta = dict(
        player=args.player,
        preset=args.preset,
        environments=B,
        ticks=args.ticks,
        tick_length=rules.tick_length,
        columns=list(SHIP_COLUMNS),
        seed=args.seed,
        card={k: [float(v) for v in getattr(types, k).tolist()] for k in api.default_card()},
    )
    (out / "meta.json").write_text(json.dumps(meta, default=str, indent=1))
    started = time.time()
    for segment in range(args.segments):
        L = args.ticks
        rows = np.zeros((L, B, N, len(SHIP_COLUMNS)), np.float32)
        power = np.zeros((L, B, N), np.int8)
        stick = np.zeros((L, B, N), np.int8)
        hit = np.zeros((L, B, N), bool)
        boundary = np.zeros((L, B, N), bool)
        zone = np.zeros((L, B, N), bool)
        done = np.zeros((L, B), bool)
        for t in range(L):
            truth = privileged.ships[:, 0]  # (B, N, C) team 0's rows, every ship's truth
            rows[t] = truth.numpy()
            view = Obs(**{part: getattr(obs, part).flatten(0, 1) for part in PARTS})
            reset = view.globals[..., GLOBAL["reset"]] > 0.5
            action, new_carry, _ = controller.step(carry, view, reset)
            if new_carry is not None:
                carry = new_carry
            packed = action.packed().reshape(B, N, 3)  # sim order (team-major)
            power[t] = packed[:, order, 0].numpy()
            stick[t] = packed[:, order, 1].numpy()
            shown = api.display(state)
            if game.mode == "frontline":
                dx = shown["ships.x"][:, :, None] - shown["mode.zone_x"][:, None, :]
                dy = shown["ships.y"][:, :, None] - shown["mode.zone_y"][:, None, :]
                inside = (dx * dx + dy * dy < shown["mode.zone_radius"][:, None, :] ** 2).any(-1)
                zone[t] = inside[:, order].numpy()
            tick_action = action.unpack(packed)
            moved, sweep = api.move(rules, types, state, tick_action, api.spread(rules, B))
            state, events = api.resolve(
                rules, types, moved, sweep, tick_action, api.uniforms(rules, B)
            )
            hit[t] = (events.damage.sum(1) > 0)[:, order].numpy()
            boundary[t] = (events.boundary_damage > 0)[:, order].numpy()
            done[t] = events.done.numpy()
            state, obs, privileged = api.perceive(rules, state)
        np.savez(
            out / f"segment_{segment}.npz",
            rows=rows,
            power=power,
            stick=stick,
            hit=hit,
            boundary=boundary,
            zone=zone,
            done=done,
        )
        elapsed = time.time() - started
        eta = elapsed / (segment + 1) * (args.segments - segment - 1)
        print(
            f"{args.player} {args.preset} segment {segment + 1}/{args.segments} "
            f"{elapsed:.0f}s, {eta:.0f}s to go",
            flush=True,
        )


if __name__ == "__main__":
    main()
