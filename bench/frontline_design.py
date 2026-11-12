"""Frontline design search: candidate capture rules side by side in one compiled rollout.

    PYTHONPATH=src python -m bench.frontline_design --variants FILE.json --out FILE.jsonl \
        [--kind frontline|deathmatch] [--ships 5 --radius 7800] [--games 256] [--slots 4]
        [--per-slot 128]

Each variant is a JSON object (every field optional, off by default):

- `label`;
- `lock` (s): after a capture the captured zone's meter stands still this long (no ship
  counts in it);
- `early`: the lock ends as soon as the capturing team has no living ship in the zone;
- `stale`: the losing team's ships inside a zone when it falls do not count in it until
  they leave it (or die);
- `forward`: a team's ships respawn at its own defense zone (its frontmost owned zone)
  instead of its spawn zone, nose toward the zone it attacks (respawns only; match start
  unchanged; the spawn's offset in the disc is kept);
- `capture_scale`: the capture time times this;
- `progressive` g: a zone's capture time times (1 + g)^depth, depth = the attacker's lead
  in captures (the front in its frame, at least 0);
- `count_scale` c: the meter rate H(c |net|) instead of H(|net|) (c = 5 / n is a capture
  by the 5v5-equivalent lead);
- `extension` E (s): a flat extension, on a capture the captured zone's next recapture
  needs E s more: its meter waits while E is drained by an attacking majority in the zone
  at the capture rate (H_n s a second for n ahead, a lone attacker 1 s a second); E never
  decays or recharges;
- `hold` H (s): a decaying hold, the extension H that also decays 1 s a second (gone
  after H s with no attacker);
- `respawn_minimum` (s) and `wave_interval` (s): respawn waves (both 0: the game's fixed
  wait). A dead ship respawns at its team's first wave after it has been dead
  `respawn_minimum` s; each team's waves come every `wave_interval` s (0: no waves, the ship
  respawns once the minimum has passed), from a random phase per team at the episode start;
- `wave_effect`: on a capture, the team that lost the zone has its wave clock 0 untouched,
  1 reset (its next wave a full interval away), 2 delayed by `wave_delay` s, 3 pushed back
  one interval (a wave skipped).

All parameters are per-environment tensors read inside the compiled tick, so `--slots`
variants run side by side in one batch of slots x per-slot environments and every variant
shares one compile. The game runs with threshold `--threshold` (default 10, effectively
off) and clock `--episode-seconds` (default 1200): the scripted agent reads neither, so
`bench.frontline_design_report` replays any lower threshold and shorter clock from the
traced front. Records: one JSON line per variant per batch (the `bench.mode_dynamics`
format, the probe histograms per variant). Bench-only: nothing here is in the game.
"""

import argparse
import dataclasses
import json
import math
import pathlib
import time

import torch

from bench import mode_dynamics as dynamics
from bnb.contracts import Pool
from bnb.controllers import scripted
from bnb.game import api
from bnb.game import state as state_module
from bnb.game.handling.characteristics import characteristics
from bnb.game.modes import frontline
from bnb.runtime.rollout import Rollout

FIELDS = ("lock", "early", "stale", "forward", "capture_scale", "progressive", "count_scale",
          "extension", "hold", "respawn_minimum", "wave_interval", "wave_effect",
          "wave_delay")  # fmt: skip
DEFAULTS = dict(lock=0.0, early=0.0, stale=0.0, forward=0.0, capture_scale=1.0,
                progressive=0.0, count_scale=1.0, extension=0.0, hold=0.0,
                respawn_minimum=0.0, wave_interval=0.0, wave_effect=0.0,
                wave_delay=0.0)  # fmt: skip
WAIT_BIN = 0.25  # s, the respawn-wait histogram's bins
WAIT_BINS = 400  # (the last one holds every longer wait)
RESET, DELAY, SKIP = 1, 2, 3  # wave_effect codes


class GroupProbe(dynamics.Probe):
    """The mode-dynamics probe with its histograms kept per slot (environment block)."""

    def __init__(self, environments: int, ships_per_team: int, device, slots: int) -> None:
        super().__init__(environments, ships_per_team, device)
        self.slots = slots
        self.group = (torch.arange(environments, device=device) * slots) // environments
        for name in dynamics.HISTOGRAMS:
            self.buffers[name] = torch.zeros(slots * dynamics.BINS, dtype=torch.long, device=device)

    def _count(self, name: str, ticks: torch.Tensor, mask: torch.Tensor) -> None:
        bins = (ticks // self.ticks_per_bin).clamp(0, dynamics.BINS - 1)
        group = self.group.view(-1, *([1] * (bins.dim() - 1))).expand_as(bins)
        index = group * dynamics.BINS + bins
        self.buffers[name].index_add_(0, index.flatten(), mask.flatten().long())

    def histogram(self, name: str, slot: int) -> list[int]:
        """One slot's histogram (host list)."""
        bins = dynamics.BINS
        return self.buffers[name][slot * bins : (slot + 1) * bins].cpu().tolist()


class Design:
    """The candidate rules as per-environment tensors (`FIELDS`), acting inside the compiled
    tick through `frontline.capture_step` and `api.resolve` (`bench.mode_dynamics.install`)."""

    hold = True  # install the capture-step wrapper
    active = True  # run after_resolve

    def __init__(
        self, environments: int, ships_per_team: int, tick_length: float, device, slots: int = 1
    ):
        self.tick_length = tick_length
        self.S = ships_per_team
        self.slots = slots
        self.group = (torch.arange(environments, device=device) * slots) // environments
        N = 2 * ships_per_team
        self.parameters = {
            name: torch.full((environments,), DEFAULTS[name], dtype=torch.float32, device=device)
            for name in FIELDS
        }
        self.buffers = dict(
            lock_zone=torch.full((environments,), -1, dtype=torch.long, device=device),
            lock_left=torch.zeros(environments, dtype=torch.float32, device=device),
            extra_left=torch.zeros(environments, dtype=torch.float32, device=device),
            winner=torch.zeros(environments, dtype=torch.long, device=device),
            stale=torch.zeros((environments, N), dtype=torch.bool, device=device),
            front=torch.zeros(environments, dtype=torch.long, device=device),
            # respawn waves: ticks each ship has been dead (0 alive or dying this tick), ticks
            # to each team's next wave (the wave respawns on that tick)
            dead_for=torch.zeros((environments, N), dtype=torch.long, device=device),
            wave_left=torch.ones((environments, 2), dtype=torch.long, device=device),
            live=torch.ones(environments, dtype=torch.bool, device=device),
            # per slot, over the first games: every respawn's wait (s) summed, and binned
            wait_sum=torch.zeros(slots, dtype=torch.float64, device=device),
            wait=torch.zeros(slots * WAIT_BINS, dtype=torch.long, device=device),
        )

    def wait_histogram(self, slot: int) -> list[int]:
        """One slot's respawn-wait histogram (host list, `WAIT_BIN` s bins)."""
        return self.buffers["wait"][slot * WAIT_BINS : (slot + 1) * WAIT_BINS].cpu().tolist()

    def _phase(self, shape) -> torch.Tensor:
        """A random wave phase per team: ticks to the first wave, uniform on 1..interval."""
        interval = torch.round(self.parameters["wave_interval"] / self.tick_length)[:, None]
        draw = torch.rand(shape, device=interval.device)
        return (draw * interval).floor().long().clamp_min(0) + 1

    def set(self, variants: list[dict]) -> None:
        """Fill the parameters in place: variant k on the k-th block of environments."""
        environments = self.buffers["front"].shape[0]
        block = environments // len(variants)
        for name in FIELDS:
            values = [float(variant.get(name, DEFAULTS[name])) for variant in variants]
            self.parameters[name].copy_(
                torch.tensor(values, dtype=torch.float32).repeat_interleave(block)
            )

    def clear(self) -> None:
        """Reset the rule state for a new batch (in place)."""
        b = self.buffers
        b["lock_zone"].fill_(-1)
        b["lock_left"].zero_()
        b["extra_left"].zero_()
        b["winner"].zero_()
        b["stale"].fill_(False)
        b["front"].zero_()
        b["dead_for"].zero_()
        b["wave_left"].copy_(self._phase(b["wave_left"].shape))
        b["live"].fill_(True)
        b["wait_sum"].zero_()
        b["wait"].zero_()

    def capture_step(self, original):
        """`frontline.capture_step` with the candidate rules."""
        p, b = self.parameters, self.buffers

        def step(team, inside, roles, progress, attack, defense, front, tick, seconds):
            Z = roles.shape[-1]
            zone = torch.arange(Z, device=roles.device)
            held = zone == b["lock_zone"][:, None]  # (B, Z)
            present = inside.transpose(-1, -2)  # (B, Z, N)
            on_winner = team[None, :] == b["winner"][:, None]  # (B, N)
            in_held = (inside & held[:, None, :]).any(-1)  # (B, N): inside the held zone
            winner_in = (in_held & on_winner).any(-1)
            lock_left = torch.where((p["early"] > 0) & ~winner_in, 0.0, b["lock_left"])
            stale = b["stale"] & in_held  # a stale ship stops being stale once it leaves
            eligible = inside & ~(stale[:, :, None] & held[:, None, :])
            # the extension or hold: an attacking majority in the held zone drains it at the
            # capture rate while the meter waits; a hold also decays 1 s a second
            on_team0 = (team == 0)[None, None, :]
            eligible_present = eligible.transpose(-1, -2)
            net = (eligible_present & on_team0).sum(-1) - (eligible_present & ~on_team0).sum(-1)
            team0_attacks = roles == frontline.TEAM1_DEFENSE
            ahead = torch.where(team0_attacks, net, -net)
            rate = frontline.harmonic(ahead.clamp_min(0)) * (ahead > 0)
            drained = torch.where(held, rate, 0.0).sum(-1) * tick
            decay = torch.where(p["hold"] > 0, tick, 0.0)
            extra_left = b["extra_left"]
            locked = held & ((lock_left > 0) | (extra_left > 0))[:, None]
            counted = eligible & ~locked[:, None, :]

            # the capture time: scale, progressive depth, count scale
            net = (present & counted.transpose(-1, -2) & on_team0).sum(-1) - (
                present & counted.transpose(-1, -2) & ~on_team0
            ).sum(-1)
            depth = torch.where(team0_attacks, front[:, None], -front[:, None]).clamp_min(0)
            lead = net.abs().clamp_min(1)
            count = frontline.harmonic(lead) / frontline.harmonic(lead * p["count_scale"][:, None])
            capture_seconds = (
                seconds * p["capture_scale"][:, None]
                * torch.pow(1.0 + p["progressive"][:, None], depth.float()) * count
            )  # fmt: skip
            capture = original(
                team, counted, roles, progress, attack, defense, front, tick,
                capture_seconds,
            )  # fmt: skip
            done = capture.team0_captured | capture.team1_captured
            winner = torch.where(capture.team0_captured, 0, 1)
            role = torch.where(
                capture.team0_captured, frontline.TEAM0_DEFENSE, frontline.TEAM1_DEFENSE
            )
            taken = (capture.roles == role[:, None]).long().argmax(-1)
            # the losing team's ships inside the zone it lost (by this tick's positions)
            in_taken = inside.gather(
                2, taken[:, None, None].expand(-1, inside.shape[1], 1)
            ).squeeze(-1)
            loser_in = in_taken & (team[None, :] != winner[:, None])
            fresh_stale = done[:, None] & (p["stale"] > 0)[:, None] & loser_in
            b["stale"].copy_(torch.where(done[:, None], fresh_stale, stale))
            b["lock_zone"].copy_(torch.where(done, taken, b["lock_zone"]))
            b["winner"].copy_(torch.where(done, winner, b["winner"]))
            b["lock_left"].copy_(torch.where(done, p["lock"], (lock_left - tick).clamp_min(0.0)))
            b["extra_left"].copy_(
                torch.where(
                    done,
                    p["extension"] + p["hold"],
                    (extra_left - drained - decay).clamp_min(0.0),
                )
            )
            return capture

        return step

    def after_resolve(self, state, events, ships_per_team: int):
        """Forward spawning: this tick's respawns moved from the team's spawn zone to its
        defense zone, nose toward the zone it attacks. Respawn waves: the dead ships' waits
        set so that they respawn on their team's wave (`waves`)."""
        b, p = self.buffers, self.parameters
        state = self.waves(state, events, ships_per_team)
        mode, ships = state.mode, state.ships
        fresh = events.done
        team = torch.arange(2 * ships_per_team, device=mode.front.device) // ships_per_team
        moving = events.spawned & ~fresh[:, None] & (p["forward"] > 0)[:, None]

        def centre(roles_by_team):
            wanted = torch.where(team == 0, roles_by_team[0], roles_by_team[1])  # (N,)
            pick = (mode.role[:, None, :] == wanted[None, :, None]).long().argmax(-1)
            return mode.zone_x.gather(1, pick), mode.zone_y.gather(1, pick)

        x, y = ships.x, ships.y
        spawn_x, spawn_y = centre((frontline.TEAM0_SPAWN, frontline.TEAM1_SPAWN))
        own_x, own_y = centre((frontline.TEAM0_DEFENSE, frontline.TEAM1_DEFENSE))
        target_x, target_y = centre((frontline.TEAM1_DEFENSE, frontline.TEAM0_DEFENSE))
        new_x, new_y = own_x + (x - spawn_x), own_y + (y - spawn_y)
        to_x, to_y = target_x - new_x, target_y - new_y
        length = torch.sqrt(to_x * to_x + to_y * to_y).clamp_min(1e-6)
        ships = state_module.replace(
            ships,
            x=torch.where(moving, new_x, x),
            y=torch.where(moving, new_y, y),
            heading_cos=torch.where(moving, to_x / length, ships.heading_cos),
            heading_sin=torch.where(moving, to_y / length, ships.heading_sin),
        )
        b["front"].copy_(mode.front.long())
        return state_module.replace(state, ships=ships)

    def waves(self, state, events, ships_per_team: int):
        """Respawn waves (inside the compiled tick, after the resolve). A ship released here
        gets respawn_wait 1, so the next tick's step spawns it; a ship held gets 2 (the step
        counts it to 1, never due). A ship's wait is (ticks dead here) + 1. The capture
        effect acts on the wave clock of the team the front stepped against. Every respawn's
        wait (episode resets aside) is counted per slot over each environment's first game."""
        b, p = self.buffers, self.parameters
        ships, mode = state.ships, state.mode
        tick = self.tick_length
        fresh = events.done
        B = fresh.shape[0]
        team = torch.arange(2 * ships_per_team, device=fresh.device) // ships_per_team
        # the waits of this tick's respawns (the step spawned them)
        spawned = events.spawned & ~fresh[:, None]
        wait = (b["dead_for"] + 1).double() * tick
        counted = spawned & b["live"][:, None]
        b["wait_sum"].index_add_(0, self.group, torch.where(counted, wait, 0.0).sum(-1))
        bins = (wait / WAIT_BIN).long().clamp(0, WAIT_BINS - 1)
        index = self.group[:, None] * WAIT_BINS + bins
        b["wait"].index_add_(0, index.flatten(), counted.flatten().long())
        dead = ~ships.alive
        dead_for = torch.where(dead & ~events.died, b["dead_for"] + 1, 0)
        # the wave clocks: the capture effect on the losing team, then this tick's waves
        interval = torch.round(p["wave_interval"] / tick).long()[:, None]  # (B, 1)
        delay = torch.round(p["wave_delay"] / tick).long()[:, None]
        step = mode.front.long() - b["front"]
        captured = (step != 0) & ~fresh
        loser = (step > 0).long()  # team 1 loses a zone when the front rises
        teams = torch.arange(2, device=fresh.device)
        losing = captured[:, None] & (teams[None, :] == loser[:, None])
        effect = p["wave_effect"][:, None]
        left = b["wave_left"]
        left = torch.where(losing & (effect == RESET), interval, left)
        left = torch.where(losing & (effect == DELAY), left + delay, left)
        left = torch.where(losing & (effect == SKIP), left + interval, left)
        fires = left <= 1  # (B, 2): this team's wave is on the next tick
        left = torch.where(fires, interval.expand(-1, 2), left - 1)
        left = torch.where(fresh[:, None], self._phase((B, 2)), left)
        b["wave_left"].copy_(left)
        on = ((p["respawn_minimum"] > 0) | (p["wave_interval"] > 0))[:, None]  # (B, 1)
        minimum = torch.round(p["respawn_minimum"] / tick)[:, None]
        team_fires = fires[:, team]  # (B, N)
        waited = (dead_for + 1).float() >= minimum
        release = dead & waited & ((interval == 0) | team_fires)
        held_wait = torch.where(release, 1, 2).to(ships.respawn_wait.dtype)
        respawn_wait = torch.where(on & dead, held_wait, ships.respawn_wait)
        b["dead_for"].copy_(torch.where(fresh[:, None], 0, dead_for))
        b["live"].copy_(b["live"] & ~fresh)
        ships = state_module.replace(ships, respawn_wait=respawn_wait)
        return state_module.replace(state, ships=ships)


def main() -> None:
    """The run the module describes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--kind", choices=("frontline", "deathmatch"), default="frontline")
    parser.add_argument("--preset", default=None, help="default real_scale_KIND")
    parser.add_argument("--ships", type=int, default=None)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--games", type=int, default=256, help="per variant")
    parser.add_argument("--slots", type=int, default=4, help="variants side by side")
    parser.add_argument("--per-slot", type=int, default=128, help="environments per variant")
    parser.add_argument("--episode-seconds", type=float, default=1200.0)
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--lineup", default="jujak,yongma")
    parser.add_argument("--fog", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args()
    variants = json.loads(arguments.variants.read_text())
    kind = arguments.kind
    if arguments.preset is None:
        arguments.preset = f"real_scale_{kind}"
    device = torch.device(arguments.device)
    shim = argparse.Namespace(
        preset=arguments.preset, ships=arguments.ships, radius=arguments.radius,
        episode_seconds=arguments.episode_seconds, respawn_seconds=None, kind=kind,
    )  # fmt: skip
    game, types, rules = dynamics.game_of(shim)
    types = api.ship_types(ship_set=game.ship_set, device=device)
    done_labels: dict[str, int] = {}
    if arguments.out.exists():
        for line in arguments.out.read_text().splitlines():
            entry = json.loads(line)
            done_labels[entry["label"]] = done_labels.get(entry["label"], 0) + len(entry["games"])
    slots, per = arguments.slots, arguments.per_slot
    environments = slots * per
    episode = int(rules.pools[0].episode_ticks)
    length = episode + 2
    card = characteristics(types, rules)
    agent = scripted.Scripted(card, rules.ships_per_team, temperature=1.0)
    probe = GroupProbe(environments, rules.ships_per_team, device, slots)
    design = Design(environments, rules.ships_per_team, rules.tick_length, device, slots)
    dynamics.install(probe, kind == "frontline", design)
    rollout = Rollout(
        dataclasses.replace(game),
        types,
        (Pool("dynamics", 1.0, ("a", "b"), fog_share=arguments.fog),),
        {"a": agent, "b": agent},
        "a",
        environments,
        32,
        device,
        staggered=False,
        ship_type=dynamics.lineup_of(arguments.lineup, game, environments),
    )
    rollout.rules = api.tuning.tuned(rollout.rules, {"mode.front_threshold": arguments.threshold})
    rollout.carry["trace"] = dynamics.blank_trace(environments, length, device)
    rollout.stages = rollout.stages + (
        torch.compile(dynamics.trace_stage(length), fullgraph=True, dynamic=False),
    )
    threshold = rollout.rules.pools[0].mode.front_threshold
    pending = [v for v in variants if done_labels.get(v["label"], 0) < arguments.games]
    rounds = [pending[i : i + slots] for i in range(0, len(pending), slots)]
    started = time.time()
    batches_left = (
        sum(math.ceil((arguments.games - done_labels.get(v["label"], 0)) / per) for v in pending)
        / slots
    )
    for group in rounds:
        group = group + [group[-1]] * (slots - len(group))  # pad with a repeat (not recorded)
        real = len({v["label"] for v in group})
        design.set(group)
        index = 0
        while min(done_labels.get(v["label"], 0) for v in group[:real]) < arguments.games:
            began = time.time()
            torch.manual_seed(arguments.seed + 1000 * index)
            trace = rollout.carry.pop("trace")
            rollout.restart()
            rollout.carry["trace"] = trace
            for name, value in dynamics.blank_trace(environments, length, device).items():
                trace[name].copy_(value)
            probe.clear()
            design.clear()
            rollout.carry["seat"].copy_(torch.arange(environments, device=device) % 2)
            for _ in range(math.ceil(length / rollout.length)):
                rollout.run()
            host = {name: value.cpu() for name, value in trace.items()}
            games = dynamics.first_games(host, episode, threshold, kind)
            first_hit = probe.buffers["first_hit"].cpu().tolist()
            for record, hit in zip(games, first_hit):
                record["first_hit"] = hit
            written = set()
            for slot, variant in enumerate(group):
                if variant["label"] in written:
                    continue
                written.add(variant["label"])
                entry = dict(
                    label=variant["label"], kind=kind, preset=arguments.preset,
                    ships=rules.ships_per_team, radius=game.arena_radius,
                    tick_length=rules.tick_length, episode_ticks=episode,
                    zone_radius=rollout.rules.pools[0].mode.zone_radius,
                    capture_seconds=rollout.rules.pools[0].mode.capture_seconds,
                    front_threshold=threshold, lineup=arguments.lineup, fog=arguments.fog,
                    respawn_seconds=rollout.rules.pools[0].respawn_ticks * rules.tick_length,
                    variant=variant,
                    histograms={n: probe.histogram(n, slot) for n in dynamics.HISTOGRAMS},
                    respawn_wait=dict(
                        bin_seconds=WAIT_BIN,
                        histogram=design.wait_histogram(slot),
                        total_seconds=float(design.buffers["wait_sum"][slot]),
                    ),
                    games=games[slot * per : (slot + 1) * per],
                )  # fmt: skip
                with arguments.out.open("a") as file:
                    file.write(json.dumps(entry) + "\n")
                done_labels[variant["label"]] = done_labels.get(variant["label"], 0) + per
            index += 1
            batches_left -= 1
            seconds = time.time() - began
            print(
                f"{','.join(sorted(written))}: batch {seconds:.0f} s, "
                f"ETA {dynamics.clock(max(0, batches_left) * seconds)} "
                f"(elapsed {time.time() - started:.0f} s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
