"""The memory hand-over test (rl-round-five brief, part 2): when a league slot's opponent is
cut from policy A to policy B mid-game, how far is B's action distribution from what it
would be had B played from the game's start, given A's carry or a reset one?

A short CPU rollout of A against itself (the learner's seat plays A too), every game started
fresh. Until the switch, A drives the opponent seat and each B (its truth) shadows it on the
same observations. At the switch (`--switch` seconds into every game) three versions of
each B are compared on the same observations from then on:

- (a) the truth: B with its own carry, warmed from the game's start; the first B's truth
  drives the seat, the other B's shadow it;
- (b) B from A's carry (the whole carry: trunk state, convolution buffer and belief);
- (c) B from a reset carry: A's carry with the trunk (state, convolution buffer) zeroed and
  the belief part (observations, last sightings, forecasts), the previous actions and the
  position offset kept, as the league's `carry_at_cut="reset"` does.

Every model draws the same position offset at a reset (the model's frame; one seed per
tick for all of them), so the comparison sees the memory, not the frame: with offsets of
their own, two copies of one policy on one game differ by the policy's sensitivity to the
offset (a first version of this bench, which drew them apart, measured a floor of KL 0.11
for 73M -> 88M that never decayed). Every shadow's
previous action is the joint action the seat actually played (it is what the ships did).
Scored per own living ship: KL(a || x) over the 42 joint actions, by seconds after the
switch, over environments that have not reset since the switch. `gap` is KL(a || A), what
the switch would cost with no hand-over at all (A playing on).

    uv run --no-sync python -m bench.memory_handover A B [B ...] [--mode frontline] \
        [--json out.json] [--progress progress.log]
"""

import argparse
import dataclasses
import json
import pathlib
import time

import torch
import torch.nn.functional as F
from torch.utils import _pytree

from bnb.contracts import Outputs, Pool
from bnb.contracts.action import from_joint
from bnb.contracts.obs import SHIP
from bnb.controllers.yemong import YemongController
from bnb.game import api
from bnb.game.modes.rules import PRESETS
from bnb.models.yemong import policy
from bnb.runs import checkpoint
from bnb.runtime.rollout import Rollout

LEARNER, SEAT = "learner", "opponent"
BINS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)  # seconds after the switch


def reset_trunk(carry):
    """The carry as a cut with `carry_at_cut="reset"` leaves it: the trunk zeroed on every
    view (`YemongController.forget`), the rest kept."""
    views = carry.state.shape[0]
    return YemongController.forget(carry, torch.ones(views, dtype=torch.bool))


def _clone(tree):
    return _pytree.tree_map_only(torch.Tensor, lambda x: x.clone(), tree)


def kl(reference: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """KL(reference || other) over the last axis of two logit tensors."""
    log_p, log_q = F.log_softmax(reference.float(), -1), F.log_softmax(other.float(), -1)
    return (log_p.exp() * (log_p - log_q)).sum(-1)


class HandOver:
    """The opponent seat (see the module docstring). Python-side state: run uncompiled.
    seconds: the B's, by name."""

    def __init__(self, first, seconds: dict, ships_per_team: int, num_ships: int, switch_tick: int):
        self.first, self.seconds = first, seconds
        self.ships_per_team, self.num_ships = ships_per_team, num_ships
        self.switch_tick, self.tick = switch_tick, 0
        self.reads = ("ships", "fields", "globals", "zones")
        self.carries: dict = {}
        self.fresh: torch.Tensor | None = None  # environments without a reset since the switch
        # (tick after the switch, B, version, KL rows)
        self.samples: list[tuple[int, str, str, torch.Tensor]] = []

    def init_carry(self, views: int, device: torch.device):
        """A fresh carry for A and each truth B; the seat keeps them (the rollout's is None)."""
        carry = self.first.initial_carry(views, self.ships_per_team, self.num_ships, device)
        self.carries = {("A", "A"): carry}
        for name in self.seconds:
            self.carries[(name, "a")] = _clone(carry)
        return None

    def _model(self, key):
        return self.first if key[0] == "A" else self.seconds[key[0]]

    @torch.no_grad()
    def step(self, carry, obs, reset):
        """One decision: every B (and A) on the observation, the driver's sample played."""
        tick, self.tick = self.tick, self.tick + 1
        if tick == self.switch_tick:
            for name in self.seconds:
                self.carries[(name, "b")] = _clone(self.carries[("A", "A")])
                self.carries[(name, "c")] = reset_trunk(_clone(self.carries[("A", "A")]))
            self.fresh = torch.ones_like(reset)
        logits = {}
        for key, value in self.carries.items():
            # one seed per tick for every model: the offsets drawn at a reset are A's
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(tick)
                decision = self._model(key).step(value, obs, reset)
            logits[key] = decision.policy_logits
            self.carries[key] = decision.carry
        driver = ("A", "A") if tick < self.switch_tick else (next(iter(self.seconds)), "a")
        joint, _, _ = policy.sample(logits[driver])
        for key, value in self.carries.items():
            self.carries[key] = dataclasses.replace(value, previous_action=joint)
        if tick >= self.switch_tick:
            self.fresh &= ~reset
            alive = obs.ships[:, : self.ships_per_team, SHIP["alive"]] > 0.5
            mask = alive & self.fresh[:, None]
            for name in self.seconds:
                truth = logits[(name, "a")]
                for version, key in (("A", ("A", "A")), ("b", (name, "b")), ("c", (name, "c"))):
                    self.samples.append(
                        (tick - self.switch_tick, name, version, kl(truth, logits[key])[mask])
                    )
        return from_joint(joint), None, Outputs()


def measure(
    first_path,
    second_paths: list,
    mode: str,
    environments: int,
    switch: float,
    follow: float,
    seed: int,
    progress: pathlib.Path | None = None,
) -> dict:
    """One game batch: the KL table by seconds after the switch (see the module docstring).
    Each second of play prints a line (and appends it to `progress`): the time, and after
    the switch that second's mean KLs."""
    device = torch.device("cpu")
    torch.manual_seed(seed)
    game = PRESETS[mode]
    first = checkpoint.load_policy(first_path, device).requires_grad_(False).eval()
    seconds = {
        path.name: checkpoint.load_policy(path, device).requires_grad_(False).eval()
        for path in second_paths
    }
    learner = checkpoint.load_policy(first_path, device).requires_grad_(False).eval()
    rules = api.rules_of(game)
    tick_length = rules.tick_length
    switch_tick = round(switch / tick_length)
    ticks = switch_tick + round(follow / tick_length)
    seat = HandOver(first, seconds, game.ships_per_team, game.num_ships, switch_tick)

    length = 30
    rollout = Rollout(
        game,
        api.ship_types(device),
        (Pool("game", 1.0, (LEARNER, SEAT)),),
        {LEARNER: YemongController(learner, game.ships_per_team, game.num_ships), SEAT: seat},
        LEARNER,
        environments,
        length,
        device,
        compiled=False,
        staggered=False,
    )
    started = time.perf_counter()
    segments = -(-ticks // length)
    for segment in range(segments):  # a segment: one second of every game, all in step
        rollout.run()
        elapsed = time.perf_counter() - started
        played, total = (segment + 1) * length * tick_length, segments * length * tick_length
        line = f"{played:.0f} of {total:.0f} s played, {elapsed:.0f} s"
        if seat.samples:
            recent = [
                sample
                for sample in seat.samples
                if sample[0] >= seat.tick - seat.switch_tick - length
            ]
            for name in seconds:
                means = []
                for version in ("A", "b", "c"):
                    rows = [v for _, n, x, v in recent if n == name and x == version]
                    means.append(f"{version} {float(torch.cat(rows).mean()):.4f}")
                line += f"; {name}: " + ", ".join(means)
        print(line, flush=True)
        if progress is not None:
            with open(progress, "a") as out:
                out.write(f"{first_path.name} {mode}: {line}\n")
    elapsed = time.perf_counter() - started
    tables = {}
    for name in seconds:
        table = tables[name] = {}
        for low, high in zip(BINS[:-1], BINS[1:], strict=True):
            row = {}
            for version in ("A", "b", "c"):
                values = [
                    v
                    for t, n, x, v in seat.samples
                    if n == name and x == version and low <= t * tick_length < high
                ]
                if values:
                    flat = torch.cat(values)
                    row[version] = dict(
                        mean=float(flat.mean()),
                        median=float(flat.median()),
                        count=int(flat.numel()),
                    )
            if row:
                table[f"{low:g}-{high:g}s"] = row
    return dict(
        first=str(first_path),
        seconds=[str(path) for path in second_paths],
        driver=str(second_paths[0]),
        mode=mode,
        environments=environments,
        switch=switch,
        follow=follow,
        wall_seconds=elapsed,
        tables=tables,
    )


def main() -> None:
    """The command line (module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("first", type=pathlib.Path)
    parser.add_argument("second", type=pathlib.Path, nargs="+", help="the B's; the first drives")
    parser.add_argument("--mode", default="frontline", choices=sorted(PRESETS))
    parser.add_argument("--environments", type=int, default=16)
    parser.add_argument("--switch", type=float, default=60.0, help="seconds into each game")
    parser.add_argument("--follow", type=float, default=32.0, help="seconds after the switch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4, help="torch's CPU threads")
    parser.add_argument("--json", type=pathlib.Path)
    parser.add_argument("--progress", type=pathlib.Path, help="append each second's line here")
    options = parser.parse_args()
    torch.set_num_threads(options.threads)
    out = measure(
        options.first,
        options.second,
        options.mode,
        options.environments,
        options.switch,
        options.follow,
        options.seed,
        options.progress,
    )
    for name, table in out["tables"].items():
        print(f"{out['first']} -> {name}, {out['mode']}, {out['wall_seconds']:.0f} s")
        print(f"{'after switch':>12} {'gap KL(a||A)':>14} {'(b) A carry':>12} {'(c) reset':>12}")
        for span, row in table.items():
            cells = [f"{row[k]['mean']:.4f}" if k in row else "-" for k in ("A", "b", "c")]
            print(f"{span:>12} {cells[0]:>14} {cells[1]:>12} {cells[2]:>12}")
    if options.json:
        options.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
