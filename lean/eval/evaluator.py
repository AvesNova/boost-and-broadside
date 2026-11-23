"""Evaluation jobs on the store: play what is missing, allocate adaptively, report, rate.

A pairing is (agent A, agent B, (A's ships, B's ships)) under the evaluator's game config.
A job reads what the store holds for its pairings and plays only the missing games, sides
alternated so that each agent plays both teams equally. The games of a job are grouped by
the networks in them (a runner call takes at most one network per seat: a network against a
scripted agent always sits in seat A) and each group is one shuffled queue.
"""

import math
import random

import torch

from lean import model
from lean.eval import runner
from lean.eval.rating import allocate, bradley_terry, standard_error
from lean.eval.store import NetworkAgent, ScriptedAgent, SobolStream, Store
from lean.learn import load_checkpoint, policy_weights
from lean.scripted import ScriptedConfig
from lean.scripted.parameters import SAMPLED, box_point


class Evaluator:
    """Jobs on one store under one game config (a new config is new conditions and a new
    compiled graph). `envs` environments per runner call, `compiled` torch.compile's tick
    (default on a GPU), `seed` the base of every call's seed (offset by the games stored, so
    no two calls replay the same draws)."""

    def __init__(self, store: Store, game_config, *, device=None, envs: int = 4096, compiled: bool | None = None,
                 segment: int = 32, seed: int = 0, seconds: float | None = None, progress=None):
        self.store, self.game_config = store, game_config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.envs, self.segment, self.seed, self.seconds, self.progress = envs, segment, seed, seconds, progress
        self.compiled = self.device.type == "cuda" if compiled is None else compiled
        self.policies = {}

    def policy(self, agent: NetworkAgent) -> tuple:
        """(the agent's lean.model.Policy, its precision: the bf16 setting it was trained with)."""
        if agent.id not in self.policies:
            checkpoint = load_checkpoint(agent.checkpoint, self.device)
            policy = model.Policy(checkpoint["model_config"]).to(self.device)
            policy.load_state_dict(policy_weights(checkpoint, agent.weights))
            self.policies[agent.id] = (policy.eval(), checkpoint["learn_config"]["bf16"])
        return self.policies[agent.id]

    def counts(self, agent_a, agent_b, sizes) -> dict[str, int]:
        conditions = self.store.add_conditions(self.game_config, sizes)
        return self.store.counts(agent_a.id, agent_b.id, conditions)

    def ensure(self, requests: list[tuple]) -> list[dict[str, int]]:
        """Play until each request (agent A, agent B, sizes, games) has at least `games` games
        in the store; returns each pairing's counts (as `counts`)."""
        groups = {}
        for agent_a, agent_b, sizes, games in requests:
            have = self.counts(agent_a, agent_b, sizes)
            missing = games - have["games"]
            if missing <= 0:
                continue
            # Sides: A's games as team 0 brought as near half of the total as the new games allow.
            team_zero = min(missing, max(0, math.ceil((have["games"] + missing) / 2) - have["as_team_zero"]))
            sides = [0] * team_zero + [1] * (missing - team_zero)
            if isinstance(agent_b, NetworkAgent) and not isinstance(agent_a, NetworkAgent):
                # A network against a scripted agent sits in seat A: the pairing turned round.
                agent_a, agent_b, sizes, sides = agent_b, agent_a, (sizes[1], sizes[0]), [1 - side for side in sides]
            key = tuple(agent.id if isinstance(agent, NetworkAgent) else None for agent in (agent_a, agent_b))
            groups.setdefault(key, []).append((agent_a, agent_b, tuple(sizes), sides))
        for entries in groups.values():
            self._play_group(entries)
        return [self.counts(agent_a, agent_b, sizes) for agent_a, agent_b, sizes, _ in requests]

    def _play_group(self, entries) -> None:
        index = {}  # each scripted agent's row of the table, in order of first appearance
        for agent_a, agent_b, _, _ in entries:
            for agent in (agent_a, agent_b):
                if isinstance(agent, ScriptedAgent) and agent not in index:
                    index[agent] = len(index)
        scripted_agents = list(index)
        table = torch.tensor(
            [agent.values for agent in scripted_agents] or [ScriptedAgent.of().values], dtype=torch.float32,
            device=self.device,
        )
        games = []  # (entry, a, b, side, size_a, size_b)
        for number, (agent_a, agent_b, sizes, sides) in enumerate(entries):
            for side in sides:
                games.append((number, index.get(agent_a, 0), index.get(agent_b, 0), side, sizes[0], sizes[1]))
        seed = self.seed + self.store.total_games()
        random.Random(seed).shuffle(games)
        columns = torch.tensor(games, dtype=torch.long)
        queue = runner.Queue(*columns[:, 1:].unbind(-1))
        seats = [self.policy(agent) if isinstance(agent, NetworkAgent) else (None, True) for agent in entries[0][:2]]
        outcome = runner.play(
            self.game_config, queue, table, seats[0][0], seats[1][0], envs=self.envs, segment=self.segment,
            seconds=self.seconds, compiled=self.compiled, seed=seed, progress=self.progress,
            bf16=(seats[0][1], seats[1][1]),
        )
        # The played games grouped by entry in one sort (not a scan of the queue per entry).
        played = (outcome.result >= 0).nonzero().squeeze(-1)
        played = played[torch.argsort(columns[played, 0], stable=True)]
        counts = torch.bincount(columns[played, 0], minlength=len(entries)).tolist()
        sides, results = columns[played, 3].tolist(), outcome.result[played].tolist()
        lengths = outcome.length[played].tolist()
        start = 0
        for (agent_a, agent_b, sizes, _), count in zip(entries, counts, strict=True):
            self.store.add_agent(agent_a)
            self.store.add_agent(agent_b)
            conditions = self.store.add_conditions(self.game_config, sizes)
            end = start + count
            self.store.add_games(agent_a.id, agent_b.id, conditions, sides[start:end], results[start:end],
                                 lengths[start:end])
            start = end

    def adaptive(self, pairings: list[tuple], *, target_error: float | None = None, budget: int | None = None,
                 round_games: int | None = None) -> list[dict]:
        """Play rounds until every pairing's standard error (rating.standard_error) is at most
        `target_error` or `budget` new games are spent (at least one of the two). Each round
        (`round_games`, default 16 x envs) goes where the error is largest (rating.allocate),
        stored games counted first. Returns `report(pairings)`."""
        if target_error is None and budget is None:
            raise ValueError("give a target standard error, a game budget or both")
        round_games = round_games or 16 * self.envs
        spent = 0
        while True:
            rows = self.report(pairings)
            if target_error is not None and all(row["error"] <= target_error for row in rows):
                return rows
            if budget is not None and spent >= budget:
                return rows
            lot = round_games if budget is None else min(round_games, budget - spent)
            # (wins plus half the draws, games): a pairing without games has no score rate.
            stats = [(row["wins"] + 0.5 * row["draws"], row["games"]) for row in rows]
            if target_error is not None:
                # No pairing already within the target takes games.
                open_rows = [index for index, row in enumerate(rows) if row["error"] > target_error]
                added_open = allocate([stats[index] for index in open_rows], lot)
                added = [0] * len(rows)
                for index, extra in zip(open_rows, added_open, strict=True):
                    added[index] = extra
            else:
                added = allocate(stats, lot)
            if sum(added) == 0:
                return rows
            self.ensure([(a, b, sizes, row["games"] + extra)
                         for (a, b, sizes), row, extra in zip(pairings, rows, added, strict=True) if extra > 0])
            spent += sum(added)

    def report(self, pairings: list[tuple]) -> list[dict]:
        """Per pairing (agent A, agent B, sizes): games, wins, draws, losses (A's), A's score
        (wins plus half the draws, a rate) and its standard error."""
        rows = []
        for agent_a, agent_b, sizes in pairings:
            counts = self.counts(agent_a, agent_b, sizes)
            score = counts["wins"] + 0.5 * counts["draws"]
            games = counts["games"]
            rows.append({**counts, "score": score / games if games else math.nan,
                         "error": standard_error(score, games)})
        return rows

    def elo(self, sizes, anchor=None, agents=None) -> list[tuple[str, float, float]]:
        """Bradley-Terry Elo of every agent with games under (the game config, `sizes`), or of
        `agents` only: (id, Elo, standard error), `anchor` (default the yardstick when it
        played, else the first agent) at 1000."""
        conditions = self.store.add_conditions(self.game_config, sizes)
        totals = self.store.pairing_totals(conditions)
        wanted = None if agents is None else {agent.id for agent in agents}
        totals = [row for row in totals if wanted is None or (row[0] in wanted and row[1] in wanted)]
        ids = sorted({row[0] for row in totals} | {row[1] for row in totals})
        if not ids:
            return []
        anchor_id = (anchor or ScriptedAgent.of()).id
        if anchor_id not in ids:
            anchor_id = ids[0]
        number = {identity: position for position, identity in enumerate(ids)}
        results = [(number[a], number[b], games, score) for a, b, games, score in totals]
        ratings, errors = bradley_terry(results, len(ids), anchor=number[anchor_id])
        return sorted(zip(ids, ratings, errors, strict=True), key=lambda row: -row[1])

    def sobol_candidates(self, name: str, count: int, seed: int = 0, centre: ScriptedConfig | None = None,
                         spread: float = 1.0) -> list[ScriptedAgent]:
        """The next `count` points of the Sobol stream `name` as scripted agents: each
        sampled row (lean.scripted.parameters.SAMPLED) on its persona box at `spread` around
        `centre` (default the yardstick), the fixed rows at the centre's values."""
        centre = centre or ScriptedConfig()
        points = SobolStream(self.store, name, len(SAMPLED), seed).next(count)
        drawn = box_point(centre, points, spread)
        agents = []
        for index in range(count):
            values = dict(centre.values)
            values.update({row: float(drawn[row][index]) for row in SAMPLED})
            agent = ScriptedAgent.of(ScriptedConfig(values))
            self.store.add_agent(agent)
            agents.append(agent)
        return agents
