"""The offline match engine (league spec 6): players named on the command line, and batches of
games between pairs of them, every pair's games split evenly between the seats.

A player is a policy file (or a training checkpoint, whose policy part plays), the newest
checkpoint of a run, a ladder entry ("scripted", "random" or "rung_T<t>"), or a teacher:
the scripted agent at its own temperature T with a uniform floor on its actions,
`scripted@T=<temperature>,floor=<floor>` (either key optional: T defaults to 1, floor to 0;
BC's teacher is `scripted`, T = 1 and floor 0). Policies of any architecture play here (only
the league's slots need the live policy's).

A batch is one rollout over every pair's games of one game preset: each pair is a pool of
its own, each player one controller over all its rows. Every environment plays one game
from a fresh clock; environment i of a pair seats the pair's first player on team i mod 2.
Each player counts its own rows' first results (`Recorded.tally`).
"""

import dataclasses
import pathlib

import torch

from bnb.contracts import GameConfig, Obs, Outputs, Pool
from bnb.contracts.tree import contract
from bnb.controllers.scripted import Scripted
from bnb.game import api
from bnb.game.handling.characteristics import characteristics
from bnb.learn.bc import TeacherConfig, TeacherOpponent
from bnb.runs import checkpoint
from bnb.runs.records import Run
from bnb.runtime.opponents import OUTCOMES, Rung, ladder_rung
from bnb.runtime.rollout import Rollout
from bnb.runtime.train import model_controller


@dataclasses.dataclass(frozen=True)
class Player:
    """One player: its name, and either a policy directory, a ladder entry's rung
    (`bnb.runtime.opponents.Rung`; +inf random), or a teacher (the scripted agent at its
    temperature with a floor)."""

    name: str
    policy: pathlib.Path | None = None
    rung: Rung | None = None
    teacher: TeacherConfig | None = None


def teacher_of(name: str) -> TeacherConfig | None:
    """The teacher a `scripted@T=<t>,floor=<f>` name means (keys in any order, each
    optional, none repeated; T defaults to 1 and floor to 0), or None for another name.
    Raises ValueError for a malformed or out-of-range one."""
    if not name.startswith("scripted@"):
        return None
    values = {"T": 1.0, "floor": 0.0}
    seen = set()
    for item in name[len("scripted@") :].split(","):
        key, equals, text = item.partition("=")
        if not equals or key not in values or key in seen:
            raise ValueError(f"{name!r}: expected scripted@T=<temperature>,floor=<floor>")
        seen.add(key)
        try:
            values[key] = float(text)
        except ValueError:
            raise ValueError(f"{name!r}: {text!r} is not a number") from None
    return TeacherConfig(temperature=values["T"], floor=values["floor"])


def player(name: str, runs_directory: pathlib.Path | None = None) -> Player:
    """The player a command-line name means: a ladder entry, a policy directory (one with a
    checkpoint.json), or a run's newest checkpoint."""
    rung = ladder_rung(name)
    if rung is not None:
        return Player(name, rung=rung)
    teacher = teacher_of(name)
    if teacher is not None:
        return Player(name, teacher=teacher)
    path = pathlib.Path(name)
    if (path / checkpoint.RECORD).exists():
        return Player(name, policy=path)
    if runs_directory is not None and (pathlib.Path(runs_directory) / name).is_dir():
        return Player(name, policy=Run.open(pathlib.Path(runs_directory), name).checkpoint_at())
    raise ValueError(
        f"{name!r} is not a policy file, a run or a ladder entry (scripted, random, "
        "rung_T<t>, rung_T<t>_p<share>)"
    )


@contract
class RecordedCarry:
    """inner: the player's own carry; finished (R,) bool: the row's first game has ended;
    started (1,) bool: a decision has been made (the first observation's reset is the
    match's start, not a game's end)."""

    inner: object
    finished: torch.Tensor
    started: torch.Tensor


class Recorded:
    """A player's controller that counts its rows' first results by pair: `tally` gives
    (pairs * 3,) counts of (wins, ties, losses) from the player's side. pairs (R,) int64:
    each row's pair index in the batch."""

    def __init__(self, controller, pairs: torch.Tensor, pair_count: int) -> None:
        self.controller, self.pairs = controller, pairs
        self.reads = controller.reads
        self.columns, self.mode_count = pair_count, 1

    def init_carry(self, views: int, device: torch.device) -> RecordedCarry:
        """The player's carry, nothing finished."""
        return RecordedCarry(
            inner=self.controller.init_carry(views, device),
            finished=torch.zeros(views, dtype=torch.bool, device=device),
            started=torch.zeros(1, dtype=torch.bool, device=device),
        )

    def step(self, carry: RecordedCarry, obs: Obs, reset: torch.Tensor):
        """The player's decision; a reset after the first decision ends the row's first
        game."""
        action, inner, _ = self.controller.step(carry.inner, obs, reset)
        finished = carry.finished | (reset & carry.started)
        return action, RecordedCarry(inner, finished, torch.ones_like(carry.started)), Outputs()

    def tally(self, carry: RecordedCarry, done: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        """The first games that ended this tick, by (pair, outcome) from the player's side
        (`result` (R,): +1 a win, 0 a tie, -1 a loss)."""
        outcome = (1 - result).to(torch.int64)  # win 0, tie 1, loss 2
        counted = (done & ~carry.finished).to(torch.int64)
        counts = torch.zeros(self.columns * len(OUTCOMES), dtype=torch.int64, device=done.device)
        return counts.index_add(0, self.pairs * len(OUTCOMES) + outcome, counted)


def controller_of(player: Player, game: GameConfig, device: torch.device):
    """A controller for `player` in `game`."""
    if player.policy is not None:
        model = checkpoint.load_policy(player.policy, device).requires_grad_(False)
        return model_controller(model, game)
    agent = characteristics(api.ship_types(device), api.rules_of(game))
    if player.teacher is not None:
        return TeacherOpponent(agent, game.ships_per_team, player.teacher)
    return Scripted(
        agent, game.ships_per_team, temperature=player.rung.temperature, share=player.rung.share
    )


def play(
    players: list[Player],
    pairs: list[tuple[int, int, int]],
    game: GameConfig,
    device: torch.device,
    length: int = 64,
) -> torch.Tensor:
    """Play `pairs` (first player, second player, games; games even) of `players` in `game`
    (a game preset's config): (len(pairs), 3) int64, the first player's wins, ties and
    losses per pair."""
    total = sum(games for _, _, games in pairs)
    if total == 0:
        return torch.zeros((len(pairs), len(OUTCOMES)), dtype=torch.int64)
    pools, rows = [], {}
    for index, (first, second, games) in enumerate(pairs):
        if games % 2:
            raise ValueError(f"pair {index}: {games} games do not split evenly between the seats")
        names = (f"player_{first}", f"player_{second}")
        pools.append(Pool(f"pair_{index}", games / total, names))
        for name in names:
            rows.setdefault(name, []).extend([index] * games)
    controllers = {
        name: Recorded(
            controller_of(players[int(name[len("player_") :])], game, device),
            torch.tensor(pair_rows, device=device),
            len(pairs),
        )
        for name, pair_rows in rows.items()
    }
    recorder = next(iter(controllers))
    rollout = Rollout(
        game,
        api.ship_types(device),
        tuple(pools),
        controllers,
        recorder,
        total,
        length,
        device,
        compiled=False,
        staggered=False,
    )
    # environment i of every pair seats the pair's first player on team i mod 2
    seat = torch.cat([torch.arange(games, device=device) % 2 for _, _, games in pairs])
    rollout.carry["seat"].copy_(seat)
    counts = {
        name: torch.zeros(len(pairs) * len(OUTCOMES), dtype=torch.int64) for name in controllers
    }
    while True:
        rollout.run()
        for name in controllers:
            counts[name] += rollout.take_tally(name)
        if all(bool(rollout.carry["controllers"][name].finished.all()) for name in controllers):
            break
    first = torch.zeros((len(pairs), len(OUTCOMES)), dtype=torch.int64)
    for index, (player_index, _, _) in enumerate(pairs):
        first[index] = counts[f"player_{player_index}"].reshape(-1, len(OUTCOMES))[index]
    return first
