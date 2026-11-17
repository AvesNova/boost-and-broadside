"""The static design of an Elo curve (the author, 2026-11-17): which pairs a learning curve's
checkpoints play and how many games each, fixed before any game, then one top-up from the
fit. No per-game planning.

Players: each chain's checkpoints in training order (BC's snapshots by steps, then RL's: RL
starts from BC's last weights, so the chain is continuous across the transition) and the
anchor (index 0, this tree's Imugi, 1000 in the fit). A champion of the same weights as a
snapshot is that snapshot (`run.load_spec`'s aliases), so it is never a second player.

**Design prior** (game counts only, never the fit): adjacent checkpoints on a chain are
`PRIOR_STEP` (100) Elo apart; checkpoints at equal steps on different chains are equal;
the anchor sits at each chain's first RL checkpoint (`PRIOR_STEP` above BC's last).

**Pairs** (`candidate_pairs`): each checkpoint against its +-1 and +-2 neighbours on its
chain (`NEIGHBOURS`); against the checkpoint at the nearest step of the same stage on every
other chain (`cross`, an extension: off for one chain); the anchor against the checkpoints
within `ANCHOR_NEAR` places of its prior position on each chain, and every second one out
to `ANCHOR_LINK` places (the links along the chain).

**Counts** (`a_optimal`): the games per pair, chosen so every checkpoint's expected
standard error (the inverse Fisher information of the Bradley-Terry fit at the design's
ratings, the anchor fixed) meets the goal, at the least total expected game seconds:
greedy on the variance drop over the unmet goals per game second (rank-one updates), then
a trim that takes back every game the goals do not need. Existing games count at their
information (they reduce the counts). The design's information matrix must be full rank
(every checkpoint connected to the anchor), checked.

**Streaming** (`streaming_schedule`): checkpoints play only near neighbours, so a resident
set of K network players can slide along the chain: the players in a fixed order (the
chain's; several chains interleaved by stage and step), the first K with games seated, a
player unloaded once its cells are all played and the next in order with games loaded in
its lane. Every cell is played while both its players are seated (a player stays until its
last cell is done, and its partners come within the next few places), the anchor plays
every row (scripted, no lane). `BucketEngine` follows the same order (`StaticEloTarget
.stream`).

**Windows** (`windows`, `assign_windows`; a spec's `windows = {size = 16, overlap = 8}`,
the author 2026-11-17): fixed overlapping windows of the order instead (`size` checkpoints,
stepping by `size - overlap`; the anchor in every one). Each pair is played in the first
window holding both its players (an overlap of at least the reach holds every pair); a
window is played to its last game, then the next window with games is loaded in place.

    uv run --no-sync python -m bnb.league.unified.static_design --demo 52
"""

import dataclasses
import math

import numpy

from bnb.league.ratings import SCALE

PRIOR_STEP = 100.0  # Elo between adjacent checkpoints of a chain (the design prior)
NEIGHBOURS = 2  # each checkpoint plays its +-1 .. +-NEIGHBOURS on its chain
ANCHOR_NEAR = 3  # the anchor plays every checkpoint within this many places of its prior
ANCHOR_LINK = 8  # and every second one out to this many places
RIDGE_STDERR = 1e5  # Elo: a negligible prior on each rating, so the greedy starts finite
TRIM_PASSES = 2


@dataclasses.dataclass(frozen=True)
class Checkpoint:
    """A field player on a chain: its name, chain, stage ("bc" or "rl") and steps."""

    name: str
    chain: str
    stage: str
    step: int


def checkpoints(players) -> tuple[list[Checkpoint], list[str]]:
    """The checkpoints among `players` (`bnb.league.matches.Player`, by their policy paths,
    `rating_pool.policy_name`: run `<chain>-bc` or `<chain>-rl`; another run is a chain of
    one RL stage) and the names of the rest (scripted players)."""
    from bnb.league.rating_pool import policy_name

    found, other = [], []
    for one in players:
        parsed = policy_name(one.policy) if one.policy is not None else None
        if parsed is None:
            other.append(one.name)
            continue
        _, run, step, _ = parsed
        chain, stage = run, "rl"
        for suffix in ("-bc", "-rl"):
            if run.endswith(suffix):
                chain, stage = run[: -len(suffix)], suffix[1:]
        found.append(Checkpoint(one.name, chain, stage, step))
    return found, other


def chains_of(found: list[Checkpoint]) -> dict[str, list[Checkpoint]]:
    """Each chain's checkpoints in training order: BC by steps, then RL by steps."""
    held: dict[str, list[Checkpoint]] = {}
    for one in found:
        held.setdefault(one.chain, []).append(one)
    return {
        chain: sorted(members, key=lambda c: (c.stage != "bc", c.step, c.name))
        for chain, members in sorted(held.items())
    }


def transition(chain: list[Checkpoint]) -> int:
    """The index of a chain's first RL checkpoint (its length when it has none)."""
    return next((i for i, c in enumerate(chain) if c.stage == "rl"), len(chain))


def prior_ratings(chains: dict[str, list[Checkpoint]]) -> dict[str, float]:
    """The design prior (module docstring), Elo relative to the anchor (0)."""
    held = {}
    for chain in chains.values():
        first = transition(chain)
        for index, one in enumerate(chain):
            held[one.name] = PRIOR_STEP * (index - first)
    return held


def candidate_pairs(
    chains: dict[str, list[Checkpoint]],
    anchor: str,
    cross: bool = False,
    neighbours: int = NEIGHBOURS,
    near: int = ANCHOR_NEAR,
    link: int = ANCHOR_LINK,
) -> list[tuple[str, str]]:
    """The pairs the design may play (module docstring), each once, in a stable order."""
    pairs: dict[tuple[str, str], None] = {}

    def add(a: str, b: str) -> None:
        if a != b and (b, a) not in pairs:
            pairs[(a, b)] = None

    for chain in chains.values():
        for index, one in enumerate(chain):
            for offset in range(1, neighbours + 1):
                if index + offset < len(chain):
                    add(one.name, chain[index + offset].name)
    for chain in chains.values():
        first = transition(chain)
        for index, one in enumerate(chain):
            distance = abs(index - first)
            if distance <= near or (distance <= link and (distance - near) % 2 == 0):
                add(anchor, one.name)
    if cross:
        names = list(chains)
        for number, a in enumerate(names):
            for b in names[number + 1 :]:
                for one in chains[a]:
                    same = [c for c in chains[b] if c.stage == one.stage]
                    if same:
                        add(one.name, min(same, key=lambda c: abs(c.step - one.step)).name)
    return list(pairs)


def pair_weights(theta: numpy.ndarray, first: numpy.ndarray, second: numpy.ndarray):
    """Each pair's Fisher information per game, p (1 - p), at ratings theta (natural units)."""
    p = 1.0 / (1.0 + numpy.exp(-(theta[first] - theta[second])))
    return p * (1.0 - p)


def precision(count: int, first, second, information) -> numpy.ndarray:
    """(K, K) the Bradley-Terry information (graph Laplacian) of games of `information`
    (per pair, games times p (1 - p)) between `first` and `second`."""
    matrix = numpy.zeros((count, count))
    numpy.add.at(matrix, (first, second), -information)
    numpy.add.at(matrix, (second, first), -information)
    numpy.fill_diagonal(matrix, 0.0)
    numpy.fill_diagonal(matrix, -matrix.sum(1))
    return matrix


@dataclasses.dataclass
class Design:
    """A design's games per pair (indices into the field, `counts[p]` new games of pair p)
    and each player's expected standard error (Elo, the anchor's 0), the existing games'
    information counted; `rank` the information matrix's rank over the free players."""

    pairs: list[tuple[int, int]]
    counts: numpy.ndarray
    stderr: numpy.ndarray
    rank: int

    @property
    def games(self) -> int:
        return int(self.counts.sum())


def a_optimal(
    count: int,
    pairs: list[tuple[int, int]],
    theta: numpy.ndarray,
    goal: float,
    existing: numpy.ndarray | None = None,
    seconds: numpy.ndarray | None = None,
    goals_for: numpy.ndarray | None = None,
) -> Design:
    """The games per pair (module docstring): field of `count` players (index 0 the anchor,
    fixed), `pairs` the candidates, theta the design's ratings (natural units), goal each
    player's standard error (Elo), existing (K, K) the information already in hand (the
    Laplacian of the games played; None: none), seconds each pair's expected game seconds
    (None: equal), goals_for: the players whose goal binds (None: every non-anchor one)."""
    first = numpy.array([a for a, _ in pairs], dtype=numpy.int64)
    second = numpy.array([b for _, b in pairs], dtype=numpy.int64)
    weights = pair_weights(theta, first, second)
    cost = numpy.ones(len(pairs)) if seconds is None else numpy.asarray(seconds, dtype=float)
    base = numpy.zeros((count, count)) if existing is None else numpy.array(existing, dtype=float)
    ridge = (SCALE / RIDGE_STDERR) ** 2
    target = (goal / SCALE) ** 2
    binding = numpy.ones(count, dtype=bool) if goals_for is None else goals_for.copy()
    binding[0] = False
    counts = numpy.zeros(len(pairs))

    def covariance_of(added: numpy.ndarray) -> numpy.ndarray:
        matrix = base + precision(count, first, second, added * weights)
        padded = numpy.zeros((count, count))
        padded[1:, 1:] = numpy.linalg.inv(matrix[1:, 1:] + ridge * numpy.eye(count - 1))
        return padded

    covariance = covariance_of(counts)
    for _ in range(100_000):
        unmet = binding & (numpy.diag(covariance) > target * (1.0 + 1e-9))
        if not unmet.any():
            break
        rows = covariance[unmet]  # (U, K)
        along = rows[:, first] - rows[:, second]  # (U, P): Sigma d for each pair, unmet rows
        quadratic = (
            numpy.diag(covariance)[first]
            + numpy.diag(covariance)[second]
            - 2.0 * covariance[first, second]
        )
        drop = weights * (along**2).sum(0) / (1.0 + weights * quadratic)
        best = int(numpy.argmax(drop / cost))
        if drop[best] <= 0.0:
            break
        batch = max(1, int(counts.sum() * 0.01))
        counts[best] += batch
        column = covariance[:, first[best]] - covariance[:, second[best]]
        n_w = batch * weights[best]
        covariance = covariance - numpy.outer(column, column) * (
            n_w / (1.0 + n_w * quadratic[best])
        )
    covariance = covariance_of(counts)  # re-inverted: no drift from the updates

    # trim: take back the games no goal needs, the least informative per second first
    def feasible(variance: numpy.ndarray) -> bool:
        return bool((variance[binding] <= target * (1.0 + 1e-6)).all())

    passes = TRIM_PASSES if feasible(numpy.diag(covariance)) else 0
    for _ in range(passes):
        for pair in numpy.argsort(weights / cost):
            if counts[pair] == 0:
                continue
            low, high = 0, int(counts[pair])  # the most games removable, by bisection
            while low < high:
                middle = (low + high + 1) // 2
                trial = counts.copy()
                trial[pair] -= middle
                if feasible(numpy.diag(covariance_of(trial))):
                    low = middle
                else:
                    high = middle - 1
            if low:
                counts[pair] -= low
                covariance = covariance_of(counts)
    information = base + precision(count, first, second, counts * weights)
    rank = int(numpy.linalg.matrix_rank(information[1:, 1:]))
    stderr = numpy.sqrt(numpy.maximum(numpy.diag(covariance), 0.0)) * SCALE
    stderr[0] = 0.0
    return Design(list(pairs), counts.astype(numpy.int64), stderr, rank)


def streaming_schedule(
    order: list[str], cells: dict[tuple[str, str], int], residents: int
) -> list[dict]:
    """The resident window's steps (module docstring), idealised: each step every cell
    whose players are both seated is played out, then every seated player without a cell
    left is unloaded and the next players in `order` with cells are loaded. cells: (a, b)
    -> games (a player outside `order`, the anchor, is always seated). Returns each step's
    loads, unloads, games and the seated players."""
    left = {pair: games for pair, games in cells.items() if games > 0}
    ranked = {name: number for number, name in enumerate(order)}

    def has_work(name: str) -> bool:
        return any(name in pair for pair in left)

    seated: list[str] = []
    steps = []
    queue = [name for name in order if has_work(name)]
    while left:
        loads = []
        while queue and len(seated) < residents:
            name = queue.pop(0)
            if has_work(name):
                seated.append(name)
                loads.append(name)
        here = set(seated)
        played = 0
        for pair in list(left):
            if all(n in here or n not in ranked for n in pair):
                played += left.pop(pair)
        unloads = [name for name in seated if not has_work(name)]
        if not played and not unloads:  # a window with nothing to play: the earliest leaves
            if not seated:
                break
            unloads = [min(seated, key=lambda n: ranked[n])]
            queue.append(unloads[0])
            queue.sort(key=lambda n: ranked[n])
        seated = [name for name in seated if name not in unloads]
        steps.append(dict(loads=loads, unloads=unloads, games=played, seated=sorted(here)))
    return steps


def windows(order: list[str], size: int, overlap: int) -> list[list[str]]:
    """Fixed overlapping windows of the streaming order (module docstring): `size` players
    each, starting every `size - overlap` places, the last one reaching the order's end."""
    if not 0 <= overlap < size:
        raise ValueError(f"windows of {size} with an overlap of {overlap}: 0 <= overlap < size")
    step = size - overlap
    held, start = [], 0
    while True:
        held.append(order[start : start + size])
        if start + size >= len(order):
            return held
        start += step


def assign_windows(
    held: list[list[str]], pairs, outside: set[str] | frozenset = frozenset()
) -> dict[tuple[str, str], int]:
    """Each pair's window: the first holding both its players (a player in `outside`, the
    anchor, is in every window). Raises for a pair no window holds."""
    where = {}
    for pair in pairs:
        number = next(
            (
                n
                for n, members in enumerate(held)
                if all(name in outside or name in members for name in pair)
            ),
            None,
        )
        if number is None:
            raise ValueError(f"no window holds both players of {pair}: overlap too small")
        where[tuple(pair)] = number
    return where


def design_order(chains: dict[str, list[Checkpoint]]) -> list[str]:
    """The streaming order: one chain's training order; several chains interleaved by stage
    and steps (a step's checkpoints of every chain together)."""
    every = [
        (c.stage != "bc", c.step, number, c.name)
        for chain in chains.values()
        for number, c in enumerate(chain)
    ]
    if len(chains) == 1:
        return [c.name for c in next(iter(chains.values()))]
    return [name for *_, name in sorted(every)]


def table(
    names: list[str], design: Design, theta: numpy.ndarray, seconds: float, played=None
) -> str:
    """The design's per-pair table, each player's expected standard error and the totals."""
    lines = [
        f"{'pair':<60} {'prior p':>8} {'games':>6}" + ("  played" if played is not None else "")
    ]
    for (a, b), games in zip(design.pairs, design.counts.tolist(), strict=True):
        if games == 0 and (played is None or not played[a, b]):
            continue
        p = 1.0 / (1.0 + math.exp(-(theta[a] - theta[b])))
        lines.append(
            f"{names[a] + ' - ' + names[b]:<60} {p:8.3f} {games:6d}"
            + (f"  {int(played[a, b]):6d}" if played is not None else "")
        )
    lines.append("")
    lines.append(f"{'player':<40} {'design Elo':>10} {'expected SE':>11}")
    for index, name in enumerate(names):
        lines.append(f"{name:<40} {1000 + theta[index] * SCALE:10.0f} {design.stderr[index]:11.1f}")
    lines.append(
        f"total: {design.games} games, {design.games * seconds / 3600:.1f} game-hours at "
        f"{seconds:.0f} s a game; worst expected SE {design.stderr[1:].max():.1f}; "
        f"information rank {design.rank} of {len(names) - 1}"
        + ("" if design.rank == len(names) - 1 else " (NOT CONNECTED)")
    )
    return "\n".join(lines)


def schedule_table(steps: list[dict], seconds: float) -> str:
    """The streaming schedule as text: per window step the loads, unloads, games, seconds."""
    lines = [f"{'step':>4} {'games':>6} {'game-h':>7}  loads / unloads"]
    for number, step in enumerate(steps):
        lines.append(
            f"{number:4d} {step['games']:6d} {step['games'] * seconds / 3600:7.1f}  "
            f"+{','.join(step['loads']) or '-'} / -{','.join(step['unloads']) or '-'}"
        )
    return "\n".join(lines)


def main() -> None:
    """A demo of the design on a synthetic chain (`--demo N` checkpoints, half BC), or a
    spec's static Elo targets (`--spec SPEC.toml --runs-directory RUNS`, the box's): prints
    the per-pair table, each player's expected standard error, the totals and the
    streaming schedule."""
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--demo", type=int, default=None)
    parser.add_argument("--spec", type=pathlib.Path, default=None)
    parser.add_argument("--runs-directory", type=pathlib.Path, default=None)
    parser.add_argument("--stderr", type=float, default=60.0)
    parser.add_argument("--seconds", type=float, default=60.0, help="expected game seconds")
    parser.add_argument("--residents", type=int, default=12)
    parser.add_argument("--prior", type=pathlib.Path, nargs="*", default=())
    args = parser.parse_args()
    if args.spec is not None:
        from .planner import Planner
        from .run import load_prior, load_spec
        from .static_target import StaticEloTarget

        spec = load_spec(args.spec, args.runs_directory)
        planner = Planner(spec.targets, spec.buckets)
        if args.prior:
            load_prior(
                planner,
                list(args.prior),
                spec.players,
                print,
                getattr(spec, "variants", {}),
                args.runs_directory,
            )
        for target in spec.targets:
            if isinstance(target, StaticEloTarget):
                target.design()
                print(f"== {target.name}: stage {target.stage}")
                print(target.design_table(args.seconds))
                print(schedule_table(target.schedule(args.residents), args.seconds))
        return
    count = args.demo or 52
    found = [Checkpoint(f"bc_{i}", "demo", "bc", i) for i in range(1, count // 2 + 1)]
    found += [Checkpoint(f"rl_{i}", "demo", "rl", i) for i in range(1, count - count // 2 + 1)]
    chains = chains_of(found)
    prior = prior_ratings(chains)
    names = ["scripted"] + [c.name for c in found]
    index = {n: i for i, n in enumerate(names)}
    theta = numpy.array([0.0] + [prior[c.name] / SCALE for c in found])
    pairs = [(index[a], index[b]) for a, b in candidate_pairs(chains, "scripted")]
    design = a_optimal(len(names), pairs, theta, args.stderr)
    print(table(names, design, theta, args.seconds))
    cells = {(names[a], names[b]): int(n) for (a, b), n in zip(pairs, design.counts, strict=True)}
    print(
        schedule_table(
            streaming_schedule(design_order(chains), cells, args.residents), args.seconds
        )
    )


if __name__ == "__main__":
    main()
