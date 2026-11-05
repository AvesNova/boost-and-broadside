"""Each side's controller: the scripted chaser, the scripted agent (with its memory), or a
trained agent from a policy file.

A side turns its team's view (1, ...) into its ships' actions, (S, 3) int32. An agent
runs with torch at B = 1 on the GPU when one is present (`agent_device`) and on the CPU
otherwise; the sim stays on the CPU, so each tick moves the view to the agent's device and
the actions back. Its carry is kept per side and handed the view's reset flag, so the
model clears it at each episode boundary as in training. It samples the policy's
distribution at a temperature (1 is the policy's own), on the CPU from its own generator,
so it never touches torch's global one (a model's own draws, such as Yemong's position
offsets, come from a second stream of its own, swapped in for the default generator, or
for the device's, and back).

A policy file loads only when the game it was trained on and its ship card equal the
match's; otherwise `load_agent` raises `AgentError` naming the difference. With `any_mode`
the mode's own settings (`bnb.contracts.pools.GAME_FIELDS`: the mode, the arena, the spawn
groups, the clocks) may differ: a policy trained on a mixed batch records the batch's first
game only (`bnb.runtime.train.mixed`) and plays every mode of the mix.

Each controller names the prediction overlays it can provide (`overlays`,
`bnb.play.overlays`) and keeps the inputs of its latest decision (`inspection`, references
to tensors it holds anyway), which `bnb.play.predictions` reads back. While a Yemong agent
is `watched` (one of its overlays is on) it decides with `Yemong.inspect`, the same
decision with every head's output from the same trunk pass, and keeps those too; it then
steps eagerly, never from its CUDA graph (whose carry lives in static buffers, so a graphed
step keeps no inspection).
"""

import contextlib
import dataclasses
import json
import math
import pathlib

import torch
from torch.utils import _pytree

from bnb.contracts import GameConfig, Obs
from bnb.contracts.action import from_joint
from bnb.contracts.pools import GAME_FIELDS
from bnb.controllers import scripted
from bnb.controllers.scripted import Chaser
from bnb.controllers.scripted_memory import ScriptedCarry
from bnb.game import api
from bnb.game.handling.characteristics import characteristics
from bnb.runs import checkpoint
from bnb.runs.serialise import to_json

from .overlays import MEMORY, NETWORK

GUMBEL_FLOOR = 1e-20
SUPPORTED_MODELS = ("yemong", "relational")
POLICY_RECORD = "checkpoint.json"
MODEL_DRAWS_OFFSET = 1_000_003  # the model's own draws: a stream apart from the sampling


class AgentError(ValueError):
    """A policy file that cannot play this match, with the reason."""


CONTROLLERS = ("scripted", "scripted_agent", "agent")  # a SideSetup's controller kinds


@dataclasses.dataclass(frozen=True)
class Inspection:
    """The inputs of a controller's latest decision: the carry it started from, the view
    (1, ...) and the reset (1,) it read; for a Yemong agent the step's replay record
    (`bnb.models.yemong.model.Decision.replay`) and, when it was watched, every head's
    output (`Yemong.inspect`'s `Unrolled`; None otherwise, and for the scripted agent)."""

    carry: object
    view: Obs
    reset: torch.Tensor
    replay: object = None
    unrolled: object = None


class Scripted:
    """The scripted chaser: each ship turns toward and shoots at its nearest visible enemy."""

    label = "scripted"
    overlays = ()

    def __init__(self, ships_per_team: int) -> None:
        self.chaser = Chaser(ships_per_team)

    def act(self, view: Obs, reset: torch.Tensor) -> torch.Tensor:
        """The team's (S, 3) actions for its view (1, ...)."""
        return self.chaser.step(None, view, reset)[0].packed()[0]


class ScriptedAgent:
    """The scripted agent (`bnb.controllers.scripted.Scripted`, memory on, temperature 1):
    it remembers each enemy it saw and dead reckons it while hidden. Its draws come from
    its own generator, swapped in for torch's default one and back, so it is reproducible
    from its seed and leaves the process's generator alone."""

    label = "scripted agent"
    overlays = MEMORY

    def __init__(self, rules: api.Rules, types: api.ShipTypes, seed: int = 0) -> None:
        self.characteristics = characteristics(types, rules)
        self.controller = scripted.Scripted(self.characteristics, rules.ships_per_team)
        self.carry: ScriptedCarry = self.controller.init_carry(1, torch.device("cpu"))
        self.draws = torch.Generator().manual_seed(seed + MODEL_DRAWS_OFFSET)
        self.inspection: Inspection | None = None

    @torch.no_grad()
    def act(self, view: Obs, reset: torch.Tensor) -> torch.Tensor:
        """The team's (S, 3) actions for its view (1, ...); the carry moves on."""
        before = self.carry
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.set_state(self.draws.get_state())
            action, self.carry, _ = self.controller.step(before, view, reset)
            self.draws.set_state(torch.default_generator.get_state())
        self.inspection = Inspection(before, view, reset)
        return action.packed()[0]


def agent_device() -> torch.device:
    """Where agents run: the GPU whenever one is present, the CPU otherwise."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample(logits: torch.Tensor, temperature: float, generator: torch.Generator) -> torch.Tensor:
    """A joint action per row of (..., NUM_JOINT) logits at a temperature, by Gumbel-max
    from `generator`; temperature 0 takes the most likely action."""
    if temperature <= 0.0:
        return logits.argmax(-1)
    uniform = torch.rand(logits.shape, generator=generator).clamp_min(GUMBEL_FLOOR)
    return torch.argmax(logits / temperature - torch.log(-torch.log(uniform)), -1)


class Agent:
    """A trained policy playing one side. `label` names it (the policy's directory, as
    the run names it); `temperature` may change between ticks. A Yemong agent provides
    its heads' overlays (`bnb.play.overlays.NETWORK`), a relational one none."""

    def __init__(
        self,
        model: torch.nn.Module,
        name: str,
        label: str,
        ships_per_team: int,
        num_ships: int,
        temperature: float = 1.0,
        seed: int = 0,
        graph: bool | None = None,
    ) -> None:
        self.model, self.name, self.label = model, name, label
        self.temperature = temperature
        self.generator = torch.Generator().manual_seed(seed)
        self.model_draws = torch.Generator().manual_seed(seed + MODEL_DRAWS_OFFSET)
        self.device = next(model.parameters()).device
        self.recurrent = name == "yemong"
        self.carry = None
        self.device_draws = None  # a GPU's own stream of the model's draws (its generator state)
        if self.device.type == "cuda":
            draws = torch.Generator(device=self.device).manual_seed(seed + MODEL_DRAWS_OFFSET)
            self.device_draws = draws.get_state()
        if self.recurrent:
            with self._local_draws():
                self.carry = model.initial_carry(1, ships_per_team, num_ships, self.device)
        # Yemong's step on a GPU is replayed from a CUDA graph (captured on the first tick):
        # eager, it is bound by kernel launches and slower than the CPU.
        self.graphed = (self.recurrent and self.device.type == "cuda") if graph is None else graph
        self._graph: torch.cuda.CUDAGraph | None = None
        self._inputs = self._outputs = None
        self.last_logits: torch.Tensor | None = None  # the latest tick's (S, NUM_JOINT) logits
        self.overlays = NETWORK if self.recurrent else ()
        self.watched = False
        self.inspection: Inspection | None = None

    def __getstate__(self) -> dict:
        """A copy (a replay forked into a live match) leaves the captured graph behind and
        captures its own on its first tick."""
        return {**self.__dict__, "_graph": None, "_inputs": None, "_outputs": None}

    def _capture(self, view: Obs, reset: torch.Tensor) -> None:
        """Capture one step on static buffers: the view and reset are copied in, the carry
        advances in place (its previous action is set after sampling), and the policy logits
        are read out. The carry, the generator states and the model's state are left as
        they were."""
        device = self.device
        self._inputs = (
            _pytree.tree_map(lambda t: t.to(device).clone(), view),
            reset.to(device).clone(),
        )
        saved = _pytree.tree_map(torch.clone, self.carry)
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))

        def step():
            decision = self.model.step(self.carry, *self._inputs)
            for field in dataclasses.fields(self.carry):
                if field.name != "previous_action":
                    for target, new in zip(
                        _pytree.tree_leaves(getattr(self.carry, field.name)),
                        _pytree.tree_leaves(getattr(decision.carry, field.name)),
                        strict=True,
                    ):
                        target.copy_(new)
            return decision.policy_logits

        with torch.random.fork_rng(devices=[device]):
            with torch.cuda.stream(side):
                for _ in range(2):
                    step()
            torch.cuda.current_stream(device).wait_stream(side)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph, capture_error_mode="thread_local"):
                self._outputs = step()
        for target, old in zip(
            _pytree.tree_leaves(self.carry), _pytree.tree_leaves(saved), strict=True
        ):
            target.copy_(old)

    def _graphed_logits(self, view: Obs, reset: torch.Tensor) -> torch.Tensor:
        if self._graph is None:
            self._capture(view, reset)
        view_in, reset_in = self._inputs
        for target, new in zip(
            _pytree.tree_leaves((view_in, reset_in)),
            _pytree.tree_leaves((view, reset)),
            strict=True,
        ):
            target.copy_(new)
        with self._local_draws():
            self._graph.replay()
        return self._outputs.cpu()

    @contextlib.contextmanager
    def _local_draws(self):
        """The model's own draws from the default generator (Yemong's position offsets)
        come from the agent's `model_draws` stream instead (on a GPU, from the device's
        generator, with its own state): torch's global generators are swapped for it and
        restored, so the agent is reproducible from its seed and the process's generators
        are untouched."""
        cuda = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda):
            torch.default_generator.set_state(self.model_draws.get_state())
            if cuda:
                torch.cuda.set_rng_state(self.device_draws, self.device)
            yield
            self.model_draws.set_state(torch.default_generator.get_state())
            if cuda:
                self.device_draws = torch.cuda.get_rng_state(self.device)

    @torch.no_grad()
    def act(self, view: Obs, reset: torch.Tensor) -> torch.Tensor:
        """The team's (S, 3) actions for its view (1, ...), on the CPU; the carry moves on."""
        if self.watched:
            self._graph = None  # the eager step replaces the carry: capture afresh after
        if self.graphed and not self.watched:
            self.inspection = None
            logits = self._graphed_logits(view, reset)
            joint = sample(logits, self.temperature, self.generator).to(torch.int32)
            self.carry.previous_action.copy_(joint)
            self.last_logits = logits
            return from_joint(joint).packed()[0]
        if self.device.type != "cpu":
            view = _pytree.tree_map(lambda tensor: tensor.to(self.device), view)
            reset = reset.to(self.device)
        before, unrolled = self.carry, None
        if self.recurrent:
            with self._local_draws():
                if self.watched:
                    decision, unrolled = self.model.inspect(self.carry, view, reset)
                else:
                    decision = self.model.step(self.carry, view, reset)
        else:
            decision = self.model(view)
        if self.recurrent:
            self.inspection = Inspection(before, view, reset, decision.replay, unrolled)
        logits = decision.policy_logits.cpu()
        joint = sample(logits, self.temperature, self.generator).to(torch.int32)
        if self.recurrent:
            self.carry = dataclasses.replace(decision.carry, previous_action=joint.to(self.device))
        self.last_logits = logits
        return from_joint(joint).packed()[0]


def ship_card(types: api.ShipTypes) -> dict:
    """The ship card as a policy record writes it: each parameter's per-type values."""
    return {field.name: getattr(types, field.name).tolist() for field in dataclasses.fields(types)}


def _close(saved, given) -> bool:
    if isinstance(saved, list) and isinstance(given, list):
        return len(saved) == len(given) and all(map(_close, saved, given))
    if isinstance(saved, float) or isinstance(given, float):
        return math.isclose(float(saved), float(given), rel_tol=1e-6, abs_tol=1e-9)
    return saved == given


def mismatch(
    record: dict,
    config: GameConfig,
    rules: api.Rules,
    types: api.ShipTypes,
    any_mode: bool = False,
) -> str | None:
    """Why a policy record cannot play this match (its game, ship card or lineup differ),
    or None when it can. any_mode: the mode's own settings may differ (`GAME_FIELDS`)."""
    if record.get("game") is None:
        return "the policy file does not record the game it was trained on"
    saved, given = record["game"], to_json(config)
    compared = [name for name in given if not (any_mode and name in GAME_FIELDS)]
    differences = [
        f"{name} {saved.get(name)!r} (this match: {given[name]!r})"
        for name in compared
        if not _close(saved.get(name), given[name])
    ]
    if differences:
        return "it was trained on another game: " + ", ".join(differences)
    if record.get("ship_card") is None:
        return "the policy file does not record its ship card"
    card = ship_card(types)
    changed = [name for name in card if not _close(record["ship_card"].get(name), card[name])]
    if changed:
        return "it was trained with another ship card: " + ", ".join(changed)
    if rules.team_types != api.rules_of(config).team_types:
        return "it was trained with the game's default lineup"
    return None


def load_agent(
    directory: str | pathlib.Path,
    config: GameConfig,
    rules: api.Rules,
    types: api.ShipTypes,
    temperature: float = 1.0,
    seed: int = 0,
    graph: bool | None = None,
    any_mode: bool = False,
) -> Agent:
    """The agent of a policy file (or training checkpoint) for this match, on `agent_device()`.
    Raises AgentError with the reason when it cannot play it. any_mode: see `mismatch`."""
    directory = pathlib.Path(directory)
    try:
        policy = checkpoint.read_policy(directory)
    except (checkpoint.CheckpointError, ValueError, OSError, KeyError) as error:
        raise AgentError(str(error)) from error
    if policy.name not in SUPPORTED_MODELS:
        raise AgentError(f"{policy.name} models do not play in the shell ({SUPPORTED_MODELS})")
    reason = mismatch(policy.raw, config, rules, types, any_mode)
    if reason is not None:
        raise AgentError(f"{directory.name} cannot play this match: {reason}")
    device = agent_device()
    drawn = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=drawn):  # the model's initialisation draws; keep it local
        model = checkpoint.load_policy(directory, device).eval()
    return Agent(
        model,
        policy.name,
        label_of(directory),
        rules.ships_per_team,
        rules.num_ships,
        temperature=temperature,
        seed=seed,
        graph=graph,
    )


def label_of(directory: pathlib.Path) -> str:
    """A policy's short name: its run and its directory ("run/policy", "run/step_100")."""
    directory = pathlib.Path(directory)
    parent = directory.parent
    run = parent.parent if parent.name in ("checkpoints", "league") else parent
    return f"{run.name}/{directory.name}"


def policy_choices(runs_directory: str | pathlib.Path) -> list[pathlib.Path]:
    """The policy files of the local runs, newest run first: each run's `policy/`, its
    training checkpoints and its league snapshots (directories with a record; nothing is
    read or checked until one loads)."""
    root = pathlib.Path(runs_directory)
    if not root.is_dir():
        return []
    found = []
    runs = sorted((path for path in root.iterdir() if path.is_dir()), reverse=True)
    for run in runs:
        candidates = [run / "policy"]
        for group in ("checkpoints", "league"):
            if (run / group).is_dir():
                candidates += sorted((run / group).iterdir(), reverse=True)
        found += [path for path in candidates if (path / POLICY_RECORD).is_file()]
    return found


def describe(directory: pathlib.Path) -> str:
    """One line about a policy file from its record, unchecked: the model and its steps."""
    try:
        record = json.loads((pathlib.Path(directory) / POLICY_RECORD).read_text())
    except OSError, ValueError:
        return "unreadable record"
    steps = record.get("steps", {}).get("environment_steps")
    model = record.get("model", {}).get("name", "?")
    return f"{model}" + (f", {steps / 1e6:.1f}M steps" if steps else "")


def controller_of(side, team: int, setup, rules: api.Rules, types: api.ShipTypes):
    """The controller of one `SideSetup` for team `team` of the setup's match (raises
    AgentError for an agent that cannot play it)."""
    seed = setup.seed + team
    if side.controller == "agent":
        return load_agent(side.policy, setup.config, rules, types, side.temperature, seed)
    if side.controller == "scripted_agent":
        return ScriptedAgent(rules, types, seed)
    return Scripted(rules.ships_per_team)


def build_sides(setup, rules: api.Rules, types: api.ShipTypes) -> list:
    """Each side's controller from the setup's `SideSetup`s (raises AgentError for an
    agent that cannot play)."""
    return [controller_of(side, team, setup, rules, types) for team, side in enumerate(setup.sides)]
