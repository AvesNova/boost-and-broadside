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
import threading

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
from bnb.runtime import built
from bnb.runtime.built import clone
from bnb.runtime.scripted_step import ScriptedStep, library_path

from .overlays import MEMORY, NETWORK

GUMBEL_FLOOR = 1e-20
SUPPORTED_MODELS = ("yemong", "relational", "nearest")
POLICY_RECORD = "checkpoint.json"
MODEL_DRAWS_OFFSET = 1_000_003  # the model's own draws: a stream apart from the sampling


def _copy_leaves(targets, sources) -> None:
    """Copy each tensor leaf of `sources` into the same leaf of `targets` (two trees of one
    shape). A carry's optional fields are None (a leaf to `_pytree`) and are skipped."""
    for target, source in zip(
        _pytree.tree_leaves(targets), _pytree.tree_leaves(sources), strict=True
    ):
        if target is not None:
            target.copy_(source)


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


class Waiting:
    """A side whose controller is being made aside (`bnb.play.app.Session.start`): its
    ships coast straight and hold their fire. It needs nothing measured, so a match starts
    at once."""

    label = "waiting"
    overlays = ()

    def __init__(self, ships_per_team: int) -> None:
        self.actions = torch.zeros((ships_per_team, 3), dtype=torch.int32)

    def act(self, view: Obs, reset: torch.Tensor) -> torch.Tensor:
        """The team's (S, 3) actions: coast, straight, no shot."""
        return self.actions


class Scripted:
    """The scripted chaser: each ship turns toward and shoots at its nearest visible enemy,
    at distances in the card's lengths."""

    label = "scripted"
    overlays = ()

    def __init__(self, rules: api.Rules, types: api.ShipTypes) -> None:
        self.chaser = Chaser(characteristics(types, rules), rules.ships_per_team)

    def retune(self, traits) -> None:
        """Play on at the distances of a tuned card's characteristics (live tuning)."""
        self.chaser.retune(traits)

    def act(self, view: Obs, reset: torch.Tensor) -> torch.Tensor:
        """The team's (S, 3) actions for its view (1, ...)."""
        return self.chaser.step(None, view, reset)[0].packed()[0]


class ScriptedAgent:
    """The scripted agent (`bnb.controllers.scripted.Scripted`, memory on, temperature 1):
    it remembers each enemy it saw and dead reckons it while hidden. Its draws come from
    its own generator, swapped in for torch's default one and back, so it is reproducible
    from its seed and leaves the process's generator alone.

    compiled: step with the built step (`bnb.runtime.scripted_step`) for the shapes of its
    first step, which runs uncompiled; without `background` it is built or loaded right
    after that step. background: get it on a thread (which waits on the child process that
    builds it) while the agent steps uncompiled, then swap it in (`uses_compiled` says which
    ran last). The two agree to float32 fusion rounding and draw the same numbers.
    """

    label = "scripted agent"
    overlays = MEMORY

    def __init__(
        self,
        rules: api.Rules,
        types: api.ShipTypes,
        seed: int = 0,
        compiled: bool = False,
        background: bool = False,
    ) -> None:
        self.characteristics = characteristics(types, rules)
        self.controller = scripted.Scripted(self.characteristics, rules.ships_per_team)
        self.carry: ScriptedCarry = self.controller.init_carry(1, torch.device("cpu"))
        self.draws = torch.Generator().manual_seed(seed + MODEL_DRAWS_OFFSET)
        self.inspection: Inspection | None = None
        self.step = ScriptedStep(rules.ships_per_team)
        self.compiled, self.background = compiled, background
        self.uses_compiled = False
        self._path: pathlib.Path | None = None  # the built step's, from the first view
        self._compile_error: BaseException | None = None
        self._thread: threading.Thread | None = None

    def retune(self, traits) -> None:
        """Play on with the characteristics of a tuned card (live tuning), keeping what the
        agent remembers (the built step takes them as an input: nothing rebuilds)."""
        self.characteristics = self.controller.characteristics = traits

    def _compile(self, arguments: tuple, outputs: tuple) -> None:
        """Load the built step (`bnb.runtime.built.library`: built in a child process first
        when no earlier run built it)."""
        try:
            built.library(self._path, self.step, arguments, outputs, name="scripted-step-build")
        except BaseException as error:  # reported by `compiling`; the agent stays uncompiled
            self._compile_error = error

    @property
    def compiling(self) -> bool:
        """The built step is still being built or loaded in the background."""
        if self._compile_error is not None:
            raise RuntimeError("building the scripted step failed") from self._compile_error
        return self._thread is not None and self._thread.is_alive()

    def wait_compiled(self) -> None:
        """Block until a background build finishes."""
        if self._thread is not None:
            self._thread.join()
        _ = self.compiling  # raises a failed build's error

    def _start_compiling(self, arguments: tuple, outputs: tuple) -> None:
        """Get the built step for the shapes of a step's arguments (the first one's), on a
        thread with `background`."""
        self._path = library_path(self.step, arguments)
        if self.background and built.loaded(self._path) is None:
            work = (clone(arguments), clone(outputs))
            self._thread = threading.Thread(target=self._compile, args=work, daemon=True)
            self._thread.start()
        else:
            self._compile(arguments, outputs)
            _ = self.compiling  # raises a failed build's error

    @torch.no_grad()
    def act(self, view: Obs, reset: torch.Tensor) -> torch.Tensor:
        """The team's (S, 3) actions for its view (1, ...); the carry moves on."""
        before = self.carry
        arguments = (before, view, reset, self.characteristics)
        loaded = built.loaded(self._path) if self._path is not None else None
        self.uses_compiled = loaded is not None
        step = self.step if loaded is None else loaded
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.set_state(self.draws.get_state())
            actions, self.carry = step(*arguments)
            self.draws.set_state(torch.default_generator.get_state())
        self.inspection = Inspection(before, view, reset)
        if self.compiled and self._path is None:
            self._start_compiling(arguments, (actions, self.carry))
        return actions


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
    its heads' overlays (`bnb.play.overlays.NETWORK`), a relational or nearest one none."""

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
        self.warning: str | None = None  # set by load_agent: trained with another card or lineup
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
                    _copy_leaves(
                        getattr(self.carry, field.name), getattr(decision.carry, field.name)
                    )
            return decision.policy_logits

        with torch.random.fork_rng(devices=[device]):
            with torch.cuda.stream(side):
                for _ in range(2):
                    step()
            torch.cuda.current_stream(device).wait_stream(side)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph, capture_error_mode="thread_local"):
                self._outputs = step()
        _copy_leaves(self.carry, saved)

    def prepare(self, view: Obs, reset: torch.Tensor) -> None:
        """Capture the graphed step ahead of play, on a view of the match (the play shell
        calls this from a thread before the agent plays, so its first tick does not stall
        the window)."""
        if self.graphed and self._graph is None:
            with torch.no_grad():
                self._capture(view, reset)

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
    if saved is None or given is None:  # a field one record lacks: not the same game
        return saved is given
    if isinstance(saved, float) or isinstance(given, float):
        return math.isclose(float(saved), float(given), rel_tol=1e-6, abs_tol=1e-9)
    return saved == given


def mismatch(
    record: dict,
    config: GameConfig,
    rules: api.Rules,
    types: api.ShipTypes,
    any_mode: bool = False,
    card: bool = True,
) -> str | None:
    """Why a policy record cannot play this match (its game, ship card or lineup differ),
    or None when it can. any_mode: the mode's own settings may differ (`GAME_FIELDS`).
    card: also check the ship card and the lineup (the play shell only warns about them:
    `trained_differently`)."""
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
    return trained_differently(record, config, rules, types) if card else None


def trained_differently(
    record: dict, config: GameConfig, rules: api.Rules, types: api.ShipTypes
) -> str | None:
    """How a policy record's ship card or lineup differs from this match's, or None. The
    agent can still play (the sizes match); it plays a game it was not trained on."""
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
    Raises AgentError with the reason when it cannot play it (another game); a ship card or
    lineup it was not trained with only sets its `warning`. any_mode: see `mismatch`."""
    directory = pathlib.Path(directory)
    try:
        policy = checkpoint.read_policy(directory)
    except (checkpoint.CheckpointError, ValueError, OSError, KeyError) as error:
        raise AgentError(str(error)) from error
    if policy.name not in SUPPORTED_MODELS:
        raise AgentError(f"{policy.name} models do not play in the shell ({SUPPORTED_MODELS})")
    reason = mismatch(policy.raw, config, rules, types, any_mode, card=False)
    if reason is not None:
        raise AgentError(f"{directory.name} cannot play this match: {reason}")
    warning = trained_differently(policy.raw, config, rules, types)
    device = agent_device()
    drawn = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=drawn):  # the model's initialisation draws; keep it local
        model = checkpoint.load_policy(directory, device).eval()
    agent = Agent(
        model,
        policy.name,
        label_of(directory),
        rules.ships_per_team,
        rules.num_ships,
        temperature=temperature,
        seed=seed,
        graph=graph,
    )
    agent.warning = f"{agent.label} plays, but {warning}" if warning else None
    return agent


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


def controller_of(
    side,
    team: int,
    setup,
    rules: api.Rules,
    types: api.ShipTypes,
    compiled: bool = False,
    background: bool = False,
):
    """The controller of one `SideSetup` for team `team` of the setup's match (raises
    AgentError for an agent that cannot play it). compiled, background: a scripted agent's
    built step (`ScriptedAgent`)."""
    seed = setup.seed + team
    if side.controller == "agent":
        return load_agent(side.policy, setup.config, rules, types, side.temperature, seed)
    if side.controller == "scripted_agent":
        return ScriptedAgent(rules, types, seed, compiled=compiled, background=background)
    return Scripted(rules, types)


def build_sides(
    setup, rules: api.Rules, types: api.ShipTypes, compiled: bool = False, background: bool = False
) -> list:
    """Each side's controller from the setup's `SideSetup`s (raises AgentError for an
    agent that cannot play); compiled, background: see `controller_of`."""
    return [
        controller_of(side, team, setup, rules, types, compiled, background)
        for team, side in enumerate(setup.sides)
    ]
