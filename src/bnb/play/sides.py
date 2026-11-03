"""Each side's controller: the scripted chaser, or a trained agent from a policy file.

A side turns its team's view (1, ...) into its ships' actions, (S, 3) int32. An agent
runs with torch on the CPU at B = 1: its carry is kept per side and handed the view's
reset flag, so the model clears it at each episode boundary as in training. It samples
the policy's distribution at a temperature (1 is the policy's own), from its own
generator, so it never touches torch's global one (a model's own draws, such as Yemong's
position offsets, come from a second stream of its own, swapped in for the default
generator and back).

A policy file loads only when the game it was trained on and its ship card equal the
match's; otherwise `load_agent` raises `AgentError` naming the difference.
"""

import contextlib
import dataclasses
import json
import math
import pathlib

import torch

from bnb.contracts import GameConfig, Obs
from bnb.contracts.action import from_joint
from bnb.controllers.scripted import Chaser
from bnb.game import api
from bnb.runs import checkpoint
from bnb.runs.serialise import to_json

GUMBEL_FLOOR = 1e-20
SUPPORTED_MODELS = ("yemong", "relational")
POLICY_RECORD = "checkpoint.json"
MODEL_DRAWS_OFFSET = 1_000_003  # the model's own draws: a stream apart from the sampling


class AgentError(ValueError):
    """A policy file that cannot play this match, with the reason."""


class Scripted:
    """The scripted chaser: each ship turns toward and shoots at its nearest visible enemy."""

    label = "scripted"

    def __init__(self, ships_per_team: int) -> None:
        self.chaser = Chaser(ships_per_team)

    def act(self, view: Obs, reset: torch.Tensor) -> torch.Tensor:
        """The team's (S, 3) actions for its view (1, ...)."""
        return self.chaser.step(None, view, reset)[0].packed()[0]


def sample(logits: torch.Tensor, temperature: float, generator: torch.Generator) -> torch.Tensor:
    """A joint action per row of (..., NUM_JOINT) logits at a temperature, by Gumbel-max
    from `generator`; temperature 0 takes the most likely action."""
    if temperature <= 0.0:
        return logits.argmax(-1)
    uniform = torch.rand(logits.shape, generator=generator).clamp_min(GUMBEL_FLOOR)
    return torch.argmax(logits / temperature - torch.log(-torch.log(uniform)), -1)


class Agent:
    """A trained policy playing one side. `label` names it (the policy's directory, as
    the run names it); `temperature` may change between ticks."""

    def __init__(
        self,
        model: torch.nn.Module,
        name: str,
        label: str,
        ships_per_team: int,
        num_ships: int,
        temperature: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.model, self.name, self.label = model, name, label
        self.temperature = temperature
        self.generator = torch.Generator().manual_seed(seed)
        self.model_draws = torch.Generator().manual_seed(seed + MODEL_DRAWS_OFFSET)
        self.recurrent = name == "yemong"
        self.carry = None
        if self.recurrent:
            with self._local_draws():
                self.carry = model.initial_carry(1, ships_per_team, num_ships, torch.device("cpu"))

    @contextlib.contextmanager
    def _local_draws(self):
        """The model's own draws from the default generator (Yemong's position offsets)
        come from the agent's `model_draws` stream instead: torch's global generator is
        swapped for it and restored, so the agent is reproducible from its seed and the
        process's generator is untouched."""
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.set_state(self.model_draws.get_state())
            yield
            self.model_draws.set_state(torch.default_generator.get_state())

    @torch.no_grad()
    def act(self, view: Obs, reset: torch.Tensor) -> torch.Tensor:
        """The team's (S, 3) actions for its view (1, ...); the carry moves on."""
        if self.recurrent:
            with self._local_draws():
                decision = self.model.step(self.carry, view, reset)
        else:
            decision = self.model(view)
        joint = sample(decision.policy_logits, self.temperature, self.generator).to(torch.int32)
        if self.recurrent:
            self.carry = dataclasses.replace(decision.carry, previous_action=joint)
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
    record: dict, config: GameConfig, rules: api.Rules, types: api.ShipTypes
) -> str | None:
    """Why a policy record cannot play this match (its game, ship card or lineup differ),
    or None when it can."""
    if record.get("game") is None:
        return "the policy file does not record the game it was trained on"
    saved, given = record["game"], to_json(config)
    differences = [
        f"{name} {saved.get(name)!r} (this match: {given[name]!r})"
        for name in given
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
) -> Agent:
    """The agent of a policy file (or training checkpoint) for this match, on the CPU.
    Raises AgentError with the reason when it cannot play it."""
    directory = pathlib.Path(directory)
    try:
        policy = checkpoint.read_policy(directory)
    except (checkpoint.CheckpointError, ValueError, OSError, KeyError) as error:
        raise AgentError(str(error)) from error
    if policy.name not in SUPPORTED_MODELS:
        raise AgentError(f"{policy.name} models do not play in the shell ({SUPPORTED_MODELS})")
    reason = mismatch(policy.raw, config, rules, types)
    if reason is not None:
        raise AgentError(f"{directory.name} cannot play this match: {reason}")
    with torch.random.fork_rng(devices=[]):  # the model's initialisation draws; keep it local
        model = checkpoint.load_policy(directory, torch.device("cpu")).eval()
    return Agent(
        model,
        policy.name,
        label_of(directory),
        rules.ships_per_team,
        rules.num_ships,
        temperature=temperature,
        seed=seed,
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


def build_sides(setup, rules: api.Rules, types: api.ShipTypes) -> list:
    """Each side's controller from the setup's `SideSetup`s (raises AgentError for an
    agent that cannot play)."""
    sides = []
    for team, side in enumerate(setup.sides):
        if side.controller == "agent":
            sides.append(
                load_agent(
                    side.policy, setup.config, rules, types, side.temperature, setup.seed + team
                )
            )
        else:
            sides.append(Scripted(rules.ships_per_team))
    return sides
