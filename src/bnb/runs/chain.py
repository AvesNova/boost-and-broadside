"""A chained run (`bnb train --objective chain`): behaviour cloning until its stop rule
fires, then PPO from the clone's exponential average, to a combined budget of environment
steps.

    runs/<chain>/chain.json     the chain's record: its command, budget, stages, handover
    runs/<chain>-bc/            the BC stage, a BC run
    runs/<chain>-rl/            the RL stage, a PPO run warm-started from the BC stage

Each stage is an ordinary run (`bnb.runs.records.Run`): its own run.json, checkpoints,
metrics and policy files, listed by `bnb runs`. Its run.json names the chain and the stage
(`chain`: name, stage, and `start_step`, the chain's environment steps before the stage),
and each record of a stage's metrics carries `counters/chain_environment_steps`, its step
on the chain's axis. The chain's record says where the handover happened: the BC stage's
last step, the rule that stopped it and the policy file the RL stage started from.

BC's budget is the combined budget (a cap: its own stop rule ends it first, in a chain
that goes as planned); the RL stage trains the whole updates that fit in what is left.
A stage starts with a checkpoint at its step 0, so the chain resumes at any point: in BC,
at the handover (the BC stage finished, the RL stage not begun or not yet past its first
checkpoint) and in RL.
"""

import json
import pathlib

from .records import Run, write_json

RECORD = "chain.json"
STAGES = ("bc", "rl")
# the BC stage's policy file the RL stage starts from, for every model (author,
# 2026-11-11): the exponential average (`bnb.runs.session.POLICY_VARIANTS`)
HANDOVER_VARIANT = "exponential"


class Chain:
    """A chain's directory and record (`record`, the parsed chain.json)."""

    def __init__(self, directory: pathlib.Path, record: dict) -> None:
        self.directory, self.record = directory, record

    @classmethod
    def create(
        cls, runs_directory: pathlib.Path, name: str, arguments: list[str], budget_steps: int
    ) -> Chain:
        """A new chain `name` under `runs_directory`: arguments, the `bnb train` arguments
        that describe both stages (each stage's options are parsed from them again);
        budget_steps, the combined budget. Neither stage's run may exist yet."""
        runs_directory = pathlib.Path(runs_directory)
        taken = [
            path.name
            for path in (runs_directory / name, *(runs_directory / f"{name}-{s}" for s in STAGES))
            if path.exists()
        ]
        if taken:
            raise FileExistsError(f"the chain {name!r} needs {', '.join(taken)}, which exist")
        directory = runs_directory / name
        directory.mkdir(parents=True)
        record = dict(
            name=name,
            arguments=list(arguments),
            budget_steps=budget_steps,
            stages={stage: f"{name}-{stage}" for stage in STAGES},
            handover=None,
        )
        chain = cls(directory, record)
        chain.save()
        return chain

    @classmethod
    def open(cls, runs_directory: pathlib.Path, name: str) -> Chain:
        """The chain `name` under `runs_directory`."""
        directory = pathlib.Path(runs_directory) / name
        if not (directory / RECORD).is_file():
            raise FileNotFoundError(f"no chain {name!r} in {runs_directory}")
        return cls(directory, json.loads((directory / RECORD).read_text()))

    @staticmethod
    def exists(runs_directory: pathlib.Path, name: str) -> bool:
        """Whether `name` under `runs_directory` is a chain."""
        return (pathlib.Path(runs_directory) / name / RECORD).is_file()

    def save(self) -> None:
        """Write the record, atomically."""
        write_json(self.directory / RECORD, self.record)

    def update(self, **fields) -> None:
        """Set fields of the record and save it."""
        self.record.update(fields)
        self.save()

    def stage_run(self, stage: str) -> str:
        """The run name of a stage ("bc" or "rl")."""
        return self.record["stages"][stage]

    def current_run(self) -> Run | None:
        """The run of the stage the chain is in: the RL stage's once it exists, else the
        BC stage's (None before either exists)."""
        runs_directory = self.directory.parent
        for stage in reversed(STAGES):
            name = self.stage_run(stage)
            if (runs_directory / name).exists():
                return Run.open(runs_directory, name)
        return None

    def stage_record(self, stage: str, start_step: int) -> dict:
        """The `chain` entry of a stage's run.json."""
        return dict(name=self.record["name"], stage=stage, start_step=start_step)

    def hand_over(self, step: int, stopped_by: str, policy: pathlib.Path) -> None:
        """Record the handover: the BC stage's last environment step, the rule that stopped
        it and the policy file the RL stage starts from."""
        self.update(handover=dict(step=step, stopped_by=stopped_by, policy=str(policy)))


def reinforcement_updates(budget_steps: int, handover_step: int, steps_per_update: int) -> int:
    """The RL stage's update budget: the whole updates that fit in the combined budget
    after the BC stage's steps (none when BC used it all)."""
    return max(0, budget_steps - handover_step) // steps_per_update
