"""Small settings of the real packages for the learner's tests, and a recorder of game steps."""

import dataclasses

import torch

from lean import game, model, scripted
from lean.learn import League, LearnConfig, Opponent, WeightAverages, start

SMALL = dict(team_size_minimum=2, team_size_maximum=2, minibatches=2, epochs=1, warmup_steps=1)


def small(envs: int = 4, *, game_config=None, league: bool = False, seed: int = 0, **overrides):
    """(config, game_config, policy, opponent, state, carry, generator): the real game (the
    core game with the training spawn unless `game_config`), the default model, the scripted
    agent, `envs` environments of 2v2 by default; a League (the scripted agent only) when
    `league`. `overrides` replace LearnConfig fields."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    config = dataclasses.replace(LearnConfig(), **{**SMALL, **overrides})
    game_config = game.GameConfig(training_spawn=True) if game_config is None else game_config
    policy = model.Policy(model.ModelConfig())
    opponent = Opponent(scripted.ScriptedConfig(), League(config) if league else None)
    state, carry = start(config, game_config, policy, opponent, envs, generator)
    return config, game_config, policy, opponent, state, carry, generator


def add_snapshots(league: League, policies, member=None, slot=None, averages=None):
    """Snapshots of `policies` into `league` through League.after_update, each at the
    schedule's next snapshot step (so the EMA, and the uniform average once `averages`
    holds an update, join with the first). `member`, `slot`: a carry's (L,) (default: no
    league environments). Returns the final member."""
    member = torch.zeros(0, dtype=torch.long) if member is None else member
    slot = torch.full_like(member, -1) if slot is None else slot
    for policy in policies:
        step = league.schedule.points(league.schedule.steps[-1] + league.taken * league.schedule.period)[league.taken]
        member, lost = league.after_update(
            policy, averages or WeightAverages(policy, LearnConfig()), step - 1, step, member, slot
        )
        assert lost is None
    return member


class StepRecorder:
    """Wraps lean.game.step (install with pytest's monkeypatch) and keeps, per call, the
    state's episode tick, the action applied and the step's bullet outcomes and done."""

    def __init__(self):
        self.ticks, self.actions, self.outcomes, self.shot_ticks, self.done = [], [], [], [], []
        self.step = game.step

    def __call__(self, game_config, state, action, noise):
        next_state, step_out = self.step(game_config, state, action, noise)
        self.ticks.append(state.tick.clone())
        self.actions.append(action.clone())
        self.outcomes.append(step_out.shot_outcome.clone())
        self.shot_ticks.append(step_out.shot_tick.clone())
        self.done.append(step_out.done.clone())
        return next_state, step_out

    def install(self, monkeypatch) -> "StepRecorder":
        monkeypatch.setattr(game, "step", self)
        return self

    def hit_labels(self) -> torch.Tensor:
        """(T, E, N) long, every tick's label found by search: the outcome of the bullet a
        ship emitted on that tick, from the bullet's end found later in the same episode
        (the latest earlier row of the episode whose episode tick is the bullet's shot tick);
        -1 where no bullet ended within the recorded ticks."""
        ticks = len(self.ticks)
        envs, slots, bullets = self.outcomes[0].shape
        label = torch.full((ticks, envs, slots), -1, dtype=torch.long)
        for end in range(ticks):
            for environment in range(envs):
                for slot in range(slots):
                    for bullet in range(bullets):
                        outcome = int(self.outcomes[end][environment, slot, bullet])
                        if outcome < 0:
                            continue
                        shot = int(self.shot_ticks[end][environment, slot, bullet])
                        row = end
                        while row >= 0:
                            if row < end and bool(self.done[row][environment]):
                                row = -1  # an earlier episode: the bullet left before the recorded ticks
                                break
                            if int(self.ticks[row][environment]) == shot:
                                break
                            row -= 1
                        if row >= 0:
                            label[row, environment, slot] = outcome
        return label
