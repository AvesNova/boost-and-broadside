"""Small settings of the real packages for the learner's tests, and a recorder of game steps."""

import dataclasses

import torch

from lean import game, model, scripted
from lean.learn import League, LearnConfig, Opponent, WeightAverages, start

# The tests' scale, and their seat layout: half the environments self-play (the tests index
# seats by it), whatever the default share.
SMALL = dict(
    team_size_minimum=2,
    team_size_maximum=2,
    minibatches=2,
    epochs=1,
    warmup_steps=1,
    self_play_share=0.5,
    # Every slot's capacity the whole league (C = L): the tests place members on rows by hand;
    # level 2 has its own tests (test_league's slot draws).
    league_capacity_factor=12.0,
    # No personas: the tests that want them name them (test_league's persona tests).
    league_personas=(),
)


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
    schedule's next snapshot step, the league opened first if it is not (the EMA joins
    there, the uniform average once `averages` holds an update). `member`, `slot`: a carry's
    (L,) (default: no league environments, so every slot can be reloaded). Returns the final
    member."""
    member = torch.zeros(0, dtype=torch.long) if member is None else member
    slot = torch.full_like(member, -1) if slot is None else slot
    averages = averages or WeightAverages(policies[0], LearnConfig())
    if not league.opened:
        league.open(policies[0], averages, slot=slot)
    for policy in policies:
        step = league.schedule.points(
            league.schedule.steps[-1] + league.taken * league.schedule.period
        )[league.taken]
        member = league.after_update(policy, averages, step - 1, step, member, slot)
    return member


def loaded_slots(league: League, member: int) -> list[int]:
    """The loaded slots holding `member`, in slot order."""
    return [
        slot
        for slot, (state, holder) in enumerate(zip(league.slot_state, league.slot_holder))
        if state == "loaded" and holder == member
    ]


def place(league: League, carry, members):
    """The carry with its league environments playing `members` (L,), each in its member's
    first loaded slot."""
    slots = torch.tensor([loaded_slots(league, int(member))[0] for member in members])
    return dataclasses.replace(carry, member=torch.as_tensor(members), slot=slots)


class StepRecorder:
    """Wraps lean.game.core.step, the step lean.game.tick calls (install with pytest's
    monkeypatch), and keeps, per call, the state's episode tick, the action applied and the
    step's bullet outcomes and done."""

    def __init__(self):
        self.ticks, self.actions, self.outcomes, self.shot_ticks, self.done = [], [], [], [], []
        self.step = game.core.step

    def __call__(self, game_config, state, action, noise):
        next_state, step_out = self.step(game_config, state, action, noise)
        self.ticks.append(state.tick.clone())
        self.actions.append(action.clone())
        self.outcomes.append(step_out.shot_outcome.clone())
        self.shot_ticks.append(step_out.shot_tick.clone())
        self.done.append(step_out.done.clone())
        return next_state, step_out

    def install(self, monkeypatch) -> StepRecorder:
        monkeypatch.setattr(game.core, "step", self)
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
                                row = (
                                    -1
                                )  # an earlier episode: the bullet left before the recorded ticks
                                break
                            if int(self.ticks[row][environment]) == shot:
                                break
                            row -= 1
                        if row >= 0:
                            label[row, environment, slot] = outcome
        return label
