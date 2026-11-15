"""The league's arm options, each a setting with today's behaviour by default: the mode mix
(`TrainConfig.mode_mix`, arm A5), one-sided self-play (`TrainConfig.self_play_seats` with
`LeagueConfig.latest_share`, arm A6) and the exploiter's fixed opponent
(`LeagueConfig.fixed_opponent`, arm R3)."""

import dataclasses

import league_cases as cases
import pytest
import torch

from bnb import cli
from bnb.contracts.pools import environments_per_pool, pool_game
from bnb.game.api import PRESETS
from bnb.league.pool import FIXED_OPPONENT, LATEST
from bnb.models.relational import RelationalConfig
from bnb.runs import checkpoint
from bnb.runs.checkpoint import CheckpointError
from bnb.runtime.opponents import OPPONENTS, LeagueConfig
from bnb.runtime.snapshots import SnapshotSchedule
from bnb.runtime.train import (
    LEAGUE_ONLY,
    LEARNER,
    LINEUP,
    Trainer,
    arranged,
    check_pools,
    training_lineup,
)

pytestmark = pytest.mark.usefixtures("global_generator")

MIX_844 = {"real_scale_frontline": 8, "real_scale_deathmatch": 4, "real_scale_elimination": 4}
GAME_FIELDS = ("mode", "arena_radius", "spawn_groups", "episode_seconds", "respawn_seconds")


def _requested(*settings: str, preset: str = "real_scale_all_modes", **options):
    return cli._requested(cli.Train(preset=preset, set=settings, device="cpu", **options))


def _mix_setting(mix: dict[str, int]) -> str:
    return "mode_mix=" + str(mix).replace("'", '"')


def _update(trainer: Trainer, league) -> None:
    trainer.collect()
    league.collected(trainer)
    trainer.ppo.environment_steps += trainer.steps_per_update
    league.updated(trainer)


# ------------------------------------------------------------------ A5: the mode mix


def test_the_default_config_keeps_its_pools():
    config = _requested()
    assert config.mode_mix is None and config.self_play_seats == "both"
    assert arranged(config, "real_scale_all_modes") is config
    sizes = environments_per_pool(config.pools, 2048)
    assert sizes == (384, 384, 320, 320, 320, 320)  # 6/5/5, today's


def test_the_844_mix_builds_each_modes_pools_with_its_own_game():
    config = _requested(_mix_setting(MIX_844))
    assert [pool.name for pool in config.pools] == [
        f"{game}/{pool.name}" for game in MIX_844 for pool in LINEUP
    ]
    assert environments_per_pool(config.pools, 2048) == (512, 512, 256, 256, 256, 256)
    for pool in config.pools:
        own = PRESETS[pool.name.split("/")[0]]
        played = pool_game(config.game, pool)
        for field in GAME_FIELDS:
            assert getattr(played, field) == getattr(own, field), (pool.name, field)
    radii = {pool.name.split("/")[0]: pool.arena_radius for pool in config.pools}
    assert radii["real_scale_deathmatch"] != radii["real_scale_frontline"]
    check_pools(config)


def test_a_mode_mix_is_checked():
    with pytest.raises(SystemExit, match="not game presets"):
        _requested(_mix_setting({"real_scale_frontline": 1, "nowhere": 1}))
    with pytest.raises(SystemExit, match="positive whole weights"):
        _requested(_mix_setting({"real_scale_frontline": 1, "real_scale_deathmatch": 0}))
    with pytest.raises(SystemExit, match="other sizes|another ship set"):
        _requested(_mix_setting({"real_scale_frontline": 1, "deathmatch": 1}))
    config = _requested()  # pools that are not the mix's are refused
    with pytest.raises(ValueError, match="not mode_mix's"):
        check_pools(dataclasses.replace(config, mode_mix=MIX_844))


# ------------------------------------------------------------------ A6: one seat


def test_one_seat_drops_the_self_play_pool_and_needs_latest():
    with pytest.raises(SystemExit, match="latest_share"):
        _requested("self_play_seats=one")
    config = _requested(
        "self_play_seats=one", "league.latest_share=0.5", "league.slots=8", environments=4096
    )
    assert training_lineup(config) == LEAGUE_ONLY
    assert all(pool.seats == (LEARNER, OPPONENTS) for pool in config.pools)
    assert environments_per_pool(config.pools, 4096) == (1536, 1280, 1280)
    both = _requested("league.latest_share=0.5", "league.slots=8")
    with pytest.raises(ValueError, match="train both seats"):
        check_pools(dataclasses.replace(both, self_play_seats="one"))
    # with a mix: each mode's league pool alone, at the mix's shares
    mixed = _requested("self_play_seats=one", "league.latest_share=0.5", _mix_setting(MIX_844))
    assert environments_per_pool(mixed.pools, 2048) == (1024, 512, 512)


def test_latest_share_is_a_whole_number_of_slots_and_leaves_one_to_draw():
    assert LeagueConfig(slots=4, latest_share=0.5).latest_slots == 2
    with pytest.raises(ValueError, match="whole number"):
        LeagueConfig(slots=4, latest_share=0.3)
    with pytest.raises(ValueError, match="cut_slots"):
        LeagueConfig(slots=4, latest_share=0.75, cut_slots=2)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        LeagueConfig(latest_share=1.0)


def _one_seat(tmp_path, **league):
    settings = dict(latest_share=0.5, cut_every=1, cut_slots=2) | league
    return cases.league(
        tmp_path,
        pools=LEAGUE_ONLY,
        self_play_seats="one",
        league=dataclasses.replace(cases.config().league, **settings),
    )


def test_one_seat_trains_only_the_learners_seat(tmp_path):
    trainer, _ = _one_seat(tmp_path)
    lineup = trainer.rollout.lineup
    environments = trainer.config.environments
    assert lineup.rows(LEARNER) == environments  # one learner row per game
    assert lineup.rows(OPPONENTS) == environments
    assert torch.equal(lineup.environments[LEARNER].sort().values, torch.arange(environments))
    both = Trainer(cases.config(), torch.device("cpu"))  # today's: self-play's two seats
    assert both.rollout.lineup.rows(LEARNER) == environments * 3 // 2


def test_latest_holds_its_share_and_the_live_weights_after_every_update(tmp_path):
    trainer, league = _one_seat(tmp_path)
    opponents, slots = trainer.opponents, trainer.config.league.slots
    latest_rows = (opponents.slot_of_row < 2).float().mean()
    assert float(latest_rows) == 0.5  # the share of the opponent rows
    for _ in range(4):
        with torch.no_grad():  # the live weights move, as a learning step moves them
            for parameter in trainer.model.parameters():
                parameter.add_(0.01)
        _update(trainer, league)
        assert league.held[:2] == [LATEST, LATEST]  # never cut
        assert LATEST not in league.held[2:] and LATEST not in league.entries
        live = trainer.model.state_dict()
        for slot in range(slots):
            entry = league.entries.get(league.held[slot])
            if entry is not None and not entry.network:
                continue  # Imugi: its set is not played
            weights = live if slot < 2 else league._weights_of(entry)
            assert all(
                torch.equal(opponents.parameters[f"model.{name}"][slot], value)
                for name, value in weights.items()
            ), slot
    metrics = league.metrics()
    assert metrics["pool/slot_share/latest"] == 0.5
    assert all(LATEST not in ratings for ratings in league.ratings.ratings.values())


def test_latest_resumes_from_the_league_state(tmp_path):
    trainer, league = _one_seat(tmp_path / "first")
    _update(trainer, league)
    from bnb.league.league import League

    restored = League(trainer, tmp_path / "second", league.state())
    assert restored.held == league.held and restored.next_slot >= 2


# ------------------------------------------------------------------ R3: a fixed opponent


def _fixed_policy(tmp_path, trainer: Trainer) -> str:
    """A policy file of the trainer's build with other weights."""
    other = Trainer(trainer.config, torch.device("cpu")).model
    with torch.no_grad():
        for parameter in other.parameters():
            parameter.add_(0.5)
    path = tmp_path / "fixed"
    checkpoint.save_policy(path, other, game=trainer.config.game, types=trainer.types)
    return str(path)


def test_a_fixed_opponent_clears_the_cheese_and_takes_no_latest():
    assert LeagueConfig(fixed_opponent="somewhere").cheese == ()
    with pytest.raises(ValueError, match="latest_share must be 0"):
        LeagueConfig(fixed_opponent="somewhere", latest_share=0.5)
    config = _requested("league.fixed_opponent=somewhere")
    assert training_lineup(config) == LEAGUE_ONLY
    assert all(pool.seats == (LEARNER, OPPONENTS) for pool in config.pools)


def test_the_exploiters_league_is_the_fixed_opponent_alone(tmp_path):
    base = cases.config(pools=LEAGUE_ONLY)
    path = _fixed_policy(tmp_path, Trainer(base, torch.device("cpu")))
    trainer, league = cases.league(
        tmp_path / "league",
        pools=LEAGUE_ONLY,
        league=LeagueConfig(
            fixed_opponent=path, cut_every=1, snapshots=SnapshotSchedule(every=16 * 16)
        ),
    )
    assert list(league.entries) == [FIXED_OPPONENT]
    fixed = checkpoint.load_policy(path, torch.device("cpu")).state_dict()
    for _ in range(3):  # a snapshot is due every update: none joins
        _update(trainer, league)
        assert list(league.entries) == [FIXED_OPPONENT] and not league.snapshots
        assert league.held == [FIXED_OPPONENT] * trainer.config.league.slots
        for name, value in fixed.items():
            stacked = trainer.opponents.parameters[f"model.{name}"]
            assert all(torch.equal(stacked[slot], value) for slot in range(stacked.shape[0]))
    metrics = league.metrics()
    assert metrics["pool/slot_share/fixed"] == 1.0


def test_a_fixed_opponent_of_another_build_is_refused(tmp_path):
    base = cases.config(pools=LEAGUE_ONLY)
    path = _fixed_policy(tmp_path, Trainer(base, torch.device("cpu")))
    with pytest.raises(CheckpointError, match="fixed opponent"):
        cases.league(
            tmp_path / "league",
            pools=LEAGUE_ONLY,
            relational=RelationalConfig(hidden=32),
            league=LeagueConfig(fixed_opponent=path),
        )
