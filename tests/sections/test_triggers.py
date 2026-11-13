"""The trigger rules (diagnostics spec 3 and 6.3-6.5): each fires on a synthetic series that
should trip it and stays quiet on one that should not, with the warm-up, the hysteresis and
the cooldown; the state survives a checkpoint; the skipped-step action; the dashboards."""

import sys

import learner_cases as lc
import pytest
import torch

from bnb.runs import triggers
from bnb.runs.triggers import Triggers

CPU = torch.device("cpu")
RULES = {rule.name: rule for rule in triggers.default_rules()}


def _flags(rule_name: str, records: list[dict]) -> list[int]:
    """The rule's flag over a run of records (updates from 1)."""
    rules = Triggers((RULES[rule_name],))
    return [
        rules.check(record, update)[f"alerts/{rule_name}"]
        for update, record in enumerate(records, 1)
    ]


def test_there_are_eleven_alerts_and_thirteen_warnings():
    severities = [rule.severity for rule in RULES.values()]
    assert severities.count(triggers.ALERT) == 11 and severities.count(triggers.WARNING) == 13


def _combat(mode: str, deaths: float, hits: float) -> dict:
    """One update's combat series of `mode`'s training games."""
    return {
        f"combat/deaths_per_ship_minute/{mode}": deaths,
        f"combat/hits_per_ship_minute/{mode}": hits,
    }


# (rule, a series that trips it, one that does not); each record is one update's
QUIET_STEPS = [{"train/environment_steps_per_second": 1000.0}] * 12
# a BC start (0.3 deaths and 26 hits per ship-minute), then the RL check's collapse
FIGHTING = [_combat("team_deathmatch", 0.3 + 0.02 * (k % 2), 26.0 + k % 3) for k in range(8)]
COLLAPSED = [_combat("team_deathmatch", 0.0, 0.2)] * 3
CASES = {
    "skipped_step": ([{"train/skipped_steps": 1}], [{"train/skipped_steps": 0}]),
    "recompile": (
        [{"counters/recompiles": 0}] * 3 + [{"counters/recompiles": 1}],
        [{"counters/recompiles": 0}] * 4,
    ),
    "memory": (
        [{"train/memory_peak_gigabytes": 7.5, "train/memory_budget_gigabytes": 8.0}],
        [{"train/memory_peak_gigabytes": 7.0, "train/memory_budget_gigabytes": 8.0}],
    ),
    "behaviour_mismatch": ([{"policy/kl_first": 0.01}], [{"policy/kl_first": 1e-6}]),
    "nonfinite_input": ([{"input/nonfinite_count": 3}], [{"input/nonfinite_count": 0}]),
    "gauge_runaway": (
        [{"pool/gauge_anomalies/frontline": 1}],
        [{"pool/gauge_anomalies/frontline": 0}],
    ),
    "never_seen_ship": ([{"fog/never_seen_share": 0.1}], [{"fog/never_seen_share": 0.0}]),
    "throughput": (
        QUIET_STEPS[:6] + [{"train/environment_steps_per_second": 100.0}] * 3,
        QUIET_STEPS[:9],
    ),
    "rating_drop": (
        [
            {"rating/live/frontline": 100.0 + 10 * k, "rating/live_stderr/frontline": 5.0}
            for k in range(12)
        ]
        + [{"rating/live/frontline": 100.0, "rating/live_stderr/frontline": 5.0}] * 3,
        [
            {"rating/live/frontline": 100.0 + 10 * k, "rating/live_stderr/frontline": 5.0}
            for k in range(15)
        ],
    ),
    "entropy_collapse": (
        [{"policy/entropy/joint": 3.0 + 0.01 * (k % 2)} for k in range(14)]
        + [{"policy/entropy/joint": 1.0}] * 3,
        [{"policy/entropy/joint": 3.0 + 0.01 * (k % 2)} for k in range(17)],
    ),
    "critic_useless": (
        [{"critic/explained_variance/outcome": -0.2}] * 13,
        [{"critic/explained_variance/outcome": 0.3}] * 13,
    ),
    "support_clipping": (
        [{"critic/support_clip/damage": 0.05}],
        [{"critic/support_clip/damage": 0.001}],
    ),
    "envelope_violation": (
        [{"input/envelope_violation_share/speed": 0.02}],
        [{"input/envelope_violation_share/speed": 0.0}],
    ),
    "representation_collapse": (
        [{"representation/effective_rank/trunk_block_0": 100.0}] * 10
        + [{"representation/effective_rank/trunk_block_0": 50.0}] * 3,
        [{"representation/effective_rank/trunk_block_0": 100.0}] * 13,
    ),
    "dormant_units": (
        [{"representation/dormant_share/trunk_block_0": 0.01 * k} for k in range(30)],
        [{"representation/dormant_share/trunk_block_0": 0.05 + 0.001 * (k % 3)} for k in range(30)],
    ),
    "update_ratio": (
        [{"train/update_ratio/trunk_block_0": 0.5}] * 13,
        [{"train/update_ratio/trunk_block_0": 1e-3}] * 13,
    ),
    "label_validity": (
        [{"enemy_action/supervised_share": 0.8}] * 10
        + [{"enemy_action/supervised_share": 0.2}] * 3,
        [{"enemy_action/supervised_share": 0.8}] * 13,
    ),
    "sampling_collapse": (
        [{"pool/sampling_entropy/frontline": 0.1}] * 13,
        [{"pool/sampling_entropy/frontline": 0.9}] * 13,
    ),
    "reward_hacking": (
        [
            {"reward/rate/damage/league": 0.01 * k, "rating/live/frontline": 500.0 + (k % 2)}
            for k in range(60)
        ],
        [
            {"reward/rate/damage/league": 0.01 * k, "rating/live/frontline": 500.0 + 5 * k}
            for k in range(60)
        ],
    ),
    "head_worse_than_trivial": (
        [{"next_state/gain/allies": -0.1}] * 13,
        [{"next_state/gain/allies": 0.2}] * 13,
    ),
    "sigma_miscalibrated": (
        [{"belief/within_one_sigma/shield/age_2_5_seconds": 0.2}] * 13,
        # the bc-v19 false alarm: a collapsed sigma's mean square error, coverage fine
        [
            {
                "belief/normalised_square_error/shield/age_0.1_0.5_seconds": 1e26,
                "belief/normalised_square_error/energy/age_0.1_0.5_seconds": float("nan"),
                "belief/within_one_sigma/shield/age_0.1_0.5_seconds": 0.7,
                "belief/within_one_sigma/energy/age_0.1_0.5_seconds": 0.98,
            }
        ]
        * 13,
    ),
    "combat_collapse/team_deathmatch": (FIGHTING + COLLAPSED, FIGHTING + FIGHTING),
    "combat_collapse/elimination": (
        [_combat("elimination", 0.4, 30.0)] * 5 + [_combat("elimination", 0.4, 1.0)] * 3,
        [_combat("elimination", 0.4, 30.0)] * 5 + [_combat("elimination", 0.2, 15.0)] * 3,
    ),
    "side_imbalance": (
        [{"episode/result/team_0": 60, "episode/result/team_1": 40, "episode/result/draw": 0}] * 12,
        [{"episode/result/team_0": 50, "episode/result/team_1": 50, "episode/result/draw": 0}] * 12,
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_each_rule_fires_on_its_case_and_stays_quiet_otherwise(name):
    tripping, quiet = CASES[name]
    assert _flags(name, tripping)[-1] == 1, name
    assert not any(_flags(name, quiet)), name


def test_every_rule_has_a_case():
    assert set(CASES) == set(RULES)


def test_the_combat_collapse_smooths_and_floors_at_the_lower_of_absolute_and_own_level():
    name = "combat_collapse/team_deathmatch"
    # one quiet update in a fighting run does not trip it (the mean of the last three)
    dip = FIGHTING + [_combat("team_deathmatch", 0.0, 0.0)] + FIGHTING[:2]
    assert not any(_flags(name, dip))
    # quiet while the baseline (the first five updates) fills, however low
    assert not any(_flags(name, [_combat("team_deathmatch", 0.0, 0.0)] * 5))
    # a run that starts fighting little (from scratch: 0.02 deaths, 1 hit per ship-minute)
    # is measured against its own level: a quarter of it, not the absolute floor
    scratch = [_combat("team_deathmatch", 0.02, 1.0)] * 5
    assert not any(_flags(name, scratch + [_combat("team_deathmatch", 0.02, 1.0)] * 5))
    assert _flags(name, scratch + [_combat("team_deathmatch", 0.002, 0.1)] * 3)[-1] == 1
    # a run that fights hard alerts only at the absolute floor (0.075 deaths, 6.5 hits)
    hard = [_combat("team_deathmatch", 2.0, 100.0)] * 5
    assert not any(_flags(name, hard + [_combat("team_deathmatch", 0.1, 10.0)] * 5))
    assert _flags(name, hard + [_combat("team_deathmatch", 0.05, 10.0)] * 3)[-1] == 1
    # the other mode's series leave it idle
    assert not any(_flags(name, [_combat("elimination", 0.0, 0.0)] * 10))


def test_the_combat_collapse_stop_ends_a_run_after_k_updates_with_the_alert():
    from types import SimpleNamespace

    from bnb.runs import session

    rules = Triggers()
    records = FIGHTING + COLLAPSED + COLLAPSED
    reasons = {}
    for after in (0, 2, 4):
        trainer = SimpleNamespace(config=SimpleNamespace(combat_collapse_stop=after))
        recorder = SimpleNamespace(triggers=Triggers())
        reasons[after] = [
            (recorder.triggers.check(record, update), session.combat_collapse(trainer, recorder))[1]
            for update, record in enumerate(records, 1)
        ]
    assert not any(reasons[0])  # off by default
    first = [index for index, reason in enumerate(reasons[2]) if reason]
    flags = [
        rules.check(record, update)["alerts/combat_collapse/team_deathmatch"]
        for update, record in enumerate(records, 1)
    ]
    alerted = flags.index(1)
    assert first[0] == alerted + 1  # the second consecutive update with the alert
    assert [index for index, reason in enumerate(reasons[4]) if reason][0] == alerted + 3
    assert reasons[2][first[0]].startswith("combat_collapse/team_deathmatch: 2 consecutive")


def test_a_rule_is_quiet_during_its_warm_up():
    rule = triggers.Above("probe", triggers.WARNING, "x", "x/value", 0.0, warmup=3)
    rules = Triggers((rule,))
    flags = [rules.check({"x/value": 1.0}, update)["alerts/probe"] for update in range(1, 6)]
    assert flags == [0, 0, 0, 1, 1]


def test_a_rule_needs_its_breach_to_hold_for_the_hysteresis():
    rule = triggers.Above("probe", triggers.WARNING, "x", "x/value", 0.0, hysteresis=3)
    rules = Triggers((rule,))
    values = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    flags = [rules.check({"x/value": v}, u)["alerts/probe"] for u, v in enumerate(values, 1)]
    assert flags == [0, 0, 0, 0, 0, 1, 1]


def test_a_fired_event_is_not_repeated_within_the_cooldown():
    rule = triggers.Above("probe", triggers.ALERT, "x", "x/value", 0.0, cooldown=3)
    rules = Triggers((rule,))
    fired = ["alerts/events" in rules.check({"x/value": 1.0}, update) for update in range(1, 8)]
    assert fired == [True, False, False, True, False, False, True]


def test_an_event_names_the_rule_series_value_threshold_and_panel():
    out = Triggers().check({"train/skipped_steps": 2, "train/nonfinite_steps/density": 2}, 1)
    events = out["alerts/events"]
    assert events["columns"] == list(triggers.EVENT_COLUMNS)
    assert events["rows"] == [
        ["skipped_step", "alert", 1, "train/skipped_steps", 2, 0, "train/skipped_steps"],
        ["skipped_step", "alert", 1, "train/nonfinite_steps/density", 2, 0, "train/skipped_steps"],
    ]
    assert out["alerts/active"] == 1


def test_an_absent_series_leaves_its_rule_idle():
    out = Triggers().check({}, 100)
    assert out["alerts/active"] == 0 and "alerts/events" not in out


def _tiny(updates: int = 2):
    sys.path.insert(0, str(lc.__file__).rsplit("/sections/", 1)[0] + "/runs")
    from runs_cases import tiny_train_config

    return tiny_train_config(updates=updates)


@pytest.mark.slow
def test_rule_state_survives_a_checkpoint_round_trip(tmp_path, global_generator):
    from bnb.runs import checkpoint, session
    from bnb.runs.recording import Recorder
    from bnb.runs.records import Run
    from bnb.runtime.train import Trainer

    config = _tiny(updates=3)
    trainer = Trainer(config, CPU)
    run = Run.create(tmp_path, "rules", config, ["test"], None)
    rules = Triggers()
    for update in range(1, 6):  # history the rules keep: EMAs, peaks, windows, counts
        rules.check(
            {"policy/entropy/joint": 3.0 + 0.1 * update, "rating/live/m": 10.0 * update}, update
        )
    recorder = Recorder(run, trainer, None, rules)
    session.save(run, trainer, recorder=recorder)
    restored = checkpoint.triggers_state(run.checkpoint_path(trainer.ppo.environment_steps))
    assert restored == rules.state
    resumed = Triggers(state=restored)
    following = {"policy/entropy/joint": 0.5, "rating/live/m": 60.0}
    assert resumed.check(following, 6) == rules.check(following, 6)
    assert resumed.state == rules.state


@pytest.mark.slow
def test_a_non_finite_step_writes_the_checkpoint_and_the_dump_and_training_continues(
    tmp_path, global_generator
):
    from bnb.runs import session
    from bnb.runs.recording import Recorder
    from bnb.runs.records import Run
    from bnb.runtime.train import Trainer

    config = _tiny(updates=2)
    trainer = Trainer(config, CPU)
    run = Run.create(tmp_path, "skips", config, ["test"], None)
    run.start_segment(0)
    hook = next(trainer.model.parameters()).register_hook(lambda gradient: gradient * torch.nan)
    records = []
    recorder = Recorder(run, trainer, None, Triggers())

    def log(record: dict) -> None:
        records.append(record)
        hook.remove()  # only the first update's steps are non-finite

    status = session.train(run, trainer, log, recorder=recorder)
    assert status == "finished"
    first, second = records
    assert first["train/skipped_steps"] == config.learner.minibatches
    assert first["alerts/skipped_step"] == 1 and second["alerts/skipped_step"] == 0
    step = first["counters/environment_steps"]
    assert run.checkpoint_path(step).is_dir()  # the action's checkpoint
    dumped = run.directory / "dumps" / f"step_{step}"
    assert (dumped / "batch.pt").is_file() and (dumped / "record.json").is_file()
    assert run.record["dumps"] == [str(dumped)]
    assert second["train/skipped_steps"] == 0
    assert all(torch.isfinite(p).all() for p in trainer.model.parameters())


def test_the_code_builds_the_headline_and_curated_views():
    pytest.importorskip("wandb_workspaces")
    from bnb.runtime import dashboards

    views = dashboards.build("2026-11-03-mixed-amber-acorn", "entity", "project")
    # the 12 of the diagnostics spec, with the ratings and the games' ends split by mode
    expected = 10 + 2 * len(dashboards.MODES)
    assert dashboards.panel_count(dashboards.sections_for(dashboards.HEADLINE, "ppo")) == expected
    headline, curated = views["headline"], views["curated"]
    assert sum(len(section.panels) for section in headline.sections) == expected
    assert [section.name for section in curated.sections][:7] == [
        "Alerts",
        "Strength",
        "Rating lanes",
        "Health",
        "Policy",
        "Value",
        "Learning signal",
    ]
    assert all(section.name != "bc/" for section in curated.sections)
    headline._to_model()  # the workspace's saved form, built offline
    curated._to_model()


# What a BC run's views leave out: reinforcement learning's ratings, league, KL and clip.
RL_SERIES = ("rating/live", "pool/score", "pool/size", "policy/kl", "policy/clip_fraction")


def test_the_bc_views_leave_out_reinforcement_learnings_panels():
    import re

    pytest.importorskip("wandb_workspaces")
    from bnb.runtime import dashboards

    views = dashboards.build("2026-11-03-mixed-amber-acorn", "entity", "project", "bc")
    headline, curated = views["headline"], views["curated"]
    assert sum(len(section.panels) for section in headline.sections) == 11 + 2 * len(
        dashboards.MODES
    )
    names = [section.name for section in curated.sections]
    # BC's live policy is rated in its lanes too, against the raw scripted agent
    assert names[:4] == ["Alerts", "Strength", "Rating lanes", "bc/"] and "League" not in names
    assert next(s for s in curated.sections if s.name == "bc/").is_open
    for view in (dashboards.HEADLINE, dashboards.CURATED):
        for section in dashboards.sections_for(view, "bc"):
            for item in section.panels:
                expression = re.compile(dashboards.regex(item.series))
                shown = [series for series in RL_SERIES if expression.match(series)]
                assert not shown, (section.name, item.title, shown)
    headline._to_model()
    curated._to_model()


@pytest.mark.parametrize(
    ("app", "link"),
    [
        ("https://forge.coreweave.com/wandb", "https://forge.coreweave.com/wandb/e/p?nw=abc"),
        ("https://forge.coreweave.com/wandb/", "https://forge.coreweave.com/wandb/e/p?nw=abc"),
        ("https://wandb.ai", "https://wandb.ai/e/p?nw=abc"),
        ("http://localhost:8080/a/b", "http://localhost:8080/a/b/e/p?nw=abc"),
    ],
)
def test_a_views_link_keeps_the_apps_path_prefix(app, link):
    """`wandb-workspaces` builds the view's URL from the app's host alone; the link printed
    is under the app's own address."""
    from bnb.runtime import dashboards

    host = app.split("/")[2]
    assert dashboards.app_link(f"https://{host}/e/p?nw=abc", app) == link


def test_the_app_address_comes_from_the_configured_base_url(monkeypatch):
    pytest.importorskip("wandb")
    import wandb

    from bnb.runtime import dashboards

    monkeypatch.setattr(wandb, "setup", lambda: _Setup("https://api.forge.coreweave.com"))
    assert dashboards.app_url() == "https://forge.coreweave.com/wandb"
    monkeypatch.setattr(wandb, "setup", lambda: _Setup("https://wandb.example.org"))
    assert dashboards.app_url() == "https://wandb.example.org"  # a server of one's own


class _Setup:
    """`wandb.setup()`'s result with settings for `base_url`."""

    def __init__(self, base_url: str) -> None:
        import wandb

        self.settings = wandb.Settings(base_url=base_url)


def test_a_bc_run_leaves_reinforcement_learnings_rules_out():
    """In BC a falling student entropy is the goal: the entropy collapse and the other
    RL-only rules are not run, and each of the 24 rules says which objectives it watches."""
    assert len(RULES) == 24
    assert all(set(rule.objectives) <= {"ppo", "bc"} for rule in RULES.values())
    rl_only = {name for name, rule in RULES.items() if rule.objectives == ("ppo",)}
    assert rl_only == {
        "behaviour_mismatch",
        "rating_drop",
        "entropy_collapse",
        "sampling_collapse",
        "gauge_runaway",
        "reward_hacking",
        "combat_collapse/team_deathmatch",
        "combat_collapse/elimination",
    }
    falling = [{"policy/entropy/joint": 3.0 * 0.8**update} for update in range(30)]
    ppo, cloning = Triggers(), Triggers(objective="bc")
    assert {rule.name for rule in cloning.rules} == set(RULES) - rl_only
    fired = [ppo.check(record, update) for update, record in enumerate(falling, 1)]
    quiet = [cloning.check(record, update) for update, record in enumerate(falling, 1)]
    assert any(out["alerts/entropy_collapse"] for out in fired)
    assert all("alerts/entropy_collapse" not in out and "alerts/events" not in out for out in quiet)
    with pytest.raises(ValueError, match="objective"):
        Triggers(objective="sft")


def test_a_panel_pattern_matches_its_slices():
    import re

    from bnb.runtime import dashboards

    expression = re.compile(dashboards.regex(("policy/usage/power/*", "policy/kl")))
    assert expression.match("policy/usage/power/2") and expression.match("policy/kl")
    assert not expression.match("policy/kl_first") and not expression.match("policy/usage/stick/0")


def test_a_value_at_the_threshold_does_not_breach():
    rules = Triggers((RULES["support_clipping"], RULES["critic_useless"]))
    record = {"critic/support_clip/damage": 0.01, "critic/explained_variance/outcome": 0.0}
    for update in range(1, 20):
        assert rules.check(record, update)["alerts/active"] == 0
