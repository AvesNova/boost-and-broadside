"""The write-up's results charts (`bnb.figures.results`): each drawn once from small
synthetic records in the shapes the runs and measurements write, its files and values
checked. The `synthetic_*` builders also feed the preview renders."""

import json
import math
import pathlib
import random

import pytest

pytest.importorskip("matplotlib")

from bnb.figures.results import arms, break_even, common, fan, game_length, health, team_size
from bnb.league import crossover

PRESETS = ("real_scale_frontline", "real_scale_deathmatch", "real_scale_elimination")


def _play(rng: random.Random, strength: float, lineup: tuple[int, int], games: int) -> dict:
    """A block of games at a lineup for a policy whose score is logistic(strength + 3 r)."""
    p = 1.0 / (1.0 + math.exp(-(strength + 3.0 * crossover.log_ratio(lineup))))
    wins = sum(rng.random() < p for _ in range(games))
    return dict(
        policy_count=lineup[0],
        opponent_count=lineup[1],
        wins=wins,
        draws=0,
        losses=games - wins,
    )


def synthetic_crossover(root: pathlib.Path, snapshots: int = 4, games: int = 64) -> pathlib.Path:
    """Per-snapshot `bnb crossover` results of a chain (`r-bc`, `r-rl`), laid out as the dry
    run lays them out, each fit the real `crossover.fit` of synthetic games."""
    rng = random.Random(0)
    lineups = [(n, 20) for n in (4, 8, 10, 12, 16, 20)] + [(20, n) for n in (16, 12, 10, 8)]
    for stage, run in enumerate(("r-bc", "r-rl")):
        for index in range(snapshots):
            step = (index + 1) * 10_000_000
            presets = {}
            for number, preset in enumerate(PRESETS):
                strength = 0.3 * number + 0.4 * (stage * snapshots + index) - 0.6
                blocks = [_play(rng, strength, lineup, games) for lineup in lineups]
                tallied = crossover.tallies(blocks)
                found = crossover.fit(tallied)
                ratio, ratio_stderr = found.ratio()
                rows = [
                    dict(
                        policy_count=lineup[0],
                        opponent_count=lineup[1],
                        games=int(held[1]),
                        wins=int(held[0]),
                        draws=0,
                        losses=int(held[1] - held[0]),
                        score=held[0] / held[1],
                    )
                    for lineup, held in tallied.items()
                ]
                presets[preset] = dict(
                    fit=found.as_dict(), ratio=ratio, ratio_stderr=ratio_stderr, lineups=rows
                )
            directory = root / run / f"step_{step}" / "artifacts" / "crossover" / "20261116-2300"
            directory.mkdir(parents=True)
            payload = dict(presets=presets, config={})
            (directory / "result.json").write_text(
                json.dumps(dict(provenance={}, kind="crossover", payload=payload, inputs={}))
            )
    return root


def synthetic_arms(path: pathlib.Path) -> pathlib.Path:
    """A forest input of four decisions and the seed-to-seed row."""
    rows = [
        dict(name="A1 map sizes", score=0.041, stderr=0.012, decision="kept"),
        dict(name="A2 team sizes", score=-0.018, stderr=0.011, decision="rejected"),
        dict(name="A7 offensive premium", score=0.006, stderr=0.010, decision="rejected"),
        dict(name="A13 no belief", key="a13", score=None, decision="pending"),
        dict(name="seed to seed", kind="seed_noise", score=0.004, stderr=0.009),
    ]
    path.write_text(json.dumps(dict(title="Arm decisions (synthetic)", delta=0.02, rows=rows)))
    return path


def synthetic_tallies(directory: pathlib.Path, games: int = 96) -> pathlib.Path:
    """Crossover-shaped records (`<preset>.batches.jsonl`) of lineups (n, k n) for n up to
    20 a side, k in 1, 1.5, 2."""
    rng = random.Random(1)
    directory.mkdir(parents=True, exist_ok=True)
    for number, preset in enumerate(PRESETS):
        blocks = []
        for size in range(1, 21):
            for ratio in (1.0, 1.5, 2.0):
                opponent = size * ratio
                if opponent > 20 or not float(opponent).is_integer():
                    continue
                strength = 1.2 + 0.2 * number - 0.08 * abs(size - 5)  # best at its own size
                blocks.append(_play(rng, strength, (size, int(opponent)), games))
        line = dict(key={}, batch=0, blocks=blocks)
        (directory / f"{preset}.batches.jsonl").write_text(json.dumps(line) + "\n")
    return directory


def synthetic_metrics(path: pathlib.Path, updates: int = 120, start: int = 0) -> pathlib.Path:
    """A metrics.jsonl with the health keys, two head losses, a table and keys to skip."""
    rng = random.Random(2)
    lines = []
    for update in range(updates):
        t = update / updates
        lines.append(
            {
                "counters/environment_steps": start + (update + 1) * 1_048_576,
                "policy/clip_fraction": 0.08 - 0.03 * t + 0.01 * rng.random(),
                "policy/kl": 0.01 + 0.004 * rng.random(),
                "critic/explained_variance/outcome": 0.4 + 0.5 * t + 0.03 * rng.random(),
                "critic/explained_variance/damage": 0.6 + 0.3 * t + 0.03 * rng.random(),
                "policy/entropy/joint": 1.2 - 0.6 * t + 0.02 * rng.random(),
                "policy/entropy/stick": 0.8 - 0.4 * t + 0.02 * rng.random(),
                "critic/value_loss/outcome": 1.0 - 0.5 * t + 0.05 * rng.random(),
                "next_state/loss": 20.0 * math.exp(-2 * t) + rng.random(),
                "shot/loss/outcome": 0.7 - 0.2 * t + 0.02 * rng.random(),
                "shot/loss/trajectory": 1.5 - 0.6 * t + 0.02 * rng.random(),
                "episode/losses/real_scale_deathmatch/league": 3,
                "pool/entries": {"columns": ["mode"], "data": [["deathmatch"]]},
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def synthetic_lengths(path: pathlib.Path, games: int = 400) -> pathlib.Path:
    """Per-game lengths per mode, some running the clock out."""
    rng = random.Random(3)
    shapes = dict(
        real_scale_frontline=(200.0, 60.0, 300.0),
        real_scale_deathmatch=(150.0, 70.0, 360.0),
        real_scale_elimination=(95.0, 25.0, 300.0),
    )
    lines = []
    for preset, (mean, spread, clock) in shapes.items():
        for _ in range(games):
            seconds = min(max(rng.gauss(mean, spread), 5.0), clock)
            lines.append(dict(preset=preset, seconds=seconds, clock_seconds=clock))
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def _drawn(written: list[pathlib.Path]) -> None:
    assert [path.suffix for path in written] == [".png", ".pdf", ".svg"]
    assert all(path.stat().st_size > 1000 for path in written)


def test_the_fan_evaluates_each_snapshot_fit_at_the_lineups(tmp_path):
    directory = synthetic_crossover(tmp_path / "break-even", snapshots=2, games=32)
    written, values = fan.draw(directory, tmp_path / "out", opponent_counts=(5, 10))
    _drawn(written)
    for mode in ("frontline", "deathmatch", "elimination"):
        rate, low, high = (values[f"fan_{mode}_5v5_{part}"] for part in ("win_rate", "low", "high"))
        assert 0.0 <= low <= rate <= high <= 1.0
        assert values[f"fan_{mode}_5v10_win_rate"] < rate  # outnumbered: a lower win rate
    from PIL import Image

    assert "Sources" in Image.open(written[0]).text


def test_a_crossing_fit_is_one_half_at_its_break_even():
    found = dict(status="crossing", intercept=0.6, slope=3.0, covariance=((0.01, 0.0), (0.0, 0.04)))
    rate, low, high = fan.win_rate(found, -0.2)
    assert rate == pytest.approx(0.5) and low < 0.5 < high
    assert fan.win_rate(dict(status="separated"), 0.0) is None


def test_break_even_draws_one_line_per_mode_of_the_chain(tmp_path):
    directory = synthetic_crossover(tmp_path / "break-even", snapshots=2, games=32)
    written, values = break_even.draw(directory, tmp_path / "out")
    _drawn(written)
    assert values["crossover_ratio_elimination_steps"] == 40_000_000  # RL after BC's last
    assert values["crossover_ratio_elimination"] > 0.0
    with pytest.raises(SystemExit, match="no crossover results"):
        break_even.draw(tmp_path / "empty", tmp_path / "out")
    by_run = break_even.draw(directory, tmp_path / "out", chain="r-rl")[1]  # a run names its chain
    assert by_run == values
    with pytest.raises(SystemExit, match="no chain"):
        break_even.draw(directory, tmp_path / "out", chain="other")


def test_the_arms_forest_reads_rows_and_marks_decisions(tmp_path):
    written, values = arms.draw(synthetic_arms(tmp_path / "arms.json"), tmp_path / "out")
    _drawn(written)
    assert values["arm_a1_map_sizes_score"] == pytest.approx(0.041)
    assert values["seed_noise_high"] == pytest.approx(0.004 + 1.959964 * 0.009)
    assert "arm_a13_score" not in values  # no estimate yet
    table = tmp_path / "arms.csv"
    table.write_text("name,score,stderr,decision\nA1,0.02,0.01,kept\n")
    rows, _ = arms.read_rows(table)
    assert rows[0]["decision"] == "kept" and rows[0]["key"] == "a1"


def test_an_arm_row_reads_its_evaluation_directory(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from bnb.league import evaluation

    monkeypatch.setattr(evaluation, "read_settings", lambda d: ({}, SimpleNamespace(delta=0.02)))
    look = SimpleNamespace(decision="B")
    summary = SimpleNamespace(overall=(-0.03, 0.01))
    monkeypatch.setattr(evaluation, "decide", lambda d, record: (summary, look, {}))
    path = tmp_path / "arms.json"
    path.write_text(json.dumps(dict(rows=[dict(name="A9", evaluation="a9", arm_side="B")])))
    rows, settings = arms.read_rows(path)
    assert rows[0]["score"] == pytest.approx(0.03) and rows[0]["decision"] == "kept"
    assert settings["delta"] == 0.02


def test_team_size_puts_each_lineup_on_its_ratio(tmp_path):
    directory = synthetic_tallies(tmp_path / "records", games=16)
    written, values = team_size.draw([directory], tmp_path / "out")
    _drawn(written)
    assert "team_size_deathmatch_ratio_1_5_size_2_win_rate" in values
    assert "team_size_deathmatch_ratio_1_5_size_3_win_rate" not in values  # 3v4.5: no lineup
    assert "team_size_deathmatch_ratio_2_size_10_win_rate" in values
    low, high = team_size.wilson(8.0, 16.0)
    assert low < 0.5 < high


def test_run_health_reads_local_records_as_a_chain(tmp_path):
    bc = synthetic_metrics(tmp_path / "r-bc" / "metrics.jsonl", updates=40)
    rl = synthetic_metrics(tmp_path / "r-rl" / "metrics.jsonl", updates=40)
    written, values = health.draw([str(bc.parent), str(rl)], tmp_path / "out")
    _drawn(written)
    assert "health_policy_clip_fraction_final" in values
    assert "health_shot_loss_trajectory_final" in values
    assert not any("episode" in name for name in values)
    titles = [title for title, _ in health.panels_for(health.read_source(str(rl)))]
    assert "next state head loss" in titles and "shot head loss" in titles


def test_game_length_leaves_clock_outs_out_of_the_density(tmp_path):
    written, values = game_length.draw(
        [synthetic_lengths(tmp_path / "games.jsonl", 200)], tmp_path / "out"
    )
    _drawn(written)
    assert 0.0 < values["game_length_frontline_clock_out_share"] < 0.5
    assert values["game_length_elimination_median_seconds"] == pytest.approx(95.0, abs=10.0)


def test_values_merge_into_the_file(tmp_path):
    path = tmp_path / "values.json"
    common.write_values(dict(a=1.0, b=2.0), path)
    common.write_values(dict(b=3.0, c=float("nan")), path)
    assert json.loads(path.read_text()) == dict(a=1.0, b=3.0, c=None)
    assert common.slug(1.5) == "1_5" and common.slug(2.0) == "2"


def test_the_command_draws_a_chart_and_writes_its_values(tmp_path):
    from bnb import cli

    directory = synthetic_crossover(tmp_path / "break-even", snapshots=1, games=16)
    values = tmp_path / "values.json"
    cli.figures_results(
        cli.FiguresResults(
            "break-even",
            (str(directory),),
            out=str(tmp_path / "out"),
            values=str(values),
            formats=("png", "svg"),
        )
    )
    assert (tmp_path / "out" / "break_even_ratio.svg").is_file()
    assert (tmp_path / "out" / "break_even_ratio_dark.png").is_file()  # both theme variants
    assert not (tmp_path / "out" / "break_even_ratio_dark.svg").exists()  # the paper is light
    assert "crossover_ratio_deathmatch" in json.loads(values.read_text())


def test_the_theme_variants_differ_and_restore():
    from bnb.figures import output, theme

    assert theme.active() == "light"
    light = (output.SURFACE, theme.mode_colour("frontline"), common.ordered_colours(3))
    with theme.use("dark"):
        assert theme.suffix() == "_dark"
        dark = (output.SURFACE, theme.mode_colour("frontline"), common.ordered_colours(3))
    assert theme.active() == "light" and theme.suffix() == ""
    assert all(a != b for a, b in zip(light, dark, strict=True))
    assert common.MODES == theme.MODE_ORDER == ("frontline", "deathmatch", "elimination")


def _synthetic_chain(root: pathlib.Path) -> list[pathlib.Path]:
    """A BC run and an RL run with the keys the training charts read."""
    runs = []
    start = 0
    for stage, rows in (("bc", 4), ("rl", 4)):
        directory = root / f"chain-{stage}"
        directory.mkdir(parents=True)
        (directory / "run.json").write_text(json.dumps({"name": f"chain-{stage}", "hours": 1.5}))
        lines = []
        for update in range(rows):
            steps = (update + 1) * 1_000_000
            row = {
                "counters/environment_steps": steps,
                "counters/chain_environment_steps": start + steps,
                "train/environment_steps_per_second": 20000.0 + update,
                "train/time/update_seconds": 150.0 if update == 0 else 50.0,
                "frontline/front_moves_per_environment_minute": 0.5,
                "physics/speed_metres_per_second": 250.0,
            }
            for mode, preset in (
                ("frontline", "real_scale_frontline"),
                ("team_deathmatch", "real_scale_deathmatch"),
                ("elimination", "real_scale_elimination"),
            ):
                row[f"combat/deaths_per_ship_minute/{mode}"] = 0.3
                row[f"episode/count/{preset}/self_play"] = 10
                row[f"episode/length_seconds/{preset}/self_play"] = 120.0
                row[f"episode/end/timeout/{preset}/self_play"] = 0.25
                for event, value in (
                    ("boundary_deaths", 1.0),
                    ("combat_deaths", 3.0),
                    ("hits", 40.0),
                    ("shots_fired", 100.0),
                ):
                    row[f"episode/event/{event}/{preset}/self_play"] = value
                if stage == "rl":
                    row[f"rating/live/{preset}"] = 1200.0 + update
                    row[f"rating/live_stderr/{preset}"] = 30.0
                    row[f"rating/average/{preset}"] = 1100.0 + update
                    row[f"rating/average_stderr/{preset}"] = 25.0
            lines.append(json.dumps(row))
        (directory / "metrics.jsonl").write_text("\n".join(lines) + "\n")
        start += rows * 1_000_000
        runs.append(directory)
    return runs


def test_the_training_charts_read_a_chain(tmp_path):
    from bnb.figures.results import training

    chain = _synthetic_chain(tmp_path / "runs")
    written, values = training.draw_behaviour(chain, tmp_path / "out")
    _drawn(written)
    assert values["behaviour_frontline_deaths_to_the_boundary_rl_final"] == pytest.approx(0.25)
    assert values["behaviour_elimination_hit_rate_bc_final"] == pytest.approx(0.4)
    assert values["game_length_deathmatch_training_mean_seconds"] == pytest.approx(120.0)
    assert values["game_length_deathmatch_training_clock_out_share"] == pytest.approx(0.25)
    written, values = training.draw_throughput(
        chain, tmp_path / "out", others={"laptop_4070": {"bc": None, "rl": None}}
    )
    _drawn(written)
    assert values["throughput_rtx_5090_bc_steps_per_second"] == pytest.approx(20002.0)
    assert values["throughput_laptop_4070_rl_steps_per_second"] is None
    assert values["compile_minutes_rtx_5090"] == pytest.approx(2 * 100.0 / 60)
    assert values["chain_hours_rtx_5090_rl"] == 1.5
    assert values["chain_steps_bc"] == 4_000_000
    written, values = training.draw_lanes(chain, tmp_path / "out")
    _drawn(written)
    assert values["lanes_deathmatch_live_final"] == 1203.0


def test_the_evaluation_charts_read_the_queue_records(tmp_path):
    from bnb.figures.results import evaluation

    curves = tmp_path / "curves.csv"
    lines = ["preset,run,kind,step,player,calibrated_elo,stderr,status,games"]
    for preset in (
        "real_scale_frontline_5v5",
        "real_scale_deathmatch_5v5",
        "real_scale_elimination_5v5",
    ):
        for run, steps, elo in (("c-bc", (50, 100), 900.0), ("c-rl", (50, 100), 1400.0)):
            for step in steps:
                for kind in ("exponential_average", "live"):
                    lines.append(
                        f"{preset},{run},{kind},{step * 1_000_000},p,{elo + step},30.0,rated,100"
                    )
    curves.write_text("\n".join(lines) + "\n")
    written, values = evaluation.draw_elo(curves, tmp_path / "out")
    _drawn(written)
    assert values["elo_frontline_bc_final"] == "1000 ± 30"
    assert values["elo_elimination_final"] == "1500 ± 30"
    assert values["elo_elimination_final_steps"] == 200_000_000.0
    assert evaluation.describe(dict(calibrated_elo=1721.4, stderr=None, status="above")) == (
        "> 1721 (a bound)"
    )

    suite = tmp_path / "suite"
    (suite / "batches").mkdir(parents=True)
    batch = dict(
        game_version=24,
        condition="elimination_5v5",
        batch=0,
        games=40,
        seconds=1.0,
        blocks=[dict(wins=30, draws=0, losses=10)],
    )
    other = dict(batch, condition="deathmatch_1v1", blocks=[dict(wins=20, draws=4, losses=16)])
    (suite / "batches" / "w.jsonl").write_text(json.dumps(batch) + "\n" + json.dumps(other) + "\n")
    written, values = evaluation.draw_suite(suite, tmp_path / "out")
    _drawn(written)
    assert values["suite_elimination_5v5_score"] == pytest.approx(0.75)
    assert values["suite_elimination_5v5_low"] < 0.75 < values["suite_elimination_5v5_high"]
    assert values["suite_deathmatch_1v1_score"] == pytest.approx(0.55)
    assert values["champion_game_version"] == 24.0

    champion = tmp_path / "champion.json"
    champion.write_text(
        json.dumps(
            dict(
                champion="c-rl/average/ema_0.999",
                path="/runs/c-rl/policies/step_100/ema_0.999",
                incumbent="c-rl/average/training_exponential_x",
                gap=40.0,
                gap_stderr=10.0,
                ranking=[
                    dict(
                        name="c-rl/average/ema_0.999",
                        gap=40.0,
                        stderr=10.0,
                        presets={"real_scale_frontline_5v5": [30.0, 12.0]},
                    )
                ],
            )
        )
    )
    written, values = evaluation.draw_averaging(champion, tmp_path / "out")
    _drawn(written)
    assert values["averaging_best_gap"] == 40.0
    assert evaluation.champion_values(champion) == dict(
        champion_name="c-rl/average/ema_0.999",
        champion_run="c-rl",
        champion_steps=100.0,
        champion_weights="ema_0.999",
    )


def test_the_champion_values_read_an_average_s_pool_name(tmp_path):
    from bnb.figures.results import evaluation

    champion = tmp_path / "champion.json"
    name = "c-rl/average/training_exponential_decay_0.95/step_98566144"
    champion.write_text(json.dumps(dict(champion=name, path=f"/runs/{name}")))
    held = evaluation.champion_values(champion)
    assert held["champion_weights"] == "training_exponential_decay_0.95"
    assert held["champion_run"] == "c-rl" and held["champion_steps"] == 98566144.0


def test_the_map_values_read_the_pool_tables(tmp_path):
    from bnb.figures.results import evaluation

    table = tmp_path / "scale_0.5" / "table.txt"
    table.parent.mkdir()
    table.write_text(
        "frontline_5v5 (scored by win_loss)\n"
        "  champion                     > 1721.4 (bound)     192 games\n"
        "  scripted                       1000.0 +-   0.0     192 games\n"
        "elimination_5v5 (scored by win_loss)\n"
        "  champion                       1243.1 +-  31.5     192 games\n"
    )
    values = evaluation.map_values(tmp_path)
    assert values["maps_scale_0_5_elimination_elo"] == "1243 ± 32"
    assert values["maps_scale_0_5_frontline_elo"].startswith("> 1721 (a bound")


def synthetic_report(directory: pathlib.Path, far_lineups: bool = True) -> pathlib.Path:
    """A unified evaluator's report.json and spec.toml: two chains' snapshots, champions,
    the anchor and an earlier Imugi in an elimination field (one RL snapshot unplayed, one
    BC snapshot a bound), win-rate cells and each champion's break-even lineups."""
    directory.mkdir(parents=True, exist_ok=True)
    ratings, status, errors = {"scripted": 1000.0, "imugi_v24": 980.0}, {}, {"imugi_v24": 3.0}
    status.update(scripted="rated", imugi_v24="rated")
    players = {"imugi_v24": "imugi_v24"}
    for index, chain in enumerate(("new-s1", "old-s1")):
        for stage, steps in (("bc", 1), ("bc", 2), ("bc", 4), ("rl", 1), ("rl", 2), ("rl", 4)):
            name = f"{chain}/{stage}_{steps * 1048576}"
            ratings[name] = 600.0 + 100 * steps + (400 if stage == "rl" else 0) - 50 * index
            status[name], errors[name] = "rated", 30.0
            players[name] = f"{chain}-{stage}/policies/step_{steps * 1048576}/exponential_average"
        ratings[f"{chain}/champion"], status[f"{chain}/champion"] = 1500.0, "rated"
        errors[f"{chain}/champion"] = 20.0
        players[f"{chain}/champion"] = f"/runs/{chain}-rl/policies/step_{4 * 1048576}/x"
    status["new-s1/rl_2097152"], ratings["new-s1/rl_2097152"] = "unplayed", None
    status["old-s1/bc_1048576"] = "below"
    elo = dict(kind="elo", ratings=ratings, status=status, errors=errors, games=900)
    targets = {"curves/elimination": elo}
    for chain, strength in (("new-s1", 1.4), ("old-s1", 1.1)):
        lineups = {"1v1": [10.0, 40.0], "5v5": [60.0, 64.0]}
        for theirs in (6, 8, 10) if far_lineups else ():
            share = 1 / (1 + math.exp(-3 * math.log(5 * strength / theirs)))
            lineups[f"5v{theirs}"] = [round(share * 64), 64.0]
        targets[f"break_even/real_scale_elimination/{chain}/champion"] = dict(
            kind="break_even",
            preset="real_scale_elimination",
            policy=f"{chain}/champion",
            lineups=lineups,
        )
        targets[f"suite/real_scale_elimination/{chain}/champion_vs_scripted"] = dict(
            kind="win_rate",
            cell=f"real_scale_elimination/{chain}/champion_vs_scripted",
            games=64,
            score=0.9,
            stderr=0.04,
        )
    path = directory / "report.json"
    path.write_text(json.dumps(dict(games=900, targets=targets)))
    spec = "\n".join(f'"{name}" = "{where}"' for name, where in players.items())
    (directory / "spec.toml").write_text(f"[players]\n{spec}\n")
    return path


def test_the_report_elo_puts_both_chains_on_one_scale(tmp_path):
    from bnb.figures.results import report

    path = synthetic_report(tmp_path / "unified")
    written, values = report.draw_elo(path, tmp_path / "out", ("elimination",), "new-s1")
    _drawn(written)
    assert values["elo_new_s1_elimination_bc_final"] == "1000 ± 30"
    assert values["elo_new_s1_elimination_final"] == "1400 ± 30"
    assert values["elo_old_s1_elimination_final"] == "1350 ± 30"
    assert values["elo_elimination_final"] == values["elo_new_s1_elimination_final"]
    assert values["elo_new_s1_elimination_champion_steps"] == 8 * 1048576  # BC 4M + RL 4M
    assert values["elo_new_s1_elimination_unrated"] == 1
    assert values["elo_imugi_v24_elimination"] == "980 ± 3"
    assert values["elo_frontline_final"] is None  # a mode not drawn is pending
    assert values["h2h_new_s1_elimination_5v5_vs_scripted_score"] == 0.9
    target = json.loads(path.read_text())["targets"]["curves/elimination"]
    runs = report.rated_segments(target, "new-s1")
    assert [len(run) for run in runs] == [4, 1]  # the unplayed RL snapshot breaks the line


def test_the_champion_break_even_fits_its_own_lineups_or_waits(tmp_path):
    from bnb.figures.results import report

    path = synthetic_report(tmp_path / "unified")
    written, values = report.draw_break_even(path, tmp_path / "out", 5)
    _drawn(written)
    assert values["break_even_new_s1_elimination_status"] == "crossing"
    assert 1.2 < values["break_even_new_s1_elimination_ratio"] < 1.7
    assert values["break_even_new_s1_elimination_games"] == 64 * 4  # the 1v1 games left out
    waiting = synthetic_report(tmp_path / "waiting", far_lineups=False)
    written, values = report.draw_break_even(waiting, tmp_path / "none", 5)
    assert written == [] and values["break_even_new_s1_elimination_ratio"] is None


def test_the_matchup_values_read_the_table(tmp_path):
    from bnb.figures.results import evaluation

    table = tmp_path / "table.txt"
    table.write_text(
        "A elimination  n=256 W/D/L 95/0/161 score 0.371 ± 0.030\n"
        "B deathmatch   n=256 W/D/L 16/10/230 score 0.082 +- 0.016\n"
    )
    values = evaluation.matchup_values(table, {"A": "run-bc"}, "imugi_v24")
    assert values["matchup_run_bc_elimination_vs_imugi_v24_score"] == 0.371
    assert values["matchup_b_deathmatch_vs_imugi_v24_losses"] == 230
