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
