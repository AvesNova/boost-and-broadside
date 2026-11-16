"""box/train_watchdog.py: a real chain log (bc-v20 then rl-v20 on the 5090, 2026-11-13) gives
its milestones and no watchdog alert; synthetic bad logs fire each alert (CPU, standard
library only)."""

import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURES = pathlib.Path(__file__).parent / "fixtures"
spec = importlib.util.spec_from_file_location("train_watchdog", ROOT / "box" / "train_watchdog.py")
watchdog = importlib.util.module_from_spec(spec)
sys.modules["train_watchdog"] = watchdog  # dataclasses look their module up
spec.loader.exec_module(watchdog)

T0 = 1_800_000_000.0
BC_LINE = (
    "update {n:4d}  steps {steps:>11,}  rollout   3064 ms  learn  {learn:5d} ms  bc kl {kl}  "
    "teacher 0.850  student 3.047  scores: learner 0.500 scripted 0.058{tail}\n"
)
RL_LINE = (
    "update {n:4d}  rollout   6012 ms  learn  {learn:5d} ms  entropy 1.35  kl {kl}  clip 0.031  "
    "drift 0.256  skipped 0  nonfinite beliefs {beliefs}  reward/s: frontline/league -0.025{tail}\n"
)


def bc(n, learn=44424, kl="1.5926", tail=""):
    return BC_LINE.format(n=n, steps=n * 1_048_576, learn=learn, kl=kl, tail=tail)


def rl(n, learn=38141, kl="+0.0033", beliefs=0, tail=""):
    return RL_LINE.format(n=n, learn=learn, kl=kl, beliefs=beliefs, tail=tail)


def snapshot(now, log="", alive=True, exit=None, gpu=95, disk_gb=150.0, ok=True):
    return dict(
        ok=ok,
        now=now,
        log=log,
        pid="4242",
        alive=alive,
        exit=exit,
        gpu=gpu,
        disk_gb=disk_gb,
        error=None if ok else "timed out",
    )


def alerts(events):
    return [text for level, text in events if level == "ALERT"]


def milestones(events):
    return [text for level, text in events if level == "MILESTONE"]


def steady_bc(watch, updates=5):
    """A BC stage through its first `updates` updates, a minute apart; returns the time."""
    now = T0
    watch.poll(
        snapshot(now, "chain dry in /workspace/runs/dry\nrun dry-bc in /workspace/runs/dry-bc\n")
    )
    for n in range(1, updates + 1):
        now += 60
        watch.poll(snapshot(now, bc(n)))
    return now


def test_real_chain_replay_reports_milestones_and_no_watchdog_alert(capsys):
    assert watchdog.main(["--replay", str(FIXTURES / "chain-v20.log")]) == 0
    lines = capsys.readouterr().out.splitlines()
    found = [line.split(" KST ", 1)[1] for line in lines if " KST " in line]
    assert [line for line in found if line.startswith("MILESTONE")] == [
        "MILESTONE BC steady: bc-v20 update 4, 21,984 steps/s (expected 21,900)",
        "MILESTONE BC done: bc-v20 finished at step 84,934,656 (81 updates logged)",
        "MILESTONE RL steady: rl-v20 update 4, 23,701 steps/s (expected 23,000)",
        "MILESTONE RL done: rl-v20 finished at step 149,946,368 (143 updates logged)",
        "MILESTONE chain finished (exit code 0)",
    ]
    # the only alerts are the trainer's own ALERT rules, each once: no stall, sps, error,
    # nonfinite, gpu or disk alert from the watchdog on a healthy run
    assert [line for line in found if line.startswith("ALERT")] == [
        "ALERT trainer gauge_runaway: first in rl-v20 at RL update 3",
        "ALERT trainer rating_drop: first in rl-v20 at RL update 48",
    ]


def test_quiet_alerts_silence_named_trainer_rules(capsys):
    watchdog.main(
        [
            "--replay",
            str(FIXTURES / "chain-v20.log"),
            "--quiet-alerts",
            "gauge_runaway",
            "rating_drop",
        ]
    )
    assert " ALERT " not in capsys.readouterr().out


def test_milestones_and_alerts_are_not_repeated_after_a_rearm(tmp_path):
    watch = watchdog.Watch(watchdog.Settings())
    now = steady_bc(watch)
    path = tmp_path / "state.json"
    watchdog.save_state(path, watch.state)
    again = watchdog.Watch(watchdog.Settings(), watchdog.load_state(path))
    assert again.poll(snapshot(now + 60, bc(6))) == []
    assert again.state["milestones"] == ["steady:dry-bc"]


def test_bc_steady_milestone_at_the_fourth_update():
    watch = watchdog.Watch(watchdog.Settings())
    now = T0
    watch.poll(snapshot(now, "run dry-bc in /workspace/runs/dry-bc\n"))
    found = []
    for n in range(1, 6):
        now += 60
        found.append(milestones(watch.poll(snapshot(now, bc(n)))))
    assert found[:3] == [[], [], []]
    assert found[3] and found[3][0].startswith("BC steady: dry-bc update 4")
    assert found[4] == []


@pytest.mark.parametrize(
    "line",
    [bc(6, kl="nan"), bc(6, kl="inf"), rl(6, kl="+nan"), rl(6, beliefs=3)],
    ids=["bc-nan", "bc-inf", "rl-nan", "nonfinite-beliefs"],
)
def test_nonfinite_losses_fire(line):
    watch = watchdog.Watch(watchdog.Settings())
    now = steady_bc(watch)
    assert any(
        text.startswith("nonfinite:") for text in alerts(watch.poll(snapshot(now + 60, line)))
    )


@pytest.mark.parametrize(
    "line",
    [
        "Traceback (most recent call last):\n",
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n",
        "RuntimeError: CUDA error: an illegal memory access was encountered\n",
        "Killed\n",
        "run dry-bc failed at step 5242880\n",
        "run dry-bc stopped at step 5242880\n",
    ],
    ids=["traceback", "oom", "cuda-error", "killed", "failed", "stopped"],
)
def test_errors_in_the_log_fire(line):
    watch = watchdog.Watch(watchdog.Settings())
    now = steady_bc(watch)
    assert any(text.startswith("error:") for text in alerts(watch.poll(snapshot(now + 60, line))))


def test_warnings_in_the_log_do_not_fire():
    text = (FIXTURES / "chain-v20.log").read_text()
    warnings = [line for line in text.splitlines() if "Warning" in line]
    assert warnings  # the real log's torch warnings
    assert not any(watchdog.ERROR.search(line) for line in warnings)


def test_trainer_alert_rules_fire_once_per_stage_and_warnings_only_note():
    watch = watchdog.Watch(watchdog.Settings())
    now = steady_bc(watch)
    events = watch.poll(
        snapshot(now + 60, bc(6, tail="  alerts: skipped_step sigma_miscalibrated"))
    )
    assert alerts(events) == ["trainer skipped_step: first in dry-bc at BC update 6"]
    assert [level for level, _ in events] == ["ALERT", "NOTE"]
    assert watch.poll(snapshot(now + 120, bc(7, tail="  alerts: skipped_step"))) == []


def test_a_stalled_step_count_fires_after_the_limit_and_repeats_only_later():
    watch = watchdog.Watch(watchdog.Settings(stall_minutes=10, repeat_minutes=30))
    now = steady_bc(watch)
    assert alerts(watch.poll(snapshot(now + 9 * 60))) == []
    fired = alerts(watch.poll(snapshot(now + 11 * 60)))
    assert fired and fired[0].startswith("stall: no update line for 11 min")
    assert alerts(watch.poll(snapshot(now + 13 * 60))) == []  # reported; repeats after 30 min
    assert alerts(watch.poll(snapshot(now + 42 * 60)))
    assert alerts(watch.poll(snapshot(now + 43 * 60, bc(6)))) == []  # progress clears it


def test_the_first_compile_has_its_own_limit():
    watch = watchdog.Watch(watchdog.Settings(compile_minutes=80))
    watch.poll(
        snapshot(T0, "chain dry in /workspace/runs/dry\nrun dry-bc in /workspace/runs/dry-bc\n")
    )
    assert alerts(watch.poll(snapshot(T0 + 79 * 60))) == []
    fired = alerts(watch.poll(snapshot(T0 + 81 * 60)))
    assert fired and "no first update 81 min after the start of dry-bc" in fired[0]


def test_the_rl_compile_after_the_handover_has_the_compile_limit_not_the_stall_one():
    watch = watchdog.Watch(watchdog.Settings())
    now = steady_bc(watch)
    events = watch.poll(
        snapshot(now + 60, "run dry-bc finished at step 5242880\nhandover at step 5,242,880\n")
    )
    assert milestones(events) == ["BC done: dry-bc finished at step 5,242,880 (5 updates logged)"]
    now += 120
    watch.poll(snapshot(now, "run dry-rl in /workspace/runs/dry-rl\n"))
    assert alerts(watch.poll(snapshot(now + 40 * 60, gpu=0))) == []  # compiling: GPU idle is fine
    assert alerts(watch.poll(snapshot(now + 42 * 60, gpu=0))) == []
    assert alerts(watch.poll(snapshot(now + 44 * 60, gpu=0))) == []
    assert alerts(watch.poll(snapshot(now + 81 * 60)))


def test_low_throughput_fires_over_three_steady_updates():
    watch = watchdog.Watch(watchdog.Settings(bc_sps=21_900, sps_fraction=0.6))
    now = T0
    watch.poll(snapshot(now, "run dry-bc in /workspace/runs/dry-bc\n"))
    fired = []
    for n in range(1, 7):  # 1,048,576 steps in 3 + 90 s: 11,300 steps/s
        now += 95
        fired += alerts(watch.poll(snapshot(now, bc(n, learn=90_000))))
    assert len(fired) == 1 and fired[0].startswith("sps: BC update 4: median of the last 3")


def test_one_slow_update_does_not_fire():
    watch = watchdog.Watch(watchdog.Settings())
    now = steady_bc(watch, updates=6)
    assert alerts(watch.poll(snapshot(now + 120, bc(7, learn=120_000)))) == []


def test_a_dead_pid_fires_and_a_finished_chain_is_a_milestone():
    watch = watchdog.Watch(watchdog.Settings())
    now = steady_bc(watch)
    fired = alerts(watch.poll(snapshot(now + 60, alive=False)))
    assert fired and fired[0].startswith("dead: trainer pid 4242 is gone (exit code unknown")
    # BC done and exit 0 is still dead: the chain should have gone on to RL
    watch = watchdog.Watch(watchdog.Settings())
    now = steady_bc(watch)
    watch.poll(snapshot(now + 60, "run dry-bc finished at step 5242880\n"))
    assert alerts(watch.poll(snapshot(now + 120, alive=False, exit="0")))
    # RL finished and exit 0: the chain's milestone
    watch.poll(snapshot(now + 180, "run dry-rl in /x\n" + "".join(rl(n) for n in range(1, 6))))
    events = watch.poll(
        snapshot(now + 240, "run dry-rl finished at step 10485760\n", alive=False, exit="0")
    )
    assert milestones(events) == [
        "RL done: dry-rl finished at step 10,485,760 (5 updates logged)",
        "chain finished (exit code 0)",
    ]
    assert alerts(events) == []


def test_gpu_idle_fires_after_three_polls_in_a_steady_stage():
    watch = watchdog.Watch(watchdog.Settings())
    now = steady_bc(watch)
    assert alerts(watch.poll(snapshot(now + 60, bc(6), gpu=0))) == []
    assert alerts(watch.poll(snapshot(now + 120, bc(7), gpu=0))) == []
    assert alerts(watch.poll(snapshot(now + 180, bc(8), gpu=0))) == [
        "gpu idle: utilisation 0 on 3 polls"
    ]


def test_low_disk_fires():
    watch = watchdog.Watch(watchdog.Settings(disk_min_gb=20))
    now = steady_bc(watch)
    assert alerts(watch.poll(snapshot(now + 60, bc(6), disk_gb=15.2))) == [
        "disk: 15.2 GB free on the volume"
    ]


def test_one_failed_ssh_poll_is_tolerated_four_fire():
    watch = watchdog.Watch(watchdog.Settings(ssh_failures=4))
    now = steady_bc(watch)
    for k in range(1, 4):
        events = watch.poll(snapshot(now + 60 * k, ok=False))
        assert alerts(events) == [] and events[0][0] == "NOTE"
    fired = alerts(watch.poll(snapshot(now + 240, ok=False)))
    assert fired and fired[0].startswith("ssh: 4 failed polls in a row")
    assert alerts(watch.poll(snapshot(now + 300, bc(6)))) == []  # back: counter reset


def test_the_box_reply_checks_the_pid_by_its_cmdline():
    def reply(cmdline, exit="none"):
        return (
            "Welcome to vast.ai\n@@PID 4242\n"
            f"@@CMDLINE {cmdline}\n@@EXIT {exit}\n@@GPU 97\n@@DISK 150000000000\n"
            "@@SIZE 30\n@@FROM 0\n@@LOG\nrun dry-bc in /x\nupdate    1  part"
        ).encode()

    fields = watchdog.parse_remote(
        reply("/workspace/venv/bin/python /workspace/venv/bin/bnb train --name dry"), "dry"
    )
    assert (
        fields["alive"]
        and fields["gpu"] == 97
        and fields["disk_gb"] == 150.0
        and fields["exit"] is None
    )
    # a reused pid (another program) is not the trainer
    assert not watchdog.parse_remote(reply("/usr/bin/python3 /usr/local/bin/tensorboard"), "dry")[
        "alive"
    ]
    assert not watchdog.parse_remote(reply("none", exit="0"), "dry")["alive"]
    text, used = watchdog.complete_lines(fields["raw_log"])
    assert text == "run dry-bc in /x\n" and used == len(
        b"run dry-bc in /x\n"
    )  # the partial line waits


def test_the_replay_state_is_json():
    watch = watchdog.Watch(watchdog.Settings())
    steady_bc(watch)
    assert json.loads(json.dumps(watch.state)) == watch.state
