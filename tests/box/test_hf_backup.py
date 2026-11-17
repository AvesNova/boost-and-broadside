"""box/hf_backup/hf_backup.py: what a backup skips and how sources are laid out in the repo
(CPU; no network, no token)."""

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "hf_backup", ROOT / "box" / "hf_backup" / "hf_backup.py"
)
backup = importlib.util.module_from_spec(spec)
sys.modules["hf_backup"] = backup
spec.loader.exec_module(backup)


def touch(path: pathlib.Path, size: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def test_plan_skips_caches_dumps_secrets(tmp_path):
    run = tmp_path / "run"
    kept = ["run.json", "checkpoints/step_1/model.safetensors", "wandb/run-1/files/out.log"]
    skipped = [
        "dumps/step_5/batch.pt",
        "sub/dumps/step_5/batch.pt",
        "inductor/xc/a.py",
        "a/__pycache__/m.pyc",
        ".git/HEAD",
        "venv/bin/python",
        "x.pid",
        "x.key",
        "secrets.env",
        "nested/secrets.env",
        "batch.dump",
    ]
    for name in kept + skipped:
        touch(run / name, 3)
    names = sorted(name for name, _ in backup.plan(backup.Source(run, "box/runs/run")))
    assert names == sorted(kept)


def test_expand_one_source_per_folder_and_loose_files(tmp_path):
    touch(tmp_path / "a" / "f.json")
    touch(tmp_path / "cache" / "big.bin")
    touch(tmp_path / "run.log")
    touch(tmp_path / "run.pid")
    touch(tmp_path / "secrets.env")
    sources = backup.expand(tmp_path, "box/runs", skip=("cache",))
    assert [s.prefix for s in sources] == ["box/runs/a", "box/runs"]
    loose = sources[-1]
    assert loose.allow == ["run.log"]
    assert backup.plan(loose) == [("run.log", 1)]  # nothing from the folders below it


def test_read_token_prefers_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / ".config" / "bnb").mkdir(parents=True)
    (tmp_path / ".config" / "bnb" / "hf.key").write_text("from-file\n")
    assert backup.read_token() == "from-file"
    monkeypatch.setenv("HF_TOKEN", "from-environment")
    assert backup.read_token() == "from-environment"
