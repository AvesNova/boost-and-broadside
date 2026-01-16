import subprocess
import sys
from pathlib import Path


def test_train_pipeline_integration():
    """
    Test the full training pipeline integration using config_test.
    Equivalent to:
    uv run main.py mode=train train.run_collect=true train.run_world_model=true --config-name config_test
    """
    # Get the project root directory
    # components of path: [root, tests, pipelines, file.py]
    project_root = Path(__file__).resolve().parents[2]
    main_py = project_root / "main.py"

    assert main_py.exists(), f"Could not find main.py at {main_py}"

    # Construct the command
    # We use sys.executable to use the current python environment (which should be the uv venv)
    command = [
        sys.executable,
        str(main_py),
        "mode=train",
        "train.run_collect=true",
        "train.run_world_model=true",
        "--config-name",
        "config_test",
    ]

    # Run the command
    result = subprocess.run(
        command, cwd=str(project_root), capture_output=True, text=True
    )

    # If it failed, print the output to help debugging
    if result.returncode != 0:
        print("\n=== COMMAND STDOUT ===\n", result.stdout)
        print("\n=== COMMAND STDERR ===\n", result.stderr)

    assert result.returncode == 0, (
        f"Integration test failed with return code {result.returncode}"
    )
