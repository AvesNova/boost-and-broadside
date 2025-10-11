#!/usr/bin/env python3
"""
Test script to verify Hydra integration works correctly
"""

import subprocess
import sys
import os


def run_command(cmd):
    """Run a command and return the result"""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        print(f"Exit code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Command timed out")
        return False
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def test_hydra_integration():
    """Test that Hydra integration works correctly"""
    print("=" * 60)
    print("TESTING HYDRA INTEGRATION")
    print("=" * 60)

    # Change to the project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Test commands
    test_commands = [
        # Test help (basic functionality)
        "python main.py --help",
        # Test training commands
        "python main.py train bc --help",
        "python main.py train rl --help",
        "python main.py train full --help",
        # Test data collection commands
        "python main.py collect bc --help",
        "python main.py collect selfplay --help",
        # Test play commands
        "python main.py play human --help",
        # Test replay commands
        "python main.py replay episode --help",
        "python main.py replay browse --help",
        # Test evaluation commands
        "python main.py evaluate model --help",
        # Test Hydra override syntax (dry run)
        "python main.py train bc model.transformer.embed_dim=128 --help",
    ]

    results = []
    for cmd in test_commands:
        print("\n" + "-" * 40)
        success = run_command(cmd)
        results.append((cmd, success))
        print(f"Result: {'SUCCESS' if success else 'FAILED'}")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("All tests passed! Hydra integration is working correctly.")
        return True
    else:
        print("Some tests failed. Please check the output above.")
        return False


if __name__ == "__main__":
    success = test_hydra_integration()
    sys.exit(0 if success else 1)
