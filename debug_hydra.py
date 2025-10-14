#!/usr/bin/env python3
"""
Debug script to test Hydra configuration loading
"""

import os
import sys
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


def test_config_loading():
    """Test loading the train/full configuration"""

    # Get the original working directory
    original_cwd = os.getcwd()
    config_dir = os.path.join(original_cwd, "src/config")

    print(f"Config directory: {config_dir}")

    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    try:
        # Initialize Hydra with the config directory
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            # Try to compose the configuration
            print("Attempting to load train/full configuration...")
            cfg = compose(config_name="train/full", return_hydra_config=False)

            print("Configuration loaded successfully!")
            print(OmegaConf.to_yaml(cfg))

    except Exception as e:
        print(f"Error loading configuration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_config_loading()
