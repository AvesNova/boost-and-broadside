import pytest
from unittest.mock import patch
from pathlib import Path
from omegaconf import OmegaConf
from modes.train import train


@pytest.fixture
def mock_pipeline_components():
    with (
        patch("modes.train.collect") as mock_collect,
        patch("modes.train.train_bc") as mock_train_bc,
        patch("modes.train.train_rl") as mock_train_rl,
    ):

        # Setup default returns
        mock_collect.return_value = Path("mock/data/path.pkl")
        mock_train_bc.return_value = Path("mock/model/path.pth")

        yield mock_collect, mock_train_bc, mock_train_rl


def test_full_pipeline_execution(mock_pipeline_components):
    mock_collect, mock_train_bc, mock_train_rl = mock_pipeline_components

    cfg = OmegaConf.create(
        {
            "train": {
                "run_collect": True,
                "run_bc": True,
                "run_rl": True,
                "use_bc": True,  # Needed if accessed anywhere
                "use_rl": True,
                "bc_data_path": "initial/path.pkl",
                "rl": {"pretrained_model_path": None},
            },
            "collect": {"num_workers": 1},
        }
    )

    train(cfg)

    # Verify Collect called
    mock_collect.assert_called_once()

    # Verify BC called with updated data path
    # Note: cfg is mutable, so we check if it was updated
    assert (
        cfg.train.bc_data_path == "mock\\data\\path.pkl"
        or cfg.train.bc_data_path == "mock/data/path.pkl"
    )
    mock_train_bc.assert_called_once()

    # Verify RL called with BC model path
    mock_train_rl.assert_called_once()
    call_args = mock_train_rl.call_args
    assert call_args.kwargs["pretrained_model_path"] == Path("mock/model/path.pth")


def test_pipeline_skip_collect(mock_pipeline_components):
    mock_collect, mock_train_bc, mock_train_rl = mock_pipeline_components

    cfg = OmegaConf.create(
        {
            "train": {
                "run_collect": False,
                "run_bc": True,
                "run_rl": True,
                "use_bc": True,
                "use_rl": True,
                "bc_data_path": "existing/data.pkl",
                "rl": {"pretrained_model_path": None},
            },
            "collect": {},
        }
    )

    train(cfg)

    mock_collect.assert_not_called()
    mock_train_bc.assert_called_once()
    mock_train_rl.assert_called_once()

    # Verify RL uses BC model
    call_args = mock_train_rl.call_args
    assert call_args.kwargs["pretrained_model_path"] == Path("mock/model/path.pth")


def test_pipeline_rl_only_pretrained(mock_pipeline_components):
    mock_collect, mock_train_bc, mock_train_rl = mock_pipeline_components

    cfg = OmegaConf.create(
        {
            "train": {
                "run_collect": False,
                "run_bc": False,
                "run_rl": True,
                "use_bc": False,
                "use_rl": True,
                "bc_data_path": "existing/data.pkl",
                "rl": {"pretrained_model_path": "pretrained/model.pth"},
            },
            "collect": {},
        }
    )

    train(cfg)

    mock_collect.assert_not_called()
    mock_train_bc.assert_not_called()
    mock_train_rl.assert_called_once()

    # Verify RL uses configured pretrained model
    call_args = mock_train_rl.call_args
    assert call_args.kwargs["pretrained_model_path"] == Path("pretrained/model.pth")
