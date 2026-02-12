"""
Main entry point for the Boost and Broadside application.
"""

import warnings

# Suppress pkg_resources deprecation warning from pygame
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

import hydra
from omegaconf import DictConfig

from boost_and_broadside.modes.collect import collect
from boost_and_broadside.modes.play import play
from boost_and_broadside.env2.collect_massive import collect_massive
from boost_and_broadside.modes.train import train
from boost_and_broadside.modes.train_rl import train_rl
from boost_and_broadside.train.pretrain import pretrain
from boost_and_broadside.eval.eval_world_model import eval_world_model


@hydra.main(version_base=None, config_path="configs", config_name="config")
def my_app(cfg: DictConfig) -> None:
    match cfg.mode:
        case "play":
            play(cfg)
        case "collect":
            collect(cfg)
        case "collect_massive":
            collect_massive(cfg)
        case "train":
            train(cfg)
        case "pretrain":
            pretrain(cfg)
        case "train_rl":
            train_rl(cfg)
        case "eval_wm":
            eval_world_model(cfg)
        case _:
            raise TypeError(
                f"Mode should be one of [play, collect, train, pretrain, eval_wm]. You used: {cfg.mode}"
            )


if __name__ == "__main__":
    my_app()
