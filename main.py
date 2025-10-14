import hydra
from omegaconf import DictConfig, OmegaConf

from modes.collect import collect
from modes.play import play
from modes.train import train


@hydra.main(version_base=None, config_path="configs", config_name="config")
def my_app(cfg: DictConfig) -> None:
    match cfg.mode:
        case "play":
            play(cfg)
        case "collect":
            collect(cfg)
        case "train":
            train(cfg)
        case _:
            raise TypeError(
                f"Mode should be one of [play, collect, train]. You used: {cfg.mode}"
            )


if __name__ == "__main__":
    my_app()
