import hydra
from omegaconf import DictConfig, OmegaConf

from modes.collect import collect
from modes.play import play
from modes.train import train


from src.train.train_world_model import train_world_model
from src.eval.eval_world_model import eval_world_model

@hydra.main(version_base=None, config_path="configs", config_name="config")
def my_app(cfg: DictConfig) -> None:
    match cfg.mode:
        case "play":
            play(cfg)
        case "collect":
            collect(cfg)
        case "train":
            train(cfg)
        case "train_wm":
            train_world_model(cfg)
        case "eval_wm":
            eval_world_model(cfg)
        case _:
            raise TypeError(
                f"Mode should be one of [play, collect, train, train_wm, eval_wm]. You used: {cfg.mode}"
            )


if __name__ == "__main__":
    my_app()
