
import torch
from omegaconf import OmegaConf
from boost_and_broadside.models.yemong.scaffolds import YemongFull

def test_instantiation():
    # Simulate config
    cfg = OmegaConf.create({
        "d_model": 128,
        "n_layers": 1,
        "num_reward_components": 4,
        "target_dim": 7,
        "context_len": 32,
        "loss": {
            "type": "composite",
            "weights": {"state": 1.0}
        }
    })
    
    print("Config:", cfg)
    
    model = YemongFull(config=cfg)
    print(f"Model instantiated. TeamEvaluator reward head: {model.team_evaluator.reward_head}")
    
    # improved check
    last_layer = model.team_evaluator.reward_head[-1]
    print(f"Last layer out_features: {last_layer.out_features}")
    
    assert last_layer.out_features == 4, f"Expected 4, got {last_layer.out_features}"

if __name__ == "__main__":
    test_instantiation()
