import torch
import pytest
from omegaconf import OmegaConf
from boost_and_broadside.train.batch_processor import BatchProcessor
from boost_and_broadside.train.world_model.validator import Validator

def test_batch_processor():
    B = 2
    T = 4
    N = 3
    
    states = torch.zeros(B, T, N, 27)
    actions = torch.arange(B * T * N * 3).view(B, T, N, 3).float()
    
    target_actions = torch.roll(actions, shifts=-1, dims=1)
    
    batch_data = {
        "states": states,
        "actions": actions,
        "target_actions": target_actions,
        "team_ids": torch.zeros(B, T, N),
        "seq_idx": torch.zeros(B, T),
        "loss_mask": torch.ones(B, T),
        "pos": torch.zeros(B, T, N, 2),
    }
    
    config = OmegaConf.create({
        "environment": {"world_size": (1000, 1000)}
    })
    device = torch.device('cpu')
    
    result = BatchProcessor.process_batch(batch_data, config, device, use_amp=False)
    
    assert result["inputs"]["state"].shape[1] == 3
    
    expected_target_actions = target_actions[:, :-1]
    assert torch.allclose(result["inputs"]["target_actions"], expected_target_actions)
    assert torch.allclose(result["targets"]["target_actions"], expected_target_actions)

def test_validator():
    class DummyModel:
        def eval(self): pass
        def __call__(self, **kwargs):
            return (torch.zeros(2,3,4), None, None, None, None)
        def get_loss(self, **kwargs):
            return {"loss": torch.tensor(0.0)}

    model = DummyModel()
    device = torch.device('cpu')
    
    from types import SimpleNamespace
    config = SimpleNamespace(
        environment=SimpleNamespace(world_size=(1000, 1000)),
        model=dict(), 
        train=dict(amp=False)
    )
    
    validator = Validator(model, device, config)

    B = 2
    T = 4
    N = 3
    states = torch.zeros(B, T, N, 27)
    actions = torch.arange(B * T * N * 3).view(B, T, N, 3).float()
    target_actions = torch.roll(actions, shifts=-1, dims=1)
    
    batch_data = {
        "states": states,
        "actions": actions,
        "target_actions": target_actions,
        "team_ids": torch.zeros(B, T, N),
        "seq_idx": torch.zeros(B, T),
        "loss_mask": torch.ones(B, T),
        "pos": torch.zeros(B, T, N, 2),
        "rewards": torch.zeros(B, T),
        "returns": torch.zeros(B, T)
    }

    loaders = [[batch_data]]
    validator.validate_validation_set(loaders, max_batches=1)
