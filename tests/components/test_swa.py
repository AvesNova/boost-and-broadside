import torch
import torch.nn as nn
from train.swa import SWAModule

def test_swa_averaging():
    """Verify SWA averages weights correctly."""
    # simple model
    model = nn.Linear(2, 1, bias=False)
    
    # 1. Initialize SWA
    # Weights starting at 1.0
    with torch.no_grad():
        model.weight.fill_(1.0)
    
    swa = SWAModule(model)
    
    # Verify SWA model is on CPU
    assert next(swa.averaged_model.parameters()).device.type == 'cpu'
    
    # 2. Update 1 (n=1)
    # Model weights = 2.0
    with torch.no_grad():
        model.weight.fill_(2.0)
        
    swa.update_parameters(model)
    
    # SWA should be (1.0 * 0 + 2.0) / 1 = 2.0 
    # Wait, cumulative average start?
    # Logic in SWAModule:
    # self.n_averaged += 1 // becomes 1
    # alpha = 1 / 1 = 1.0
    # swa = swa * 0 + param * 1 = param
    # So after first update, it becomes the model weights.
    # But wait, we initialized it with deepcopy of start weights (1.0).
    # If we update with 2.0...
    # swa (1.0) * (0.0) + 2.0 * 1.0 = 2.0.
    # Yes.
    
    assert torch.allclose(swa.averaged_model.weight, torch.tensor([[2.0, 2.0]]))
    
    # 3. Update 2 (n=2)
    # Model weights = 4.0
    with torch.no_grad():
        model.weight.fill_(4.0)
        
    swa.update_parameters(model)
    
    # SWA should be (2.0 * 1 + 4.0) / 2 = 3.0
    # Logic:
    # n=2, alpha=0.5
    # swa = 2.0 * 0.5 + 4.0 * 0.5 = 1.0 + 2.0 = 3.0.
    
    assert torch.allclose(swa.averaged_model.weight, torch.tensor([[3.0, 3.0]]))
    
    # 4. Check device persistence
    assert next(swa.averaged_model.parameters()).device.type == 'cpu'
    
    print("SWA Averaging Test Passed!")

if __name__ == "__main__":
    test_swa_averaging()
