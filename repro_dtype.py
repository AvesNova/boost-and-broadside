import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Mismatch here if x is bf16 and weight is f32?
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dim = 128
    # Initialize in BF16 to match our new setup.py logic
    model = RMSNorm(dim).to(device, dtype=torch.bfloat16)
    
    # Input in bf16
    x = torch.randn(1, 10, dim, device=device, dtype=torch.bfloat16)
    
    print("Testing eager with autocast(bf16)...")
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        try:
            out = model(x)
            print("Eager success")
        except Exception as e:
            print(f"Eager failed: {e}")

    if hasattr(torch, "compile") and device.type == "cuda":
        print("Testing compiled with autocast(bf16)...")
        compiled_model = torch.compile(model)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            try:
                out = compiled_model(x)
                print("Compile success")
            except Exception as e:
                print(f"Compile failed: {e}")

if __name__ == "__main__":
    test()
