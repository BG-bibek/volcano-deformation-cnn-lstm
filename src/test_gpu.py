import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
x = torch.randn(1000,1000, device=device)
print("Using:", device)
print(x.shape)