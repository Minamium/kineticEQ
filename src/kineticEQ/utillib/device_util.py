import torch

def resolve_device(device: str) -> str:
    d = device.strip().lower()
    if d == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' was requested, but torch.cuda.is_available() is False. "
                               "Install a CUDA-enabled PyTorch on an NVIDIA machine, or use 'cpu'/'mps'.")
        return "cuda"
    if d == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("device='mps' was requested, but MPS is not available on this system.")
        return "mps"
    return "cpu"
