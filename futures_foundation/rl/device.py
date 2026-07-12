"""Auto device selection for the RL pipeline: CUDA → MPS → CPU.

Used by the context-head precompute (transformer forward) and PPO. Local
by default (the env loop is light — cached embeddings); Colab is at most
an optional one-shot for the precompute.
"""
import torch


def get_device(prefer: str = "auto") -> torch.device:
    """`prefer` in {'auto','cuda','mps','cpu'}. 'auto' picks the best
    available: CUDA, else Apple MPS (Metal), else CPU."""
    if prefer and prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_str(prefer: str = "auto") -> str:
    return get_device(prefer).type
