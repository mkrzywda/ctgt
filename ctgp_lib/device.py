"""CUDA / CPU device selection (cuda:0, cuda:1, cpu)."""

from __future__ import annotations

import torch

_device: torch.device | None = None


def set_device(device: str = "cuda:0") -> torch.device:
    """Configure the compute device used by all CTGP modules."""
    global _device
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device!r} but CUDA is not available.")
        _device = torch.device(device)
    else:
        _device = torch.device(device)
    return _device


def get_device() -> torch.device:
    if _device is None:
        default = "cuda:0" if torch.cuda.is_available() else "cpu"
        return set_device(default)
    return _device
