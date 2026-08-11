"""Device selection helpers for CUDA, Apple MPS, and CPU."""

import torch


def get_device(preference: str | torch.device | None = None) -> torch.device:
    """Return the requested device, or the fastest available device."""
    if isinstance(preference, torch.device):
        return preference

    requested = (preference or "auto").lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    if requested not in {"cuda", "mps", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, mps, cpu")

    return torch.device(requested)
