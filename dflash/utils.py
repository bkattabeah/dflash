"""Utility functions for dflash."""

import logging
import time
from contextlib import contextmanager
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def get_device(device: Optional[str] = None) -> torch.device:
    """Resolve the best available device or use the specified one."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(dtype: Optional[str] = None) -> torch.dtype:
    """Resolve torch dtype from string or return a sensible default."""
    _map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype is None:
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if dtype not in _map:
        raise ValueError(f"Unsupported dtype '{dtype}'. Choose from {list(_map)}.")
    return _map[dtype]


@contextmanager
def timer(label: str = "", verbose: bool = True):
    """Simple wall-clock timer context manager.

    Example::

        with timer("model load"):
            model = load_model(...)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if verbose:
            tag = f"[{label}] " if label else ""
            logger.info("%s%.3f s", tag, elapsed)


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """Return the total number of parameters in a model."""
    params = (
        model.parameters()
        if not trainable_only
        else filter(lambda p: p.requires_grad, model.parameters())
    )
    return sum(p.numel() for p in params)


def pretty_size(num: int, suffix: str = "") -> str:
    """Format a large integer with K/M/B suffix for readability."""
    for unit in ("", "K", "M", "B"):
        if abs(num) < 1_000:
            return f"{num:.1f}{unit}{suffix}"
        num /= 1_000  # type: ignore[assignment]
    return f"{num:.1f}T{suffix}"


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility across torch (and numpy if available)."""
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    logger.debug("Global seed set to %d", seed)
