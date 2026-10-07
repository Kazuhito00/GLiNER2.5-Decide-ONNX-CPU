"""Torch-free runtime for the GLiNER2.5-Decide classification (decide) path."""

from .protocol import DecideEncoder, normalize_text, split_words
from .runtime import VARIANTS, Decider

__all__ = ["Decider", "DecideEncoder", "VARIANTS", "normalize_text", "split_words"]
__version__ = "0.1.0"
