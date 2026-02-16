from __future__ import annotations
from typing import Dict, Any, Optional
import torch


def save_checkpoint(path: str, obj: Dict[str, Any]) -> None:
    torch.save(obj, path)


def load_checkpoint(path: str, map_location=None) -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)