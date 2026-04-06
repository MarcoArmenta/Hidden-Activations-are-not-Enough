"""Atomic file write utilities: write to .tmp then rename.

Prevents corrupt files from interrupted writes (OOM kill, SIGKILL, wall-time).
"""

import os
import json
import torch
from pathlib import Path


def atomic_torch_save(obj, path):
    """Save a PyTorch object atomically via tmp+rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    tmp.rename(path)


def atomic_json_dump(data, path, indent=2):
    """Write JSON atomically via tmp+rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=indent)
    tmp.rename(path)
