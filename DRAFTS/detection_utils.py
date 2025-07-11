from __future__ import annotations

from typing import Tuple, List

import numpy as np
import torch

from . import config

__all__ = [
    "detect_candidates",
    "prep_patch",
    "classify_patch",
]


def detect_candidates(model: torch.nn.Module, img: np.ndarray) -> Tuple[List, List | None]:
    """Run the detection network on ``img`` and return confidences and boxes."""
    from ObjectDet.centernet_utils import get_res

    with torch.no_grad():
        hm, wh, offset = model(
            torch.from_numpy(img).to(config.DEVICE).float().unsqueeze(0)
        )
    conf, boxes = get_res(hm, wh, offset, confidence=config.DET_PROB)

    if boxes is None:
        return [], None

    if isinstance(conf, np.ndarray):
        conf = conf.tolist()
    if isinstance(boxes, np.ndarray):
        boxes = boxes.tolist()

    return conf, boxes


def prep_patch(patch: np.ndarray) -> np.ndarray:
    """Normalize ``patch`` before classification."""
    patch = patch.copy()
    patch += 1
    patch /= np.mean(patch, axis=0)
    vmin, vmax = np.nanpercentile(patch, [5, 95])
    patch = np.clip(patch, vmin, vmax)
    patch = (patch - patch.min()) / (patch.max() - patch.min())
    return patch


def classify_patch(model: torch.nn.Module, patch: np.ndarray) -> Tuple[float, np.ndarray]:
    """Return probability from classifier for ``patch`` along with processed patch."""
    proc = prep_patch(patch)
    tensor = torch.from_numpy(proc[None, None, :, :]).float().to(config.DEVICE)
    with torch.no_grad():
        out = model(tensor)
        prob = out.softmax(dim=1)[0, 1].item()
    return prob, proc
