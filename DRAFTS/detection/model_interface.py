"""Model interface for FRB detection and classification - handles neural network inference."""
import numpy as np
import logging
from .. import config

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)

def detect(model, img_tensor: np.ndarray):
    """Run the detection model and return confidences and boxes."""
    from training.ObjectDet.centernet_utils import get_res
    try:
        with torch.no_grad():
            hm, wh, offset = model(
                torch.from_numpy(img_tensor)
                .to(config.DEVICE)
                .float()
                .unsqueeze(0)
            )
        top_conf, top_boxes = get_res(hm, wh, offset, confidence=config.DET_PROB)
        if top_boxes is None:
            return [], None
        if isinstance(top_conf, np.ndarray):
            top_conf = top_conf.tolist()
        if isinstance(top_boxes, np.ndarray):
            top_boxes = top_boxes.tolist()
        return top_conf, top_boxes
    except Exception as e:
        logger.error(f"Error en detect: {e}")
        return [], None

def prep_patch(patch: np.ndarray) -> np.ndarray:
    """Normalize patch for classification."""
    patch = patch.copy()
    patch += 1
    patch /= np.mean(patch, axis=0)
    vmin, vmax = np.nanpercentile(patch, [5, 95])
    patch = np.clip(patch, vmin, vmax)
    patch = (patch - patch.min()) / (patch.max() - patch.min())
    return patch

def classify_patch(model, patch: np.ndarray):
    """Return probability from binary model for patch along with the processed patch."""
    proc = prep_patch(patch)
    tensor = torch.from_numpy(proc[None, None, :, :]).float().to(config.DEVICE)
    with torch.no_grad():
        out = model(tensor)
        prob = out.softmax(dim=1)[0, 1].item()
    return prob, proc
