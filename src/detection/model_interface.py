"""Model interface for FRB detection and classification - handles neural network inference."""
# Standard library imports
import logging

# Third-party imports
import numpy as np

# Local imports
from ..config import config

# Model imports
try:
    from ..models.ObjectDet.centernet_utils import get_res
except ImportError:
    get_res = None

# Optional third-party imports
try:
    import torch
except ImportError:
    torch = None

# Setup logger
logger = logging.getLogger(__name__)

def detect(model, img_tensor: np.ndarray):
    """Run the detection model and return confidences and boxes."""
    if get_res is None:
        logger.error("get_res no está disponible. Verifique que models.ObjectDet.centernet_utils esté instalado.")
        return [], []
    
    try:
        with torch.no_grad():
            hm, wh, offset = model(
                torch.from_numpy(img_tensor)
                .to(config.DEVICE)
                .float()
                .unsqueeze(0)
            )
        top_conf, top_boxes = get_res(hm, wh, offset, confidence=config.DET_PROB)
        # Normalizar salidas a listas Python siempre
        if top_boxes is None:
            return [], []
        try:
            if torch.is_tensor(top_conf):
                top_conf = top_conf.detach().cpu().numpy()
            if torch.is_tensor(top_boxes):
                top_boxes = top_boxes.detach().cpu().numpy()
        except Exception:
            pass
        if isinstance(top_conf, np.ndarray):
            top_conf = top_conf.tolist()
        if isinstance(top_boxes, np.ndarray):
            top_boxes = top_boxes.tolist()
        return top_conf, top_boxes
    except Exception as e:
        logger.error(f"Error en detect: {e}")
        return [], []

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
    # Fallback robusto: permitir que el modelo sea None (p. ej., pipeline alta frecuencia)
    try:
        if model is None or torch is None:
            from ..analysis.snr_utils import compute_snr_profile
            snr_profile, _, _ = compute_snr_profile(proc)
            snr_peak = float(np.max(snr_profile)) if snr_profile is not None and snr_profile.size > 0 else 0.0
            thresh = float(getattr(config, 'SNR_THRESH', 3.0))
            # Mapear SNR a probabilidad con sigmoide centrada en el umbral
            prob = 1.0 / (1.0 + np.exp(-((snr_peak - thresh) / 2.0)))
            return float(prob), proc
        tensor = torch.from_numpy(proc[None, None, :, :]).float().to(config.DEVICE)
        with torch.no_grad():
            out = model(tensor)
            prob = out.softmax(dim=1)[0, 1].item()
        return prob, proc
    except Exception as e:
        logger.error(f"Error en classify_patch (fallback): {e}")
        return 0.0, proc
