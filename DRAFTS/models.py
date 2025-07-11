from __future__ import annotations

import torch

from . import config

__all__ = [
    "load_detection_model",
    "load_classification_model",
]


def load_detection_model() -> torch.nn.Module:
    """Load the CenterNet detection model."""
    from ObjectDet.centernet_model import centernet

    model = centernet(model_name=config.MODEL_NAME).to(config.DEVICE)
    state = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def load_classification_model() -> torch.nn.Module:
    """Load the binary classification model."""
    from BinaryClass.binary_model import BinaryNet

    model = BinaryNet(config.CLASS_MODEL_NAME, num_classes=2).to(config.DEVICE)
    state = torch.load(config.CLASS_MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model
