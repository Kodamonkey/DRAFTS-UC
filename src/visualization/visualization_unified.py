# This module coordinates unified visualization outputs.

from __future__ import annotations

                          
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

                     
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec
from matplotlib.colors import ListedColormap

               
from ..analysis.snr_utils import compute_snr_profile, find_snr_peak
from ..config import config
from ..preprocessing.dedispersion import dedisperse_block
from ..preprocessing.dm_candidate_extractor import extract_candidate_dm
from .visualization_ranges import get_dynamic_dm_range_for_candidate
from .plot_composite import save_composite_plot

              
logger = logging.getLogger(__name__)

                                                    
if "mako" not in plt.colormaps():
    plt.register_cmap(
        name="mako",
        cmap=ListedColormap(sns.color_palette("mako", as_cmap=True)(np.linspace(0, 1, 256)))
    )

def _calculate_dynamic_dm_range(
    top_boxes: Iterable | None,
    slice_len: int,
    fallback_dm_min: int = None,
    fallback_dm_max: int = None,
    confidence_scores: Iterable | None = None
) -> Tuple[float, float]:
    """Unified delegate: uses visualization_ranges for dynamic DM range."""
    if (not getattr(config, 'DM_DYNAMIC_RANGE_ENABLE', True)
        or top_boxes is None
        or len(top_boxes) == 0):
        dm_min = fallback_dm_min if fallback_dm_min is not None else config.DM_min
        dm_max = fallback_dm_max if fallback_dm_max is not None else config.DM_max
        return float(dm_min), float(dm_max)

    dm_candidates: List[float] = []
    for box in top_boxes:
        x1, y1, x2, y2 = map(int, box)
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        dm_val, _, _ = extract_candidate_dm(center_x, center_y, slice_len)
        dm_candidates.append(dm_val)
    if not dm_candidates:
        dm_min = fallback_dm_min if fallback_dm_min is not None else config.DM_min
        dm_max = fallback_dm_max if fallback_dm_max is not None else config.DM_max
        return float(dm_min), float(dm_max)

    if confidence_scores is not None and len(confidence_scores) > 0:
        best_idx = int(np.argmax(confidence_scores))
        dm_optimal = float(dm_candidates[best_idx])
        confidence = float(confidence_scores[best_idx])
    else:
        dm_optimal = float(np.median(dm_candidates))
        confidence = 0.8

    try:
        return get_dynamic_dm_range_for_candidate(
            dm_optimal=dm_optimal,
            config_module=config,
            visualization_type=getattr(config, 'DM_RANGE_DEFAULT_VISUALIZATION', 'detailed'),
            confidence=confidence,
            range_factor=getattr(config, 'DM_RANGE_FACTOR', 0.2),
            min_range_width=getattr(config, 'DM_RANGE_MIN_WIDTH', 50.0),
            max_range_width=getattr(config, 'DM_RANGE_MAX_WIDTH', 200.0),
        )
    except Exception as e:
        print(f"[WARNING] Error calculating dynamic DM range: {e}")
        dm_min = fallback_dm_min if fallback_dm_min is not None else config.DM_min
        dm_max = fallback_dm_max if fallback_dm_max is not None else config.DM_max
        return float(dm_min), float(dm_max)


def preprocess_img(img: np.ndarray) -> np.ndarray:
    img = (img - img.min()) / np.ptp(img)
    img = (img - img.mean()) / img.std()
    img = cv2.resize(img, (512, 512))
    img = np.clip(img, *np.percentile(img, (0.1, 99.9)))
    img = (img - img.min()) / np.ptp(img)
    img = plt.get_cmap("mako")(img)[..., :3]
    img -= [0.485, 0.456, 0.406]
    img /= [0.229, 0.224, 0.225]
    return img.transpose(2, 0, 1)


def postprocess_img(img_tensor: np.ndarray) -> np.ndarray:
    img = img_tensor.transpose(1, 2, 0)
    img *= [0.229, 0.224, 0.225]
    img += [0.485, 0.456, 0.406]
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_all_plots(
    waterfall_block,
    dedisp_block,
    img_rgb,
    first_patch,
    first_start,
    first_dm,
    top_conf,
    top_boxes,
    class_probs_list,
    comp_path,
    j,
    time_slice,
    band_name,
    band_suffix,
    fits_stem,
    slice_len,
    normalize,
    off_regions,
    thresh_snr,
    band_idx,
    patch_path,
    absolute_start_time=None,
    chunk_idx=None,  
    force_plots: bool = False,
    candidate_times_abs: Optional[Iterable[float]] = None,
):

    if comp_path is not None:
        comp_path.parent.mkdir(parents=True, exist_ok=True)
        
                                                                                   
        real_slice_samples = (
            waterfall_block.shape[0]
            if waterfall_block is not None and hasattr(waterfall_block, "shape")
            else slice_len
        )

                                                                                
        save_slice_summary(
            waterfall_block,
            dedisp_block if dedisp_block is not None and dedisp_block.size > 0 else waterfall_block,
            img_rgb,
            first_patch,
            first_start if first_start is not None else 0.0,
            first_dm if first_dm is not None else 0.0,
            top_conf if len(top_conf) > 0 else [],
            top_boxes if len(top_boxes) > 0 else [],
            class_probs_list,
            comp_path,
            j,
            time_slice,
            band_name,
            band_suffix,
            fits_stem,
            slice_len,
            normalize=normalize,
            off_regions=off_regions,
            thresh_snr=thresh_snr,
            band_idx=band_idx,
            absolute_start_time=absolute_start_time,  
            chunk_idx=chunk_idx,  
            slice_samples=real_slice_samples,
            candidate_times_abs=candidate_times_abs,
        )
        
        logger.info(f"Plot composite generado en: {comp_path}")
        logger.info(f"Individual plots automatically generated in: {comp_path.parent}/individual_plots/")

def get_band_frequency_range(band_idx: int) -> Tuple[float, float]:
    freq_ds = np.mean(
        config.FREQ.reshape(config.FREQ_RESO // config.DOWN_FREQ_RATE, config.DOWN_FREQ_RATE),
        axis=1,
    )
    
    if band_idx == 0:             
        return freq_ds.min(), freq_ds.max()
    elif band_idx == 1:             
        mid_channel = len(freq_ds) // 2
        return freq_ds.min(), freq_ds[mid_channel]
    elif band_idx == 2:             
        mid_channel = len(freq_ds) // 2  
        return freq_ds[mid_channel], freq_ds.max()
    else:
        logger.warning(f"Invalid band index {band_idx}, using Full Band range")
        return freq_ds.min(), freq_ds.max()


def get_band_name_with_freq_range(band_idx: int, band_name: str) -> str:
    freq_min, freq_max = get_band_frequency_range(band_idx)
    return f"{band_name} ({freq_min:.0f}-{freq_max:.0f} MHz)"

def save_slice_summary(
    waterfall_block: np.ndarray,
    dedispersed_block: np.ndarray,
    img_rgb: np.ndarray,
    patch_img: np.ndarray,
    patch_start: float,
    dm_val: float,
    top_conf: Iterable,
    top_boxes: Iterable | None,
    class_probs: Iterable | None,
    out_path: Path,
    slice_idx: int,
    time_slice: int,
    band_name: str,
    band_suffix: str,
    fits_stem: str,
    slice_len: int,
    normalize: bool = False,
    off_regions: Optional[List[Tuple[int, int]]] = None,
    thresh_snr: Optional[float] = None,
    band_idx: int = 0,                                        
    absolute_start_time: Optional[float] = None, 
    chunk_idx: Optional[int] = None,  
    slice_samples: Optional[int] = None,  
    candidate_times_abs: Optional[Iterable[float]] = None,
) -> None:
    
    save_composite_plot(
        waterfall_block=waterfall_block,
        dedispersed_block=dedispersed_block,
        img_rgb=img_rgb,
        patch_img=patch_img,
        patch_start=patch_start,
        dm_val=dm_val,
        top_conf=top_conf,
        top_boxes=top_boxes,
        class_probs=class_probs,
        out_path=out_path,
        slice_idx=slice_idx,
        time_slice=time_slice,
        band_name=band_name,
        band_suffix=band_suffix,
        fits_stem=fits_stem,
        slice_len=slice_len,
        normalize=normalize,
        off_regions=off_regions,
        thresh_snr=thresh_snr,
        band_idx=band_idx,
        absolute_start_time=absolute_start_time,
        chunk_idx=chunk_idx,
        slice_samples=slice_samples,
        candidate_times_abs=candidate_times_abs,
    )

