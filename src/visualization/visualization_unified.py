# This module coordinates unified visualization outputs.

from __future__ import annotations

                          
import logging
from typing import Iterable, Optional

                     
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

               
from .plot_composite import save_composite_plot

              
logger = logging.getLogger(__name__)

# Seven modules in this package pass cmap="mako" to imshow without registering
# it themselves, so importing this one is what makes that name resolve. What
# registers it today is `import seaborn` above; the guard is the fallback for a
# seaborn that stops doing that. Two things had to change for the fallback to
# be able to run at all:
#   * plt.register_cmap was removed in matplotlib 3.9 and this project needs
#     >= 3.10, so the call itself raised AttributeError.
#   * the colormap came from sns.color_palette("mako"), which resolves the name
#     through matplotlib's own registry -- the thing the guard says is missing.
#     It raised ValueError: 'mako' is not a valid palette name.
# seaborn's colormap object carries the same 256 colors and needs no registry.
if "mako" not in plt.colormaps():
    matplotlib.colormaps.register(sns.cm.mako, name="mako")



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
    absolute_start_time=None,
    chunk_idx=None,
    candidate_times_abs: Optional[Iterable[float]] = None,
    dedisp_block_linear: Optional[np.ndarray] = None,
    dedisp_block_circular: Optional[np.ndarray] = None,
    class_probs_linear_list: Optional[Iterable[float]] = None,  # NEW: Linear classification probs
    snr_waterfall_linear_list: Optional[Iterable[float | None]] = None,  # NEW: SNR from Linear waterfall
    snr_patch_linear_list: Optional[Iterable[float | None]] = None,  # NEW: SNR from dedispersed Linear patch
    snr_waterfall_intensity_list: Optional[Iterable[float | None]] = None,  # NEW: SNR from Intensity waterfall
    snr_patch_intensity_list: Optional[Iterable[float | None]] = None,  # NEW: SNR from dedispersed Intensity patch
):

    def _safe_float(arr):
        """Coerce array to float64 if it has dtype=object (FITS edge case)."""
        if arr is None:
            return None
        a = np.asarray(arr)
        if a.dtype == object or not np.issubdtype(a.dtype, np.number):
            try:
                a = np.asarray(a, dtype=np.float64)
            except (TypeError, ValueError):
                a = np.array(a.tolist(), dtype=np.float64)
        return a

    waterfall_block = _safe_float(waterfall_block)
    dedisp_block = _safe_float(dedisp_block)
    if dedisp_block_linear is not None:
        dedisp_block_linear = _safe_float(dedisp_block_linear)
    if dedisp_block_circular is not None:
        dedisp_block_circular = _safe_float(dedisp_block_circular)
    if img_rgb is not None:
        img_rgb = np.asarray(img_rgb)
        if img_rgb.dtype == object:
            try:
                img_rgb = np.asarray(img_rgb, dtype=np.float32)
            except (TypeError, ValueError):
                img_rgb = np.array(img_rgb.tolist(), dtype=np.float32)

    if comp_path is not None:
        comp_path.parent.mkdir(parents=True, exist_ok=True)
        
                                                                                   
        real_slice_samples = (
            waterfall_block.shape[0]
            if waterfall_block is not None and hasattr(waterfall_block, "shape")
            else slice_len
        )


        save_composite_plot(
            waterfall_block=waterfall_block,
            dedispersed_block=(
                dedisp_block if dedisp_block is not None and dedisp_block.size > 0 else waterfall_block
            ),
            img_rgb=img_rgb,
            patch_img=first_patch,
            patch_start=first_start if first_start is not None else 0.0,
            dm_val=first_dm if first_dm is not None else 0.0,
            top_conf=top_conf if len(top_conf) > 0 else [],
            top_boxes=top_boxes if len(top_boxes) > 0 else [],
            class_probs=class_probs_list,
            out_path=comp_path,
            slice_idx=j,
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
            slice_samples=real_slice_samples,
            candidate_times_abs=candidate_times_abs,
            dedisp_block_linear=dedisp_block_linear,
            dedisp_block_circular=dedisp_block_circular,
            class_probs_linear=class_probs_linear_list,  # NEW: Pass Linear probs
            snr_waterfall_linear=snr_waterfall_linear_list,  # NEW: Pass SNR from Linear waterfall
            snr_patch_linear=snr_patch_linear_list,  # NEW: Pass SNR from dedispersed Linear patch
            snr_waterfall_intensity=snr_waterfall_intensity_list,  # NEW: Pass SNR from Intensity waterfall
            snr_patch_intensity=snr_patch_intensity_list,  # NEW: Pass SNR from dedispersed Intensity patch
        )

        logger.info(f"Composite plot generated at: {comp_path}")
        logger.info(f"Individual plots automatically generated in: {comp_path.parent}/individual_plots/")

