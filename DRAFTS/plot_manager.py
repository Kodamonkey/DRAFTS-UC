"""Plot management utilities for FRB pipeline visualization."""
from pathlib import Path
import numpy as np
from ..visualization.image_utils import plot_waterfall_block
from ..visualization.visualization import save_plot, save_patch_plot, save_slice_summary

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
    waterfall_dedispersion_dir,
    freq_down,
    time_reso_ds,
    detections_dir,
    out_img_path,
    absolute_start_time=None
):
    """Guarda todos los plots con tiempo absoluto para continuidad temporal.
    
    Args:
        absolute_start_time: Tiempo absoluto de inicio del slice en segundos desde el inicio del archivo
    """
    # Composite plot
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
        absolute_start_time=absolute_start_time,  # 🕐 PASAR TIEMPO ABSOLUTO
    )
    # Patch plot
    if first_patch is not None:
        save_patch_plot(
            first_patch,
            patch_path,
            freq_down,
            time_reso_ds,
            first_start,
            off_regions=off_regions,
            thresh_snr=thresh_snr,
            band_idx=band_idx,
            band_name=band_name,
        )
    # Waterfall dedispersed
    if dedisp_block is not None and dedisp_block.size > 0:
        plot_waterfall_block(
            data_block=dedisp_block,
            freq=freq_down,
            time_reso=time_reso_ds,
            block_size=dedisp_block.shape[0],
            block_idx=j,
            save_dir=waterfall_dedispersion_dir,
            filename=f"{fits_stem}_dm{first_dm:.2f}_{band_suffix}",
            normalize=normalize,
            absolute_start_time=absolute_start_time,  # 🕐 PASAR TIEMPO ABSOLUTO
        )
    # Detections plot
    save_plot(
        img_rgb,
        top_conf if len(top_conf) > 0 else [],
        top_boxes if len(top_boxes) > 0 else [],
        class_probs_list,
        out_img_path,
        j,
        time_slice,
        band_name,
        band_suffix,
        fits_stem,
        slice_len,
        band_idx=band_idx,
        absolute_start_time=absolute_start_time,  # 🕐 PASAR TIEMPO ABSOLUTO
    )

