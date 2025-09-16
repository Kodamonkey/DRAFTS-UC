# This module generates individual diagnostic plots.

"""Individual plot components generator for FRB pipeline - creates separate plots for each composite component."""
from __future__ import annotations

                          
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

               
from .plot_dm_time import save_dm_time_plot
from .plot_waterfall_dispersed import save_waterfall_dispersed_plot
from .plot_waterfall_dedispersed import save_waterfall_dedispersed_plot
from .plot_patches import save_patches_plot

              
logger = logging.getLogger(__name__)


def generate_individual_plots(
    waterfall_block,
    dedispersed_block,
    img_rgb,
    patch_img,
    patch_start,
    dm_val,
    top_conf,
    top_boxes,
    class_probs,
    base_out_path: Path,
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
    output_dir: str = "individual_plots",
) -> None:
    """Generate individual plots for each component of the composite plot.
    
    This function creates separate plot files for:
    1. DM-Time plot (detections)
    2. Waterfall dispersed plot
    3. Waterfall dedispersed plot
    4. Patches plot (candidate centered)
    
    All plots are IDENTICAL to their corresponding panels in the composite plot.
    """
    
                                                  
    if chunk_idx is not None:
        individual_dir = base_out_path.parent / output_dir / f"chunk_{chunk_idx:03d}" / f"slice_{slice_idx:03d}"
    else:
        individual_dir = base_out_path.parent / output_dir / f"slice_{slice_idx:03d}"
    
    individual_dir.mkdir(parents=True, exist_ok=True)
    
                            
    base_filename = f"{fits_stem}_slice_{slice_idx:03d}"
    if chunk_idx is not None:
        base_filename = f"{fits_stem}_chunk_{chunk_idx:03d}_slice_{slice_idx:03d}"
    
    logger.info(f"Generating individual plots in: {individual_dir}")
    
    try:
                                                                                 
        dm_time_path = individual_dir / f"{base_filename}_dm_time.png"
        save_dm_time_plot(
            img_rgb=img_rgb,
            top_conf=top_conf,
            top_boxes=top_boxes,
            class_probs=class_probs,
            out_path=dm_time_path,
            slice_idx=slice_idx,
            time_slice=time_slice,
            band_name=band_name,
            band_suffix=band_suffix,
            fits_stem=fits_stem,
            slice_len=slice_len,
            band_idx=band_idx,
            absolute_start_time=absolute_start_time,
            chunk_idx=chunk_idx,
            slice_samples=slice_samples,
            candidate_times_abs=candidate_times_abs,
        )
        logger.info(f"✓ DM-Time plot saved: {dm_time_path}")
        
                                                                                 
        waterfall_dispersed_path = individual_dir / f"{base_filename}_waterfall_dispersed.png"
        save_waterfall_dispersed_plot(
            waterfall_block=waterfall_block,
            out_path=waterfall_dispersed_path,
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
            dm_value=dm_val,                                            
        )
        logger.info(f"✓ Waterfall dispersed plot saved: {waterfall_dispersed_path}")
        
                                                                                 
        waterfall_dedispersed_path = individual_dir / f"{base_filename}_waterfall_dedispersed.png"
        save_waterfall_dedispersed_plot(
            dedispersed_block=dedispersed_block,
            waterfall_block=waterfall_block,
            top_conf=top_conf,
            top_boxes=top_boxes,
            out_path=waterfall_dedispersed_path,
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
        )
        logger.info(f"✓ Waterfall dedispersed plot saved: {waterfall_dedispersed_path}")
        
                                                                                        
        patches_path = individual_dir / f"{base_filename}_patches.png"
        save_patches_plot(
            patch_img=patch_img,
            patch_start=patch_start,
            out_path=patches_path,
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
        )
        logger.info(f"✓ Patches plot saved: {patches_path}")
        
        logger.info(f"All individual plots generated successfully in: {individual_dir}")
        
    except Exception as e:
        logger.error(f"Error generating individual plots: {e}")
        raise


def generate_individual_plots_from_composite_params(
    composite_params: dict,
    base_out_path: Path,
    output_dir: str = "individual_plots",
) -> None:
    """Generate individual plots using parameters from a composite plot call.
    
    This is a convenience function that extracts parameters from a dictionary
    and calls generate_individual_plots.
    """
    
                                 
    required_params = [
        'waterfall_block', 'dedispersed_block', 'img_rgb', 'patch_img',
        'patch_start', 'dm_val', 'top_conf', 'top_boxes', 'class_probs',
        'slice_idx', 'time_slice', 'band_name', 'band_suffix', 'fits_stem', 'slice_len'
    ]
    
                                                  
    missing_params = [param for param in required_params if param not in composite_params]
    if missing_params:
        raise ValueError(f"Missing required parameters: {missing_params}")
    
                                               
    normalize = composite_params.get('normalize', False)
    off_regions = composite_params.get('off_regions', None)
    thresh_snr = composite_params.get('thresh_snr', None)
    band_idx = composite_params.get('band_idx', 0)
    absolute_start_time = composite_params.get('absolute_start_time', None)
    chunk_idx = composite_params.get('chunk_idx', None)
    slice_samples = composite_params.get('slice_samples', None)
    candidate_times_abs = composite_params.get('candidate_times_abs', None)
    
                            
    generate_individual_plots(
        waterfall_block=composite_params['waterfall_block'],
        dedispersed_block=composite_params['dedispersed_block'],
        img_rgb=composite_params['img_rgb'],
        patch_img=composite_params['patch_img'],
        patch_start=composite_params['patch_start'],
        dm_val=composite_params['dm_val'],
        top_conf=composite_params['top_conf'],
        top_boxes=composite_params['top_boxes'],
        class_probs=composite_params['class_probs'],
        base_out_path=base_out_path,
        slice_idx=composite_params['slice_idx'],
        time_slice=composite_params['time_slice'],
        band_name=composite_params['band_name'],
        band_suffix=composite_params['band_suffix'],
        fits_stem=composite_params['fits_stem'],
        slice_len=composite_params['slice_len'],
        normalize=normalize,
        off_regions=off_regions,
        thresh_snr=thresh_snr,
        band_idx=band_idx,
        absolute_start_time=absolute_start_time,
        chunk_idx=chunk_idx,
        slice_samples=slice_samples,
        candidate_times_abs=candidate_times_abs,
        output_dir=output_dir,
    )

