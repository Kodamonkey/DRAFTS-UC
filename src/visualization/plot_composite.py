# This module creates composite diagnostic visualizations.

"""Composite plot generation module for FRB pipeline.

``create_composite_plot`` used to be one 868-line function with thirty
parameters; it is now an orchestrator over one function per panel. Each panel
takes what it draws with, returns what the panel beneath it needs, and can be
called on an axes of its own. What the panels share travels as the three small
objects near the top of this module rather than as free variables in one very
long scope.

``tests/test_golden_images.py`` pins what the assembled figure looks like; every
step of that decomposition was made against it.
"""
from __future__ import annotations


import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

                     
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

               
from ..domain.physics import K_DM_MS
from ..analysis.snr_utils import compute_snr_profile, find_snr_peak
from ..config import config
from ..preprocessing.dm_candidate_extractor import extract_candidate_dm
from ..core.mjd_utils import calculate_candidate_mjd
from .normalization import normalize_block, percentile_limits
from .plot_multi_pol_panels import create_multi_pol_panels
from .visualization_ranges import get_dynamic_dm_range_for_candidate

              
logger = logging.getLogger(__name__)


def _coerce_float_image(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Force waterfall / dedispersed panels to a real dtype (FITS can yield dtype=object)."""

    if arr is None or getattr(arr, "size", 0) == 0:
        return arr
    a = np.ascontiguousarray(arr)
    if a.dtype == object or not np.issubdtype(a.dtype, np.number):
        try:
            a = np.asarray(a, dtype=np.float64)
        except (TypeError, ValueError):
            a = np.array(a.tolist(), dtype=np.float64)
    else:
        a = a.astype(np.float64, copy=False)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _coerce_img_rgb_for_imshow(arr: np.ndarray) -> np.ndarray:
    """DM-time RGB: matplotlib rejects dtype=object; normalize float RGB to 0–1 if needed."""

    if arr is None or getattr(arr, "size", 0) == 0:
        return arr
    a = np.ascontiguousarray(arr)
    if a.dtype == object or not np.issubdtype(a.dtype, np.number):
        try:
            a = np.asarray(a, dtype=np.float32)
        except (TypeError, ValueError):
            a = np.array(a.tolist(), dtype=np.float32)
    if np.issubdtype(a.dtype, np.floating):
        amax = float(np.nanmax(a)) if a.size else 0.0
        if amax > 1.0:
            a = np.clip(a / 255.0, 0.0, 1.0)
        else:
            a = np.clip(a, 0.0, 1.0)
    return a


def _calculate_label_positions(
    ax,
    candidate_boxes: list,
    initial_offset: float = 10.0,
    min_spacing: float = 5.0,
    title_bottom: float = None
) -> list:
    """
    Calculate label positions avoiding collisions with title and between labels.
    
    Args:
        ax: Matplotlib axis
        candidate_boxes: List of (x1, y1, x2, y2) tuples for each candidate
        initial_offset: Initial vertical offset from box top
        min_spacing: Minimum spacing between labels
        title_bottom: Y coordinate of title bottom (in data coordinates)
    
    Returns:
        List of (x, y) tuples for label positions
    """
    if not candidate_boxes:
        return []
    
    # Get axis limits to work in data coordinates
    ylim = ax.get_ylim()
    
    # Estimate label height in data coordinates
    # A typical label has ~6-8 lines, estimate ~100-150 pixels total height
    # Convert to data coordinates
    bbox_ax = ax.get_window_extent()
    height_pixels = bbox_ax.height
    data_height = ylim[1] - ylim[0]
    pixels_per_data_unit = height_pixels / data_height if data_height > 0 else 1.0
    
    # Estimate label height in data coordinates (assuming ~120-150 pixels for full label)
    estimated_label_height = 130.0 / pixels_per_data_unit
    
    label_positions = []
    
    for idx, box in enumerate(candidate_boxes):
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2.0
        
        # Start position: above the box
        y_start = float(y2) + initial_offset
        
        # If title position is provided, ensure label is below title
        if title_bottom is not None:
            y_start = max(y_start, float(title_bottom) - min_spacing)
        
        # Check collisions with existing labels
        y_pos = y_start
        max_iterations = 100
        iteration = 0
        
        while iteration < max_iterations:
            # Check if this position collides with any existing label
            collision = False
            label_top = y_pos + estimated_label_height / 2
            label_bottom = y_pos - estimated_label_height / 2
            
            for existing_x, existing_y in label_positions:
                existing_top = existing_y + estimated_label_height / 2
                existing_bottom = existing_y - estimated_label_height / 2
                
                # Check vertical overlap (horizontal overlap is OK, we stack vertically)
                if not (label_bottom > existing_top or label_top < existing_bottom):
                    collision = True
                    break
            
            if not collision:
                break
            
            # Move down
            y_pos += estimated_label_height + min_spacing
            iteration += 1
        
        # Ensure label doesn't go below plot area (with some margin)
        if y_pos > ylim[1]:
            y_pos = float(ylim[1]) - estimated_label_height / 2 - min_spacing
        
        label_positions.append((center_x, y_pos))
    
    return label_positions


def _calculate_dynamic_dm_range(
    top_boxes: Iterable | None,
    slice_len: int,
    fallback_dm_min: int = None,
    fallback_dm_max: int = None,
    confidence_scores: Iterable | None = None
) -> Tuple[float, float]:
    """Unified delegate: use visualization_ranges for dynamic DM range."""
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


def get_band_frequency_range(band_idx: int) -> Tuple[float, float]:
    """Get the frequency range (min, max) for a specific band."""
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
    """Get band name with frequency range information."""
    freq_min, freq_max = get_band_frequency_range(band_idx)
    return f"{band_name} ({freq_min:.0f}-{freq_max:.0f} MHz)"


# ---------------------------------------------------------------------------
# The data clumps the panels share
#
# These three objects exist because the same handful of values travelled
# together through the whole of the old create_composite_plot and every panel
# read some of them. Passing one object beats repeating five parameters on five
# signatures, and it makes what a panel actually needs legible at its call site.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SliceWindow:
    """Time and frequency geometry shared by every bottom panel."""

    start_abs: float
    end_abs: float
    freq_ds: np.ndarray
    time_reso_ds: float
    #: Bound once in create_composite_plot and handed to both waterfalls. It
    #: used to be assigned inside the block guarded on the raw waterfall and
    #: read inside the block guarded on the dedispersed one (audit P1-23).
    freq_tick_positions: np.ndarray

    @property
    def freq_min(self) -> float:
        return float(self.freq_ds.min())

    @property
    def freq_max(self) -> float:
        return float(self.freq_ds.max())


@dataclass(frozen=True)
class RawWaterfallWindow:
    """The raw waterfall's own time window, and the SNR its panels display.

    The raw waterfall shows the burst still dispersed, so it arrives later in
    the lowest channel than the slice start suggests. The window is therefore
    the slice window shifted back by the sweep of the best candidate. Computed
    once, read by the SNR profile and by the waterfall under it.
    """

    start: float
    end: float
    #: SNR of the best candidate as the *detector* measured it (PRESTO-style).
    #: The panels must display this number and not one recomputed here, or the
    #: waterfall title and the DM-time label disagree.
    candidate_snr_intensity: Optional[float]


@dataclass(frozen=True)
class SnrProfilePanel:
    """What an SNR profile panel computed, for the waterfall drawn beneath it.

    This is what replaced two ``if 'peak_snr_wf' in locals()`` sniffs. The
    waterfall needs the profile's time axis and peak sample, and needed a way
    to ask whether the profile above it had drawn anything at all; ``None`` is
    that answer now, and it is a real condition rather than a question about
    the interpreter's stack frame.
    """

    time_axis: np.ndarray
    peak_idx: int

    @property
    def peak_time(self) -> float:
        return float(self.time_axis[self.peak_idx])


@dataclass(frozen=True)
class CandidateSnr:
    """The four per-candidate SNR lists, coerced once and indexed safely.

    They arrive as arbitrary iterables and were indexed repeatedly, so each one
    had to be materialised before the first read; and every read repeated the
    same three bounds and None checks. Both happen here instead.
    """

    waterfall_intensity: Optional[list]
    patch_intensity: Optional[list]
    waterfall_linear: Optional[list]
    patch_linear: Optional[list]

    @staticmethod
    def _materialise(values, name: str, quiet_when_absent: bool):
        if values is None:
            if quiet_when_absent:
                logger.debug("create_composite_plot: %s is None "
                             "(expected in standard/single-pol mode)", name)
            else:
                logger.warning("create_composite_plot: %s is None", name)
            return None
        seq = values if isinstance(values, (list, tuple)) else list(values)
        logger.info("create_composite_plot: Received %s with %d values: %s",
                    name, len(seq),
                    [f"{v:.2f}" if v is not None else "None" for v in seq[:5]])
        return seq

    @classmethod
    def collect(cls, *, waterfall_intensity, patch_intensity,
                waterfall_linear, patch_linear) -> "CandidateSnr":
        return cls(
            waterfall_intensity=cls._materialise(
                waterfall_intensity, "snr_waterfall_intensity", quiet_when_absent=False),
            patch_intensity=cls._materialise(
                patch_intensity, "snr_patch_intensity", quiet_when_absent=False),
            waterfall_linear=cls._materialise(
                waterfall_linear, "snr_waterfall_linear", quiet_when_absent=True),
            patch_linear=cls._materialise(
                patch_linear, "snr_patch_linear", quiet_when_absent=True),
        )

    @staticmethod
    def _at(values, idx: int, name: str) -> Optional[float]:
        if values is None:
            logger.warning("%s is None for candidate idx=%d", name, idx)
            return None
        if idx >= len(values):
            logger.warning("idx=%d >= len(%s)=%d", idx, name, len(values))
            return None
        value = values[idx]
        if value is None:
            logger.warning("%s[%d] is None", name, idx)
            return None
        return float(value)

    def intensity_at(self, idx: int) -> Optional[float]:
        value = self._at(self.waterfall_intensity, idx, "snr_waterfall_intensity")
        if value is not None:
            logger.info("Candidate idx=%d: SNR_I from detection=%.2f (PRESTO-style)",
                        idx, value)
        return value

    def linear_at(self, idx: int) -> Optional[float]:
        value = self._at(self.waterfall_linear, idx, "snr_waterfall_linear")
        if value is not None:
            logger.info("Candidate idx=%d: snr_L_wf=%.2f", idx, value)
        return value

    def intensity_for_best(self, best_idx: int):
        """The detector's intensity SNR for the best candidate, raw (not float)."""
        seq = self.waterfall_intensity
        if seq is None:
            return None
        if best_idx < len(seq):
            return seq[best_idx]
        if len(seq) > 0:
            return seq[0]
        return None


# ---------------------------------------------------------------------------
# Small shared decisions, made once each
# ---------------------------------------------------------------------------


def _best_candidate_time(top_conf, top_boxes, candidate_times_abs) -> Optional[float]:
    """Absolute time of the highest-confidence candidate, or None."""
    if top_boxes is None or len(top_boxes) == 0:
        return None
    if top_conf is not None and len(top_conf) > 0:
        best_idx = int(np.argmax(top_conf))
    else:
        best_idx = 0
    if candidate_times_abs is not None and best_idx < len(candidate_times_abs):
        return candidate_times_abs[best_idx]
    return None


def _mark_time(time_axis, peak_idx, candidate_snr_intensity,
               top_conf, top_boxes, candidate_times_abs) -> float:
    """Where the red dot goes: the candidate's time when known, else the peak."""
    if candidate_snr_intensity is not None:
        candidate_time = _best_candidate_time(top_conf, top_boxes, candidate_times_abs)
        if candidate_time is not None:
            idx = int(np.argmin(np.abs(time_axis - candidate_time)))
            if 0 <= idx < len(time_axis):
                return float(time_axis[idx])
    return float(time_axis[peak_idx])


def _profile_title(panel_name: str, display_snr: float, candidate_snr_intensity,
                   top_conf, top_boxes, candidate_times_abs) -> str:
    if candidate_snr_intensity is not None:
        candidate_time = _best_candidate_time(top_conf, top_boxes, candidate_times_abs)
        if candidate_time is not None:
            return f"{panel_name}\nTime: {candidate_time:.6f}s | SNR: {display_snr:.1f}σ"
    return f"{panel_name}\nPeak SNR: {display_snr:.1f}σ"


def _no_data_profile(ax, message: str, title: str) -> None:
    """The empty arm of both SNR profile panels."""
    ax.text(0.5, 0.5, message, transform=ax.transAxes,
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    ax.set_ylabel('SNR (σ)', fontsize=8, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([])
    ax.set_title(title, fontsize=9, fontweight="bold")


def _no_data_waterfall(ax, message: str) -> None:
    """The empty arm of both waterfall panels."""
    ax.text(0.5, 0.5, message, transform=ax.transAxes,
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Frequency (MHz)", fontsize=9)


def _mark_display_snr(ax, mark_time: float, display_snr: float) -> None:
    """The red dot and the boxed 'NNσ' beside it, shared by both profiles."""
    ax.plot(mark_time, display_snr, 'ro', markersize=5)
    # Offset to the right of and above the mark so the box clears the title.
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    ax.text(mark_time + 0.02 * x_range, display_snr + 0.15 * y_range,
            f'{display_snr:.1f}σ', ha='left', va='bottom', fontsize=8,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8,
                      edgecolor='red', linewidth=0.5))


def _prepare_blocks(waterfall_block, dedispersed_block, dedisp_block_linear,
                    dedisp_block_circular, *, multi_pol_mode: bool, normalize: bool):
    """Copy, coerce to a real dtype and optionally normalize the four blocks."""
    wf_block = (waterfall_block.copy()
                if waterfall_block is not None and waterfall_block.size > 0 else None)
    dw_block = (dedispersed_block.copy()
                if dedispersed_block is not None and dedispersed_block.size > 0 else None)

    dw_linear = None
    dw_circular = None
    if multi_pol_mode:
        if dedisp_block_linear is not None and dedisp_block_linear.size > 0:
            dw_linear = _coerce_float_image(dedisp_block_linear.copy())
        if dedisp_block_circular is not None and dedisp_block_circular.size > 0:
            dw_circular = _coerce_float_image(dedisp_block_circular.copy())

    wf_block = _coerce_float_image(wf_block)
    dw_block = _coerce_float_image(dw_block)

    if normalize:
        wf_block = normalize_block(wf_block)
        dw_block = normalize_block(dw_block)
        if multi_pol_mode:
            dw_linear = normalize_block(dw_linear)
            dw_circular = normalize_block(dw_circular)

    return wf_block, dw_block, dw_linear, dw_circular


# ---------------------------------------------------------------------------
# Panel 1: the DM-time detection image, its axes, and the candidate overlay
# ---------------------------------------------------------------------------


def draw_detection_panel(fig, gs_main, img_rgb, *, slice_start_abs: float,
                         slice_end_abs: float, top_conf, top_boxes, slice_len: int):
    """Panel 1a: the DM-time image with absolute-time and DM axes."""
    ax_det = fig.add_subplot(gs_main[0, 0])
    img_rgb = _coerce_img_rgb_for_imshow(img_rgb)
    ax_det.imshow(img_rgb, origin="lower", aspect="auto")

    # Set first: the label layout below measures where the title ends.
    ax_det.set_title("Detection Results", fontsize=10, fontweight="bold")
    ax_det.set_xlabel("Time (s)", fontsize=9)
    ax_det.set_ylabel("Dispersion Measure (pc cm⁻³)", fontsize=9)

    n_time_ticks_det = 8
    time_positions_det = np.linspace(0, img_rgb.shape[1] - 1, n_time_ticks_det)
    denom = float(max(img_rgb.shape[1] - 1, 1))
    time_values_det = slice_start_abs + (time_positions_det / denom) * (
        slice_end_abs - slice_start_abs)
    ax_det.set_xticks(time_positions_det)
    ax_det.set_xticklabels([f"{t:.6f}" for t in time_values_det], rotation=45)
    ax_det.set_xlabel("Time (s)", fontsize=10, fontweight="bold")

    n_dm_ticks = 8
    dm_positions = np.linspace(0, img_rgb.shape[0] - 1, n_dm_ticks)
    dm_plot_min, dm_plot_max = _calculate_dynamic_dm_range(
        top_boxes=top_boxes,
        slice_len=slice_len,
        fallback_dm_min=config.DM_min,
        fallback_dm_max=config.DM_max,
        confidence_scores=top_conf if top_conf is not None else None,
    )
    dm_values = dm_plot_min + (dm_positions / img_rgb.shape[0]) * (dm_plot_max - dm_plot_min)
    ax_det.set_yticks(dm_positions)
    ax_det.set_yticklabels([f"{dm:.0f}" for dm in dm_values])
    ax_det.set_ylabel("Dispersion Measure (pc cm⁻³)", fontsize=10, fontweight="bold")
    return ax_det


def candidate_label_positions(fig, ax_det, top_boxes) -> list:
    """Panel 1b: where each candidate's label goes, clear of the title and of
    the other labels.

    Needs a rendered figure: the title's extent is only known after a draw, and
    it is wanted in data coordinates.
    """
    if top_boxes is None:
        return []

    fig.canvas.draw()
    title_bbox = ax_det.title.get_window_extent(fig.canvas.get_renderer())
    bbox_ax = ax_det.get_window_extent()
    ylim = ax_det.get_ylim()

    title_bottom_fig = title_bbox.y0
    ax_top_fig = bbox_ax.y1
    ax_bottom_fig = bbox_ax.y0

    if title_bottom_fig <= ax_top_fig:
        # The title reaches into the axes; find where it stops, in data units.
        overlap_fig = ax_top_fig - title_bottom_fig
        data_range = ylim[1] - ylim[0]
        fig_range = ax_top_fig - ax_bottom_fig
        if fig_range > 0:
            title_bottom_data = ylim[1] - (overlap_fig / fig_range) * data_range
        else:
            title_bottom_data = ylim[1]
    else:
        title_bottom_data = ylim[1]

    return _calculate_label_positions(
        ax_det,
        top_boxes,
        initial_offset=10.0,
        min_spacing=5.0,
        title_bottom=title_bottom_data,
    )


def _candidate_label(idx: int, conf, dm_val_cand: float, detection_time: float,
                     mjd_data: dict, *, class_probs, class_probs_linear,
                     candidate_snr: CandidateSnr) -> Tuple[str, str]:
    """The text and the colour of one candidate's annotation.

    Three shapes, and which one applies is decided by what the caller supplied:
    no classification at all, intensity only, or intensity plus linear (the HF
    pipeline, where a candidate is green only if both agree).
    """
    mjd_str = f"MJD_topo: {mjd_data.get('mjd_utc', 0.0):.8f}"
    mjd_bary_utc_inf = mjd_data.get('mjd_bary_utc_inf')
    if mjd_bary_utc_inf is not None:
        mjd_str += f"\nMJD_bary_inf: {mjd_bary_utc_inf:.8f}"

    if class_probs is None or idx >= len(class_probs):
        return (
            f"#{idx+1}\nDM: {dm_val_cand:.1f}\nTime: {detection_time:.3f}s\n"
            f"{mjd_str}\nDet: {conf:.2f}",
            "lime",
        )

    # The SNR shown here is the detector's, never one recomputed from the
    # dedispersed waterfall: this label and the waterfall titles must agree.
    snr_I_wf = candidate_snr.intensity_at(idx)
    thresh_snr_I = config.SNR_THRESH
    thresh_class_I = config.CLASS_PROB
    snr_I_str = (f"{snr_I_wf:.1f}σ (≥{thresh_snr_I:.1f})" if snr_I_wf is not None
                 else f"N/A (≥{thresh_snr_I:.1f})")

    class_prob_I = class_probs[idx]
    is_burst_I = class_prob_I >= config.CLASS_PROB
    has_linear_classification = (class_probs_linear is not None
                                 and idx < len(class_probs_linear))
    logger.info("Candidate idx=%d: has_linear_classification=%s, "
                "snr_waterfall_linear is not None=%s",
                idx, has_linear_classification,
                candidate_snr.waterfall_linear is not None)

    head = (
        f"#{idx+1}\n"
        f"DM: {dm_val_cand:.1f} | Time: {detection_time:.3f}s\n"
        f"{mjd_str}\n"
        f"SNR_I: {snr_I_str} | Class_I: {class_prob_I:.2f} (≥{thresh_class_I:.2f})"
    )

    if not has_linear_classification:
        color = "lime" if is_burst_I else "orange"
        burst_status = "BURST" if is_burst_I else "NO BURST"
        return f"{head}\n{burst_status}", color

    class_prob_L = class_probs_linear[idx]
    thresh_class_L = getattr(config, 'CLASS_PROB_LINEAR', config.CLASS_PROB)
    is_burst_L = class_prob_L >= thresh_class_L
    # HF mode: green only when BOTH polarisations classify the candidate a burst.
    color = "lime" if (is_burst_I and is_burst_L) else "orange"
    if is_burst_I and is_burst_L:
        burst_status = "BURST (I+L)"
    elif is_burst_I and not is_burst_L:
        burst_status = "I:BURST L:NO"
    else:
        burst_status = "NO BURST"

    snr_L_wf = candidate_snr.linear_at(idx)
    thresh_snr_L = getattr(config, 'SNR_THRESH_LINEAR', config.SNR_THRESH)
    snr_L_str = f"{snr_L_wf:.1f}σ" if snr_L_wf is not None else "N/A"
    return (
        f"{head}"
        f"\nSNR_L: {snr_L_str} (≥{thresh_snr_L:.1f}) | "
        f"Class_L: {class_prob_L:.2f} (≥{thresh_class_L:.2f})"
        f"\n{burst_status}",
        color,
    )


def annotate_candidates(ax_det, *, top_conf, top_boxes, class_probs,
                        class_probs_linear, candidate_snr: CandidateSnr,
                        label_positions: list, slice_idx: int, slice_len: int,
                        slice_samples: Optional[int],
                        absolute_start_time: Optional[float],
                        candidate_times_abs) -> None:
    """Panel 1c: one box and one label per candidate on the detection panel."""
    if top_boxes is None:
        return

    for idx, (conf, box) in enumerate(zip(top_conf, top_boxes)):
        x1, y1, x2, y2 = map(int, box)
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

        effective_len_det = slice_samples if slice_samples is not None else slice_len
        dm_val_cand, t_sec_real, _ = extract_candidate_dm(
            center_x, center_y, effective_len_det)

        if candidate_times_abs is not None and idx < len(candidate_times_abs):
            detection_time = float(candidate_times_abs[idx])
        elif absolute_start_time is not None:
            detection_time = absolute_start_time + t_sec_real
        else:
            detection_time = (slice_idx * slice_len * config.TIME_RESO
                              * config.DOWN_TIME_RATE + t_sec_real)

        mjd_data = calculate_candidate_mjd(
            t_sec=detection_time, compute_bary=True, dm=dm_val_cand)

        label, color = _candidate_label(
            idx, conf, dm_val_cand, detection_time, mjd_data,
            class_probs=class_probs,
            class_probs_linear=class_probs_linear,
            candidate_snr=candidate_snr,
        )

        ax_det.add_patch(plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
        ))

        if idx < len(label_positions):
            label_x, label_y = label_positions[idx]
        else:
            label_x, label_y = center_x, y2 + 10

        ax_det.annotate(
            label,
            xy=(center_x, center_y),
            xytext=(label_x, label_y),
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
            fontsize=7,
            ha="center",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            zorder=10,  # above the title
        )


# ---------------------------------------------------------------------------
# The bottom row, multi-polarisation variant
# ---------------------------------------------------------------------------


def draw_multi_pol_row(fig, gs_bottom_row, *, dw_block, dw_linear, dw_circular,
                       dm_val: float, window: SliceWindow, thresh_snr,
                       top_conf, top_boxes, candidate_times_abs,
                       candidate_snr: CandidateSnr) -> None:
    """The HF pipeline's bottom row: dedispersed I, L and V side by side."""
    candidate_time_abs = None
    candidate_time_intensity = None
    candidate_snr_intensity_wf = None
    candidate_snr_linear_wf = None

    if top_boxes is not None and len(top_boxes) > 0:
        if top_conf is not None and len(top_conf) > 0:
            best_idx = int(np.argmax(top_conf))
            if candidate_times_abs is not None and best_idx < len(candidate_times_abs):
                candidate_time_abs = candidate_times_abs[best_idx]
                candidate_time_intensity = candidate_times_abs[best_idx]
            if (candidate_snr.waterfall_intensity is not None
                    and best_idx < len(candidate_snr.waterfall_intensity)):
                candidate_snr_intensity_wf = candidate_snr.waterfall_intensity[best_idx]
            if (candidate_snr.waterfall_linear is not None
                    and best_idx < len(candidate_snr.waterfall_linear)):
                candidate_snr_linear_wf = candidate_snr.waterfall_linear[best_idx]
        elif candidate_times_abs is not None and len(candidate_times_abs) > 0:
            candidate_time_abs = candidate_times_abs[0]
            candidate_time_intensity = candidate_times_abs[0]
            if (candidate_snr.waterfall_intensity is not None
                    and len(candidate_snr.waterfall_intensity) > 0):
                candidate_snr_intensity_wf = candidate_snr.waterfall_intensity[0]
            if (candidate_snr.waterfall_linear is not None
                    and len(candidate_snr.waterfall_linear) > 0):
                candidate_snr_linear_wf = candidate_snr.waterfall_linear[0]

    create_multi_pol_panels(
        fig=fig,
        gs_bottom_row=gs_bottom_row,
        dedisp_intensity=dw_block,
        dedisp_linear=dw_linear,
        dedisp_circular=dw_circular,
        dm_val=dm_val,
        slice_start_abs=window.start_abs,
        slice_end_abs=window.end_abs,
        freq_ds=window.freq_ds,
        time_reso_ds=window.time_reso_ds,
        thresh_snr=thresh_snr,
        candidate_time_abs=candidate_time_abs,
        candidate_time_intensity=candidate_time_intensity,
        candidate_snr_intensity_wf=candidate_snr_intensity_wf,
        candidate_snr_linear_wf=candidate_snr_linear_wf,
    )


# ---------------------------------------------------------------------------
# The bottom row, classic variant: raw waterfall left, dedispersed right
# ---------------------------------------------------------------------------


def raw_waterfall_window(*, window: SliceWindow, top_conf, top_boxes,
                         slice_len: int, slice_samples: Optional[int],
                         band_idx: int,
                         candidate_snr: CandidateSnr) -> RawWaterfallWindow:
    """Shift the slice window back by the best candidate's dispersion sweep.

    Only the classic bottom row draws a raw waterfall, so this only runs there.
    It used to run in both branches and have its result discarded in the
    multi-polarisation one.
    """
    start = window.start_abs
    end = window.end_abs
    candidate_snr_intensity = None

    if (top_boxes is not None and len(top_boxes) > 0
            and top_conf is not None and len(top_conf) > 0):
        best_idx = int(np.argmax(top_conf))
        x1, y1, x2, y2 = map(int, top_boxes[best_idx])
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        effective_len = slice_samples if slice_samples is not None else slice_len
        dm_best, _, _ = extract_candidate_dm(center_x, center_y, effective_len)

        candidate_snr_intensity = candidate_snr.intensity_for_best(best_idx)

        if dm_best > 0:
            freq_min, freq_max = get_band_frequency_range(band_idx)
            delta_t_max = K_DM_MS * dm_best * (1.0 / (freq_min ** 2) - 1.0 / (freq_max ** 2))
            margin = 0.1 * delta_t_max
            delta_t_max_correction = delta_t_max + margin
            start = window.start_abs - delta_t_max_correction
            end = window.end_abs - delta_t_max_correction
            logger.debug(
                f"[RAW WATERFALL CORRECTION] DM={dm_best:.1f}, Δt_max={delta_t_max:.3f}s")

    return RawWaterfallWindow(start=start, end=end,
                              candidate_snr_intensity=candidate_snr_intensity)


def draw_raw_snr_profile(ax, wf_block, *, raw_window: RawWaterfallWindow,
                         thresh_snr, top_conf, top_boxes,
                         candidate_times_abs) -> Optional[SnrProfilePanel]:
    """Panel 2: the SNR profile of the still-dispersed waterfall."""
    if wf_block is None or wf_block.size == 0:
        _no_data_profile(ax, 'No waterfall data\navailable', "No Raw Waterfall Data")
        return None

    snr_wf, _, _ = compute_snr_profile(wf_block)
    peak_snr_wf, _, peak_idx_wf = find_snr_peak(snr_wf)

    candidate_snr_intensity = raw_window.candidate_snr_intensity
    # The detector's SNR when there is one, the profile's own peak otherwise.
    # Used for the dot, for the text beside it and for the title, which is the
    # whole point: those three must not disagree.
    display_snr_wf = (candidate_snr_intensity if candidate_snr_intensity is not None
                      else peak_snr_wf)

    snr_samples = len(snr_wf)
    raw_end_snr = (raw_window.start
                   + snr_samples * config.TIME_RESO * config.DOWN_TIME_RATE)
    time_axis_wf = np.linspace(raw_window.start, raw_end_snr, snr_samples)

    ax.plot(time_axis_wf, snr_wf, color="royalblue", alpha=0.8, lw=1.5,
            label='SNR Profile')

    if thresh_snr is not None and config.SNR_SHOW_PEAK_LINES:
        above_thresh_wf = snr_wf >= thresh_snr
        if np.any(above_thresh_wf):
            ax.plot(time_axis_wf[above_thresh_wf], snr_wf[above_thresh_wf],
                    color=config.SNR_HIGHLIGHT_COLOR, lw=2, alpha=0.9)
        ax.axhline(y=thresh_snr, color=config.SNR_HIGHLIGHT_COLOR,
                   linestyle='--', alpha=0.7, linewidth=1)

    mark_time_wf = _mark_time(time_axis_wf, peak_idx_wf, candidate_snr_intensity,
                              top_conf, top_boxes, candidate_times_abs)
    _mark_display_snr(ax, mark_time_wf, display_snr_wf)

    ax.set_xlim(time_axis_wf[0], time_axis_wf[-1])
    ax.set_ylabel('SNR (σ)', fontsize=8, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([])
    ax.set_title(
        _profile_title("Raw Waterfall", display_snr_wf, candidate_snr_intensity,
                       top_conf, top_boxes, candidate_times_abs),
        fontsize=9, fontweight="bold", pad=12,
    )
    return SnrProfilePanel(time_axis=time_axis_wf, peak_idx=peak_idx_wf)


def draw_raw_waterfall(ax, wf_block, *, raw_window: RawWaterfallWindow,
                       window: SliceWindow,
                       profile: Optional[SnrProfilePanel]) -> None:
    """Panel 3: the still-dispersed waterfall, on its corrected time window."""
    if wf_block is None or wf_block.size == 0:
        _no_data_waterfall(ax, 'No waterfall data available')
        return

    wf_vmin, wf_vmax = percentile_limits(wf_block)
    ax.imshow(
        wf_block.T,
        origin="lower",
        cmap="mako",
        aspect="auto",
        vmin=wf_vmin,
        vmax=wf_vmax,
        extent=[raw_window.start, raw_window.end, window.freq_min, window.freq_max],
    )
    ax.set_xlim(raw_window.start, raw_window.end)
    ax.set_ylim(window.freq_min, window.freq_max)
    ax.set_yticks(window.freq_tick_positions)

    n_time_ticks = 5
    time_tick_positions = np.linspace(raw_window.start, raw_window.end, n_time_ticks)
    ax.set_xticks(time_tick_positions)
    ax.set_xticklabels([f"{t:.6f}" for t in time_tick_positions], rotation=45)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Frequency (MHz)", fontsize=9)

    # Was `if 'peak_snr_wf' in locals()`. The real question is whether the SNR
    # profile above this panel drew anything, and that is what `profile` says.
    if profile is not None and config.SNR_SHOW_PEAK_LINES:
        ax.axvline(x=profile.peak_time, color=config.SNR_HIGHLIGHT_COLOR,
                   linestyle='-', alpha=0.8, linewidth=2)


def draw_dedispersed_snr_profile(ax, dw_block, *, window: SliceWindow,
                                 candidate_snr_intensity, thresh_snr, 
                                 top_conf, top_boxes,
                                 candidate_times_abs) -> Optional[SnrProfilePanel]:
    """Panel 4: the SNR profile of the dedispersed waterfall."""
    if dw_block is None or dw_block.size == 0:
        _no_data_profile(ax, 'No dedispersed\ndata available', "No Dedispersed Data")
        return None

    snr_dw, _, _ = compute_snr_profile(dw_block)
    peak_snr_dw, _, peak_idx_dw = find_snr_peak(snr_dw)

    display_snr_dw = (candidate_snr_intensity if candidate_snr_intensity is not None
                      else peak_snr_dw)
    # No dispersion correction here: the block is already dedispersed, so it
    # sits on the slice's own time window.
    time_axis_dw = np.linspace(window.start_abs, window.end_abs, len(snr_dw))

    ax.plot(time_axis_dw, snr_dw, color="green", alpha=0.8, lw=1.5,
            label='Dedispersed SNR')

    if thresh_snr is not None and config.SNR_SHOW_PEAK_LINES:
        above_thresh_dw = snr_dw >= thresh_snr
        if np.any(above_thresh_dw):
            ax.plot(time_axis_dw[above_thresh_dw], snr_dw[above_thresh_dw],
                    color=config.SNR_HIGHLIGHT_COLOR, lw=2.5, alpha=0.9)
        ax.axhline(y=thresh_snr, color=config.SNR_HIGHLIGHT_COLOR,
                   linestyle='--', alpha=0.7, linewidth=1)

    mark_time_dw = _mark_time(time_axis_dw, peak_idx_dw, candidate_snr_intensity,
                              top_conf, top_boxes, candidate_times_abs)
    _mark_display_snr(ax, mark_time_dw, display_snr_dw)

    ax.set_xlim(window.start_abs, window.end_abs)
    ax.set_ylabel('SNR (σ)', fontsize=8, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([])
    ax.set_title(
        _profile_title("Dedispersed Waterfall", display_snr_dw,
                       candidate_snr_intensity, top_conf, top_boxes,
                       candidate_times_abs),
        fontsize=9, fontweight="bold", pad=12,
    )
    return SnrProfilePanel(time_axis=time_axis_dw, peak_idx=peak_idx_dw)


def draw_dedispersed_waterfall(ax, dw_block, *, window: SliceWindow,
                               profile: Optional[SnrProfilePanel]) -> None:
    """Panel 5: the dedispersed waterfall, on the slice's own time window."""
    if dw_block is None or dw_block.size == 0:
        _no_data_waterfall(ax, 'No dedispersed data available')
        return

    dw_vmin, dw_vmax = percentile_limits(dw_block)
    ax.imshow(
        dw_block.T,
        origin="lower",
        cmap="mako",
        aspect="auto",
        vmin=dw_vmin,
        vmax=dw_vmax,
        extent=[window.start_abs, window.end_abs, window.freq_min, window.freq_max],
    )
    ax.set_xlim(window.start_abs, window.end_abs)
    ax.set_ylim(window.freq_min, window.freq_max)

    n_time_ticks_dw = 5
    time_tick_positions_dw = np.linspace(window.start_abs, window.end_abs, n_time_ticks_dw)

    ax.set_yticks(window.freq_tick_positions)
    ax.set_yticklabels([f"{f:.0f}" for f in window.freq_tick_positions])
    ax.set_xticks(time_tick_positions_dw)
    ax.set_xticklabels([f"{t:.6f}" for t in time_tick_positions_dw], rotation=45)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Frequency (MHz)", fontsize=9)

    # Was `if 'peak_snr_dw' in locals()`; see draw_raw_waterfall.
    if profile is not None and config.SNR_SHOW_PEAK_LINES:
        ax.axvline(x=profile.peak_time, color=config.SNR_HIGHLIGHT_COLOR,
                   linestyle='-', alpha=0.8, linewidth=2)


def draw_classic_bottom_row(fig, gs_bottom_row, *, wf_block, dw_block,
                            window: SliceWindow, raw_window: RawWaterfallWindow,
                            thresh_snr, top_conf, top_boxes,
                            candidate_times_abs) -> None:
    """The classic pipeline's bottom row: raw waterfall left, dedispersed right.

    Each column is an SNR profile over a waterfall, and the waterfall needs the
    profile's peak, which is the only thing the two panels in a column share.
    """
    gs_waterfall_nested = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_bottom_row[0, 0], height_ratios=[1, 4], hspace=0.05
    )
    ax_prof_wf = fig.add_subplot(gs_waterfall_nested[0, 0])
    raw_profile = draw_raw_snr_profile(
        ax_prof_wf, wf_block,
        raw_window=raw_window, thresh_snr=thresh_snr, 
        top_conf=top_conf, top_boxes=top_boxes,
        candidate_times_abs=candidate_times_abs,
    )
    ax_wf = fig.add_subplot(gs_waterfall_nested[1, 0])
    draw_raw_waterfall(ax_wf, wf_block, raw_window=raw_window, window=window,
                       profile=raw_profile)

    gs_dedisp_nested = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_bottom_row[0, 1], height_ratios=[1, 4], hspace=0.05
    )
    ax_prof_dw = fig.add_subplot(gs_dedisp_nested[0, 0])
    dedisp_profile = draw_dedispersed_snr_profile(
        ax_prof_dw, dw_block, window=window,
        candidate_snr_intensity=raw_window.candidate_snr_intensity,
        thresh_snr=thresh_snr, 
        top_conf=top_conf, top_boxes=top_boxes,
        candidate_times_abs=candidate_times_abs,
    )
    ax_dw = fig.add_subplot(gs_dedisp_nested[1, 0])
    draw_dedispersed_waterfall(ax_dw, dw_block, window=window, profile=dedisp_profile)


# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------


def create_composite_plot(
    waterfall_block: np.ndarray,
    dedispersed_block: np.ndarray,
    img_rgb: np.ndarray,
    patch_img: np.ndarray,
    patch_start: float,
    dm_val: float,
    top_conf: Iterable,
    top_boxes: Iterable | None,
    class_probs: Iterable | None,
    slice_idx: int,
    time_slice: int,
    band_name: str,
    band_suffix: str,
    fits_stem: str,
    slice_len: int,
    normalize: bool = False,
    thresh_snr: Optional[float] = None,
    band_idx: int = 0,
    absolute_start_time: Optional[float] = None,
    chunk_idx: Optional[int] = None,
    slice_samples: Optional[int] = None,
    candidate_times_abs: Optional[Iterable[float]] = None,
    dedisp_block_linear: Optional[np.ndarray] = None,
    dedisp_block_circular: Optional[np.ndarray] = None,
    class_probs_linear: Optional[Iterable[float]] = None,  # NEW: Linear classification probs
    snr_waterfall_linear: Optional[Iterable[float | None]] = None,  # NEW: SNR from Linear waterfall
    snr_patch_linear: Optional[Iterable[float | None]] = None,  # NEW: SNR from dedispersed Linear patch
    snr_waterfall_intensity: Optional[Iterable[float | None]] = None,  # NEW: SNR from Intensity waterfall
    snr_patch_intensity: Optional[Iterable[float | None]] = None,  # NEW: SNR from dedispersed Intensity patch
) -> plt.Figure:
    """Assemble the composite figure from the panels above.

    Top: the DM-time detection image with one box and one label per candidate.
    Bottom, one of two rows:

    * multi-polarisation (HF pipeline, both ``dedisp_block_linear`` and
      ``dedisp_block_circular`` given) -- dedispersed Intensity, Linear and
      Circular side by side;
    * classic -- the still-dispersed waterfall and the dedispersed one, each
      under its own SNR profile.

    Unused parameters
    -----------------
    Six parameters are accepted and not read: ``patch_img``, ``patch_start``,
    ``time_slice``, ``band_suffix``, ``band_name`` and ``chunk_idx``. The first
    four fed the candidate-patch panel, which was removed from the classic
    layout; ``band_name`` only ever reached a band label that was computed and
    discarded; ``chunk_idx`` selected between two branches that built the same
    title string. They stay in the signature because it is the one both
    pipelines reach through ``save_composite_plot``, which still needs all six
    for the individual-component and polarization plots -- letting the two
    signatures diverge would buy nothing and cost a translation layer.
    """
    candidate_snr = CandidateSnr.collect(
        waterfall_intensity=snr_waterfall_intensity,
        patch_intensity=snr_patch_intensity,
        waterfall_linear=snr_waterfall_linear,
        patch_linear=snr_patch_linear,
    )

    freq_ds = np.mean(
        config.FREQ.reshape(
            config.FREQ_RESO // config.DOWN_FREQ_RATE,
            config.DOWN_FREQ_RATE,
        ),
        axis=1,
    )
    time_reso_ds = config.TIME_RESO * config.DOWN_TIME_RATE

    # Shared by the dispersed and the dedispersed waterfall panels. It used to be
    # computed inside the dispersed panel, which is guarded on wf_block, and read
    # inside the dedispersed one, which is guarded on dw_block: an empty waterfall
    # next to a non-empty dedispersed block raised NameError here.
    n_freq_ticks = 6
    freq_tick_positions = np.linspace(freq_ds.min(), freq_ds.max(), n_freq_ticks)

    multi_pol_mode = (dedisp_block_linear is not None and dedisp_block_circular is not None)
    if multi_pol_mode:
        logger.info("Composite plot: Multi-polarization mode (HF pipeline) - "
                    "Using 3 dedispersed waterfalls (I, L, V)")
        logger.debug("Multi-pol blocks: Linear shape=%s, Circular shape=%s",
                     dedisp_block_linear.shape, dedisp_block_circular.shape)
    else:
        logger.info("Composite plot: Standard mode - Using dispersed/dedispersed/patch layout")
        if dedisp_block_linear is None:
            logger.debug("dedisp_block_linear is None")
        if dedisp_block_circular is None:
            logger.debug("dedisp_block_circular is None")

    wf_block, dw_block, dw_linear, dw_circular = _prepare_blocks(
        waterfall_block, dedispersed_block, dedisp_block_linear, dedisp_block_circular,
        multi_pol_mode=multi_pol_mode, normalize=normalize,
    )

    if absolute_start_time is not None:
        slice_start_abs = absolute_start_time
    else:
        slice_start_abs = slice_idx * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE

    # The real block size, not slice_len: the last slice of a chunk is short.
    if wf_block is not None and wf_block.size > 0 and wf_block.ndim >= 1:
        real_samples = wf_block.shape[0]
    else:
        real_samples = slice_samples if slice_samples is not None else slice_len
    slice_end_abs = slice_start_abs + real_samples * config.TIME_RESO * config.DOWN_TIME_RATE

    window = SliceWindow(
        start_abs=slice_start_abs,
        end_abs=slice_end_abs,
        freq_ds=freq_ds,
        time_reso_ds=time_reso_ds,
        freq_tick_positions=freq_tick_positions,
    )

    fig = plt.figure(figsize=(14, 12))
    gs_main = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1], hspace=0.3, figure=fig)

    ax_det = draw_detection_panel(
        fig, gs_main, img_rgb,
        slice_start_abs=slice_start_abs, slice_end_abs=slice_end_abs,
        top_conf=top_conf, top_boxes=top_boxes, slice_len=slice_len,
    )
    label_positions = candidate_label_positions(fig, ax_det, top_boxes)
    annotate_candidates(
        ax_det,
        top_conf=top_conf, top_boxes=top_boxes,
        class_probs=class_probs, class_probs_linear=class_probs_linear,
        candidate_snr=candidate_snr, label_positions=label_positions,
        slice_idx=slice_idx, slice_len=slice_len, slice_samples=slice_samples,
        absolute_start_time=absolute_start_time,
        candidate_times_abs=candidate_times_abs,
    )

    if multi_pol_mode:
        gs_bottom_row = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=gs_main[1, 0], width_ratios=[1, 1, 1], wspace=0.3
        )
        draw_multi_pol_row(
            fig, gs_bottom_row,
            dw_block=dw_block, dw_linear=dw_linear, dw_circular=dw_circular,
            dm_val=dm_val, window=window, thresh_snr=thresh_snr,
            top_conf=top_conf, top_boxes=top_boxes,
            candidate_times_abs=candidate_times_abs, candidate_snr=candidate_snr,
        )
    else:
        gs_bottom_row = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs_main[1, 0], width_ratios=[1, 1], wspace=0.3
        )
        draw_classic_bottom_row(
            fig, gs_bottom_row,
            wf_block=wf_block, dw_block=dw_block, window=window,
            raw_window=raw_waterfall_window(
                window=window, top_conf=top_conf, top_boxes=top_boxes,
                slice_len=slice_len, slice_samples=slice_samples,
                band_idx=band_idx, candidate_snr=candidate_snr,
            ),
            thresh_snr=thresh_snr, 
            top_conf=top_conf, top_boxes=top_boxes,
            candidate_times_abs=candidate_times_abs,
        )

    fig.suptitle(f"{fits_stem} - Slice {slice_idx:03d}",
                 fontsize=14, fontweight="bold", y=0.97)

    return fig


def save_composite_plot(
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
    thresh_snr: Optional[float] = None,
    band_idx: int = 0,
    absolute_start_time: Optional[float] = None, 
    chunk_idx: Optional[int] = None,  
    slice_samples: Optional[int] = None,  
    candidate_times_abs: Optional[Iterable[float]] = None,
    generate_individual_plots: bool = True,
    individual_plots_dir: str = "individual_plots",
    dedisp_block_linear: Optional[np.ndarray] = None,
    dedisp_block_circular: Optional[np.ndarray] = None,
    class_probs_linear: Optional[Iterable[float]] = None,  # NEW: Linear classification probs
    snr_waterfall_linear: Optional[Iterable[float | None]] = None,  # NEW: SNR from Linear waterfall
    snr_patch_linear: Optional[Iterable[float | None]] = None,  # NEW: SNR from dedispersed Linear patch
    snr_waterfall_intensity: Optional[Iterable[float | None]] = None,  # NEW: SNR from Intensity waterfall
    snr_patch_intensity: Optional[Iterable[float | None]] = None,  # NEW: SNR from dedispersed Intensity patch
) -> None:
    """Save composite plot by creating the figure and saving it to file.
    
    Args:
        ... (existing parameters) ...
        generate_individual_plots: If True, also generate individual plot components
        individual_plots_dir: Directory name for individual plots (relative to composite plot location)
        dedisp_block_linear: Dedispersed waterfall in Linear polarization (for HF pipeline)
        dedisp_block_circular: Dedispersed waterfall in Circular polarization (for HF pipeline)
    """
    
                                 
    fig = create_composite_plot(
        waterfall_block=waterfall_block,
        dedispersed_block=dedispersed_block,
        img_rgb=img_rgb,
        patch_img=patch_img,
        patch_start=patch_start,
        dm_val=dm_val,
        top_conf=top_conf,
        top_boxes=top_boxes,
        class_probs=class_probs,
        slice_idx=slice_idx,
        time_slice=time_slice,
        band_name=band_name,
        band_suffix=band_suffix,
        fits_stem=fits_stem,
        slice_len=slice_len,
        normalize=normalize,
        thresh_snr=thresh_snr,
        band_idx=band_idx,
        absolute_start_time=absolute_start_time,
        chunk_idx=chunk_idx,
        slice_samples=slice_samples,
        candidate_times_abs=candidate_times_abs,
        dedisp_block_linear=dedisp_block_linear,
        dedisp_block_circular=dedisp_block_circular,
        class_probs_linear=class_probs_linear,  # NEW: Pass Linear probs
        snr_waterfall_linear=snr_waterfall_linear,  # NEW: Pass SNR from Linear waterfall
        snr_patch_linear=snr_patch_linear,  # NEW: Pass SNR from dedispersed Linear patch
        snr_waterfall_intensity=snr_waterfall_intensity,  # NEW: Pass SNR from Intensity waterfall
        snr_patch_intensity=snr_patch_intensity,  # NEW: Pass SNR from dedispersed Intensity patch
    )
    
                                    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
                               
    plt.savefig(out_path, dpi=config.PLOT_DPI, bbox_inches=config.PLOT_BBOX_INCHES,
                pad_inches=config.PLOT_PAD_INCHES, facecolor="white", edgecolor="none")
    plt.close(fig)
    
                                            
    if generate_individual_plots:
        try:
            from .plot_individual_components import generate_individual_plots
            
            generate_individual_plots(
                waterfall_block=waterfall_block,
                dedispersed_block=dedispersed_block,
                img_rgb=img_rgb,
                patch_img=patch_img,
                patch_start=patch_start,
                dm_val=dm_val,
                top_conf=top_conf,
                top_boxes=top_boxes,
                class_probs=class_probs,
                base_out_path=out_path,
                slice_idx=slice_idx,
                time_slice=time_slice,
                band_name=band_name,
                band_suffix=band_suffix,
                fits_stem=fits_stem,
                slice_len=slice_len,
                normalize=normalize,
                thresh_snr=thresh_snr,
                band_idx=band_idx,
                absolute_start_time=absolute_start_time,
                chunk_idx=chunk_idx,
                slice_samples=slice_samples,
                candidate_times_abs=candidate_times_abs,
                output_dir=individual_plots_dir,
                class_probs_linear=class_probs_linear,  # NEW: Pass Linear probs
                dedisp_block_linear=dedisp_block_linear,  # NEW: Pass Linear polarization block
                dedisp_block_circular=dedisp_block_circular,  # NEW: Pass Circular polarization block
            )
        except Exception as e:
            logger.warning(f"Could not generate individual plots: {e}")
    
    # Generate polarization time series plots for each candidate
    if candidate_times_abs is not None and len(candidate_times_abs) > 0:
        try:
            # `config` is imported at module scope; re-importing it here made it
            # a function-local name, so every earlier reference to it in this
            # function raised UnboundLocalError.
            from .plot_polarization_timeseries import save_polarization_timeseries_plot

            logger.info(f"Generating polarization time series plots for {len(candidate_times_abs)} candidate(s)")
            
            # Calculate frequency array
            freq_ds = np.mean(
                config.FREQ.reshape(
                    config.FREQ_RESO // config.DOWN_FREQ_RATE,
                    config.DOWN_FREQ_RATE,
                ),
                axis=1,
            )
            time_reso_ds = config.TIME_RESO * config.DOWN_TIME_RATE
            
            # Calculate slice time boundaries
            if absolute_start_time is not None:
                slice_start_abs = absolute_start_time
            else:
                slice_start_abs = slice_idx * slice_len * time_reso_ds
            
            real_samples = slice_samples if slice_samples is not None else slice_len
            slice_end_abs = slice_start_abs + real_samples * time_reso_ds
            
            # Determine polarization mode
            has_multipol = (dedisp_block_linear is not None) or (dedisp_block_circular is not None)
            pol_mode = "all" if has_multipol else "intensity"
            
            # CRITICAL: Prepare normalized blocks EXACTLY as in create_composite_plot (lines 178-204)
            # This ensures the waterfall data matches exactly what's shown in the composite plot
            dw_block = dedispersed_block.copy() if dedispersed_block is not None and dedispersed_block.size > 0 else None
            dw_linear = None
            dw_circular = None
            if has_multipol:
                if dedisp_block_linear is not None and dedisp_block_linear.size > 0:
                    dw_linear = dedisp_block_linear.copy()
                if dedisp_block_circular is not None and dedisp_block_circular.size > 0:
                    dw_circular = dedisp_block_circular.copy()
            
            # Apply the same normalization as create_composite_plot
            if normalize:
                dw_block = normalize_block(dw_block)
                if has_multipol:
                    dw_linear = normalize_block(dw_linear)
                    dw_circular = normalize_block(dw_circular)
            
            # Generate plot for each candidate
            for cand_idx, cand_time_abs in enumerate(candidate_times_abs):
                # Determine output path - EXACTLY same structure as individual_plots
                # individual_plots uses: base_out_path.parent / output_dir / f"chunk_{chunk_idx:03d}" / f"slice_{slice_idx:03d}"
                # where base_out_path is the composite plot path
                # So polarization_timeseries should be at the same level as individual_plots
                if chunk_idx is not None:
                    # Match exact format: chunk_000 (not chunk_0)
                    pol_dir = out_path.parent / "polarization_timeseries" / f"chunk_{chunk_idx:03d}" / f"slice_{slice_idx:03d}"
                else:
                    pol_dir = out_path.parent / "polarization_timeseries" / f"slice_{slice_idx:03d}"
                
                pol_dir.mkdir(parents=True, exist_ok=True)
                
                # Create filename
                pol_filename = f"{fits_stem}_slice{slice_idx:03d}_cand{cand_idx:02d}_t{cand_time_abs:.3f}s_pol.png"
                pol_path = pol_dir / pol_filename
                
                # Generate and save plot
                # IMPORTANT: Pass the normalized blocks (dw_block, dw_linear, dw_circular)
                # that were prepared above using EXACTLY the same normalization as create_composite_plot
                # This ensures the waterfall matches exactly what's shown in the composite plot
                save_polarization_timeseries_plot(
                    dedisp_intensity=dw_block if dw_block is not None else dedispersed_block,
                    dedisp_linear=dw_linear,
                    dedisp_circular=dw_circular,
                    dm_val=dm_val,
                    candidate_time_abs=cand_time_abs,
                    slice_start_abs=slice_start_abs,
                    slice_end_abs=slice_end_abs,
                    freq_ds=freq_ds,
                    time_reso_ds=time_reso_ds,
                    fits_filename=fits_stem,
                    slice_idx=slice_idx,
                    pol_mode=pol_mode,
                    out_path=pol_path,
                    normalize=False,  # Data is already normalized above using same logic as create_composite_plot
                )
                logger.info(f"✓ Polarization time series plot saved: {pol_path}")
        except Exception as e:
            logger.warning(f"Could not generate polarization time series plots: {e}", exc_info=True)
    else:
        logger.debug(f"No candidate_times_abs provided or empty (candidate_times_abs={candidate_times_abs})")
                                                                                  
