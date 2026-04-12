# This module implements dedispersion routines.

"""Dedispersion helpers using GPU/CPU."""
from __future__ import annotations

                          
import logging

                     
import numpy as np
from ..analysis.science_metrics import K_DM_MS

try:
    from numba import cuda
except ImportError:
    cuda = None

               
from ..config import config

                              
try:
    import torch
except ImportError:
    torch = None

              
logger = logging.getLogger(__name__)


def delay_from_dm(dm: float, freq_mhz: float) -> float:
    """
    Calculate dispersion delay in seconds for a given DM and frequency.
    
    Based on PRESTO's delay_from_dm() function.
    
    Args:
        dm: Dispersion Measure in pc cm^-3
        freq_mhz: Frequency in MHz
    
    Returns:
        Delay in seconds
    """
    if freq_mhz == 0.0:
        return 0.0
    return (K_DM_MS * dm * (freq_mhz ** -2)) / 1000.0


def calculate_dispersion_bandwidth_delay(
    dm_max: float,
    freq_low_mhz: float,
    freq_high_mhz: float
) -> float:
    """
    Calculate the maximum dispersion delay across the bandwidth.
    
    Based on PRESTO's BW_ddelay calculation.
    This is the difference in delay between the lowest and highest frequencies.
    
    Args:
        dm_max: Maximum DM to consider
        freq_low_mhz: Lowest frequency in MHz
        freq_high_mhz: Highest frequency in MHz
    
    Returns:
        Maximum delay difference in seconds
    """
    delay_low = delay_from_dm(dm_max, freq_low_mhz)
    delay_high = delay_from_dm(dm_max, freq_high_mhz)
    return delay_low - delay_high


if cuda is not None:
    @cuda.jit
    def _de_disp_gpu(dm_time, data, freq, index, dm_values, mid_channel):
        x, y = cuda.grid(2)
        if x < dm_time.shape[1] and y < dm_time.shape[2]:
            total_val = 0.0
            total_cnt = 0
            mid_val = 0.0
            DM = dm_values[x]

            for idx in index:
                delay = (
                    K_DM_MS
                    * DM
                    * ((freq[idx]) ** -2 - (freq[-1] ** -2))
                    / (config.TIME_RESO * config.DOWN_TIME_RATE * 1000.0)
                )
                pos = int(round(delay) + y)
                if 0 <= pos < data.shape[0]:
                    total_val += data[pos, idx]
                    total_cnt += 1
                    if idx == mid_channel:
                        mid_val = data[pos, idx]

            if total_cnt > 0:
                dm_time[0, x, y] = total_val / total_cnt
            else:
                dm_time[0, x, y] = 0.0

            dm_time[1, x, y] = mid_val
            dm_time[2, x, y] = dm_time[0, x, y] - mid_val


def _d_dm_time_cpu(
    data: np.ndarray,
    height: int,
    width: int,
    dm_min: float,
    dm_max: float,
    freq_ds: np.ndarray,
    dm_values: np.ndarray | None = None,
) -> np.ndarray:
    """CPU dedispersion with exposure normalization and edge handling.

    Uses Numba prange for parallel execution across DM values when available,
    falling back to sequential NumPy otherwise.
    """
    time_reso = float(config.TIME_RESO)
    down_time_rate = int(config.DOWN_TIME_RATE)
    if dm_values is None:
        dm_values = np.linspace(dm_min, dm_max, height).astype(np.float32)
    return _d_dm_time_cpu_core(
        data, height, width, dm_min, dm_max, freq_ds,
        time_reso, down_time_rate, dm_values.astype(np.float32),
    )


try:
    from numba import njit, prange as _prange

    @njit(parallel=True, cache=True, fastmath=True)
    def _d_dm_time_cpu_core(
        data, height, width, dm_min, dm_max, freq_ds,
        time_reso, down_time_rate, dm_values,
    ):
        out = np.zeros((3, height, width), dtype=np.float32)
        nchan_ds = freq_ds.shape[0]
        mid_channel = nchan_ds // 2
        n_time = data.shape[0]
        n_chan = data.shape[1]
        freq_max = freq_ds.max()
        inv_tr_dt = np.float32(1.0 / (time_reso * down_time_rate * 1000.0))

        for i in _prange(height):
            DM = np.float32(dm_values[i])

            total_series = np.zeros(width, dtype=np.float32)
            count_series = np.zeros(width, dtype=np.int32)
            mid_series = np.zeros(width, dtype=np.float32)

            for j in range(nchan_ds):
                if j >= n_chan:
                    continue
                delay = int(np.rint(np.float64(K_DM_MS) * np.float64(DM)
                            * (np.float64(freq_ds[j]) ** -2 - np.float64(freq_max) ** -2)
                            * np.float64(inv_tr_dt)))

                if delay >= 0:
                    src_lo = delay
                    dst_lo = 0
                else:
                    src_lo = 0
                    dst_lo = -delay
                src_hi = delay + width
                if src_hi > n_time:
                    src_hi = n_time
                if src_hi <= src_lo:
                    continue
                length = src_hi - src_lo
                dst_hi = dst_lo + length

                for k in range(length):
                    total_series[dst_lo + k] += data[src_lo + k, j]
                    count_series[dst_lo + k] += 1

                if j == mid_channel:
                    for k in range(length):
                        mid_series[dst_lo + k] = data[src_lo + k, j]

            for k in range(width):
                norm = np.float32(count_series[k]) if count_series[k] > 0 else np.float32(1.0)
                out[0, i, k] = total_series[k] / norm
                out[1, i, k] = mid_series[k]
                out[2, i, k] = out[0, i, k] - mid_series[k]

        return out

except ImportError:
    def _d_dm_time_cpu_core(
        data, height, width, dm_min, dm_max, freq_ds,
        time_reso, down_time_rate, dm_values,
    ):
        """Pure-NumPy fallback (sequential) when Numba is not installed."""
        out = np.zeros((3, height, width), dtype=np.float32)
        nchan_ds = freq_ds.shape[0]
        mid_channel = nchan_ds // 2
        for i in range(height):
            DM = dm_values[i]
            delays = (
                K_DM_MS * DM
                * (freq_ds ** -2 - freq_ds.max() ** -2)
                / (time_reso * down_time_rate * 1000.0)
            ).round().astype(np.int64)

            total_series = np.zeros(width, dtype=np.float32)
            count_series = np.zeros(width, dtype=np.int32)
            mid_series = np.zeros(width, dtype=np.float32)

            for j in range(nchan_ds):
                d = delays[j]
                if d >= 0:
                    src_lo = d
                    dst_lo = 0
                else:
                    src_lo = 0
                    dst_lo = -d
                src_hi = d + width
                if src_hi > data.shape[0]:
                    src_hi = data.shape[0]
                if src_hi <= src_lo or j >= data.shape[1]:
                    continue
                length = src_hi - src_lo
                dst_hi = dst_lo + length

                total_series[dst_lo:dst_hi] += data[src_lo:src_hi, j]
                count_series[dst_lo:dst_hi] += 1

                if j == mid_channel:
                    mid_series[dst_lo:dst_hi] = data[src_lo:src_hi, j]

            norm = count_series.astype(np.float32)
            norm[norm <= 0] = 1.0
            out[0, i] = total_series / norm
            out[1, i] = mid_series
            out[2, i] = out[0, i] - out[1, i]

        return out


def d_dm_time_g(data: np.ndarray, height: int, width: int, chunk_size: int = 128, dm_min: float = None, dm_max: float = None) -> np.ndarray:
    # CRITICAL: Validate memory before creating full result array
    # This cube can be HUGE: 3 × height × width × 4 bytes
    cube_size_bytes = 3 * height * width * 4
    cube_size_gb = cube_size_bytes / (1024**3)
    if cube_size_gb > 4.0:  # Warn if > 4GB
        logger.warning(
            f"Creating large DM-time cube: {cube_size_gb:.2f} GB "
            f"(height={height}, width={width:,}). "
            f"This will use significant GPU/CPU memory."
        )
    
    result = np.zeros((3, height, width), dtype=np.float32)
    
                                                        
    if dm_min is None:
        dm_min = config.DM_min
    if dm_max is None:
        dm_max = config.DM_max
    try:
        from ..core.pipeline_parameters import calculate_dm_values
        dm_values_full = calculate_dm_values(dm_min, dm_max).astype(np.float32)
        if dm_values_full.size != height:
            dm_values_full = np.linspace(dm_min, dm_max, height, dtype=np.float32)
    except Exception:
        dm_values_full = np.linspace(dm_min, dm_max, height, dtype=np.float32)
    
                                                                     
    try:
        if getattr(config, 'PREWHITEN_BEFORE_DM', False):
            eps = np.float32(1e-6)
            # Ensure data is writable float32 for in-place ops
            if not data.flags.writeable or data.dtype != np.float32:
                data = data.astype(np.float32)
            mean_ch = np.mean(data, axis=0)
            std_ch = np.std(data, axis=0)
            std_ch = np.where(std_ch < eps, np.float32(1.0), std_ch)
            # In-place operations: 0 extra copies (was 3x peak with broadcast)
            data -= mean_ch
            data /= std_ch
    except Exception as e:
        logger.warning("Prewhitening failed: %s", e)
    
                                                 
    if config.FREQ is None or config.FREQ.size == 0:
        raise ValueError("config.FREQ invalid during dedispersion (empty)")
    if config.FREQ_RESO == 0 or config.DOWN_FREQ_RATE == 0:
        raise ValueError(f"Invalid frequency parameters: FREQ_RESO={config.FREQ_RESO}, DOWN_FREQ_RATE={config.DOWN_FREQ_RATE}")
    if (config.FREQ_RESO // config.DOWN_FREQ_RATE) * config.DOWN_FREQ_RATE != config.FREQ_RESO:
                                                       
        n_groups = config.FREQ_RESO // config.DOWN_FREQ_RATE
        config.FREQ_RESO = n_groups * config.DOWN_FREQ_RATE
        config.FREQ = config.FREQ[:config.FREQ_RESO]
    freq_values = np.mean(config.FREQ.reshape(config.FREQ_RESO // config.DOWN_FREQ_RATE, config.DOWN_FREQ_RATE), axis=1)

                                                                
    use_cuda_device = str(getattr(config, 'DEVICE', 'cpu')).startswith('cuda')

    if torch is not None and torch.cuda.is_available() and use_cuda_device:
        try:
            return _d_dm_time_torch_gpu(data, height, width, dm_min, dm_max, freq_values, dm_values_full)
        except Exception as e:
            logger.warning("Torch GPU dedispersion failed (%s); attempting Numba GPU...", e)

    # Only attempt Numba-CUDA when the pipeline is actually configured for CUDA.
    # On CPU-only Windows environments, importing numba.cuda may succeed even
    # though NVVM/driver pieces are missing, which only produces noisy fallbacks.
    if cuda is None or not use_cuda_device:
        return _d_dm_time_cpu(data, height, width, dm_min, dm_max, freq_values, dm_values_full)

    try:
        if hasattr(cuda, "is_available") and not cuda.is_available():
            return _d_dm_time_cpu(data, height, width, dm_min, dm_max, freq_values, dm_values_full)
    except Exception:
        return _d_dm_time_cpu(data, height, width, dm_min, dm_max, freq_values, dm_values_full)

    try:
        logger.info("Attempting GPU dedispersion...")

        freq_gpu = cuda.to_device(freq_values)
        nchan_ds = config.FREQ_RESO // config.DOWN_FREQ_RATE
        index_values = np.arange(0, nchan_ds)
        mid_channel = nchan_ds // 2
        index_gpu = cuda.to_device(index_values)
        data_gpu = cuda.to_device(data)
        
        # Use default chunk_size (128) - don't modify dynamically to avoid breaking detection
        # The chunking is already optimized in the GPU kernels
        
        for start_dm in range(0, height, chunk_size):
            end_dm = min(start_dm + chunk_size, height)
            current_height = end_dm - start_dm
            
                                                            
            dm_values = dm_values_full[start_dm:end_dm].astype(np.float32)
            dm_values_gpu = cuda.to_device(dm_values)
            
            dm_time_gpu = cuda.to_device(np.zeros((3, current_height, width), dtype=np.float32))
            nthreads = (8, 128)
            nblocks = (current_height // nthreads[0] + 1, width // nthreads[1] + 1)
            _de_disp_gpu[nblocks, nthreads](dm_time_gpu, data_gpu, freq_gpu, index_gpu, dm_values_gpu, mid_channel)
            cuda.synchronize()
            result[:, start_dm:end_dm, :] = dm_time_gpu.copy_to_host()
            del dm_time_gpu, dm_values_gpu
            cuda.synchronize()
        
        # CRITICAL: Free GPU arrays after processing
        del freq_gpu, index_gpu, data_gpu
        cuda.synchronize()
        
        logger.info("GPU dedispersion completed successfully")
        return result
    except (cuda.cudadrv.driver.CudaAPIError, Exception) as e:
        logger.warning("GPU dedispersion failed (%s); falling back to CPU", e)
        # CRITICAL: Clean up GPU memory on failure
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logger.debug("GPU cleanup after failure: %s", e)
        return _d_dm_time_cpu(data, height, width, dm_min, dm_max, freq_values, dm_values_full)


def _d_dm_time_torch_gpu(
    data_np: np.ndarray,
    height: int,
    width: int,
    dm_min: float,
    dm_max: float,
    freq_ds_np: np.ndarray,
    dm_values_np: np.ndarray | None = None,
) -> np.ndarray:
    """GPU dedispersion using PyTorch with vectorized channel batching.

    Instead of iterating per-channel (for j in range(C)), processes channels
    in batches of ``chan_batch`` to maximise GPU parallelism while controlling
    VRAM usage.
    """
    device = torch.device('cuda')
    data_t = torch.from_numpy(data_np).to(device=device, dtype=torch.float32)
    T, C = data_t.shape
    freq_ds = torch.from_numpy(freq_ds_np.astype(np.float32)).to(device)
    if dm_values_np is None:
        dm_values = torch.linspace(float(dm_min), float(dm_max), steps=height, device=device, dtype=torch.float32)
    else:
        dm_values = torch.from_numpy(dm_values_np.astype(np.float32)).to(device)
    time_reso = float(config.TIME_RESO * config.DOWN_TIME_RATE)

    out0 = torch.zeros((height, width), device=device, dtype=torch.float32)
    out1 = torch.zeros((height, width), device=device, dtype=torch.float32)
    out2 = torch.zeros((height, width), device=device, dtype=torch.float32)

    mid_channel = C // 2
    base = torch.arange(width, device=device, dtype=torch.int64)

    # Adaptive DM chunk size based on available VRAM
    vram_free = torch.cuda.mem_get_info(0)[0]
    # Each DM-chunk of size D×C×W needs ~D*C*W*4 bytes for index/val tensors
    bytes_per_dm = C * width * 4 * 3  # idx + vals + valid masks
    dm_chunk = max(16, min(256, int(vram_free * 0.3 / max(1, bytes_per_dm))))

    # Channel batch size: process multiple channels at once
    chan_batch = min(C, 32)

    for start in range(0, height, dm_chunk):
        end = min(start + dm_chunk, height)
        D = end - start
        dms = dm_values[start:end]

        # Compute all delays: (D, C)
        delays = (K_DM_MS * dms[:, None] * (freq_ds[None, :] ** -2 - freq_ds.max() ** -2) / (time_reso * 1000.0))
        delays = torch.round(delays).to(dtype=torch.int64)

        acc = torch.zeros((D, width), device=device, dtype=torch.float32)
        cnt = torch.zeros((D, width), device=device, dtype=torch.int32)
        mid_vals = torch.zeros((D, width), device=device, dtype=torch.float32)

        for j0 in range(0, C, chan_batch):
            j1 = min(j0 + chan_batch, C)
            B = j1 - j0

            # idx shape: (D, B, width) — gather indices for this channel batch
            ch_delays = delays[:, j0:j1]  # (D, B)
            idx = ch_delays[:, :, None] + base[None, None, :]  # (D, B, W)
            valid = (idx >= 0) & (idx < T)  # (D, B, W)
            safe_idx = idx.clamp(0, max(T - 1, 0))

            # Gather channel data for the batch: data_t[:, j0:j1] is (T, B)
            ch_data = data_t[:, j0:j1]  # (T, B)
            # Flatten and gather: safe_idx -> (D*B*W,)
            flat_idx = safe_idx.reshape(-1)
            # For each channel in the batch, gather independently
            # ch_data.T is (B, T), we need vals (D, B, W)
            vals = ch_data.T[:, None, :].expand(B, D, T)  # not memory-efficient
            # Better approach: use advanced indexing
            # vals[d, b, w] = ch_data[safe_idx[d, b, w], b]
            # Reshape for gather: index into T dimension
            vals = torch.zeros((D, B, width), device=device, dtype=torch.float32)
            for bi in range(B):
                ch_ts = ch_data[:, bi]  # (T,)
                bi_idx = safe_idx[:, bi, :]  # (D, W)
                vals[:, bi, :] = ch_ts[bi_idx]

            vals = torch.where(valid, vals, torch.zeros_like(vals))

            # Sum over the batch dimension (channels)
            acc += vals.sum(dim=1)  # (D, W)
            cnt += valid.to(torch.int32).sum(dim=1)  # (D, W)

            # Check if mid_channel is in this batch
            if j0 <= mid_channel < j1:
                bi = mid_channel - j0
                mid_vals = vals[:, bi, :]

            del idx, valid, safe_idx, vals, ch_data

        cnt_f = cnt.to(torch.float32)
        cnt_f = torch.where(cnt_f <= 0, torch.ones_like(cnt_f), cnt_f)
        block0 = acc / cnt_f
        out0[start:end] = block0
        out1[start:end] = mid_vals
        out2[start:end] = block0 - mid_vals

    result = torch.stack([out0, out1, out2], dim=0).detach().cpu().numpy().astype(np.float32)

    del data_t, freq_ds, dm_values, out0, out1, out2, base
    torch.cuda.empty_cache()

    return result

def dedisperse_patch(
    data: np.ndarray,
    freq_down: np.ndarray,
    dm: float,
    sample: int,
    patch_len: int = 512,
) -> tuple[np.ndarray, int]:
    """Dedisperse ``data`` at ``dm`` around ``sample`` and return the patch.

    Returns
    -------
    patch : np.ndarray
        Dedispersed patch of shape (patch_len, n_freq).
        If data is smaller than patch_len, the patch will be padded with zeros.
    start : int
        Start sample used on the original data array.
    """
    if data.shape[0] == 0:
        # Return zero patch if no data
        return np.zeros((patch_len, freq_down.size), dtype=np.float32), 0
    
    delays = (
        K_DM_MS
        * dm
        * (freq_down ** -2 - freq_down.max() ** -2)
        / (config.TIME_RESO * config.DOWN_TIME_RATE * 1000.0)
    ).round().astype(np.int64)
    max_delay = int(delays.max())
    
    # CRITICAL: Adapt patch_len if data is too small (edge cases at chunk boundaries)
    # With the new slice_len calculation, this should rarely happen, but we keep it
    # as a safety measure for edge cases at chunk boundaries
    available_samples = data.shape[0] - max_delay
    if available_samples < patch_len:
        # Use available samples, but ensure minimum size
        actual_patch_len = max(32, available_samples)  # Minimum 32 samples
        if actual_patch_len < patch_len:
            logger.warning(
                f"WARNING: Data insufficient for full patch. Adapting patch_len from {patch_len} "
                f"to {actual_patch_len} (data has {data.shape[0]} samples, max_delay={max_delay}). "
                f"This may occur at chunk boundaries. Consider increasing chunk overlap or slice_len."
            )
    else:
        actual_patch_len = patch_len
    
    start = sample - actual_patch_len // 2
    if start < 0:
        start = 0
    if start + actual_patch_len + max_delay > data.shape[0]:
        start = max(0, data.shape[0] - (actual_patch_len + max_delay))
    
    # Extract segment (may be smaller than needed)
    segment_end = min(start + actual_patch_len + max_delay, data.shape[0])
    segment = data[start : segment_end]
    
    # Create patch - will pad if segment is too small
    patch = np.zeros((patch_len, freq_down.size), dtype=np.float32)
    
    # Calculate how much we can actually fill
    if segment.shape[0] > max_delay:
        segment_available = segment.shape[0] - max_delay
        fill_len = min(actual_patch_len, segment_available, patch_len)
        
        if fill_len > 0:
            # Fill what we can, centered in the patch
            for idx in range(freq_down.size):
                delay = delays[idx]
                src_start = delay
                src_end = min(src_start + fill_len, segment.shape[0])
                
                if src_end > src_start and src_start < segment.shape[0]:
                    # Center the available data in the patch
                    dst_start = max(0, (patch_len - fill_len) // 2)
                    actual_fill = src_end - src_start
                    dst_end = dst_start + actual_fill
                    
                    if dst_end <= patch_len and src_end <= segment.shape[0]:
                        patch[dst_start:dst_end, idx] = segment[src_start:src_end, idx]
    
    return patch, start

def dedisperse_block(
    data: np.ndarray,
    freq_down: np.ndarray,
    dm: float,
    start: int,
    block_len: int,
) -> np.ndarray:
    """Dedisperse a continuous block of data.

    Parameters
    ----------
    data : np.ndarray
        Time--frequency array already downsampled in frequency.
    freq_down : np.ndarray
        Array of downsampled frequency values.
    dm : float
        Dispersion measure used for the correction.
    start : int
        Starting sample of the block within ``data``.
    block_len : int
        Number of samples to include in the output block.

    Returns
    -------
    np.ndarray
        Dedispersed block of shape ``(block_len, n_freq)`` whose time span
        matches that of the original slice.
    """

    delays = (
        K_DM_MS
        * dm
        * (freq_down ** -2 - freq_down.max() ** -2)
        / (config.TIME_RESO * config.DOWN_TIME_RATE * 1000.0)
    ).round().astype(np.int64)

                                   
    if config.DEBUG_FREQUENCY_ORDER:
        logger.debug(f"[DEBUG DEDISPERSION] DM: {dm:.2f} pc cm⁻³")
        logger.debug(f"[DEBUG DEDISPERSION] freq_down shape: {freq_down.shape}")
        logger.debug(f"[DEBUG DEDISPERSION] First 3 freq_down: {freq_down[:3]}")
        logger.debug(f"[DEBUG DEDISPERSION] Last 3 freq_down: {freq_down[-3:]}")
        logger.debug(f"[DEBUG DEDISPERSION] freq_down.max(): {freq_down.max():.2f} MHz")
        logger.debug(f"[DEBUG DEDISPERSION] First 3 delays: {delays[:3]} samples")
        logger.debug(f"[DEBUG DEDISPERSION] Last 3 delays: {delays[-3:]} samples")
        logger.debug(f"[DEBUG DEDISPERSION] max_delay: {delays.max()} samples")
        logger.debug(f"[DEBUG DEDISPERSION] Expected dedispersion: HIGH freq arrive first (delay=0), LOW freq arrive later (delay>0)")
        if freq_down[0] < freq_down[-1]:              
            expected_delay_pattern = "DECREASING delays (from max to 0)"
        else:               
            expected_delay_pattern = "INCREASING delays (from 0 to max)"
        logger.debug(f"[DEBUG DEDISPERSION] Expected delay pattern: {expected_delay_pattern}")
        logger.debug("[DEBUG DEDISPERSION] " + "="*60)

    max_delay = int(delays.max())
    
    # Ensure we have enough data for the requested block
    if start + block_len + max_delay > data.shape[0]:
        # Adjust start to fit within bounds, but keep original block_len
        start = max(0, data.shape[0] - (block_len + max_delay))
        # If still not enough data, we'll handle it in the loop below

    # Extract segment with safety bounds
    segment_end = min(start + block_len + max_delay, data.shape[0])
    segment = data[start:segment_end]

    # Always create block with the requested size (maintain expected structure)
    block = np.zeros((block_len, freq_down.size), dtype=np.float32)
    
    for idx in range(freq_down.size):
        delay = delays[idx]
        # Calculate how much data we can actually copy
        if delay < segment.shape[0]:
            available_samples = min(block_len, segment.shape[0] - delay)
            if available_samples > 0:
                block[:available_samples, idx] = segment[delay:delay + available_samples, idx]
        # Remaining samples stay as zeros (padding)
    return block
