# This module tests slice alignment across data sources.

"""Slicing alignment test for PSRFITS and filterbank files.

Usage:
  python src/scripts/test_slicing_alignment.py --fits Data/raw/<file>.fits --slice-sec 0.3
  # Optionally compare a .fil file as well
  python src/scripts/test_slicing_alignment.py --fil Data/raw/<file>.fil --slice-sec 0.3

Verifies that:
  - Chunks are contiguous in sample indices: 0..N, N..2N, ...
  - The time increment between chunk starts matches slice-sec exactly
  - The total duration is not exceeded
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

import numpy as np

from src.input.fits_handler import get_obparams, stream_fits
from src.input.filterbank_handler import get_obparams_fil, stream_fil
from src.config import config


def _assert_close(a: float, b: float, tol: float, what: str) -> None:
    if not (abs(a - b) <= tol):
        raise AssertionError(f"{what}: expected {b}, obtained {a}, tol={tol}")


def test_psrfits_slicing(fits_path: str, slice_sec: float) -> None:
                                   
    get_obparams(fits_path)
    tbin = float(config.TIME_RESO)
    nsamples = int(config.FILE_LENG)
    if tbin <= 0:
        raise AssertionError(f"Invalid TBIN: {tbin}")
    chunk_samples = int(round(slice_sec / tbin))
    if chunk_samples <= 0:
        raise AssertionError(
            f"Invalid chunk_samples: {chunk_samples} (slice_sec={slice_sec}, tbin={tbin})"
        )

    print(f"[PSRFITS] tbin={tbin:.9f} s, nsamples={nsamples}, slice={slice_sec}s, chunk_samples={chunk_samples}")

                                                    
    tstart_mjd = getattr(config, "TSTART_MJD_CORR", None)
    if tstart_mjd is None:
        tstart_mjd = getattr(config, "TSTART_MJD", None)
    if tstart_mjd is None:
        raise AssertionError("Unable to determine TSTART_MJD(_CORR)")

    last_start = None
    last_end = None
    expected_start = 0
    expected_delta_t_mjd = slice_sec / 86400.0
    tol_time = max(tbin / 86400.0, 1e-12)                       

    num_chunks = 0
    total_emitted = 0

    for block, meta in stream_fits(fits_path, chunk_samples=chunk_samples, overlap_samples=0):
        start_sample = int(meta["start_sample"])                              
        end_sample = int(meta["end_sample"])                               
                               
        if last_end is None:
            _assert_close(start_sample, 0, 0, "initial start_sample")
        else:
            _assert_close(start_sample, expected_start, 0, "non-contiguous start_sample")

                                                      
        expected_len = min(chunk_samples, nsamples - start_sample)
        actual_len = end_sample - start_sample
        if start_sample + expected_len < nsamples:
            _assert_close(actual_len, expected_len, 0, "unexpected chunk size")

                                              
        if last_start is not None:
            tprev = tstart_mjd + (last_start * tbin) / 86400.0
            tcurr = tstart_mjd + (start_sample * tbin) / 86400.0
            _assert_close(tcurr - tprev, expected_delta_t_mjd, tol_time, "time delta between chunks")

        num_chunks += 1
        total_emitted += actual_len
        last_start = start_sample
        last_end = end_sample
        expected_start += chunk_samples

                                  
    if total_emitted > nsamples:
        raise AssertionError(
            f"More samples emitted than available: {total_emitted}>{nsamples}"
        )

    print(f"[PSRFITS] OK: {num_chunks} chunks, total_emitted={total_emitted}/{nsamples}")


def test_filterbank_slicing(fil_path: str, slice_sec: float) -> None:
    get_obparams_fil(fil_path)
    tsamp = float(config.TIME_RESO)
    nsamples = int(config.FILE_LENG)
    if tsamp <= 0:
        raise AssertionError(f"Invalid tsamp: {tsamp}")
    chunk_samples = int(round(slice_sec / tsamp))
    if chunk_samples <= 0:
        raise AssertionError(
            f"Invalid chunk_samples: {chunk_samples} (slice_sec={slice_sec}, tsamp={tsamp})"
        )

    print(f"[FIL] tsamp={tsamp:.9f} s, nsamples={nsamples}, slice={slice_sec}s, chunk_samples={chunk_samples}")

    last_start = None
    last_end = None
    expected_start = 0
    num_chunks = 0
    total_emitted = 0

    for block, meta in stream_fil(fil_path, chunk_samples=chunk_samples, overlap_samples=0):
        start_sample = int(meta["start_sample"])                              
        end_sample = int(meta["end_sample"])                               
                     
        if last_end is None:
            _assert_close(start_sample, 0, 0, "initial start_sample")
        else:
            _assert_close(start_sample, expected_start, 0, "non-contiguous start_sample")

        expected_len = min(chunk_samples, nsamples - start_sample)
        actual_len = end_sample - start_sample
        if start_sample + expected_len < nsamples:
            _assert_close(actual_len, expected_len, 0, "unexpected chunk size")

        num_chunks += 1
        total_emitted += actual_len
        last_start = start_sample
        last_end = end_sample
        expected_start += chunk_samples

    if total_emitted > nsamples:
        raise AssertionError(
            f"More samples emitted than available: {total_emitted}>{nsamples}"
        )

    print(f"[FIL] OK: {num_chunks} chunks, total_emitted={total_emitted}/{nsamples}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Slicing test for PSRFITS and filterbank files")
    parser.add_argument("--fits", type=str, default=None, help="Path to .fits (PSRFITS) file")
    parser.add_argument("--fil", type=str, default=None, help="Path to .fil (filterbank) file")
    parser.add_argument("--slice-sec", type=float, default=0.3, help="Slice duration in seconds")
    args = parser.parse_args()

    if args.fits is None and args.fil is None:
        print("You must specify --fits or --fil")
        sys.exit(2)

    try:
        if args.fits:
            test_psrfits_slicing(args.fits, args.slice_sec)
        if args.fil:
            test_filterbank_slicing(args.fil, args.slice_sec)
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)

    print("PASS: slicing verified")


if __name__ == "__main__":
    main()


