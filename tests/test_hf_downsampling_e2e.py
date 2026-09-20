"""A full high-frequency run with ``DOWN_TIME_RATE > 1``, which nothing else does.

Why this module exists
----------------------
``tests/test_e2e_pipeline.py::TestTemporalDownsamplingEndToEnd`` closed this hole
on the LOW-frequency path and its docstring says what the hole is: at
``DOWN_TIME_RATE = 1`` the sampling interval and the *effective* interval after
decimation are the same number, so every expression that must choose between
them is unobservable, and swapping one for the other passes the whole suite.

The high-frequency driver was left in exactly that state. Every test that
reaches it -- ``test_hf_e2e.py``, ``test_hf_band_function.py``,
``test_golden_csv.py`` -- pins the rate at 1. That matters more here than on the
low-frequency path, because this driver has arithmetic the other one does not:

    chunk_start_time_sec = metadata["start_sample"] * obs_meta.time_reso
    absolute_start_time  = chunk_start_time_sec + start_idx * dt_ds

``start_sample`` counts RAW samples (``psrfits_chunking.chunk_metadata`` builds
it from the span, untouched by decimation) so the first term must use the RAW
resolution; ``start_idx`` indexes the DECIMATED block, so the second must use
the effective one. The two terms need *different* resolutions, and at rate 1
nothing can tell them apart. It also owns the hand-written multi-polarisation
decimation block, which no test had ever executed with anything to do.

What is real here
-----------------
Everything. The file, the header load, ``select_pipeline_path``, the chunk plan,
the multi-polarisation reader, both decimators, the chunk loop, the REAL slice
processor, the REAL band function, the candidate writer and the CSV. Nothing is
stubbed except ``save_all_plots`` -- rendering figures is slow and this module
asserts on numbers, not pictures.

The single-chunk trap
---------------------
The low-frequency class warns that with one chunk every ``start_sample`` is 0,
so ``start_sample * resolution`` is 0 whichever resolution is used and the
mix-up cancels exactly. On this path the trap is worse, because it can be
sprung by the decimation itself: ``file_driver.resolve_memory_safe_chunk_size``
computes

    min_required_raw = required_min_size * max(1, config.DOWN_TIME_RATE)

and a request below that is *silently upgraded* to the memory-safe size, which
on any normal machine is the whole file -- one chunk, ``start_sample`` 0, test
vacuous. Measured on this file: the minimum is 613 raw samples at rate 1, 1224
at rate 2, 2448 at rate 4 and 4896 at rate 8. ``CHUNK_SAMPLES`` is 4096, which
clears rates 1, 2 and 4 and is why 8 is not in ``RATES``. Every run asserts it
really got more than one chunk, so a machine where this reasoning fails says so
instead of passing.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.config import config

# 350 GHz, so the bow-tie criterion collapses and the dispatch picks this
# driver on the physics rather than on a flag. Raising DOWN_TIME_RATE only
# makes that decision more certain: should_use_hf_pipeline divides by
# time_reso * down_time_rate, so a larger rate lowers the ratio further.
FCH1_MHZ = 350_000.0
FOFF_MHZ = -10.0
NCHAN = 32
TSAMP = 1.0e-3

#: 128 x 64 = 8192 samples = 8.192 s. Two chunks of 4096.
NSUBINT = 128
NSBLK = 64
TOTAL_SAMPLES = NSUBINT * NSBLK
CHUNK_SAMPLES = 4096

#: The burst sits at 6.0 s -- raw sample 6000, which is inside chunk 1, whose
#: ``start_sample`` is 4096. That non-zero start is the whole assertion.
BURST_T = 6.0
BURST_SAMPLE = int(round(BURST_T / TSAMP))
BURST_CHUNK = BURST_SAMPLE // CHUNK_SAMPLES
DM_INJECTED = 300.0

#: Measured, not guessed: at 512 ms slices this burst is recovered with an SNR
#: in the thousands at rates 1, 2 and 4, so detection is not marginal and a
#: missing candidate means something broke rather than that the data was thin.
SLICE_DURATION_MS = 512.0
BURST_AMPLITUDE = 400.0
BURST_WIDTH_SAMPLES = 8

#: Rate 8 is excluded: its minimum chunk (4896 raw) exceeds CHUNK_SAMPLES and
#: the run would collapse to a single chunk. See the module docstring.
RATES = (1, 2, 4)

#: ``calculate_slice_len_from_duration`` never returns fewer decimated samples
#: than this: 512 for the classifier's patch plus a 100-sample dispersion
#: margin. It is also what ``resolve_memory_safe_chunk_size`` multiplies by the
#: rate to get its raw minimum, which is why the two numbers in the docstring
#: above are 612 x rate.
PATCH_FLOOR_SAMPLES = 612

#: A 10 GHz band with a 310 MHz span, which is where SPEC-HF-002 actually
#: decides: at DM 500 the sweep is 1.349 ms, so 1.35 effective samples at rate 1
#: and 0.67 at rate 2. The dispatch sends it to this driver at both rates
#: (1.35 and 0.67 are each under the bow-tie's 2.0), so the only thing the rate
#: changes is whether the DM-time cube gets built.
RESOLVED_FCH1_MHZ = 10_000.0
RESOLVED_FOFF_MHZ = -10.0


def _write_file(tmp_path: Path) -> Path:
    from src.input.fits_handler import get_obparams
    from tests.synthetic_psrfits import write_psrfits

    path = tmp_path / "hf_downsampled.fits"
    write_psrfits(
        path, nsubint=NSUBINT, nsblk=NSBLK, nchan=NCHAN, npol=4,
        pol_type="IQUV", tsamp=TSAMP, fch1=FCH1_MHZ, foff=FOFF_MHZ,
        dm=DM_INJECTED, burst_time_s=BURST_T, dtype="float32",
        amplitude=BURST_AMPLITUDE, burst_width_samples=BURST_WIDTH_SAMPLES,
    )
    # Reads the header into config. Must run before the rate is set, because it
    # rewrites the observation parameters.
    get_obparams(str(path))
    return path


#: Enough free RAM for the requested chunk to be honoured, reported as a fixed
#: number so the chunk geometry is a property of this test and not of how busy
#: the machine is. See ``_pin_the_hardware``.
PINNED_RAM_BYTES = 64 * 1024 ** 3


def _pin_the_hardware(monkeypatch) -> None:
    """Stop the chunk geometry from depending on free memory.

    ``calculate_memory_safe_chunk_size`` sizes the chunk from
    ``detect_hardware().ram_available_bytes``, and
    ``resolve_memory_safe_chunk_size`` then *reduces* a request that does not
    fit. Run on an idle machine this file gets the 4096 samples it asks for;
    run inside the full suite, after other modules have taken memory, the same
    request came back as 3672 -- which moves every chunk boundary, every
    ``start_sample`` and so every absolute time this module asserts on. The
    failure looks like a bug in the arithmetic and is not one.

    Only the memory figures are replaced; the rest of the probe (CPU, platform,
    disk) is whatever the machine really reports. The GPU is pinned off because
    the budget takes a different branch when VRAM is available, and a test whose
    chunk size depends on which machine ran it is the thing being fixed here.
    """
    import dataclasses

    from src.core import hardware_profile

    real = hardware_profile.detect_hardware

    def _fixed(results_dir=None):
        return dataclasses.replace(
            real(results_dir),
            ram_total_bytes=PINNED_RAM_BYTES,
            ram_available_bytes=PINNED_RAM_BYTES,
            gpu_available=False,
            gpu_vram_total_bytes=0,
            gpu_vram_available_bytes=0,
        )

    monkeypatch.setattr(hardware_profile, "detect_hardware", _fixed)


def _configure(monkeypatch, save_dir: Path, rate: int) -> None:
    for key, value in {
        "DM_min": 0.0, "DM_max": 500.0,
        "DOWN_TIME_RATE": rate, "DOWN_FREQ_RATE": 1,
        "SLICE_DURATION_MS": SLICE_DURATION_MS,
        "AUTO_HIGH_FREQ_PIPELINE": True,
        "SAVE_ONLY_BURST": False,
        "FORCE_PLOTS": False,
        "USE_MULTI_BAND": False,
        "RESULTS_DIR": save_dir,
        "SNR_THRESH": 5.0,
        # Phase 3a on so a candidate reaches the CSV at all: with every
        # classification phase off, decide_candidate has no verdict and writes
        # no row (audit P1-10). No model is needed -- classify_patch falls back
        # to a deterministic SNR sigmoid when cls_model is None.
        "ENABLE_INTENSITY_CLASSIFICATION": True,
        "ENABLE_LINEAR_CLASSIFICATION": False,
        "ENABLE_LINEAR_VALIDATION": False,
        # Both are defaults, pinned because the decimation assertions depend on
        # them. "sum" is the mode in which the shared decimator and the
        # driver's own reshape compute the same thing; under
        # "phase_preserving"/"snr_preserving" the intensity block takes a
        # max-over-phase path the raw block does not, and the two stop agreeing
        # at any rate above 1. "intensity" is what makes plane 0 of the raw
        # block the same quantity as the intensity block at all.
        "TEMPORAL_DOWNSAMPLING_MODE": "sum",
        "POLARIZATION_MODE": "intensity",
        # Also defaults, and also pinned rather than inherited: the slice
        # boundaries come from the planner, the DM grid decides the cube height
        # and so the memory budget, and the two RAM factors scale the budget
        # directly. Any of them left to whatever an earlier module set moves the
        # chunk geometry underneath this one.
        "USE_PLANNED_CHUNKING": True,
        "DM_GRID_MODE": "legacy_uniform",
        "MAX_RAM_FRACTION": 0.25,
        "OVERHEAD_FACTOR": 1.3,
        # The one that actually bit. `calculate_memory_safe_chunk_size` clamps
        # the chunk to whole `SLICE_LEN * DOWN_TIME_RATE` blocks of this, so a
        # 4000 left behind by another module turns a requested 4096 into
        # (4000 // 612) * 612 = 3672 and retiles the file. The high-frequency
        # driver passes `max_chunk_limit=None` to `plan_chunking`, so this is
        # the only place the value reaches it -- and the only reason a number
        # this large is not simply "no limit".
        "MAX_CHUNK_SAMPLES": 1_000_000,
        # No source block: the barycentric columns are reported unavailable
        # rather than guessed, and nothing reaches for the network.
        "SOURCE_RA": None, "SOURCE_DEC": None, "REF_FREQ_MHZ": None,
        "OBSERVATORY": None, "EPHEMERIS": None,
    }.items():
        monkeypatch.setattr(config, key, value, raising=False)


def _stokes_i_difference(block, block_raw):
    """How far the two decimated blocks disagree about Stokes I.

    ``block`` is the intensity waterfall, decimated by the shared
    ``downsample_data``. ``block_raw`` is the multi-polarisation array,
    decimated by the driver's own reshape. Plane 0 of the second is the same
    quantity as the first, and they must stay the same numbers: Phase 2 reads
    the linear SNR out of one at the index where it found the intensity peak in
    the other.

    Returns ``None`` when the shapes make the comparison impossible, which is a
    finding rather than a pass.
    """
    import numpy as np

    if block_raw is None or block_raw.ndim != 3:
        return None
    if block_raw.shape[0] != block.shape[0] or block_raw.shape[2] != block.shape[-1]:
        return None
    return float(np.max(np.abs(np.asarray(block) - np.asarray(block_raw[:, 0, :]))))


def _spy_on_slices(monkeypatch) -> list[dict]:
    """Record what the driver hands each slice, then run the real thing.

    The CSV shows the answer; this shows the arithmetic that produced it. It is
    what separates "the burst came back at the right second" from "the driver
    used the right resolution to say so" -- two claims that coincide on correct
    code and diverge the moment either resolution is substituted for the other.
    """
    from src.core import high_freq_pipeline as hfp

    real = hfp.process_slice_with_multiple_bands_high_freq
    calls: list[dict] = []

    def _recording(**kwargs):
        block = kwargs["block"]
        block_raw = kwargs.get("block_raw")
        calls.append({
            "chunk_idx": kwargs["chunk_idx"],
            "j": kwargs["j"],
            "slice_len": kwargs["slice_len"],
            "time_reso_ds": kwargs["time_reso_ds"],
            "absolute_start_time": kwargs["absolute_start_time"],
            "slice_start_idx": kwargs["slice_start_idx"],
            "slice_end_idx": kwargs["slice_end_idx"],
            "block_samples": block.shape[0],
            "block_raw_shape": None if block_raw is None else tuple(block_raw.shape),
            # The two decimators compared on the arrays the driver actually
            # built, not on a reimplementation of them in the test. ``None``
            # when the shapes rule the comparison out, which is itself a
            # failure the assertions below name.
            "stokes_i_max_diff": _stokes_i_difference(block, block_raw),
        })
        return real(**kwargs)

    monkeypatch.setattr(
        hfp, "process_slice_with_multiple_bands_high_freq", _recording
    )
    return calls


def _spy_on_chunks(monkeypatch) -> list[dict]:
    """Record the raw-sample geometry of every chunk the reader yields.

    Patched on ``fits_handler`` rather than on the driver: the driver imports
    the reader inside the function, so the name it resolves is the one on the
    source module.
    """
    from src.input import fits_handler

    real_stream = fits_handler.stream_fits_multi_pol
    chunks: list[dict] = []

    def _recording(file_name, chunk_samples, overlap_samples=0):
        for block, block_raw, metadata, pol_type in real_stream(
            file_name, chunk_samples, overlap_samples=overlap_samples
        ):
            chunks.append({
                "chunk_idx": metadata["chunk_idx"],
                "start_sample": metadata["start_sample"],
                "end_sample": metadata["end_sample"],
                "overlap_left": metadata["overlap_left"],
                "overlap_right": metadata["overlap_right"],
                "block_samples": block.shape[0],
                "raw_shape": None if block_raw is None else tuple(block_raw.shape),
            })
            yield block, block_raw, metadata, pol_type

    monkeypatch.setattr(fits_handler, "stream_fits_multi_pol", _recording)
    return chunks


def _read_rows(save_dir: Path, stem: str) -> list[dict]:
    from src.output.candidate_manager import CandidateWriter

    CandidateWriter.flush_all()
    path = save_dir / "Summary" / stem / f"{stem}.candidates.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run_at(rate: int, tmp_root: Path) -> dict:
    """One complete run at *rate*, with its own patches undone on the way out.

    A single ``MonkeyPatch`` shared across the rates does not work here and the
    way it fails is quiet: each run's spy would wrap the *previous* run's spy
    instead of the production function, so every earlier rate's record would go
    on collecting the later rates' calls and the per-rate assertions would read
    a mixture. Undoing between runs is what keeps each record its own.
    """
    from src.core.file_driver import select_pipeline_path
    from src.core.pipeline import _process_file_chunked
    from src.visualization import visualization_unified

    monkeypatch = pytest.MonkeyPatch()
    try:
        save_dir = tmp_root / f"rate{rate}"
        save_dir.mkdir(parents=True, exist_ok=True)
        fits_path = _write_file(tmp_root)
        _configure(monkeypatch, save_dir, rate)
        _pin_the_hardware(monkeypatch)

        # Figures are slow and this module asserts on numbers. Patched on the
        # module the slice processor imports from, which is where its local
        # import resolves the name.
        monkeypatch.setattr(
            visualization_unified, "save_all_plots", lambda *a, **k: None
        )

        slices = _spy_on_slices(monkeypatch)
        chunks = _spy_on_chunks(monkeypatch)

        use_hf, reason = select_pipeline_path()
        result = _process_file_chunked(None, None, fits_path, save_dir, CHUNK_SAMPLES)
        rows = _read_rows(save_dir, fits_path.stem)
    finally:
        monkeypatch.undo()

    return {
        "rate": rate,
        "use_hf": use_hf,
        "dispatch_reason": reason,
        "result": result,
        "rows": rows,
        "slices": slices,
        "chunks": chunks,
    }


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """One real run per rate, shared by every assertion in this module.

    Module-scoped because each run streams a real file through the real driver;
    doing it once per test would multiply an already heavy job by the number of
    assertions. What the tests read back is plain data, so the per-test config
    isolation in ``conftest.py`` is unaffected.
    """
    pytest.importorskip("astropy")
    # Without it ``stream_fits_multi_pol`` raises rather than falling back, so
    # the driver would return an ERROR result and every assertion here would
    # fail for a missing dependency instead of skipping.
    pytest.importorskip("your")
    return {
        rate: _run_at(rate, tmp_path_factory.mktemp(f"hfds{rate}"))
        for rate in RATES
    }


def _burst_row(run: dict) -> dict:
    assert run["rows"], (
        f"rate {run['rate']}: the high-frequency driver produced no candidates. "
        f"status={run['result'].get('status')}"
    )
    return max(run["rows"], key=lambda r: float(r["snr_pre_dedisp"] or 0.0))


# --------------------------------------------------------------------------- #
# the run is what it claims to be
# --------------------------------------------------------------------------- #
class TestTheRunIsNotVacuous:
    """Four ways this test could pass while testing nothing. Each is checked."""

    @pytest.mark.parametrize("rate", RATES)
    def test_the_dispatch_really_chose_the_high_frequency_driver(self, runs, rate):
        run = runs[rate]
        assert run["use_hf"], run["dispatch_reason"]
        assert run["result"]["status"].endswith("HIGH_FREQ"), run["result"]["status"]

    @pytest.mark.parametrize("rate", RATES)
    def test_more_than_one_chunk_was_processed(self, runs, rate):
        """The single-chunk trap. With one chunk every ``start_sample`` is 0 and
        the resolution it is multiplied by stops mattering."""
        run = runs[rate]
        indices = sorted({c["chunk_idx"] for c in run["chunks"]})
        assert len(indices) > 1, (
            f"rate {rate}: the run collapsed to {len(indices)} chunk(s); "
            f"chunk_samples={CHUNK_SAMPLES} was probably upgraded by "
            f"resolve_memory_safe_chunk_size, which scales its minimum by the rate"
        )

    @pytest.mark.parametrize("rate", RATES)
    def test_the_requested_chunk_size_was_honoured(self, runs, rate):
        """Said directly, because when it is not honoured every other failure in
        this module is a confusing offset rather than a clear message. The
        memory budget can *reduce* a request as well as upgrade one -- inside
        the full suite this file was handed 3672 samples instead of 4096 purely
        because earlier modules had taken memory."""
        run = runs[rate]
        starts = sorted(c["start_sample"] for c in run["chunks"])
        assert starts == [i * CHUNK_SAMPLES for i in range(len(starts))], (
            f"rate {rate}: the chunk geometry is not the requested "
            f"{CHUNK_SAMPLES}-sample tiling; got starts {starts}"
        )

    @pytest.mark.parametrize("rate", RATES)
    def test_the_burst_sits_in_a_chunk_that_does_not_start_at_zero(self, runs, rate):
        run = runs[rate]
        carrier = [c for c in run["chunks"] if c["chunk_idx"] == BURST_CHUNK]
        assert carrier, f"rate {rate}: chunk {BURST_CHUNK} was never streamed"
        assert carrier[0]["start_sample"] == BURST_CHUNK * CHUNK_SAMPLES
        assert carrier[0]["start_sample"] > 0

    @pytest.mark.parametrize("rate", RATES)
    def test_the_decimation_actually_happened(self, runs, rate):
        """If the rate were ignored the decimated block would keep its raw
        length, and every 'it still works at rate 2' assertion below would be an
        assertion about rate 1.

        The expected length is exact rather than approximate, because getting it
        exactly right means the two conversions agree: the block is decimated by
        floor division and the overlap is converted by ``ceil`` division
        (``calculate_overlap_decimated``), and the valid window is what is left
        after trimming the second from the first.
        """
        run = runs[rate]
        carrier = [c for c in run["chunks"] if c["chunk_idx"] == BURST_CHUNK][0]
        decimated = [s for s in run["slices"] if s["chunk_idx"] == BURST_CHUNK]
        assert decimated, f"rate {rate}: chunk {BURST_CHUNK} produced no slices"

        overlap_left_ds = -(-carrier["overlap_left"] // rate)
        overlap_right_ds = -(-carrier["overlap_right"] // rate)
        expected = carrier["block_samples"] // rate - overlap_left_ds - overlap_right_ds
        assert decimated[0]["block_samples"] == expected


# --------------------------------------------------------------------------- #
# the two resolutions
# --------------------------------------------------------------------------- #
class TestTheTwoResolutions:
    @pytest.mark.parametrize("rate", RATES)
    def test_the_slice_resolution_is_the_effective_one(self, runs, rate):
        """``time_reso_ds`` is handed to the slice processor and from there to
        every time the band function computes. It is the decimated interval, not
        the sampling one."""
        run = runs[rate]
        assert run["slices"], f"rate {rate}: no slice was processed"
        for call in run["slices"]:
            assert call["time_reso_ds"] == pytest.approx(TSAMP * rate, rel=1e-12)

    @pytest.mark.parametrize("rate", RATES)
    def test_the_chunk_offset_uses_the_raw_resolution(self, runs, rate):
        """The other half, and the one that cannot be seen at rate 1: the
        chunk's contribution to a slice's absolute time is
        ``start_sample * TIME_RESO`` with ``start_sample`` in RAW samples. Using
        the effective resolution here would multiply it by the rate."""
        run = runs[rate]
        for call in run["slices"]:
            expected = (
                call["chunk_idx"] * CHUNK_SAMPLES * TSAMP
                + call["slice_start_idx"] * TSAMP * rate
            )
            assert call["absolute_start_time"] == pytest.approx(expected, abs=1e-9), (
                f"rate {rate}, chunk {call['chunk_idx']}, slice {call['j']}: "
                "the slice's absolute start time is not "
                "start_sample * TIME_RESO + start_idx * TIME_RESO * DOWN_TIME_RATE"
            )

    @pytest.mark.parametrize("rate", RATES)
    def test_the_slice_len_argument_is_pinned_at_the_patch_floor(self, runs, rate):
        """``calculate_slice_len_from_duration`` computes
        ``round(SLICE_DURATION_MS / 1000 / dt_ds)`` and then clamps it up to 612
        decimated samples -- 512 for the classifier's patch plus 100 for the
        dispersion margin. At this module's durations the clamp wins at every
        rate, so the number handed down does not move with the rate at all.
        """
        expected = round(SLICE_DURATION_MS / (TSAMP * rate * 1000.0))
        assert expected < PATCH_FLOOR_SAMPLES, (
            "this test assumes the patch floor dominates; raise "
            "SLICE_DURATION_MS above it and the assertion below is wrong"
        )
        assert runs[rate]["slices"][0]["slice_len"] == PATCH_FLOOR_SAMPLES

    @pytest.mark.parametrize("rate", RATES)
    def test_the_real_slice_boundaries_come_from_the_planner_and_do_scale(
        self, runs, rate
    ):
        """And the clamped number is not what cuts the block.

        ``USE_PLANNED_CHUNKING`` is true by default, so ``plan_slices`` ignores
        the ``slice_len`` it is given and calls ``plan_slices_for_chunk`` with
        ``SLICE_DURATION_MS`` and the effective interval instead. The widths it
        returns DO scale with the rate -- 512, 256 and 128 decimated samples --
        which is the same 512 ms of sky at each. So the two lengths disagree by
        design in the shipped configuration, and it is the planner's that
        decides where a candidate's slice begins.
        """
        widths = {
            call["slice_end_idx"] - call["slice_start_idx"]
            for call in runs[rate]["slices"]
        }
        nominal = round(SLICE_DURATION_MS / (TSAMP * rate * 1000.0))
        # The planner adjusts the count to divide the chunk, so the last slice
        # of a chunk can be one sample short of the rest.
        assert widths <= {nominal, nominal - 1}, widths
        assert PATCH_FLOOR_SAMPLES not in widths, (
            "the planner is no longer deciding the boundaries; the clamped "
            "slice_len is, and the slice duration has stopped being constant"
        )

    def test_the_two_slice_lengths_disagree_and_that_is_the_shipped_behaviour(
        self, runs
    ):
        """Recorded rather than asserted to be right. ``slice_len`` reaches
        ``process_slice_with_multiple_bands_high_freq`` as 612 at every rate and
        is then used for a plot count and handed to ``plan_slices``, which
        discards it. The band function receives ``end_idx - start_idx``, the
        planner's width, so the number that matters is never the number that was
        computed. This fails the day the two are reconciled -- update it then.
        """
        for rate in RATES:
            call = runs[rate]["slices"][0]
            planned = call["slice_end_idx"] - call["slice_start_idx"]
            assert call["slice_len"] != planned, (
                f"rate {rate}: slice_len and the planned width now agree "
                f"({planned}); the two-lengths situation this pins is gone"
            )

    def test_the_three_rates_did_not_all_produce_the_same_run(self, runs):
        """A guard on the parametrisation itself: if ``DOWN_TIME_RATE`` stopped
        reaching the driver, all three runs would be the same run and every
        test above would still pass. The slice LENGTH is the same at all three
        by design, so it cannot be the discriminator -- the resolution and the
        decimated block size are."""
        resolutions = {runs[r]["slices"][0]["time_reso_ds"] for r in RATES}
        blocks = {runs[r]["slices"][0]["block_samples"] for r in RATES}
        assert len(resolutions) == len(RATES), resolutions
        assert len(blocks) == len(RATES), blocks


# --------------------------------------------------------------------------- #
# the burst
# --------------------------------------------------------------------------- #
class TestTheBurstKeepsItsAbsoluteTime:
    @pytest.mark.parametrize("rate", RATES)
    def test_the_burst_comes_back_at_the_second_it_was_injected(self, runs, rate):
        """The assertion that carries the module. Substituting either resolution
        for the other in the chunk loop moves this by whole seconds: at rate 2
        the carrying chunk starts at 4.096 s, so using the effective resolution
        there would report the burst near 10 s instead of 6 s, and using the raw
        one for the in-chunk offset would report it near 5 s."""
        run = runs[rate]
        row = _burst_row(run)
        tolerance = 8 * TSAMP * rate
        assert float(row["t_sec_dm_time"]) == pytest.approx(BURST_T, abs=tolerance), (
            f"rate {rate}: the burst's absolute time moved under temporal "
            "decimation; the usual cause is TIME_RESO used where "
            "TIME_RESO * DOWN_TIME_RATE belongs, or the reverse"
        )

    @pytest.mark.parametrize("rate", RATES)
    def test_it_is_reported_from_the_chunk_that_contains_it(self, runs, rate):
        row = _burst_row(runs[rate])
        assert int(row["chunk_id"]) == BURST_CHUNK

    def test_every_rate_agrees_with_every_other_on_when_it_happened(self, runs):
        """Pairwise, not against a constant: the rates must agree with each
        other to within the coarsest sample, which is a tighter statement than
        each one separately landing near 6 s."""
        times = {r: float(_burst_row(runs[r])["t_sec_dm_time"]) for r in RATES}
        coarsest = TSAMP * max(RATES)
        for a in RATES:
            for b in RATES:
                assert times[a] == pytest.approx(times[b], abs=4 * coarsest), times

    @pytest.mark.parametrize("rate", RATES)
    def test_the_two_time_columns_agree_with_each_other(self, runs, rate):
        row = _burst_row(runs[rate])
        assert float(row["t_sec_waterfall"]) == pytest.approx(
            float(row["t_sec_dm_time"]), abs=4 * TSAMP * rate
        )

    @pytest.mark.parametrize("rate", RATES)
    def test_the_topocentric_mjd_follows_the_detection_time(self, runs, rate):
        """A second, independent expression of the same instant: if the arrival
        time were wrong by the decimation factor the MJD would be wrong with it,
        and this pins that they are computed from the same number."""
        row = _burst_row(runs[rate])
        expected = config.TSTART_MJD + float(row["t_sec_dm_time"]) / 86400.0
        assert float(row["mjd_utc"]) == pytest.approx(expected, abs=1e-9)

    @pytest.mark.parametrize("rate", RATES)
    def test_no_candidate_is_reported_outside_the_file(self, runs, rate):
        span = TOTAL_SAMPLES * TSAMP
        for row in runs[rate]["rows"]:
            assert 0.0 <= float(row["t_sec_dm_time"]) <= span, row


# --------------------------------------------------------------------------- #
# the multi-polarisation decimator, which only this driver has
# --------------------------------------------------------------------------- #
class TestMultiPolarisationDecimation:
    """``_process_file_chunked_high_freq`` decimates the RAW (time, npol, chan)
    array with its own hand-written reshape instead of calling the shared
    downsampler, and wraps it in ``except Exception: block_raw_ds = None``. At
    rate 1 the reshape is an identity and the block cannot be wrong; these are
    the first assertions that see it do anything.
    """

    @pytest.mark.parametrize("rate", RATES)
    def test_the_raw_block_reaches_the_slice_processor_decimated(self, runs, rate):
        run = runs[rate]
        call = [s for s in run["slices"] if s["chunk_idx"] == BURST_CHUNK][0]
        assert call["block_raw_shape"] is not None, (
            f"rate {rate}: the multi-polarisation block was dropped. The "
            "decimation block swallows failures into block_raw_ds = None, so "
            "this is what a silent exception there looks like"
        )
        assert call["block_raw_shape"] == (call["block_samples"], 4, NCHAN)

    @pytest.mark.parametrize("rate", RATES)
    def test_the_linear_column_is_filled_so_the_block_was_usable(self, runs, rate):
        """Shape alone does not prove the samples line up. ``snr_waterfall_linear``
        is read out of the decimated raw block at the same index as the intensity
        peak, so a filled column means the two blocks agree on where the burst
        is."""
        row = _burst_row(runs[rate])
        assert row["snr_waterfall_linear"] != ""
        assert float(row["snr_waterfall_linear"]) > 0.0

    @pytest.mark.parametrize("rate", RATES)
    def test_the_two_decimators_agree_on_stokes_i(self, runs, rate):
        """The intensity waterfall comes from the shared ``downsample_data``;
        the polarisation waterfalls come from the driver's own reshape. Plane 0
        of the second is the same quantity as the first, and they must be the
        same numbers -- one summing where the other averages would scale them
        apart by the rate, and at rate 1 nothing would show it, because at rate
        1 a sum over one sample and a mean over one sample are equal.

        Compared on the arrays the driver built, so mutating the driver moves
        this assertion; a reimplementation in the test would not.
        """
        run = runs[rate]
        diffs = [s["stokes_i_max_diff"] for s in run["slices"]]
        assert diffs, f"rate {rate}: no slice was processed"
        assert None not in diffs, (
            f"rate {rate}: the intensity block and the decimated raw block "
            "could not be compared -- their shapes disagree, so the two "
            "decimators no longer produce blocks that line up"
        )
        assert max(diffs) == pytest.approx(0.0, abs=1e-3), (
            f"rate {rate}: the two decimators disagree about Stokes I by up to "
            f"{max(diffs)}"
        )


# --------------------------------------------------------------------------- #
# SPEC-HF-002 under decimation
# --------------------------------------------------------------------------- #
class TestUnresolvedDMUnderDecimation:
    """SPEC-HF-002 skips the DM-time cube when the dispersive sweep is under one
    sample, and it measures that in EFFECTIVE samples:

        dm_smear_samples = dm_delay_s / obs_meta.effective_time_reso

    so decimating makes the skip *more* likely, never less. At 350 GHz the sweep
    is already far below one sample at every rate, so the branch is the same one
    throughout this module and the DM columns say so rather than naming a number
    nobody measured.
    """

    @pytest.mark.parametrize("rate", RATES)
    def test_the_dm_is_reported_unresolved_rather_than_invented(self, runs, rate):
        import math

        row = _burst_row(runs[rate])
        assert row["dm_status"] == "unresolved_high_freq"
        assert math.isnan(float(row["dm_pc_cm-3"]))
        assert row["dm_uncertainty"] == ""

    def test_decimation_can_only_push_the_band_further_into_unresolved(self):
        """The monotonicity itself, as arithmetic rather than as a run: whatever
        the band, raising the rate lowers the smearing measured in samples. A
        file that skips the cube at rate 1 can never build it at rate 2."""
        from src.domain.physics import K_DM_MS

        delay_s = K_DM_MS * 500.0 * (349_690.0 ** -2 - 350_000.0 ** -2)
        smear = [delay_s / (TSAMP * rate) for rate in RATES]
        assert smear == sorted(smear, reverse=True)
        assert all(value < 1.0 for value in smear)

    def test_decimating_alone_stops_the_cube_from_being_built(self, tmp_path, monkeypatch):
        """The branch change, driven rather than derived.

        At 350 GHz the sweep is far under one sample at every rate, so this
        module's main runs never see SPEC-HF-002 decide anything: it skips the
        cube each time and the comparison with the raw resolution is invisible.
        A 10 GHz band with a 310 MHz span at DM 500 is the case where it
        decides -- the sweep is 1.349 ms, so it is 1.35 effective samples at
        rate 1 and 0.67 at rate 2. Same file, same configuration, one rate
        apart: the cube is built and then it is not.

        That is also the only arrangement in which the test can tell
        ``effective_time_reso`` from ``time_reso`` on this line, because both
        answers agree everywhere the sweep is not within a factor of the rate
        of one sample.
        """
        pytest.importorskip("astropy")
        pytest.importorskip("your")
        from src.core import data_flow_manager as dfm
        from src.core.high_freq_pipeline import _process_file_chunked_high_freq
        from src.input.fits_handler import get_obparams
        from src.visualization import visualization_unified
        from tests.synthetic_psrfits import write_psrfits

        built: dict[int, int] = {}
        for rate in (1, 2):
            path = tmp_path / f"resolved_rate{rate}.fits"
            write_psrfits(
                path, nsubint=16, nsblk=64, nchan=NCHAN, npol=4,
                pol_type="IQUV", tsamp=TSAMP,
                fch1=RESOLVED_FCH1_MHZ, foff=RESOLVED_FOFF_MHZ,
                dm=DM_INJECTED, burst_time_s=0.5, dtype="float32",
            )
            get_obparams(str(path))

            save_dir = tmp_path / f"out{rate}"
            save_dir.mkdir(parents=True, exist_ok=True)
            patcher = pytest.MonkeyPatch()
            try:
                _configure(patcher, save_dir, rate)
                _pin_the_hardware(patcher)
                patcher.setattr(
                    visualization_unified, "save_all_plots", lambda *a, **k: None
                )
                seen: list[int] = []
                real_build = dfm.build_dm_time_cube

                def _spy(block, height, dm_min, dm_max, collector=None):
                    seen.append(height)
                    return real_build(block, height=height, dm_min=dm_min,
                                      dm_max=dm_max, collector=collector)

                patcher.setattr(dfm, "build_dm_time_cube", _spy)
                _process_file_chunked_high_freq(
                    cls_model=None, fits_path=path, save_dir=save_dir,
                    chunk_samples=1024,
                )
                built[rate] = len(seen)
            finally:
                patcher.undo()

        assert built[1] > 0, (
            "the cube was never built at rate 1, so this band does not sit "
            "where SPEC-HF-002 decides and the test below proves nothing"
        )
        assert built[2] == 0, (
            "the cube was still built at rate 2, so SPEC-HF-002 measured the "
            "smearing against the raw sampling interval rather than the "
            "effective one"
        )


# --------------------------------------------------------------------------- #
# the rate-dependent arithmetic the runs above cannot reach
# --------------------------------------------------------------------------- #
class TestTheRateReachesTheParametersThemselves:
    """Three computations that take ``DOWN_TIME_RATE`` and that the end-to-end
    runs cannot discriminate, because in the regime those runs are in the rate
    is dominated by something else -- a clamp, a floor, a band far past the
    threshold. They are exercised directly instead, which is the honest way to
    cover a line whose effect a full run hides.
    """

    @pytest.mark.parametrize("rate", (1, 2))
    def test_the_slice_length_is_computed_from_the_effective_interval(self, rate):
        """Above the patch floor the requested duration wins, and then the
        length must be the duration divided by the EFFECTIVE interval. The
        end-to-end runs sit below the floor, where this is unobservable.

        Only rates 1 and 2: the window between the floor (612) and
        ``SLICE_LEN_MAX`` (2048) is a factor of 3.3, so no single duration can
        put rates 1 and 4 both inside it. A 2000 ms slice gives 2000 samples at
        rate 1 and 1000 at rate 2, both clear of either clamp -- which is what
        makes the division by the effective interval the only thing deciding.
        """
        from src.preprocessing.slice_len_calculator import (
            calculate_slice_len_from_duration,
        )

        duration_ms = 2000.0
        patcher = pytest.MonkeyPatch()
        try:
            patcher.setattr(config, "TIME_RESO", TSAMP, raising=False)
            patcher.setattr(config, "DOWN_TIME_RATE", rate, raising=False)
            patcher.setattr(config, "SLICE_DURATION_MS", duration_ms, raising=False)
            slice_len, real_ms = calculate_slice_len_from_duration()
        finally:
            patcher.undo()

        expected = round(duration_ms / (TSAMP * rate * 1000.0))
        assert PATCH_FLOOR_SAMPLES < expected <= config.SLICE_LEN_MAX, (
            "a clamp would mask the rate and this test would prove nothing"
        )
        assert slice_len == expected
        assert real_ms == pytest.approx(duration_ms)

    @pytest.mark.parametrize("rate", (2, 4))
    def test_a_chunk_below_the_rate_scaled_minimum_is_upgraded(self, rate):
        """``resolve_memory_safe_chunk_size`` scales its physical minimum by the
        rate, and a request under it is replaced rather than honoured. Without
        that scaling a decimated run would be handed a chunk too short for one
        slice -- and, because the replacement is the memory-safe size, the whole
        file in a single chunk, which is the trap the module docstring names.
        """
        from src.core.file_driver import resolve_memory_safe_chunk_size
        from src.preprocessing.slice_len_calculator import (
            calculate_memory_safe_chunk_size,
        )

        class _NullCollector:
            def __getattr__(self, _name):
                return lambda *a, **k: None

        patcher = pytest.MonkeyPatch()
        try:
            patcher.setattr(config, "TIME_RESO", TSAMP, raising=False)
            patcher.setattr(config, "DOWN_TIME_RATE", rate, raising=False)
            patcher.setattr(config, "DOWN_FREQ_RATE", 1, raising=False)
            patcher.setattr(config, "SLICE_DURATION_MS", SLICE_DURATION_MS, raising=False)
            # The upgrade it must perform is to the memory-safe size, so a
            # starved machine could otherwise hand back something below the
            # minimum and fail this for the wrong reason.
            _pin_the_hardware(patcher)
            _, diagnostics = calculate_memory_safe_chunk_size()
            minimum_raw = diagnostics["required_min_size"] * rate
            # One sample under the rate-scaled minimum, and comfortably over
            # the unscaled one, so only the scaling can decide.
            requested = minimum_raw - 1
            assert requested > diagnostics["required_min_size"]
            resolved = resolve_memory_safe_chunk_size(requested, _NullCollector())
        finally:
            patcher.undo()

        assert resolved >= minimum_raw, (
            f"rate {rate}: a chunk of {requested} raw samples was accepted "
            f"although the physical minimum at this rate is {minimum_raw}"
        )

    def test_the_dispatch_threshold_moves_with_the_rate(self):
        """``should_use_hf_pipeline`` divides by ``time_reso * down_time_rate``.
        Every other test of it pins the rate at 1, where the factor is invisible.
        This is a band the rate alone moves across the threshold: resolved at
        rate 1, collapsed at rate 4.
        """
        from src.core.pipeline_parameters import should_use_hf_pipeline

        band = {"freq_low_mhz": 9690.0, "freq_high_mhz": 10_000.0, "dm_max": 2000.0,
                "time_reso_s": TSAMP}
        at_one, reason_one = should_use_hf_pipeline(**band, down_time_rate=1)
        at_four, reason_four = should_use_hf_pipeline(**band, down_time_rate=4)

        assert not at_one, reason_one
        assert at_four, reason_four
