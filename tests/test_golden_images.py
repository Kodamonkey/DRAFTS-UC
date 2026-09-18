"""A characterisation test that pins what the composite figure looks like.

This is the precondition for REF-04. ``create_composite_plot`` is ~870 lines of
drawing code with thirty parameters, and before this file nothing in the suite
asserted anything about its output: ``test_e2e_pipeline`` only checked that a
PNG existed. Decomposing it without a picture of the "before" would have been a
rewrite, not a refactor.

How the comparison works, and why
---------------------------------
Matplotlib PNGs are not byte-stable across matplotlib, FreeType or Pillow
versions -- a glyph hinting change moves every character by a fraction of a
pixel -- so a single byte comparison would either fail for reasons having
nothing to do with the code or, if loosened enough to survive, stop catching
anything. There are therefore **two layers**, and they catch different things.

**Layer 1: the structural digest** (``<scene>.digest.json``), which is the
primary assertion. It walks the ``Figure`` object and records, for every Axes in
order: its position in figure coordinates, its title / xlabel / ylabel, its
limits, its tick locations and tick label *strings*, and then for every artist
on it -- images (extent, shape, colour limits, colormap name, data min/max/sum),
lines (colour, width, style, marker, point count, endpoints, y statistics),
rectangles (position, size, edge colour) and texts/annotations (the string, its
position, and the point an annotation arrow points at). Plus the suptitle.

  * It **catches**: a panel that stopped being drawn, an axes that moved, a
    title or label that changed, a tick that moved or got a different string, a
    waterfall drawn from the wrong array or with the wrong extent or colour
    limits, an SNR profile whose values or length changed, a candidate box at
    the wrong place or in the wrong colour, a label with different text or at a
    different position, and the whole label-collision layout of Panel 1b.
  * It **cannot catch**: anything purely visual that is not an artist property
    -- font rendering, z-order between artists that do not overlap in the
    digest, alpha compositing, dpi, and the exact rasterisation of the "mako"
    colormap. Layer 2 covers those.
  * It is **portable** up to the numbers: the strings are exact everywhere, and
    the numbers are compared with a small tolerance (see ``_close``) because the
    SNR profile accumulates in float32 through ``np.convolve``. The one
    genuinely fragile spot is that a panel title embeds an SNR formatted
    ``.1f``: a platform that moves the value across a rounding boundary changes
    a string. If that ever happens it fails loudly with the two strings side by
    side, which is the right failure.

**Layer 2: the rendered pixels** (``<scene>.png``), compared as arrays with a
tolerance of at most 2 levels on any channel and a mean absolute difference
below 0.05/255. That is far tighter than "looks the same" -- one glyph moving
one pixel breaks it -- so it is gated on the environment stamp in
``environment.json`` (matplotlib and FreeType versions). When the stamp does not
match, the test **skips and says so**, naming both stamps. It does not silently
pass: layer 1 is still asserted on every machine. This is the same bargain
``test_golden_csv.py`` makes, for the same reason.

There is also a layer that needs no stored file and holds everywhere:
``test_two_renders_are_byte_identical``. If that ever fails, something in the
drawing path has become non-deterministic and both comparisons below are
meaningless until it is found.

Guarding against a vacuous pass
-------------------------------
``TestTheBaselinesAreNotVacuous`` asserts the stored baselines actually contain
a drawn figure: the expected number of axes for the layout, at least one image
artist, and a PNG with real colour variance. A golden test that compares two
blank canvases is worse than no test.

The scene list was then checked by mutation: single-line changes to
``plot_composite.py`` -- a panel title string, a colormap, a colour limit moved
by 0.1%, the label-collision spacing, a candidate box colour, a DM tick off by
one, the dispersion correction on the raw waterfall's time axis, both peak
``axvline``s, the candidate SNR the multi-pol panels are handed, the window the
dedispersed column is given, and the two waterfalls' time windows swapped -- and
every one of them is caught, before the decomposition and after it. The first
pass caught all but one: nothing reached the "Peak SNR" arm of the waterfall
titles, which is why ``no_candidate_times`` exists. ``empty_dedispersed``
covers the mirror of ``empty_raw_waterfall``.

Regenerating the baselines
--------------------------
``DRAFTS_UPDATE_GOLDEN=1 .venv/Scripts/python.exe -m pytest tests/test_golden_images.py``

Do that only when a change is *meant* to alter the figures, and say in the
commit message what moved. A silently regenerated baseline is worse than none.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from src.config import config

# Importing the package is what registers the "mako" colormap the panels use,
# and what forces the Agg backend before pyplot picks one.
import src.visualization  # noqa: F401
from src.visualization.plot_composite import (
    CandidateSnr,
    RawWaterfallWindow,
    SliceWindow,
    annotate_candidates,
    candidate_label_positions,
    create_composite_plot,
    draw_dedispersed_snr_profile,
    draw_dedispersed_waterfall,
    draw_detection_panel,
    draw_raw_snr_profile,
    draw_raw_waterfall,
    raw_waterfall_window,
    save_composite_plot,
)
from src.visualization.visualization_unified import save_all_plots
from tests.observatory import effelsberg

GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "images"
ENV_STAMP = GOLDEN_DIR / "environment.json"

UPDATING = bool(os.environ.get("DRAFTS_UPDATE_GOLDEN"))

#: Baselines are rendered small on purpose: the figure is 14x12 inches, so this
#: is 420x360 px per scene (~130 kB of PNG, ~0.3 s to render). Production uses
#: config.PLOT_DPI = 300, which would be 4200x3600 and ~8 MB per baseline for no
#: extra assurance: the panels are laid out in figure coordinates, so a
#: decomposition that moves one moves it at every dpi.
BASELINE_DPI = 30

#: Geometry of the synthetic data. n_time is the slice length; n_freq is small
#: because the waterfalls are drawn with aspect="auto" and the channel count
#: only has to survive the median/MAD weighting in compute_snr_profile.
N_TIME = 512
N_FREQ = 64


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def _environment() -> dict:
    import matplotlib
    from matplotlib import ft2font

    return {
        "matplotlib": matplotlib.__version__,
        "freetype": ft2font.__freetype_version__,
    }


# ---------------------------------------------------------------------------
# Config: every global the figure reads, pinned through monkeypatch so nothing
# leaks into the rest of the suite. conftest's autouse fixture only snapshots a
# subset of these, which is exactly why they go through monkeypatch here.
# ---------------------------------------------------------------------------

def _pin_every_config_key_the_figure_depends_on(monkeypatch) -> None:
    freq = np.linspace(1500.0, 1000.0, N_FREQ, dtype=np.float64)
    values = {
        "FREQ": freq,
        "FREQ_RESO": N_FREQ,
        "DOWN_FREQ_RATE": 1,
        "TIME_RESO": 0.001,
        "DOWN_TIME_RATE": 1,
        "DM_min": 100.0,
        "DM_max": 500.0,
        # False keeps _calculate_dynamic_dm_range on its fallback arm, so the DM
        # tick labels are a pure function of DM_min/DM_max. The dynamic arm is
        # visualization_ranges' business and has its own tests.
        "DM_DYNAMIC_RANGE_ENABLE": False,
        "DM_RANGE_DEFAULT_VISUALIZATION": "detailed",
        "DM_RANGE_FACTOR": 0.3,
        "DM_RANGE_MIN_WIDTH": 80.0,
        "DM_RANGE_MAX_WIDTH": 300.0,
        "SNR_THRESH": 5.0,
        "SNR_THRESH_LINEAR": 4.0,
        "CLASS_PROB": 0.5,
        "CLASS_PROB_LINEAR": 0.5,
        # True on purpose: the threshold highlight, the axhline and the peak
        # axvline in both waterfalls are only drawn under this flag, and the
        # axvlines sit behind the `'peak_snr_wf' in locals()` sniffs that REF-04
        # has to replace. With it off those branches would never be rendered and
        # the baseline would not protect them.
        "SNR_SHOW_PEAK_LINES": True,
        "SNR_HIGHLIGHT_COLOR": "red",
        # Empty means compute_snr_profile uses its built-in PRESTO width set,
        # which does not depend on config.yaml.
        "DETECTION_WIDTHS_MS": [],
        "PLOT_DPI": BASELINE_DPI,
        # None, not "tight": with "tight" the saved image size depends on text
        # extents, so a font metric change would resize the whole PNG and the
        # array comparison could not even line the two up.
        "PLOT_BBOX_INCHES": None,
        "PLOT_PAD_INCHES": 0.1,
        # The candidate labels carry an MJD, so the file start time has to be
        # pinned or every label string moves.
        "TSTART_MJD": 60000.0,
        "TSTART_MJD_CORR": None,
        # Barycentric correction runs (create_composite_plot hardcodes
        # compute_bary=True). The bundled ephemeris means no download, no
        # network and the same answer on every machine -- and so does the
        # pinned EarthLocation, which replaces the name "Effelsberg" because
        # resolving that name is a network call. See tests/observatory.py: when
        # the lookup fails, every candidate annotation silently loses its
        # MJD_bary_inf line, which moves both the digest and the pixels.
        "SOURCE_RA": "05:31:58.70",
        "SOURCE_DEC": "33:08:52.5",
        "REF_FREQ_MHZ": 1400.0,
        "OBSERVATORY": effelsberg(),
        "EPHEMERIS": "builtin",
    }
    for key, val in values.items():
        monkeypatch.setattr(config, key, val, raising=False)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def _waterfall(seed: int, *, burst_sample: int, dispersed: bool) -> np.ndarray:
    """A (time, freq) block with noise and one burst, optionally dispersed."""
    rng = np.random.default_rng(seed)
    block = rng.normal(loc=10.0, scale=1.0, size=(N_TIME, N_FREQ)).astype(np.float32)
    width = 4.0
    t = np.arange(N_TIME, dtype=np.float32)
    for ch in range(N_FREQ):
        # A quadratic sweep stands in for the cold-plasma delay; the exact law
        # does not matter, only that the two blocks differ and both are fixed.
        offset = (ch / (N_FREQ - 1)) ** 2 * 60.0 if dispersed else 0.0
        centre = burst_sample + offset
        block[:, ch] += 18.0 * np.exp(-0.5 * ((t - centre) / width) ** 2)
    return block


def _img_rgb(seed: int) -> np.ndarray:
    """A 512x512x3 uint8 DM-time image, the shape postprocess_img returns.

    The background is a coarse 64x64 field blown up eightfold rather than
    per-pixel noise: it is just as fixed, just as clearly not blank, and it lets
    the baseline PNG compress to tens of kilobytes instead of a third of a
    megabyte.
    """
    rng = np.random.default_rng(seed)
    coarse = rng.integers(20, 90, size=(64, 64, 3), dtype=np.uint8)
    base = np.kron(coarse, np.ones((8, 8, 1), dtype=np.uint8))
    yy, xx = np.mgrid[0:512, 0:512]
    blob = np.exp(-(((xx - 256) / 18.0) ** 2 + ((yy - 316) / 14.0) ** 2))
    base = np.clip(base.astype(np.float32) + 160.0 * blob[..., None], 0, 255)
    return base.astype(np.uint8)


@dataclass
class Scene:
    """One call to save_composite_plot, named so its baseline can be found."""

    name: str
    kwargs: dict
    expected_axes: int
    build: Callable[[], dict] = field(default=lambda: {})

    def call_kwargs(self) -> dict:
        return {**self.build(), **self.kwargs}


def _common(**over) -> dict:
    base = dict(
        waterfall_block=_waterfall(1, burst_sample=300, dispersed=True),
        dedispersed_block=_waterfall(2, burst_sample=300, dispersed=False),
        img_rgb=_img_rgb(3),
        patch_img=None,
        patch_start=0.0,
        dm_val=300.0,
        top_conf=[0.95, 0.72],
        top_boxes=[[240, 300, 272, 332], [96, 120, 128, 152]],
        class_probs=[0.91, 0.28],
        slice_idx=7,
        time_slice=4,
        band_name="Full Band",
        band_suffix="fullband",
        fits_stem="golden_scene",
        slice_len=N_TIME,
        normalize=True,
        off_regions=None,
        thresh_snr=5.0,
        band_idx=0,
        absolute_start_time=1.0,
        chunk_idx=2,
        slice_samples=N_TIME,
        candidate_times_abs=[1.300, 1.120],
        snr_waterfall_intensity=[12.34, 6.10],
        snr_patch_intensity=[11.02, 5.51],
    )
    base.update(over)
    return base


def _scenes() -> list[Scene]:
    return [
        # The classic single-polarisation layout: detection panel on top, raw
        # and dedispersed waterfalls with their SNR profiles below, two
        # candidates so the label-collision layout of Panel 1b has work to do.
        Scene(
            name="classic_single_pol",
            expected_axes=5,
            build=lambda: _common(),
            kwargs={},
        ),
        # The multi-polarisation layout: the bottom row delegates to
        # create_multi_pol_panels and the standard panels must not be drawn.
        Scene(
            name="multi_pol",
            expected_axes=7,
            build=lambda: _common(
                dedisp_block_linear=_waterfall(4, burst_sample=300, dispersed=False),
                dedisp_block_circular=_waterfall(5, burst_sample=300, dispersed=False),
                class_probs_linear=[0.83, 0.21],
                snr_waterfall_linear=[9.41, 4.22],
                snr_patch_linear=[8.13, 3.90],
            ),
            kwargs={},
        ),
        # Candidates whose absolute times were never supplied, and no
        # classification probabilities either. This is the only scene that
        # reaches the "Peak SNR" arm of both waterfall titles, the plain
        # `color = "lime"` label with no burst status, and the fallback that
        # marks the global peak instead of the candidate time. Mutation testing
        # found those branches uncovered by the three scenes above.
        Scene(
            name="no_candidate_times",
            expected_axes=5,
            build=lambda: _common(
                class_probs=None,
                candidate_times_abs=None,
            ),
            kwargs={},
        ),
        # An empty dedispersed block beside a real raw waterfall: the mirror of
        # empty_raw_waterfall, and the only scene that draws the "No
        # Dedispersed Data" arms.
        Scene(
            name="empty_dedispersed",
            expected_axes=5,
            build=lambda: _common(
                dedispersed_block=np.zeros((0, 0), dtype=np.float32),
            ),
            kwargs={},
        ),
        # A slice with no candidate at all: no boxes, no labels, no candidate
        # SNR, so every panel falls back to its global-peak arm.
        Scene(
            name="candidate_free",
            expected_axes=5,
            build=lambda: _common(
                top_conf=[],
                top_boxes=[],
                class_probs=None,
                candidate_times_abs=None,
                snr_waterfall_intensity=None,
                snr_patch_intensity=None,
            ),
            kwargs={},
        ),
        # The asymmetric case test_p5_rendering documents: an empty raw
        # waterfall beside a real dedispersed block. Both "no data" arms of the
        # left column draw, the right column draws normally, and the shared
        # frequency ticks have to be bound outside both.
        Scene(
            name="empty_raw_waterfall",
            expected_axes=5,
            build=lambda: _common(
                waterfall_block=np.zeros((0, 0), dtype=np.float32),
            ),
            kwargs={},
        ),
    ]


SCENES = {s.name: s for s in _scenes()}
SCENE_NAMES = sorted(SCENES)


# ---------------------------------------------------------------------------
# The structural digest
# ---------------------------------------------------------------------------

def _num(x) -> float:
    return round(float(x), 9)


def _nums(seq) -> list[float]:
    return [_num(v) for v in np.asarray(seq).ravel()]


def _colour(c) -> list[float]:
    from matplotlib.colors import to_rgba

    return [_num(v) for v in to_rgba(c)]


def _image_digest(im) -> dict:
    arr = np.asarray(im.get_array(), dtype=np.float64)
    finite = arr[np.isfinite(arr)] if arr.size else arr
    return {
        "shape": list(arr.shape),
        "extent": _nums(im.get_extent()),
        "clim": _nums(im.get_clim()),
        "cmap": im.get_cmap().name,
        "origin": im.origin,
        "min": _num(finite.min()) if finite.size else None,
        "max": _num(finite.max()) if finite.size else None,
        "sum": _num(finite.sum()) if finite.size else None,
    }


def _line_digest(ln) -> dict:
    x, y = ln.get_xdata(orig=False), ln.get_ydata(orig=False)
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    fy = y[np.isfinite(y)]
    return {
        "color": _colour(ln.get_color()),
        "lw": _num(ln.get_linewidth()),
        "ls": str(ln.get_linestyle()),
        "marker": str(ln.get_marker()),
        "alpha": _num(ln.get_alpha()) if ln.get_alpha() is not None else None,
        "n": int(x.size),
        "x_first": _num(x[0]) if x.size else None,
        "x_last": _num(x[-1]) if x.size else None,
        "y_min": _num(fy.min()) if fy.size else None,
        "y_max": _num(fy.max()) if fy.size else None,
        "y_sum": _num(fy.sum()) if fy.size else None,
    }


def _patch_digest(p) -> dict:
    out = {
        "type": type(p).__name__,
        "edgecolor": _colour(p.get_edgecolor()),
        "lw": _num(p.get_linewidth()),
    }
    if hasattr(p, "get_xy") and hasattr(p, "get_width"):
        out["xy"] = _nums(p.get_xy())
        out["width"] = _num(p.get_width())
        out["height"] = _num(p.get_height())
    return out


def _text_digest(t) -> dict:
    from matplotlib.text import Annotation

    out = {
        "s": t.get_text(),
        "pos": _nums(t.get_position()),
        "ha": t.get_ha(),
        "va": t.get_va(),
        "size": _num(t.get_fontsize()),
        "coords": "axes" if t.get_transform() is t.axes.transAxes else "data",
    }
    if isinstance(t, Annotation):
        out["arrow_to"] = _nums(t.xy)
    return out


def _axes_digest(ax) -> dict:
    pos = ax.get_position()
    return {
        "position": [_num(pos.x0), _num(pos.y0), _num(pos.x1), _num(pos.y1)],
        "title": ax.get_title(),
        "xlabel": ax.get_xlabel(),
        "ylabel": ax.get_ylabel(),
        "xlim": _nums(ax.get_xlim()),
        "ylim": _nums(ax.get_ylim()),
        "xticks": _nums(ax.get_xticks()),
        "yticks": _nums(ax.get_yticks()),
        "xticklabels": [t.get_text() for t in ax.get_xticklabels()],
        "yticklabels": [t.get_text() for t in ax.get_yticklabels()],
        "images": [_image_digest(im) for im in ax.images],
        "lines": [_line_digest(ln) for ln in ax.lines],
        "patches": [_patch_digest(p) for p in ax.patches],
        "texts": [_text_digest(t) for t in ax.texts],
    }


def figure_digest(fig) -> dict:
    """Everything about the figure that a decomposition could plausibly break."""
    # The tick label strings only exist after a draw, and Panel 1b draws anyway.
    fig.canvas.draw()
    return {
        "suptitle": fig._suptitle.get_text() if fig._suptitle is not None else None,
        "size_inches": _nums(fig.get_size_inches()),
        "n_axes": len(fig.axes),
        "axes": [_axes_digest(ax) for ax in fig.axes],
    }


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(1e-6, 1e-6 * max(abs(a), abs(b)))


def _diff(want: Any, got: Any, path: str = "") -> list[str]:
    """Recursive compare; numbers get a tolerance, everything else is exact."""
    if isinstance(want, bool) or isinstance(got, bool):
        return [] if want == got else [f"{path}: baseline {want!r} != now {got!r}"]
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        return [] if _close(float(want), float(got)) else [
            f"{path}: baseline {want!r} != now {got!r}"
        ]
    if isinstance(want, dict) and isinstance(got, dict):
        out: list[str] = []
        for key in sorted(set(want) | set(got)):
            if key not in want:
                out.append(f"{path}.{key}: absent in baseline, now {got[key]!r}")
            elif key not in got:
                out.append(f"{path}.{key}: in baseline {want[key]!r}, now absent")
            else:
                out += _diff(want[key], got[key], f"{path}.{key}")
        return out
    if isinstance(want, list) and isinstance(got, list):
        if len(want) != len(got):
            return [f"{path}: baseline has {len(want)} entries, now {len(got)}"]
        out = []
        for i, (a, b) in enumerate(zip(want, got)):
            out += _diff(a, b, f"{path}[{i}]")
        return out
    return [] if want == got else [f"{path}: baseline {want!r} != now {got!r}"]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_png(scene: Scene, out_path: Path) -> bytes:
    kwargs = scene.call_kwargs()
    save_composite_plot(out_path=out_path, generate_individual_plots=False, **kwargs)
    return out_path.read_bytes()


def _render_figure(scene: Scene):
    import matplotlib.pyplot as plt

    fig = create_composite_plot(**scene.call_kwargs())
    try:
        return figure_digest(fig)
    finally:
        plt.close(fig)


def _png_array(data: bytes | Path) -> np.ndarray:
    import io

    from PIL import Image

    handle = io.BytesIO(data) if isinstance(data, bytes) else data.open("rb")
    with Image.open(handle) as img:
        return np.asarray(img.convert("RGB"), dtype=np.int16)


def _baseline_png(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.png"


def _baseline_digest(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.digest.json"


@pytest.fixture
def pinned(monkeypatch):
    _pin_every_config_key_the_figure_depends_on(monkeypatch)
    return monkeypatch


# ---------------------------------------------------------------------------
# Layer 0: reproducibility, needs no stored file, holds on any machine
# ---------------------------------------------------------------------------

class TestTheFiguresAreReproducible:
    @pytest.mark.parametrize("name", SCENE_NAMES)
    def test_two_renders_are_byte_identical(self, name, tmp_path, pinned):
        """If this fails, both comparisons below are meaningless.

        Every source of non-determinism the drawing path could acquire -- an
        unseeded RNG, a set iteration reaching a colour, a wall-clock value in a
        title -- shows up here first, and here it needs no baseline at all.
        """
        scene = SCENES[name]
        first = _render_png(scene, tmp_path / "a.png")
        second = _render_png(scene, tmp_path / "b.png")
        assert first == second

    @pytest.mark.parametrize("name", SCENE_NAMES)
    def test_the_scene_draws_the_layout_it_claims(self, name, pinned):
        """A digest of an empty figure would match another empty figure."""
        scene = SCENES[name]
        digest = _render_figure(scene)
        assert digest["n_axes"] == scene.expected_axes, (
            f"{name}: expected {scene.expected_axes} axes, drew {digest['n_axes']}"
        )
        n_images = sum(len(ax["images"]) for ax in digest["axes"])
        assert n_images >= 1, f"{name}: no image artist was drawn at all"
        assert digest["suptitle"], f"{name}: no suptitle"


# ---------------------------------------------------------------------------
# Layer 1: the structural digest, asserted everywhere
# ---------------------------------------------------------------------------

class TestTheStructureMatchesTheStoredBaseline:
    @pytest.mark.parametrize("name", SCENE_NAMES)
    def test_the_figure_digest_is_unchanged(self, name, pinned):
        scene = SCENES[name]
        digest = _render_figure(scene)
        path = _baseline_digest(name)

        if UPDATING:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(digest, indent=1, sort_keys=True) + "\n",
                            encoding="utf-8")
            pytest.skip(f"baseline regenerated at {path}")
        if not path.exists():
            pytest.skip(f"no baseline at {path}; create it with DRAFTS_UPDATE_GOLDEN=1")

        want = json.loads(path.read_text(encoding="utf-8"))
        mismatches = _diff(want, digest, name)
        assert not mismatches, (
            f"the composite figure changed ({len(mismatches)} differences):\n  "
            + "\n  ".join(mismatches[:25])
            + "\n\nIf the change was intended, regenerate with "
              "DRAFTS_UPDATE_GOLDEN=1 and say in the commit message what moved."
        )


# ---------------------------------------------------------------------------
# Layer 2: the rendered pixels, gated on the environment stamp
# ---------------------------------------------------------------------------

class TestThePixelsMatchTheStoredBaseline:
    @pytest.mark.parametrize("name", SCENE_NAMES)
    def test_the_png_is_unchanged(self, name, tmp_path, pinned):
        scene = SCENES[name]
        path = _baseline_png(name)

        if UPDATING:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            # Render into tmp_path, not into the baseline directory:
            # save_composite_plot also drops per-candidate polarization figures
            # next to whatever path it is given, and those are not baselines.
            rendered = tmp_path / f"{name}.png"
            _render_png(scene, rendered)
            # Re-encode losslessly before storing. PNG is lossless, and the
            # comparison below is on the decoded array, so this changes no
            # pixel -- it only stops a noisy waterfall costing a third of a
            # megabyte in the repository.
            from PIL import Image

            with Image.open(rendered) as img:
                img.convert("RGB").save(path, optimize=True, compress_level=9)
            assert np.array_equal(_png_array(path), _png_array(rendered)), (
                "re-encoding the baseline changed a pixel"
            )
            ENV_STAMP.write_text(
                json.dumps(_environment(), indent=1, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            pytest.skip(f"baseline regenerated at {path}")
        if not path.exists() or not ENV_STAMP.exists():
            pytest.skip(f"no baseline at {path}; create it with DRAFTS_UPDATE_GOLDEN=1")

        stamp = json.loads(ENV_STAMP.read_text(encoding="utf-8"))
        here = _environment()
        if stamp != here:
            pytest.skip(
                "the pixel baseline was rendered by a different renderer, so a "
                "difference here would say nothing about the code: baseline "
                f"{stamp}, this machine {here}. The structural digest still "
                "holds and is asserted."
            )

        want = _png_array(path)
        got = _png_array(_render_png(scene, tmp_path / f"{name}.png"))
        assert want.shape == got.shape, (
            f"{name}: baseline is {want.shape}, now {got.shape}"
        )

        delta = np.abs(want - got)
        max_diff = int(delta.max())
        mean_diff = float(delta.mean())
        n_changed = int((delta > 0).sum())
        assert max_diff <= 2 and mean_diff <= 0.05, (
            f"{name}: the rendered figure changed -- max channel difference "
            f"{max_diff}, mean {mean_diff:.4f}, {n_changed} subpixel channels "
            f"of {delta.size} differ.\nIf the change was intended, regenerate "
            "with DRAFTS_UPDATE_GOLDEN=1 and say what moved."
        )


# ---------------------------------------------------------------------------
# The entry point the pipelines actually call
# ---------------------------------------------------------------------------

class TestTheWrapperPathDrawsTheSameFigure:
    def test_save_all_plots_reproduces_the_classic_baseline(self, tmp_path, pinned):
        """Both pipelines reach the composite through save_all_plots.

        This pins the argument threading of visualization_unified as well as the
        figure: a parameter dropped or reordered on the way through shows up as
        a changed composite, not as a silent default.
        """
        scene = SCENES["classic_single_pol"]
        k = scene.call_kwargs()
        out = tmp_path / "composite.png"
        save_all_plots(
            k["waterfall_block"],
            k["dedispersed_block"],
            k["img_rgb"],
            k["patch_img"],
            k["patch_start"],
            k["dm_val"],
            k["top_conf"],
            k["top_boxes"],
            k["class_probs"],
            out,
            k["slice_idx"],
            k["time_slice"],
            k["band_name"],
            k["band_suffix"],
            k["fits_stem"],
            k["slice_len"],
            k["normalize"],
            k["off_regions"],
            k["thresh_snr"],
            k["band_idx"],
            absolute_start_time=k["absolute_start_time"],
            chunk_idx=k["chunk_idx"],
            candidate_times_abs=k["candidate_times_abs"],
            snr_waterfall_intensity_list=k["snr_waterfall_intensity"],
            snr_patch_intensity_list=k["snr_patch_intensity"],
        )
        assert out.exists(), "save_all_plots wrote no composite"

        direct = _render_png(scene, tmp_path / "direct.png")
        # save_all_plots derives slice_samples from the block itself, which for
        # this scene is the same N_TIME the scene passes, so the two must agree
        # byte for byte in the same process.
        assert out.read_bytes() == direct


# ---------------------------------------------------------------------------
# The baselines themselves
# ---------------------------------------------------------------------------

class TestTheBaselinesAreNotVacuous:
    @pytest.mark.parametrize("name", SCENE_NAMES)
    def test_the_stored_digest_describes_a_real_figure(self, name):
        path = _baseline_digest(name)
        if not path.exists():
            pytest.skip(f"no baseline at {path}")
        want = json.loads(path.read_text(encoding="utf-8"))
        assert want["n_axes"] == SCENES[name].expected_axes
        assert sum(len(ax["images"]) for ax in want["axes"]) >= 1
        assert sum(len(ax["lines"]) for ax in want["axes"]) >= 1
        assert want["suptitle"]

    @pytest.mark.parametrize("name", SCENE_NAMES)
    def test_the_stored_png_is_not_blank(self, name):
        path = _baseline_png(name)
        if not path.exists():
            pytest.skip(f"no baseline at {path}")
        arr = _png_array(path)
        assert arr.shape[0] > 100 and arr.shape[1] > 100
        assert len(np.unique(arr.reshape(-1, 3), axis=0)) > 500, (
            "the stored baseline is nearly a single colour; comparing against "
            "it would prove nothing"
        )


# ---------------------------------------------------------------------------
# What the decomposition bought
# ---------------------------------------------------------------------------

def _standalone_window() -> SliceWindow:
    freq_ds = np.asarray(config.FREQ, dtype=np.float64)
    return SliceWindow(
        start_abs=1.0,
        end_abs=1.0 + N_TIME * config.TIME_RESO * config.DOWN_TIME_RATE,
        freq_ds=freq_ds,
        time_reso_ds=config.TIME_RESO * config.DOWN_TIME_RATE,
        freq_tick_positions=np.linspace(freq_ds.min(), freq_ds.max(), 6),
    )


def _one_candidate() -> dict:
    return dict(top_conf=[0.95], top_boxes=[[240, 300, 272, 332]],
                candidate_times_abs=[1.300])


class TestThePanelsDrawIndependently:
    """A panel is now a function you can call on an axes of your own.

    Before REF-04 none of this was reachable: each panel was a block inside one
    868-line function, reading names bound hundreds of lines earlier and gated
    on a ``_skip_standard_panels`` flag that only ever restated
    ``multi_pol_mode``. These tests are the assertion that the seams are real,
    and they cost one small figure each.
    """

    def test_the_raw_column_draws_on_a_bare_figure(self, pinned):
        import matplotlib.pyplot as plt

        window = _standalone_window()
        raw_window = RawWaterfallWindow(start=0.9, end=1.4,
                                        candidate_snr_intensity=12.34)
        block = _waterfall(1, burst_sample=300, dispersed=True)

        fig, (ax_prof, ax_wf) = plt.subplots(2, 1)
        try:
            profile = draw_raw_snr_profile(
                ax_prof, block, raw_window=raw_window, thresh_snr=5.0,
                off_regions=None, **_one_candidate(),
            )
            assert profile is not None
            assert "Raw Waterfall" in ax_prof.get_title()
            # The detector's SNR, not one recomputed from the block.
            assert "12.3σ" in ax_prof.texts[0].get_text()

            draw_raw_waterfall(ax_wf, block, raw_window=raw_window,
                               window=window, profile=profile)
            assert len(ax_wf.images) == 1
            assert list(ax_wf.images[0].get_extent()[:2]) == [0.9, 1.4]
            # The peak line is the only Line2D on a waterfall panel.
            assert len(ax_wf.lines) == 1
            assert float(ax_wf.lines[0].get_xdata()[0]) == pytest.approx(profile.peak_time)
        finally:
            plt.close(fig)

    def test_the_dedispersed_column_draws_on_a_bare_figure(self, pinned):
        import matplotlib.pyplot as plt

        window = _standalone_window()
        block = _waterfall(2, burst_sample=300, dispersed=False)

        fig, (ax_prof, ax_wf) = plt.subplots(2, 1)
        try:
            profile = draw_dedispersed_snr_profile(
                ax_prof, block, window=window, candidate_snr_intensity=None,
                thresh_snr=5.0, off_regions=None, **_one_candidate(),
            )
            assert profile is not None
            assert ax_prof.get_title().startswith("Dedispersed Waterfall")

            draw_dedispersed_waterfall(ax_wf, block, window=window, profile=profile)
            assert len(ax_wf.images) == 1
            assert list(ax_wf.images[0].get_extent()[:2]) == [window.start_abs, window.end_abs]
            assert len(ax_wf.lines) == 1
        finally:
            plt.close(fig)

    def test_a_waterfall_without_a_profile_draws_no_peak_line(self, pinned):
        """This is the condition that replaced ``'peak_snr_wf' in locals()``.

        The sniff could only ever answer "did the interpreter bind that name in
        this frame", which is not a question a panel should be asking. ``None``
        is the same answer as a value.
        """
        import matplotlib.pyplot as plt

        window = _standalone_window()
        raw_window = RawWaterfallWindow(start=0.9, end=1.4, candidate_snr_intensity=None)
        block = _waterfall(1, burst_sample=300, dispersed=True)

        fig, ax = plt.subplots()
        try:
            draw_raw_waterfall(ax, block, raw_window=raw_window, window=window,
                               profile=None)
            assert len(ax.images) == 1
            assert len(ax.lines) == 0
        finally:
            plt.close(fig)

    def test_an_absent_block_leaves_a_labelled_placeholder(self, pinned):
        import matplotlib.pyplot as plt

        window = _standalone_window()
        raw_window = RawWaterfallWindow(start=0.9, end=1.4, candidate_snr_intensity=None)

        fig, (ax_raw, ax_dedisp) = plt.subplots(2, 1)
        try:
            assert draw_raw_snr_profile(
                ax_raw, None, raw_window=raw_window, thresh_snr=5.0,
                off_regions=None, top_conf=[], top_boxes=[],
                candidate_times_abs=None,
            ) is None
            assert ax_raw.get_title() == "No Raw Waterfall Data"

            assert draw_dedispersed_snr_profile(
                ax_dedisp, np.zeros((0, 0)), window=window,
                candidate_snr_intensity=None, thresh_snr=5.0, off_regions=None,
                top_conf=[], top_boxes=[], candidate_times_abs=None,
            ) is None
            assert ax_dedisp.get_title() == "No Dedispersed Data"
        finally:
            plt.close(fig)

    def test_the_detection_panel_and_its_overlay_draw_on_a_bare_figure(self, pinned):
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        boxes = [[240, 300, 272, 332], [96, 120, 128, 152]]
        fig = plt.figure(figsize=(14, 12))
        try:
            gs = gridspec.GridSpec(2, 1, figure=fig)
            ax_det = draw_detection_panel(
                fig, gs, _img_rgb(3),
                slice_start_abs=1.0, slice_end_abs=1.512,
                top_conf=[0.95, 0.72], top_boxes=boxes, slice_len=N_TIME,
            )
            assert ax_det.get_title() == "Detection Results"
            assert len(ax_det.images) == 1

            positions = candidate_label_positions(fig, ax_det, boxes)
            assert len(positions) == len(boxes)
            # The collision layout must not stack two labels at one height.
            assert positions[0][1] != positions[1][1]

            annotate_candidates(
                ax_det,
                top_conf=[0.95, 0.72], top_boxes=boxes,
                class_probs=[0.91, 0.28], class_probs_linear=None,
                candidate_snr=CandidateSnr.collect(
                    waterfall_intensity=[12.34, 6.10], patch_intensity=None,
                    waterfall_linear=None, patch_linear=None,
                ),
                label_positions=positions, slice_idx=7, slice_len=N_TIME,
                slice_samples=N_TIME, absolute_start_time=1.0,
                candidate_times_abs=[1.300, 1.120],
            )
            assert len(ax_det.patches) == 2, "one box per candidate"
            assert len(ax_det.texts) == 2, "one label per candidate"
            assert ax_det.texts[0].get_text().startswith("#1")
            assert "BURST" in ax_det.texts[0].get_text()
        finally:
            plt.close(fig)

    def test_the_raw_window_is_shifted_back_by_the_dispersion_sweep(self, pinned):
        """The precompute used to run in both branches and be discarded in one."""
        window = _standalone_window()
        snr = CandidateSnr.collect(
            waterfall_intensity=[12.34], patch_intensity=None,
            waterfall_linear=None, patch_linear=None,
        )
        shifted = raw_waterfall_window(
            window=window, top_conf=[0.95], top_boxes=[[240, 300, 272, 332]],
            slice_len=N_TIME, slice_samples=N_TIME, band_idx=0, candidate_snr=snr,
        )
        assert shifted.start < window.start_abs
        # Shifted, not stretched: both edges move by the same amount.
        assert window.start_abs - shifted.start == pytest.approx(
            window.end_abs - shifted.end)
        assert shifted.candidate_snr_intensity == 12.34

    def test_the_raw_window_is_the_slice_window_with_no_candidate(self, pinned):
        window = _standalone_window()
        unshifted = raw_waterfall_window(
            window=window, top_conf=[], top_boxes=[], slice_len=N_TIME,
            slice_samples=N_TIME, band_idx=0,
            candidate_snr=CandidateSnr.collect(
                waterfall_intensity=None, patch_intensity=None,
                waterfall_linear=None, patch_linear=None,
            ),
        )
        assert unshifted.start == window.start_abs
        assert unshifted.end == window.end_abs
        assert unshifted.candidate_snr_intensity is None
