"""Regression tests for the rendering-cost fixes, audit P1-23 / PERF-01.

Rendering dominates a real run: on a 5.5 TiB file the audit measured roughly
1.94 million slices x 4 images, between 860 and 3200 CPU-hours of matplotlib,
one to two orders of magnitude above dedispersion and inference combined. Three
things fed that: plots forced on for every slice, a backend chosen by
autodetection, and a dpi pinned in seven places with no way to lower it.

Two of the tests here guard bugs rather than cost -- a shadowed import and a
name bound in one conditional block and read in another. Both were found while
making the above configurable, and both are the kind that only fire on a branch
the test suite does not usually take.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.config import config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIS_DIR = PROJECT_ROOT / "src" / "visualization"
CORE_DIR = PROJECT_ROOT / "src" / "core"


def _plot_modules():
    return sorted(
        p for p in VIS_DIR.glob("plot_*.py")
    )


class TestHeadlessBackend:
    def test_importing_the_package_selects_agg(self):
        """Autodetection loads a GUI toolkit that no compute node has."""
        pytest.importorskip("matplotlib")
        import matplotlib

        import src.visualization  # noqa: F401  (import is the thing under test)

        assert matplotlib.get_backend().lower() == "agg"

    def test_an_explicit_mplbackend_still_wins(self, monkeypatch):
        """Backend choice is a deployment decision; the env var expresses it."""
        pytest.importorskip("matplotlib")
        import matplotlib

        from src.visualization.mpl_backend import select_headless_backend

        before = matplotlib.get_backend()
        monkeypatch.setenv("MPLBACKEND", "pdf")
        try:
            # With MPLBACKEND set, the helper must not call use("Agg").
            assert select_headless_backend().lower() == before.lower()
        finally:
            matplotlib.use(before, force=True)

    def test_the_backend_module_is_imported_before_pyplot(self):
        """Importing pyplot is what instantiates a backend, so order matters."""
        init = (VIS_DIR / "__init__.py").read_text(encoding="utf-8")
        assert "mpl_backend" in init
        assert init.index("mpl_backend") < init.index("plot_composite")

        pipeline = (CORE_DIR / "pipeline.py").read_text(encoding="utf-8")
        assert "select_headless_backend" in pipeline
        assert pipeline.index("select_headless_backend") < pipeline.index(
            "import matplotlib.pyplot"
        )


class TestFigureOutputIsConfigurable:
    def test_config_exposes_the_output_knobs(self):
        assert isinstance(config.PLOT_DPI, int)
        assert config.PLOT_DPI > 0
        assert hasattr(config, "PLOT_BBOX_INCHES")
        assert isinstance(config.PLOT_PAD_INCHES, float)

    def test_no_plot_module_hardcodes_dpi(self):
        """Seven savefig calls pinned dpi=300 with no way to lower it."""
        offenders = []
        for path in _plot_modules():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = getattr(func, "attr", None)
                if name != "savefig":
                    continue
                for kw in node.keywords:
                    if kw.arg == "dpi" and isinstance(kw.value, ast.Constant):
                        offenders.append(f"{path.name}:{node.lineno}")
        assert not offenders, f"savefig with a literal dpi: {offenders}"

    def test_dpi_actually_reaches_the_rendered_file(self, tmp_path, monkeypatch):
        """A knob nothing reads is not a knob."""
        plt = pytest.importorskip("matplotlib.pyplot")
        from PIL import Image

        sizes = {}
        for dpi in (50, 100):
            monkeypatch.setattr(config, "PLOT_DPI", dpi, raising=False)
            fig = plt.figure(figsize=(2, 2))
            out = tmp_path / f"f{dpi}.png"
            fig.savefig(out, dpi=config.PLOT_DPI, bbox_inches=config.PLOT_BBOX_INCHES,
                        pad_inches=config.PLOT_PAD_INCHES)
            plt.close(fig)
            sizes[dpi] = Image.open(out).size

        assert sizes[100][0] > sizes[50][0]


class TestPlotsAreNotForcedByDefault:
    def test_force_plots_defaults_off(self):
        """Rendering every slice, candidate or not, dominated the run."""
        assert config.FORCE_PLOTS is False

    def test_postprocess_img_is_not_called_before_the_plot_gate(self):
        """Its only consumer is the composite plot.

        It used to run once per band per slice regardless, which with
        force_plots off is once per band per slice that draws nothing.
        """
        for rel in ("detection_engine.py", "high_freq_pipeline.py"):
            source = (CORE_DIR / rel).read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                if not (isinstance(node.value, ast.Call)
                        and getattr(node.value.func, "id", None) == "postprocess_img"):
                    continue
                pytest.fail(
                    f"{rel}:{node.lineno} computes postprocess_img eagerly; "
                    "it belongs at the plot call site"
                )


class TestPerSliceGarbageCollection:
    def test_the_slice_loop_does_not_collect_on_every_iteration(self):
        """A full generational collection per slice bought only cyclic garbage.

        The `del` of the slice arrays frees them by refcount already, and
        _optimize_memory() still runs every tenth slice. The audit measured the
        per-slice version at about 11 hours of pure GC on a 5.5 TiB file
        (PERF-04).
        """
        source = (CORE_DIR / "pipeline.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        target = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_process_block"
        )
        # Find the `if j % 10 == 0:` housekeeping branch and check its else arm.
        for node in ast.walk(target):
            if not isinstance(node, ast.If) or not node.orelse:
                continue
            test_src = ast.unparse(node.test)
            if "% 10" not in test_src:
                continue
            else_src = "\n".join(ast.unparse(stmt) for stmt in node.orelse)
            assert "gc.collect()" not in else_src, (
                "the slice loop collects on every iteration again"
            )
            assert "plt.close" in else_src, (
                "figures must still be closed every slice or matplotlib leaks"
            )
            break
        else:
            pytest.fail("could not find the per-slice housekeeping branch")


class TestLatentNameErrors:
    """Two failure modes that only fire on an unusual branch."""

    def test_no_function_reimports_a_module_level_name(self):
        """A function-local `from ..config import config` makes `config` local
        for the WHOLE function, so every earlier reference to it raises
        UnboundLocalError. That is what save_composite_plot did, and the
        end-to-end test caught it only because a candidate was found.
        """
        offenders = []
        for path in sorted(VIS_DIR.glob("*.py")) + sorted(CORE_DIR.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            module_level = set()
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        module_level.add(alias.asname or alias.name.split(".")[0])
            for func in ast.walk(tree):
                if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for node in ast.walk(func):
                    if not isinstance(node, (ast.Import, ast.ImportFrom)):
                        continue
                    for alias in node.names:
                        bound = alias.asname or alias.name.split(".")[0]
                        if bound in module_level:
                            offenders.append(
                                f"{path.name}:{node.lineno} re-imports '{bound}' "
                                f"inside {func.name}()"
                            )
        assert not offenders, "\n".join(offenders)

    def test_shared_tick_positions_are_bound_outside_both_panels(self):
        """freq_tick_positions was assigned inside the block guarded on
        wf_block and read inside the block guarded on dw_block. An empty
        waterfall next to a non-empty dedispersed block raised NameError.
        """
        source = (VIS_DIR / "plot_composite.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        target = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "create_composite_plot"
        )
        assignments = [
            node for node in ast.walk(target)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "freq_tick_positions"
                    for t in node.targets)
        ]
        assert len(assignments) == 1, "expected exactly one binding"

        # It must be a direct statement of the function body, not nested in an if.
        top_level_lines = {stmt.lineno for stmt in target.body}
        assert assignments[0].lineno in top_level_lines, (
            "freq_tick_positions is bound inside a conditional again"
        )
