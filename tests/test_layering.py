"""The dependency direction between packages, enforced. Audit REF-09.

``src/core`` orchestrates; ``src/preprocessing``, ``src/input``, ``src/output``,
``src/analysis``, ``src/domain`` and ``src/config`` are below it. Before this,
four modules in the lower layers reached back up into ``core`` for parameters
derived from configuration, and they did it with imports placed inside functions
-- so the cycle never raised on import and nothing ever showed it. The audit
found twelve such late imports.

A cycle behind a late import is not harmless. It means the two modules cannot be
reasoned about separately, it makes import order load-bearing, and it turns any
attempt to import one of them standalone -- a test, a script, a notebook -- into
a coin flip.

The fix was to move the three configuration-derived functions down into
``src/config/derived.py``, where both layers can see them, rather than leaving
them in ``core`` where only an upward import could reach them.
"""
from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"

#: Packages that must never import from ``src.core``.
LOWER_LAYERS = ["preprocessing", "input", "output", "analysis", "domain", "config"]

#: The two exceptions, and they are only tolerable because they are leaves:
#: ``test_the_permitted_exceptions_are_really_leaves`` below checks that neither
#: imports anything from this project, so neither can close a cycle. If one ever
#: grows a project import, that test fails and this list stops being safe.
PERMITTED_LEAF_UTILITIES = {"src.core.retry", "src.core.hardware_profile"}

#: Directories that are not pipeline library code.
NOT_LIBRARY = {SRC / "scripts", SRC / "tests", SRC / "training"}


def _is_library(path: Path) -> bool:
    return not any(_within(path, d) for d in NOT_LIBRARY)


def _within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _core_imports(path: Path) -> list[tuple[int, str]]:
    """Every import of src.core in *path*, as (line, dotted target)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[int, str]] = []
    package_depth = len(path.relative_to(SRC).parts) - 1

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                # `from ..core.retry import x` inside src/<pkg>/mod.py
                if node.level - 1 > package_depth:
                    continue
                module = node.module or ""
                if node.level == package_depth + 1 and module.startswith("core"):
                    found.append((node.lineno, "src." + module))
            elif (node.module or "").startswith("src.core"):
                found.append((node.lineno, node.module))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src.core"):
                    found.append((node.lineno, alias.name))
    return found


class TestNoLowerLayerImportsCore:
    def test_lower_layers_do_not_reach_up_into_core(self):
        violations = []
        for package in LOWER_LAYERS:
            directory = SRC / package
            if not directory.is_dir():
                continue
            for path in sorted(directory.rglob("*.py")):
                if not _is_library(path):
                    continue
                for lineno, target in _core_imports(path):
                    root = ".".join(target.split(".")[:3])
                    if root in PERMITTED_LEAF_UTILITIES:
                        continue
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{lineno} imports {target}"
                    )
        assert not violations, (
            "a lower layer imports from src.core:\n  " + "\n  ".join(violations)
            + "\n\nIf the imported thing is derived from configuration, it belongs "
              "in src/config/derived.py, not in core."
        )

    def test_the_permitted_exceptions_are_really_leaves(self):
        """The exception list is only safe while these two import nothing of ours."""
        for dotted in sorted(PERMITTED_LEAF_UTILITIES):
            path = PROJECT_ROOT / (dotted.replace(".", "/") + ".py")
            assert path.exists(), f"{dotted} moved; update PERMITTED_LEAF_UTILITIES"
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert not node.level, (
                        f"{dotted} now has a relative import at line {node.lineno}; "
                        "it is no longer a leaf and cannot stay on the exception list"
                    )
                    assert not (node.module or "").startswith("src."), (
                        f"{dotted} now imports {node.module}; it is no longer a leaf"
                    )
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith("src."), (
                            f"{dotted} now imports {alias.name}; it is no longer a leaf"
                        )


class TestTheDerivedParametersLiveBelowCore:
    def test_the_module_exists_and_imports_nothing_from_core(self):
        path = SRC / "config" / "derived.py"
        assert path.exists()
        assert not _core_imports(path), "src/config/derived.py must not import core"

    def test_pipeline_parameters_still_exposes_them(self):
        """Callers and tests import these from core.pipeline_parameters; the move
        is an implementation detail and must stay one."""
        from src.core import pipeline_parameters

        for name in ("calculate_dm_height", "calculate_dm_values",
                     "calculate_frequency_downsampled"):
            assert hasattr(pipeline_parameters, name)

    def test_both_import_paths_give_the_same_function(self):
        from src.config.derived import calculate_dm_values as from_config
        from src.core.pipeline_parameters import calculate_dm_values as from_core

        assert from_config is from_core

    def test_the_lower_layers_can_be_imported_without_core(self):
        """The property the late imports were hiding.

        Importing a preprocessing module used to pull core in behind your back,
        or not, depending on which function you happened to call first.
        """
        import subprocess
        import sys

        code = (
            "import sys; "
            "import src.preprocessing.slice_len_calculator, src.preprocessing.dedispersion, "
            "src.output.validation_metrics; "
            "loaded = [m for m in sys.modules if m.startswith('src.core.')]; "
            "print(sorted(loaded))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=PROJECT_ROOT,
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        loaded = result.stdout.strip()
        # core.retry and core.hardware_profile are the permitted leaves; nothing
        # else from core may be dragged in by importing a lower layer.
        for permitted in PERMITTED_LEAF_UTILITIES:
            loaded = loaded.replace(f"'{permitted}'", "")
        assert "src.core." not in loaded, (
            f"importing the lower layers pulled in core modules: {result.stdout.strip()}"
        )
