"""SPEC-DM-001: K_DM_MS defined once; no hardcoded duplicates in pipeline modules."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"

# Canonical source: allowed to define the literal. It is the domain layer, which
# imports nothing from the project -- that is what lets every other layer import
# it without creating a cycle (SPEC-DM-001).
EXCLUDED_FILES = {
    SRC / "domain" / "physics.py",
}
# Nothing under ``src/`` is exempt any more. ``src/scripts`` and ``src/tests``
# were the two exemptions; audit item 42 moved them to the top-level ``tools/``,
# so the scan below now covers the whole pipeline library with no holes in it.
#
# The tools did not stop being checked, they moved to their own test:
# ``test_no_new_hardcoded_kdm_literals_in_tools`` below, with the four literals
# they already carry named explicitly so new ones cannot join them quietly.
EXCLUDED_DIRS: set = set()

#: Top-level standalone tools (audit item 42). Not pipeline library: they are
#: run by hand, are not importable packages and ship in no image.
TOOLS = SRC.parent / "tools"

#: Files under ``tools/`` that already hardcode a K_DM literal. They predate the
#: move and fixing them is a separate change -- they are one-shot analysis
#: scripts, and two of the three would need the project importable from their
#: new location to do it properly. Listed here so the guard still fails on a
#: NEW one.
TOOLS_WITH_KNOWN_KDM_LITERALS = {
    TOOLS / "diagnostics" / "MJD.py",
    TOOLS / "analysis" / "simulate_5_5tb_processing.py",
    TOOLS / "analysis" / "analisis_chunk_reduccion.py",
}

# Forbidden: any hardcoded K_DM numeric literal outside the canonical file.
_FORBIDDEN = re.compile(r"4\.148808e3|4\.1488e3|4\.15e3")


def _is_excluded(path: Path) -> bool:
    if path in EXCLUDED_FILES:
        return True
    for d in EXCLUDED_DIRS:
        try:
            path.relative_to(d)
            return True
        except ValueError:
            pass
    return False


def test_kdm_ms_value():
    """K_DM_MS must equal the PRESTO canonical value to 7 significant figures."""
    from src.domain.physics import K_DM_MS

    assert K_DM_MS == pytest.approx(4.148808e3, rel=1e-9)


def test_no_hardcoded_kdm_literals():
    """No pipeline module may hardcode K_DM literal values; import from src.domain.physics."""
    violations = []
    for py_file in SRC.rglob("*.py"):
        if _is_excluded(py_file):
            continue
        text = py_file.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if _FORBIDDEN.search(line):
                violations.append(
                    f"{py_file.relative_to(ROOT)}:{lineno}: {line.strip()}"
                )
    assert not violations, (
        "Hardcoded K_DM literals found (import from src.domain.physics instead):\n"
        + "\n".join(violations)
    )


def test_pipeline_modules_import_from_the_domain_layer():
    """SPEC-DM-001 also fixes the DIRECTION of the import, not just the value.

    The SPEC said every pipeline module imports the constant from
    ``src/domain/physics.py``. Until this test existed, none did: the constant
    was defined in ``analysis/science_metrics.py`` and ``domain/physics.py``
    imported it from there, so the documented arrow pointed the wrong way and
    the domain layer was not a layer at all.

    ``science_metrics`` itself is exempt -- it re-exports the name on purpose,
    for callers that already asked it for it.
    """
    exempt = {
        SRC / "analysis" / "science_metrics.py",
        SRC / "domain" / "physics.py",
        SRC / "domain" / "__init__.py",
    }
    pattern = re.compile(r"science_metrics\s+import\s+[^\n]*\bK_DM_MS\b")
    violations = []
    for py_file in SRC.rglob("*.py"):
        if py_file in exempt or _is_excluded(py_file):
            continue
        text = py_file.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                violations.append(f"{py_file.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not violations, (
        "K_DM_MS imported from the analysis layer instead of the domain layer:\n"
        + "\n".join(violations)
    )


def test_no_new_hardcoded_kdm_literals_in_tools():
    """``tools/`` may not grow NEW hardcoded K_DM literals.

    The standalone tools moved out of ``src/`` in audit item 42. That made the
    scan above exemption-free, but it would also have taken the tools out of its
    reach entirely -- silently dropping a guard rather than deciding about it.
    So they are checked here instead, with the files that already carry a
    literal named one by one.

    Three files, four literals, all predating the move. They are one-shot
    analysis scripts; importing ``src.domain.physics`` from them is a separate
    change, not part of moving directories.
    """
    if not TOOLS.exists():
        pytest.skip("no tools/ directory")

    violations = []
    for py_file in sorted(TOOLS.rglob("*.py")):
        if py_file in TOOLS_WITH_KNOWN_KDM_LITERALS:
            continue
        text = py_file.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if _FORBIDDEN.search(line):
                violations.append(f"{py_file}:{lineno}: {line.strip()}")

    assert not violations, (
        "a tool hardcodes a K_DM literal instead of importing it from "
        "src.domain.physics:\n  " + "\n  ".join(violations)
    )


def test_the_known_tools_list_is_still_accurate():
    """Guard the guard, both ways.

    If one of those files is cleaned up, it should leave the list rather than
    sit there implying debt that is gone. If one is deleted, the stale entry
    should not linger either.
    """
    if not TOOLS.exists():
        pytest.skip("no tools/ directory")

    for path in sorted(TOOLS_WITH_KNOWN_KDM_LITERALS):
        assert path.exists(), f"{path} is on the known-literals list but is gone"
        text = path.read_text(encoding="utf-8", errors="replace")
        has_literal = any(
            _FORBIDDEN.search(line)
            for line in text.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert has_literal, (
            f"{path} no longer hardcodes a K_DM literal -- take it off "
            "TOOLS_WITH_KNOWN_KDM_LITERALS so the guard covers it"
        )
