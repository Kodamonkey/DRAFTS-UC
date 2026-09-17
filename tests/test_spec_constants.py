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
# Standalone tools / legacy scripts: not pipeline library, not enforced.
EXCLUDED_DIRS = {
    SRC / "scripts",
    SRC / "tests",
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
