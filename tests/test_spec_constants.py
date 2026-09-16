"""SPEC-DM-001: K_DM_MS defined once; no hardcoded duplicates in pipeline modules."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"

# Canonical source: allowed to define the literal.
EXCLUDED_FILES = {
    SRC / "analysis" / "science_metrics.py",
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
