"""Architecture gate: domain layer stays physics-only (Etapa 5)."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOMAIN_DIR = PROJECT_ROOT / "src" / "domain"


class TestDomainIsolation(unittest.TestCase):
    def test_domain_modules_avoid_pipeline_imports(self):
        banned = ("src.core.pipeline", "src.core.high_freq_pipeline", "src.config.config")
        for py_file in DOMAIN_DIR.glob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(
                            any(alias.name.startswith(b) for b in banned),
                            f"{py_file.name} imports {alias.name}",
                        )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(
                        any(node.module.startswith(b) for b in banned),
                        f"{py_file.name} imports from {node.module}",
                    )

    def test_k_dm_reexported(self):
        from src.domain import K_DM_MS
        from src.analysis.science_metrics import K_DM_MS as canonical

        self.assertEqual(K_DM_MS, canonical)


if __name__ == "__main__":
    unittest.main()
