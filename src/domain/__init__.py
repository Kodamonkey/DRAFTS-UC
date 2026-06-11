"""Pure physics contracts — no pipeline orchestration or global config imports.

Etapa 5 (architecture): domain layer re-exports science primitives used by specs.
Heavy pipeline logic stays in ``src/core``; this package is the stable import
surface for physics-as-code tests and downstream modules.
"""
from .physics import K_DM_MS, dispersion_delay_ms

__all__ = ["K_DM_MS", "dispersion_delay_ms"]
