# This module fixes the matplotlib backend before any pyplot import happens.

"""Select a non-interactive matplotlib backend.

Importing ``matplotlib.pyplot`` is what resolves and instantiates a backend, and
until this module existed nothing in the repository called ``matplotlib.use()``.
That left the choice to matplotlib's autodetection, which on a machine with a
display loads a GUI toolkit the pipeline never uses, and on a headless compute
node has to discover that no display exists before falling back.

This pipeline only ever writes PNG files, so Agg is the correct backend in every
environment it runs in. ``MPLBACKEND``, if the operator set it, still wins --
selecting a backend is a deployment decision and the environment variable is how
matplotlib expects that decision to be expressed.

Import this module before ``matplotlib.pyplot`` anywhere pyplot is imported at
module scope. Importing it afterwards is not an error (matplotlib can switch
backends while no figure exists) but it has already paid the cost.
"""
from __future__ import annotations

import os


def select_headless_backend() -> str | None:
    """Force Agg unless MPLBACKEND says otherwise. Returns the backend in use."""
    try:
        import matplotlib
    except ImportError:
        return None

    if not os.environ.get("MPLBACKEND"):
        matplotlib.use("Agg", force=True)
    return matplotlib.get_backend()


BACKEND = select_headless_backend()
