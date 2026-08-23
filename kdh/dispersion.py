"""Dispersion corrections for double-hybrid functionals."""
from __future__ import annotations

from typing import Callable


def resolve_dispersion_correction(functional, injected=None) -> Callable | None:
    """Resolve the dispersion correction callable for a functional."""
    if injected is not None:
        return injected
    if not getattr(functional, "dispersion", None):
        return None
    from .dispersion_d3 import correction

    return correction

