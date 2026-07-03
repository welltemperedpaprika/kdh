"""Dispersion correction hook for double-hybrid energies (molecular + periodic).

The correction is a callable ``correction(system, functional, kpts) -> float``
returning the signed dispersion energy in Hartree, normalized to the simulation
cell (attractive, so ``<= 0``); the driver adds it as-is. Molecular callers
pass ``kpts=None`` and ``kpts`` never influences the result.

Resolution: an injected callable always wins; otherwise, if the functional
carries ``dispersion`` metadata, the built-in D3(BJ) adapter in
``kdh.dispersion_d3`` is used (functional-matched damping from the ``dftd3``
library, never defaulting ``s6=1.0``); with neither, ``None`` is returned.
Requires the optional ``dftd3`` package (``pip install kdh[dispersion]``).
"""
from __future__ import annotations

from typing import Callable


def resolve_dispersion_correction(functional, injected) -> Callable | None:
    if injected is not None:
        return injected
    if not functional.dispersion:
        return None
    from .dispersion_d3 import correction

    return correction
