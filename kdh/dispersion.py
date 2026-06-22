"""dftd3-backed D3(BJ) dispersion for double-hybrid energies.

A single backend serves both molecular (``Mole``) and periodic (``Cell``)
structures. D3(BJ) is an additive correction: the electronic functional is
unchanged and a classical pairwise dftd3 energy is added to the total energy.
Requires the optional ``dftd3`` package (``pip install kdh[dispersion]``).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pyscf.data.elements as elements


def _atomic_numbers(structure: Any) -> np.ndarray:
    """Return true atomic numbers from element symbols (not pseudized charges)."""
    return np.array(
        [elements.charge(structure.atom_symbol(i)) for i in range(structure.natm)]
    )


def _is_periodic(structure: Any) -> bool:
    """Return True for a 3D periodic ``Cell``."""
    return getattr(structure, "dimension", 0) == 3


def dftd3_correction(structure: Any, functional: Any) -> float:
    """Return the D3(BJ) dispersion energy (Hartree) for *structure*.

    Uses true atomic numbers, Bohr positions, and — for a periodic cell — the
    Bohr lattice with full periodicity. Damping parameters come from
    ``functional.dispersion['params']`` and must include an explicit ``s9``
    (the dftd3 API defaults ``s9=1.0``/ATM-on; standard B2PLYP-D3(BJ) is
    two-body, ``s9=0.0``).
    """
    try:
        from dftd3.interface import DispersionModel, RationalDampingParam
    except Exception as exc:
        raise RuntimeError(
            "dispersion requires the optional 'dftd3' package: "
            "pip install kdh[dispersion]"
        ) from exc

    meta = dict(functional.dispersion)
    if meta.get("method") != "d3bj":
        raise NotImplementedError(
            f"only 'd3bj' dispersion is supported, got {meta.get('method')!r}"
        )
    params = meta.get("params")
    if not params or "s9" not in params:
        raise ValueError(
            "dispersion params must be functional-matched and include an "
            "explicit s9 (no ATM default)"
        )

    numbers = _atomic_numbers(structure)
    positions = structure.atom_coords()
    if _is_periodic(structure):
        model = DispersionModel(
            numbers,
            positions,
            lattice=structure.lattice_vectors(),
            periodic=np.array([True, True, True]),
        )
    else:
        model = DispersionModel(numbers, positions)
    param = RationalDampingParam(**params)
    return float(model.get_dispersion(param, grad=False)["energy"])


def resolve_dispersion_correction(functional: Any, injected) -> Callable | None:
    """Resolve the dispersion callable: injected wins; else dftd3 if metadata set."""
    if injected is not None:
        return injected
    if not functional.dispersion:
        return None
    return dftd3_correction
