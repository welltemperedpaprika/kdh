"""Built-in D3(BJ) / D3(0) dispersion adapter for double-hybrid energies.

Backs the ``kdh.dispersion`` seam: ``correction(system, functional, kpts)``
returns the signed dispersion energy in Hartree for a molecular ``Mole`` or a
periodic ``Cell`` through the installed ``dftd3`` package.

Units: ``DispersionModel`` consumes positions and lattice vectors in Bohr and
returns Hartree, so the adapter feeds ``atom_coords()`` / ``lattice_vectors()``
(both Bohr in PySCF) directly.

Atomic numbers: D3 is keyed by the true atomic number Z. For a pseudopotential
cell ``atom_charges()`` returns the *valence* charge (4 for carbon under
gth-pade), which would select the wrong element; Z is taken from the element
symbol via ``pyscf.data.elements.charge`` instead.

k-points do not enter the real-space pairwise sum and are ignored.

Damping is looked up by the ``xc`` name in the dispersion metadata against the
dftd3 library database; an explicit ``params`` dict overrides it. The explicit
path defaults ``s9 = 0.0`` (Axilrod-Teller-Muto three-body off) to match the
database convention, which was fitted two-body-only; pass ``s9 = 1.0`` to opt
into ATM. An unknown method, or D3(BJ) with neither a resolvable ``xc`` nor
``params``, raises rather than defaulting ``s6 = 1.0``; ``method="d4"`` is
refused because ``dftd4`` is not a dependency here.
"""
from __future__ import annotations

from typing import Any

import numpy as np

_D3BJ_METHODS = frozenset({"d3bj", "d3(bj)", "bj"})
_D3ZERO_METHODS = frozenset({"d3zero", "d3(0)", "zero"})


def _atomic_numbers(system: Any) -> np.ndarray:
    """Return true atomic numbers Z for *system* from element symbols."""
    from pyscf.data import elements

    return np.array(
        [elements.charge(system.atom_pure_symbol(i)) for i in range(system.natm)],
        dtype=np.int64,
    )


def _is_periodic(system: Any) -> bool:
    """Return True for a periodic ``Cell`` (exposes ``lattice_vectors``)."""
    return hasattr(system, "lattice_vectors")


def _resolved_damping_param(metadata, param_cls, label: str):
    """Resolve a *param_cls* damping set from *metadata* (explicit params win).

    Explicit params default ``s9 = 0.0`` (ATM three-body off) to match the
    library database convention, which was fitted two-body-only; pass
    ``s9 = 1.0`` explicitly to opt into ATM. Missing both an ``xc`` name and
    ``params`` raises rather than defaulting ``s6 = 1.0``.
    """
    params = metadata.get("params")
    if params is not None:
        params = dict(params)
        params.setdefault("s9", 0.0)
        return param_cls(**params)

    xc = metadata.get("xc")
    if xc is None:
        raise ValueError(
            f"{label} dispersion metadata has neither an 'xc' method name for "
            "the library damping-parameter database nor an explicit 'params' "
            "dict; refusing to default s6=1.0 (double-counting guard). Provide "
            f"dispersion={{'method': ..., 'xc': '<functional>'}} or explicit "
            "params={'s6': ..., ...}."
        )
    try:
        return param_cls(method=str(xc))
    except (RuntimeError, ValueError, KeyError) as err:
        raise ValueError(
            f"dftd3 has no built-in {label} damping parameters for xc={xc!r}; "
            "supply an explicit params dict in the dispersion metadata instead."
        ) from err


def _damping_param(metadata):
    """Dispatch on the dispersion ``method`` and return a damping parameter set."""
    method = str(metadata.get("method", "")).strip().lower()
    if method == "d4":
        raise NotImplementedError(
            "D4 dispersion (method='d4') requires the 'dftd4' python package, "
            "which is not installed in this environment. Install 'dftd4' to "
            "enable the D4 branch, or select a D3(BJ) functional."
        )
    if method in _D3BJ_METHODS:
        from dftd3.interface import RationalDampingParam

        return _resolved_damping_param(metadata, RationalDampingParam, "D3(BJ)")
    if method in _D3ZERO_METHODS:
        from dftd3.interface import ZeroDampingParam

        return _resolved_damping_param(metadata, ZeroDampingParam, "D3(0)")
    raise ValueError(
        f"unknown dispersion method {metadata.get('method')!r}; supported "
        "methods are D3(BJ) ('d3bj'), D3 zero-damping ('d3zero'), and, when "
        "dftd4 is installed, 'd4'. Refusing to default s6=1.0."
    )


def _dispersion_model(system: Any):
    """Build a ``DispersionModel`` for a molecule or a periodic cell (Bohr in)."""
    from dftd3.interface import DispersionModel

    numbers = _atomic_numbers(system)
    positions = system.atom_coords()
    if _is_periodic(system):
        dimension = int(getattr(system, "dimension", 3))
        periodic = np.array([axis < dimension for axis in range(3)])
        return DispersionModel(
            numbers,
            positions,
            lattice=system.lattice_vectors(),
            periodic=periodic,
        )
    return DispersionModel(numbers, positions)


def correction(system: Any, functional: Any, kpts: Any = None) -> float:
    """Return the signed D3 dispersion energy for *system* in Hartree.

    ``system`` is a ``Mole`` or a ``Cell`` (positions/lattice read in Bohr, Z
    from element symbols); ``functional.dispersion`` selects the damping method
    and parameters; ``kpts`` is accepted for signature compatibility and
    ignored. Attractive, so a physical value is ``<= 0``.
    """
    metadata = dict(getattr(functional, "dispersion", {}) or {})
    if not metadata:
        raise ValueError(
            "dispersion_d3.correction called on a functional with empty "
            "dispersion metadata; nothing to evaluate."
        )
    param = _damping_param(metadata)
    model = _dispersion_model(system)
    result = model.get_dispersion(param, grad=False)
    return float(result["energy"])
