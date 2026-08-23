"""D3(BJ) and D3(0) dispersion adapter for double-hybrid energies."""
from __future__ import annotations

from typing import Any

import numpy as np

_D3BJ_METHODS = frozenset({"d3bj", "d3(bj)", "bj"})
_D3ZERO_METHODS = frozenset({"d3zero", "d3(0)", "zero"})


def _atomic_numbers(system: Any) -> np.ndarray:
    """Return true atomic numbers Z for atoms in system."""
    from pyscf.data import elements

    return np.array(
        [elements.charge(system.atom_pure_symbol(i)) for i in range(system.natm)],
        dtype=np.int64,
    )


def _is_periodic(system: Any) -> bool:
    return hasattr(system, "lattice_vectors")


def _resolved_damping_param(metadata: dict, param_cls: type, label: str):
    params = metadata.get("params")
    if params is not None:
        params = dict(params)
        params.setdefault("s9", 0.0)
        return param_cls(**params)

    xc = metadata.get("xc")
    if xc is None:
        raise ValueError(
            f"{label} dispersion metadata requires an 'xc' functional name or explicit 'params' dict; "
            "refusing to default s6=1.0."
        )
    try:
        return param_cls(method=str(xc))
    except (RuntimeError, ValueError, KeyError) as err:
        raise ValueError(
            f"dftd3 has no built-in D3 damping parameters for xc={xc!r}; "
            "provide explicit params instead."
        ) from err


def _damping_param(metadata: dict):
    method = str(metadata.get("method", "")).strip().lower()
    if method == "d4":
        raise NotImplementedError(
            "D4 dispersion requires the 'dftd4' package, which is not installed."
        )
    if method in _D3BJ_METHODS:
        from dftd3.interface import RationalDampingParam

        return _resolved_damping_param(metadata, RationalDampingParam, "D3(BJ)")
    if method in _D3ZERO_METHODS:
        from dftd3.interface import ZeroDampingParam

        return _resolved_damping_param(metadata, ZeroDampingParam, "D3(0)")
    raise ValueError(
        f"unknown dispersion method {metadata.get('method')!r}; supported methods are 'd3bj' and 'd3zero'."
    )


def _dispersion_model(system: Any):
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
    """Return the signed D3 dispersion energy for system in Hartree."""
    metadata = dict(getattr(functional, "dispersion", {}) or {})
    if not metadata:
        raise ValueError("Functional has no dispersion metadata.")
    param = _damping_param(metadata)
    model = _dispersion_model(system)
    result = model.get_dispersion(param, grad=False)
    return float(result["energy"])

