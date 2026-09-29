"""D3(BJ) and D3(0) dispersion energies via simple-dftd3."""
from __future__ import annotations

import numpy as np
from pyscf.data import elements


def correction(mol, functional) -> float:
    """D3 dispersion energy (Hartree) of a Mole or Cell.

    ``functional.dispersion`` holds ``method`` ('d3bj' or 'd3zero') and either
    ``xc`` (a dftd3 database name) or ``params`` (explicit damping parameters).
    """
    from dftd3.interface import DispersionModel, RationalDampingParam, ZeroDampingParam

    meta = functional.dispersion
    param_cls = {"d3bj": RationalDampingParam, "d3zero": ZeroDampingParam}.get(meta.get("method"))
    if param_cls is None:
        raise ValueError(
            f"unknown dispersion method {meta.get('method')!r}; "
            "supported methods are 'd3bj' and 'd3zero'."
        )
    if "params" in meta:
        # Two-body only by default, like the database parameters.
        param = param_cls(**{"s9": 0.0, **meta["params"]})
    elif "xc" in meta:
        try:
            param = param_cls(method=meta["xc"])
        except (RuntimeError, ValueError, KeyError) as err:
            raise ValueError(
                f"dftd3 has no D3 damping parameters for xc={meta['xc']!r}; "
                "provide explicit params instead."
            ) from err
    else:
        raise ValueError("dispersion metadata needs an 'xc' name or a 'params' dict.")

    # True atomic numbers: atom_charges() is the valence charge under pseudopotentials.
    numbers = np.array([elements.charge(mol.atom_pure_symbol(i)) for i in range(mol.natm)])
    if hasattr(mol, "lattice_vectors"):
        model = DispersionModel(
            numbers,
            mol.atom_coords(),
            lattice=mol.lattice_vectors(),
            periodic=np.arange(3) < mol.dimension,
        )
    else:
        model = DispersionModel(numbers, mol.atom_coords())
    return float(model.get_dispersion(param, grad=False)["energy"])
