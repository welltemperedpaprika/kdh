"""Geometry optimization with numerical gradients (needs geomeTRIC or pyberny).

Numerical gradients cover every functional, including xDH (XYG3) and
spin-component-scaled ones that lack analytic gradients.
"""
from pyscf import gto

from kdh import RDFDH, optimize

mol = gto.M(atom="H 0 0 0; H 0 0 0.80", basis="cc-pVDZ", verbose=0)


def factory(coords):
    return RDFDH(mol.set_geom_(coords, unit="Bohr", inplace=False), xc="XYG3")


mol_eq = optimize(factory, mol)
print(mol_eq.atom_coords(unit="Angstrom"))
