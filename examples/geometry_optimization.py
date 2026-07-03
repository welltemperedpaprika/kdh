"""Numerical-gradient geometry optimization of H2 (B2PLYP/cc-pVDZ).

Guarded by geometry-optimizer availability: without geomeTRIC or pyberny the
example prints a hint and exits instead of failing.
"""
import numpy as np
from pyscf import gto

from kdh import optimize
from kdh.rdfdh import RDFDH

BOHR = 0.52917721092


def _backend_available():
    try:
        from pyscf.geomopt import geometric_solver  # noqa: F401

        return True
    except ImportError:
        pass
    try:
        from pyscf.geomopt import berny_solver  # noqa: F401

        return True
    except ImportError:
        return False


def main():
    if not _backend_available():
        print("geometry optimization needs geomeTRIC or pyberny; skipping")
        return

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.80", basis="cc-pVDZ", unit="Angstrom", verbose=0
    )

    def factory(coords):
        m = mol.copy()
        m.set_geom_(coords, unit="Bohr")
        return RDFDH(m, xc="B2PLYP")

    mol_opt = optimize(factory, mol)
    coords = mol_opt.atom_coords()
    r = float(np.linalg.norm(coords[1] - coords[0])) * BOHR
    print(f"optimized B2PLYP/cc-pVDZ H2 bond length: {r:.4f} Angstrom")


if __name__ == "__main__":
    main()
