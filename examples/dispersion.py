"""D3(BJ) dispersion via the built-in dftd3 backend (metadata-driven).

Selecting a ``*-D3BJ`` functional turns dispersion on: the correction is
resolved from the functional's registry metadata alone -- no injected callable.
Works identically for molecular (``RDFDH``) and periodic (``KRDH``) systems.
"""
from pyscf import gto

from kdh import KRDH, RDFDH
from examples.periodic_energy import silicon_cell


def main():
    # Molecular: built-in backend, no dispersion_correction argument.
    mol = gto.M(atom="Ne 0 0 0; Ne 0 0 3.0", basis="def2-SVP", verbose=0)
    plain_mol = RDFDH(mol, xc="B2PLYP").kernel()
    corr_mol = RDFDH(mol, xc="B2PLYP-D3BJ")
    e_mol = corr_mol.kernel()
    print(f"Ne2 B2PLYP:        {plain_mol:.8f} Ha")
    print(f"Ne2 B2PLYP-D3(BJ): {e_mol:.8f} Ha  (E_disp {corr_mol.e_disp:.8f})")

    # Periodic: same metadata-driven backend on a Si diamond cell.
    cell = silicon_cell()
    kpts = cell.make_kpts([2, 2, 2])
    plain = KRDH(cell, xc="B2PLYP", kpts=kpts).kernel()
    corrected = KRDH(cell, xc="B2PLYP-D3BJ", kpts=kpts)
    e = corrected.kernel()
    print(f"Si B2PLYP:         {plain:.8f} Ha")
    print(f"Si B2PLYP-D3(BJ):  {e:.8f} Ha  (E_disp {corrected.e_disp:.8f})")


if __name__ == "__main__":
    main()
