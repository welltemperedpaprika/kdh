"""k-point periodic double-hybrid energy with KRDH (Si diamond)."""
import numpy as np
from pyscf.pbc import gto

from kdh import KRDH


def silicon_cell():
    a = 5.43
    fcc = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    dia = [(0, 0, 0), (0.25, 0.25, 0.25)]
    frac = np.array([[b[i] + d[i] for i in range(3)] for b in fcc for d in dia]) % 1.0
    cell = gto.Cell()
    cell.atom = [("Si", tuple(x)) for x in frac @ (np.eye(3) * a)]
    cell.a = np.eye(3) * a
    cell.unit = "A"
    cell.basis = "gth-dzvp"
    cell.pseudo = "gth-pade"
    cell.verbose = 0
    cell.build()
    return cell


def main():
    cell = silicon_cell()
    dh = KRDH(cell, xc="B2PLYP", kpts=cell.make_kpts([2, 2, 2]))
    e = dh.kernel()
    print(f"Si B2PLYP total energy: {e:.8f} Ha")
    print(f"  OS correlation: {dh.e_corr_os:.8f} Ha")
    print(f"  SS correlation: {dh.e_corr_ss:.8f} Ha")


if __name__ == "__main__":
    main()
