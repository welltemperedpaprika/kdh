"""k-point periodic double hybrid with KRDH (diamond Si, 2x2x2 k-mesh)."""
from pyscf.pbc import gto

from kdh import KRDH

cell = gto.M(
    atom="Si 0 0 0; Si 1.3575 1.3575 1.3575",
    a=[[0, 2.715, 2.715], [2.715, 0, 2.715], [2.715, 2.715, 0]],
    basis="gth-dzvp",
    pseudo="gth-pade",
)
dh = KRDH(cell, xc="B2PLYP", kpts=cell.make_kpts([2, 2, 2]))
dh.kernel()
print(f"E_corr(OS) = {dh.e_corr_os:.8f}  E_corr(SS) = {dh.e_corr_ss:.8f}")
print(f"min gap = {dh.min_gap:.4f} Ha")
