"""Open-shell double hybrid with UDFDH (triplet O atom)."""
from pyscf import gto

from kdh import UDFDH

mol = gto.M(atom="O 0 0 0", spin=2, basis="cc-pVDZ")
dh = UDFDH(mol, xc="B2PLYP")
dh.kernel()
print(f"E_corr(OS) = {dh.e_corr_os:.8f}  E_corr(SS) = {dh.e_corr_ss:.8f}")
