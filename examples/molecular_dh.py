"""Closed-shell molecular double hybrid with an analytic gradient."""
from pyscf import gto

from kdh import RDFDH

mol = gto.M(atom="O 0 0 0; H 0 0 0.96; H 0.93 0 -0.26", basis="def2-SVP")
dh = RDFDH(mol, xc="B2PLYP")
dh.kernel()
print(f"E(B2PLYP) = {dh.e_tot:.8f}  E_PT2 = {dh.e_pt2:.8f}")
print(dh.Gradients().kernel())
