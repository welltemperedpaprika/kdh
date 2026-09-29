"""D3(BJ) dispersion: selecting a *-D3(BJ) functional enables it (needs dftd3)."""
from pyscf import gto

from kdh import RDFDH

mol = gto.M(atom="Ne 0 0 0; Ne 0 0 3.0", basis="def2-SVP")
dh = RDFDH(mol, xc="B2PLYP-D3BJ")
dh.kernel()
print(f"E_disp = {dh.e_disp:.8f}")
