"""Unrestricted molecular double-hybrid DFT driver."""
from pyscf import dft, mp, scf

from .rdfdh import RDFDH
from .xc import is_pure_hf


class UDFDH(RDFDH):
    """Unrestricted (open-shell) molecular double-hybrid DFT."""

    def _check_spin(self, mol):
        pass

    def _new_ks(self, xc):
        mf = scf.UHF(self.mol) if is_pure_hf(xc) else dft.UKS(self.mol, xc=xc)
        return mf.density_fit() if self.df else mf

    def _new_mp2(self):
        return mp.UMP2(self.mf_s, frozen=self.frozen)

    def nuc_grad_method(self):
        raise NotImplementedError("Analytic gradients are not implemented for UDFDH.")
