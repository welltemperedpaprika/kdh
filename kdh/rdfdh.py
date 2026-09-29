"""Restricted molecular double-hybrid DFT driver."""
from __future__ import annotations

from pyscf import dft, lib, mp, scf
from pyscf.lib import logger

from .xc import is_pure_hf, parse_dh_xc


class DHBase(lib.StreamObject):
    """Common double-hybrid driver: E_tot = E_DFA + E_PT2 + E_disp.

    Subclasses provide ``_new_ks`` (SCF object for an xc string) and
    ``_new_mp2`` (MP2 object on ``mf_s``).

    Attributes:
        dispersion_correction : callable (mol, functional) -> float
            Overrides the D3 correction selected by ``xc_dh.dispersion``.
        conv_tol, conv_tol_grad, grids_level :
            Applied to the SCF object when not None.
    """

    def __init__(self, mol, xc="B2PLYP", *, frozen=None, with_t2=False,
                 dispersion_correction=None, conv_tol=None, conv_tol_grad=None,
                 grids_level=None):
        self._check_spin(mol)
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory
        self.xc_dh = parse_dh_xc(xc)
        self.frozen = frozen
        self.with_t2 = with_t2
        self.dispersion_correction = dispersion_correction
        self.conv_tol = conv_tol
        self.conv_tol_grad = conv_tol_grad
        self.grids_level = grids_level
        self.reset()
        self._keys = set(self.__dict__)

    def _check_spin(self, mol):
        if mol.spin != 0:
            raise NotImplementedError(
                f"{self.__class__.__name__} only supports closed-shell (spin=0) systems."
            )

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        xc = self.xc_dh
        log.info("")
        log.info("******** %s ********", self.__class__)
        log.info("xc = %s", xc.name)
        log.info("xc_scf = %s", xc.xc_scf)
        if xc.xc_nscf is not None:
            log.info("xc_nscf = %s", xc.xc_nscf)
        log.info("c_pt2 = %s  c_os = %s  c_ss = %s", xc.c_pt2, xc.c_os, xc.c_ss)
        log.info("dispersion = %s", xc.dispersion)
        log.info("frozen = %s", self.frozen)
        return self

    def reset(self, mol=None):
        if mol is not None:
            self._check_spin(mol)
            self.mol = mol
        self.mf_s = self.mf_n = self.mmp = None
        self.e_scf = self.e_dfa = self.e_pt2 = self.e_disp = self.e_tot = None
        self.e_corr_os = self.e_corr_ss = None
        return self

    def run_scf(self, **kwargs):
        mf = self._new_ks(self.xc_dh.xc_scf)
        if self.conv_tol is not None:
            mf.conv_tol = self.conv_tol
        if self.conv_tol_grad is not None:
            mf.conv_tol_grad = self.conv_tol_grad
        if self.grids_level is not None and hasattr(mf, "grids"):
            mf.grids.level = self.grids_level
        self.e_scf = mf.kernel(**kwargs)
        if not mf.converged:
            raise RuntimeError(f"{self.__class__.__name__} SCF did not converge")
        self.mf_s = mf
        return mf

    def energy_dfa(self, **kwargs):
        """Total DFA energy: the SCF energy, or xc_nscf evaluated on the SCF density."""
        if self.mf_s is None:
            self.run_scf(**kwargs)
        if self.xc_dh.xc_nscf is None:
            self.mf_n = self.mf_s
            self.e_dfa = float(self.mf_s.e_tot)
            return self.e_dfa
        self.mf_n = self._new_ks(self.xc_dh.xc_nscf)
        self.mf_n.grids = self.mf_s.grids
        if getattr(self.mf_s, "with_df", None) is not None:
            self.mf_n.with_df = self.mf_s.with_df
        self.e_dfa = float(self.mf_n.energy_tot(dm=self.mf_s.make_rdm1()))
        return self.e_dfa

    def energy_pt2(self, **kwargs):
        xc = self.xc_dh
        if not xc.eval_pt2:
            self.e_corr_os = self.e_corr_ss = self.e_pt2 = 0.0
            return self.e_pt2
        if self.mf_s is None:
            self.run_scf(**kwargs)
        self.mmp = self._new_mp2()
        self.mmp.kernel(with_t2=self.with_t2)
        self.e_corr_os = float(self.mmp.e_corr_os)
        self.e_corr_ss = float(self.mmp.e_corr_ss)
        self.e_pt2 = xc.c_pt2 * (xc.c_os * self.e_corr_os + xc.c_ss * self.e_corr_ss)
        return self.e_pt2

    def energy_dispersion(self):
        correction = self.dispersion_correction
        if correction is None and self.xc_dh.dispersion:
            from .dispersion_d3 import correction
        self.e_disp = 0.0 if correction is None else float(correction(self.mol, self.xc_dh))
        return self.e_disp

    def kernel(self, **kwargs):
        self.check_sanity()
        self.dump_flags()
        self.e_tot = self.energy_dfa(**kwargs) + self.energy_pt2() + self.energy_dispersion()
        logger.note(self, "E(%s) = %.15g", self.xc_dh.name, self.e_tot)
        return self.e_tot

    def nuc_grad_method(self):
        raise NotImplementedError(
            f"Analytic gradients are not implemented for {self.__class__.__name__}."
        )

    Gradients = lib.alias(nuc_grad_method, alias_name="Gradients")

    def as_scanner(self):
        class SinglePointScanner(self.__class__, lib.SinglePointScanner):
            def __init__(self, dh):
                self.__dict__.update(dh.__dict__)

            def __call__(self, mol_or_geom, **kwargs):
                if isinstance(mol_or_geom, str) or getattr(mol_or_geom, "ndim", None) == 2:
                    mol = self.mol.set_geom_(mol_or_geom, inplace=False)
                else:
                    mol = mol_or_geom
                self.reset(mol)
                return self.kernel(**kwargs)

        return SinglePointScanner(self)


class RDFDH(DHBase):
    """Restricted (closed-shell) molecular double-hybrid DFT.

    Attributes:
        df : bool
            Density-fit the SCF (MP2 then follows the SCF's DF object).
    """

    def __init__(self, mol, xc="B2PLYP", *, df=False, **kwargs):
        super().__init__(mol, xc, **kwargs)
        self.df = df
        self._keys = set(self.__dict__)

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        logger.info(self, "df = %s", self.df)
        return self

    def _new_ks(self, xc):
        mf = scf.RHF(self.mol) if is_pure_hf(xc) else dft.RKS(self.mol, xc=xc)
        return mf.density_fit() if self.df else mf

    def _new_mp2(self):
        return mp.MP2(self.mf_s, frozen=self.frozen)

    def nuc_grad_method(self):
        """Analytic gradient; conventional (no xc_nscf), unscaled PT2 only."""
        xc = self.xc_dh
        if xc.xc_nscf is not None:
            raise NotImplementedError(
                f"Analytic gradients for xDH functionals (xc_nscf={xc.xc_nscf!r}) "
                "are not implemented."
            )
        if xc.c_os != xc.c_ss:
            raise NotImplementedError(
                "Analytic gradients for spin-component-scaled PT2 "
                f"(c_os={xc.c_os} != c_ss={xc.c_ss}) are not implemented."
            )
        if xc.dispersion or self.dispersion_correction is not None:
            raise NotImplementedError("Analytic gradients with dispersion are not implemented.")
        if self.frozen is not None:
            raise NotImplementedError("Analytic gradients with frozen core are not implemented.")
        if self.df:
            raise NotImplementedError(
                "Analytic gradients for density-fitted double hybrids are not implemented."
            )
        from .grad.rdfdh import Gradients

        return Gradients(self)
