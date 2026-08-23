"""Unrestricted molecular double-hybrid DFT driver."""
from __future__ import annotations

from typing import Any

from pyscf import dft, lib, mp, scf
from pyscf.lib import logger

from .dispersion import resolve_dispersion_correction
from .pt2_assembly import assemble_pt2_energy
from .xc import DoubleHybridFunctional, is_pure_hf, parse_dh_xc


class UDFDH(lib.StreamObject):
    """Unrestricted molecular double-hybrid DFT driver (open-shell)."""

    def __init__(
        self,
        mol: Any,
        xc: str | dict[str, Any] | DoubleHybridFunctional = "B2PLYP",
        *,
        frozen: int | list[int] | None = None,
        with_t2: bool = False,
        dispersion_correction=None,
        df: bool = False,
    ) -> None:
        self.mol = mol
        self.stdout = getattr(mol, "stdout", None)
        self.verbose = getattr(mol, "verbose", 0)
        self.max_memory = getattr(mol, "max_memory", 4000)
        self.xc_dh = parse_dh_xc(xc)
        self.frozen = frozen
        self.with_t2 = with_t2
        self.dispersion_correction = dispersion_correction
        self.df = df

        self.mf_s = None
        self.mf_n = None
        self.mmp = None
        self.e_scf = None
        self.e_dfa = None
        self.e_pt2 = None
        self.e_corr_os = None
        self.e_corr_ss = None
        self.e_disp = None
        self.e_tot = None
        self._keys = set(self.__dict__.keys())

    @property
    def xc(self) -> str:
        return self.xc_dh.xc_scf

    @property
    def xc_n(self) -> str | None:
        return self.xc_dh.xc_nscf

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info("")
        log.info("******** %s ********", self.__class__)
        log.info("xc = %s", self.xc_dh.name)
        log.info("xc_scf = %s", self.xc_dh.xc_scf)
        if self.xc_dh.xc_nscf is not None:
            log.info("xc_nscf = %s", self.xc_dh.xc_nscf)
        log.info("c_pt2 = %s", self.xc_dh.c_pt2)
        log.info("c_os = %s", self.xc_dh.c_os)
        log.info("c_ss = %s", self.xc_dh.c_ss)
        log.info("spin = %s", getattr(self.mol, "spin", 0))
        log.info("frozen = %s", self.frozen)
        log.info("with_t2 = %s", self.with_t2)
        log.info("df = %s", self.df)
        log.info("dispersion = %s", self.xc_dh.dispersion)
        log.info("dispersion_correction = %s", self.dispersion_correction)
        return self

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self.mf_s = None
        self.mf_n = None
        self.mmp = None
        self.e_scf = None
        self.e_dfa = None
        self.e_pt2 = None
        self.e_corr_os = None
        self.e_corr_ss = None
        self.e_disp = None
        self.e_tot = None
        return self

    def density_fit(self, auxbasis=None):
        self.df = True
        return self

    def _new_ks(self, xc: str):
        if is_pure_hf(xc):
            mf = scf.UHF(self.mol)
        else:
            mf = dft.UKS(self.mol, xc=xc)
        if self.df:
            return mf.density_fit()
        return mf

    def run_scf(self, **kwargs):
        self.mf_s = self._new_ks(self.xc_dh.xc_scf)
        self.e_scf = self.mf_s.kernel(**kwargs)
        if not self.mf_s.converged:
            raise RuntimeError("UDFDH SCF did not converge")
        return self.mf_s

    def energy_dfa(self, **kwargs) -> float:
        if self.mf_s is None:
            self.run_scf(**kwargs)

        if self.xc_dh.xc_nscf is None:
            self.mf_n = self.mf_s
            self.e_dfa = float(self.mf_s.e_tot)
            return self.e_dfa

        self.mf_n = self._new_ks(self.xc_dh.xc_nscf)
        self.mf_n.grids = self.mf_s.grids
        dm = self.mf_s.make_rdm1()
        self.e_dfa = float(self.mf_n.energy_tot(dm=dm))
        return self.e_dfa

    def energy_pt2(self, **kwargs) -> float:
        if self.xc_dh.requires_lr_pt2:
            raise NotImplementedError(
                "Range-separated double hybrids requiring long-range PT2 are not supported."
            )

        if not self.xc_dh.eval_pt2:
            self.e_corr_os = 0.0
            self.e_corr_ss = 0.0
            self.e_pt2 = 0.0
            return self.e_pt2

        if self.mf_s is None:
            self.run_scf(**kwargs)

        self.mmp = mp.UMP2(self.mf_s, frozen=self.frozen)
        e_corr, _ = self.mmp.kernel(with_t2=self.with_t2)
        e_corr_os = getattr(self.mmp, "e_corr_os", None)
        e_corr_ss = getattr(self.mmp, "e_corr_ss", None)
        self.e_corr_os = None if e_corr_os is None else float(e_corr_os)
        self.e_corr_ss = None if e_corr_ss is None else float(e_corr_ss)
        self.e_pt2 = assemble_pt2_energy(
            self.xc_dh, e_corr, self.e_corr_os, self.e_corr_ss
        )
        return self.e_pt2

    def energy_dispersion(self) -> float:
        correction = resolve_dispersion_correction(
            self.xc_dh, self.dispersion_correction
        )
        if correction is None:
            self.e_disp = 0.0
            return self.e_disp
        self.e_disp = float(correction(self.mol, self.xc_dh, None))
        return self.e_disp

    def nuc_grad_method(self):
        raise NotImplementedError(
            "Analytic gradients for unrestricted (open-shell) double hybrids are not implemented."
        )

    Gradients = nuc_grad_method

    def as_scanner(self):
        class SinglePointScanner(self.__class__, lib.SinglePointScanner):
            def __init__(self, mf):
                lib.StreamObject.__init__(self)
                self.__dict__.update(mf.__dict__)

            def __call__(self, mol_or_geom, **kwargs):
                if isinstance(mol_or_geom, str) or (
                    hasattr(mol_or_geom, "ndim") and mol_or_geom.ndim == 2
                ):
                    mol = self.mol.set_geom_(mol_or_geom, inplace=False)
                else:
                    mol = mol_or_geom
                self.reset(mol)
                return self.kernel(**kwargs)

        return SinglePointScanner(self)

    def kernel(self, **kwargs) -> float:
        self.check_sanity()
        self.dump_flags()
        if self.xc_dh.requires_lr_pt2:
            raise NotImplementedError(
                "Range-separated double hybrids requiring long-range PT2 are not supported."
            )
        self.e_tot = (
            self.energy_dfa(**kwargs)
            + self.energy_pt2()
            + self.energy_dispersion()
        )
        return self.e_tot

