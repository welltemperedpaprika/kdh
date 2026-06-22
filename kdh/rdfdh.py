from __future__ import annotations

from typing import Any

from pyscf import dft, lib, mp
from pyscf.lib import logger

from .xc import DoubleHybridFunctional, parse_dh_xc


class RDFDH(lib.StreamObject):
    def __init__(
        self,
        mol: Any,
        xc: str | dict[str, Any] | DoubleHybridFunctional = "B2PLYP",
        *,
        frozen: int | list[int] | None = None,
        with_t2: bool = False,
        dispersion_correction=None,
    ) -> None:
        self._check_closed_shell(mol)
        self.mol = mol
        self.stdout = getattr(mol, "stdout", None)
        self.verbose = getattr(mol, "verbose", 0)
        self.max_memory = getattr(mol, "max_memory", 4000)
        self.xc_dh = parse_dh_xc(xc)
        self.frozen = frozen
        self.with_t2 = with_t2
        self.dispersion_correction = dispersion_correction

        self.mf_s = None
        self.mf_n = None
        self.mmp = None
        self.e_scf = None
        self.e_dfa = None
        self.e_pt2 = None
        self.e_corr_os = None
        self.e_corr_ss = None
        self.e_tot = None
        self.e_disp = None
        self._keys = set(self.__dict__.keys())

    def _check_closed_shell(self, mol: Any) -> None:
        if getattr(mol, "spin", 0) != 0:
            raise NotImplementedError(
                "The molecular RDFDH checkpoint only supports closed-shell "
                "spin=0 molecules."
            )

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
        log.info("frozen = %s", self.frozen)
        log.info("with_t2 = %s", self.with_t2)
        log.info("dispersion = %s", self.xc_dh.dispersion)
        return self

    def reset(self, mol=None):
        if mol is not None:
            self._check_closed_shell(mol)
            self.mol = mol
        self.mf_s = None
        self.mf_n = None
        self.mmp = None
        self.e_scf = None
        self.e_dfa = None
        self.e_pt2 = None
        self.e_corr_os = None
        self.e_corr_ss = None
        self.e_tot = None
        self.e_disp = None
        return self

    def run_scf(self, **kwargs):
        self.mf_s = dft.RKS(self.mol, xc=self.xc_dh.xc_scf)
        self.e_scf = self.mf_s.kernel(**kwargs)
        if not self.mf_s.converged:
            raise RuntimeError("RDFDH SCF did not converge")
        return self.mf_s

    def energy_dfa(self, **kwargs) -> float:
        if self.mf_s is None:
            self.run_scf(**kwargs)

        if self.xc_dh.xc_nscf is None:
            self.mf_n = self.mf_s
            self.e_dfa = float(self.mf_s.e_tot)
            return self.e_dfa

        self.mf_n = dft.RKS(self.mol, xc=self.xc_dh.xc_nscf)
        self.mf_n.grids = self.mf_s.grids
        dm = self.mf_s.make_rdm1()
        self.e_dfa = float(self.mf_n.energy_tot(dm=dm))
        return self.e_dfa

    def energy_pt2(self, **kwargs) -> float:
        if self.xc_dh.requires_lr_pt2:
            raise NotImplementedError(
                "Range-separated double hybrids requiring long-range PT2 are not "
                "supported by the molecular checkpoint wrapper."
            )

        if not self.xc_dh.eval_pt2:
            self.e_corr_os = 0.0
            self.e_corr_ss = 0.0
            self.e_pt2 = 0.0
            return self.e_pt2

        if self.mf_s is None:
            self.run_scf(**kwargs)

        self.mmp = mp.MP2(self.mf_s, frozen=self.frozen)
        e_corr, _ = self.mmp.kernel(with_t2=self.with_t2)
        e_corr_os = getattr(self.mmp, "e_corr_os", None)
        e_corr_ss = getattr(self.mmp, "e_corr_ss", None)
        if e_corr_os is None or e_corr_ss is None:
            if abs(self.xc_dh.c_os - self.xc_dh.c_ss) > 1e-14:
                raise RuntimeError(
                    "OS/SS MP2 components are required for spin-scaled "
                    "double-hybrid PT2."
                )
            self.e_corr_os = None
            self.e_corr_ss = None
            self.e_pt2 = self.xc_dh.c_pt2 * self.xc_dh.c_os * float(e_corr)
        else:
            self.e_corr_os = float(e_corr_os)
            self.e_corr_ss = float(e_corr_ss)
            self.e_pt2 = self.xc_dh.c_pt2 * (
                self.xc_dh.c_os * self.e_corr_os
                + self.xc_dh.c_ss * self.e_corr_ss
            )
        return self.e_pt2

    def energy_dispersion(self) -> float:
        """Add the optional dftd3-backed D3(BJ) dispersion correction."""
        from .dispersion import resolve_dispersion_correction

        correction = resolve_dispersion_correction(
            self.xc_dh, self.dispersion_correction
        )
        if correction is None:
            self.e_disp = 0.0
            return self.e_disp
        self.e_disp = float(correction(self.mol, self.xc_dh))
        return self.e_disp

    def nuc_grad_method(self):
        raise NotImplementedError(
            "Gradients are not supported by the molecular RDFDH checkpoint."
        )

    Gradients = nuc_grad_method

    def kernel(self, **kwargs) -> float:
        self.check_sanity()
        self.dump_flags()
        if self.xc_dh.requires_lr_pt2:
            raise NotImplementedError(
                "Range-separated double hybrids requiring long-range PT2 are not "
                "supported by the molecular checkpoint wrapper."
            )
        self.e_tot = (
            self.energy_dfa(**kwargs)
            + self.energy_pt2()
            + self.energy_dispersion()
        )
        return self.e_tot
