from __future__ import annotations

from typing import Any

from pyscf import dft, lib, mp, scf
from pyscf.lib import logger

from .pt2_assembly import assemble_pt2_energy
from .xc import DoubleHybridFunctional, parse_dh_xc


def _is_pure_hf(xc: str) -> bool:
    """Return True when *xc* is an unmixed Hartree-Fock reference.

    Routed through ``scf.UHF`` (not ``dft.UKS(xc="HF")``) so the SCF skips the
    unused XC grid; the two paths are energy-identical (parity-tested).
    """
    return isinstance(xc, str) and xc.strip().upper() == "HF"


class UDFDH(lib.StreamObject):
    """Unrestricted (open-shell) molecular double-hybrid DFT driver.

    UKS (or UHF for a pure-HF reference) orbitals with UMP2 correlation. Any
    spin is accepted, including ``spin=0`` where it reproduces the RKS result of
    ``RDFDH``. The PT2 spin-scaling reuses the spin-agnostic
    ``pt2_assembly.assemble_pt2_energy``. Density fitting is opt-in via ``df``
    (``mp.UMP2`` then routes to DF-MP2). Periodic open-shell double hybrids
    remain unsupported (see ``KRDH``).
    """

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

    def _new_ks(self, xc: str):
        """Build a UKS (or UHF for pure HF) object; density_fit when ``self.df``."""
        if _is_pure_hf(xc):
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
                "Range-separated double hybrids requiring long-range PT2 are not "
                "supported by UDFDH."
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
        """Add the optional dftd3-backed D3(BJ) dispersion correction."""
        from .dispersion import resolve_dispersion_correction

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
            "Analytic gradients for the open-shell (unrestricted) UDFDH driver "
            "are not implemented: they need the spin-resolved (alpha/beta) UMP2 "
            "relaxed-density Lagrangian and the U-CPHF (UKS orbital-Hessian) "
            "Z-vector response, neither of which is assembled here. The analytic "
            "double-hybrid gradient is closed-shell (RDFDH) only."
        )

    Gradients = nuc_grad_method

    def kernel(self, **kwargs) -> float:
        self.check_sanity()
        self.dump_flags()
        if self.xc_dh.requires_lr_pt2:
            raise NotImplementedError(
                "Range-separated double hybrids requiring long-range PT2 are not "
                "supported by UDFDH."
            )
        self.e_tot = (
            self.energy_dfa(**kwargs)
            + self.energy_pt2()
            + self.energy_dispersion()
        )
        return self.e_tot
