from __future__ import annotations

from typing import Any

from pyscf import dft, lib, mp, scf
from pyscf.lib import logger

from .xc import DoubleHybridFunctional, parse_dh_xc


def _is_pure_hf(xc: str) -> bool:
    """Return True when *xc* is an unmixed Hartree-Fock reference.

    Routed through ``scf.RHF`` (not ``dft.RKS(xc="HF")``) so the SCF skips the
    unused XC grid; the two paths are energy-identical (parity-tested).
    """
    return isinstance(xc, str) and xc.strip().upper() == "HF"


class RDFDH(lib.StreamObject):
    """Restricted molecular double-hybrid DFT driver (closed-shell only).

    Density fitting is opt-in via ``df``. The default ``df=False`` keeps the
    conventional RKS SCF + conventional 4-index RMP2 path (no energy changes).
    ``df=True`` builds the SCF with ``.density_fit()`` and the ``mp.MP2`` factory
    then routes the PT2 step through DF-MP2, which also exposes the OS/SS
    components used by spin-scaled functionals.
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
        conv_tol: float | None = None,
        conv_tol_grad: float | None = None,
        grids_level: int | None = None,
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
        self.df = df
        self.conv_tol = conv_tol
        self.conv_tol_grad = conv_tol_grad
        self.grids_level = grids_level

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
                "RDFDH only supports closed-shell spin=0 molecules."
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
        log.info("df = %s", self.df)
        log.info("dispersion = %s", self.xc_dh.dispersion)
        log.info("dispersion_correction = %s", self.dispersion_correction)
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

    def _new_ks(self, xc: str):
        """Build an RKS (or RHF for pure HF) molecular mean-field object.

        When ``self.df`` is set the object is wrapped with ``.density_fit()``;
        the ``mp.MP2`` factory then routes the PT2 step to DF-MP2.
        """
        if _is_pure_hf(xc):
            mf = scf.RHF(self.mol)
        else:
            mf = dft.RKS(self.mol, xc=xc)
        if self.df:
            return mf.density_fit()
        return mf

    def _apply_scf_options(self, mf):
        """Apply opt-in SCF-tightness options (conv_tol, conv_tol_grad, grids_level).

        All default to ``None`` and leave PySCF's defaults untouched, so existing
        energies are unchanged. They let a caller (notably the analytic-gradient
        validation) converge the SCF and grid tightly enough that a
        finite-difference reference is trustworthy.
        """
        if self.conv_tol is not None:
            mf.conv_tol = self.conv_tol
        if self.conv_tol_grad is not None:
            mf.conv_tol_grad = self.conv_tol_grad
        if self.grids_level is not None and hasattr(mf, "grids"):
            mf.grids.level = self.grids_level
        return mf

    def run_scf(self, **kwargs):
        self.mf_s = self._apply_scf_options(self._new_ks(self.xc_dh.xc_scf))
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

        self.mf_n = self._new_ks(self.xc_dh.xc_nscf)
        self.mf_n.grids = self.mf_s.grids
        dm = self.mf_s.make_rdm1()
        self.e_dfa = float(self.mf_n.energy_tot(dm=dm))
        return self.e_dfa

    def energy_pt2(self, **kwargs) -> float:
        if self.xc_dh.requires_lr_pt2:
            raise NotImplementedError(
                "Range-separated double hybrids requiring long-range PT2 are not "
                "supported by RDFDH."
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
        self.e_disp = float(correction(self.mol, self.xc_dh, None))
        return self.e_disp

    def nuc_grad_method(self):
        raise NotImplementedError(
            "Gradients are not supported by RDFDH."
        )

    Gradients = nuc_grad_method

    def kernel(self, **kwargs) -> float:
        self.check_sanity()
        self.dump_flags()
        if self.xc_dh.requires_lr_pt2:
            raise NotImplementedError(
                "Range-separated double hybrids requiring long-range PT2 are not "
                "supported by RDFDH."
            )
        self.e_tot = (
            self.energy_dfa(**kwargs)
            + self.energy_pt2()
            + self.energy_dispersion()
        )
        return self.e_tot
