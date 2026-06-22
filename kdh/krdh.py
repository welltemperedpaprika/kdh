from __future__ import annotations

from typing import Any

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import dft, mp

from .xc import DoubleHybridFunctional, parse_dh_xc


class KRDH(lib.StreamObject):
    """Restricted periodic double-hybrid DFT driver (k-point sampling); closed-shell (spin=0) only."""

    def __init__(
        self,
        cell: Any,
        xc: str | dict[str, Any] | DoubleHybridFunctional = "B2PLYP",
        *,
        kpts: Any | None = None,
        frozen: int | list[int] | None = None,
        with_t2: bool = False,
        df_backend: str = "gdf",
        exxdiv: str | None = "ewald",
        min_gap_ha: float = 0.01,
        allow_small_gap: bool = False,
        allow_fractional_occ: bool = False,
        scf_stabilization: Any | None = None,
        dispersion_correction=None,
    ) -> None:
        if getattr(cell, "spin", 0) != 0:
            raise NotImplementedError(
                "Open-shell periodic double hybrids are not supported: PySCF "
                "has no working periodic unrestricted KMP2 "
                "(pyscf.pbc.mp.kump2.KUMP2.kernel raises NotImplementedError). "
                "Only closed-shell spin=0 cells are supported."
            )

        self.cell = cell
        self.mol = cell
        self.stdout = getattr(cell, "stdout", None)
        self.verbose = getattr(cell, "verbose", 0)
        self.max_memory = getattr(cell, "max_memory", 4000)
        self.xc_dh = parse_dh_xc(xc)
        self.kpts = kpts if kpts is not None else cell.make_kpts([1, 1, 1])
        self.frozen = frozen
        self.with_t2 = with_t2
        self.df_backend = df_backend
        self.exxdiv = exxdiv
        self.min_gap_ha = min_gap_ha
        self.allow_small_gap = allow_small_gap
        self.allow_fractional_occ = allow_fractional_occ
        self.scf_stabilization = scf_stabilization
        self.dispersion_correction = dispersion_correction

        self.mf_s = None
        self.mf_n = None
        self.kmp2 = None
        self.reference_safety = None
        self._scf_stabilizer_handle = None
        self.e_scf = None
        self.e_dfa = None
        self.e_pt2 = None
        self.e_corr_os = None
        self.e_corr_ss = None
        self.e_tot = None
        self.e_disp = None
        self._keys = set(self.__dict__.keys())

    @property
    def xc(self) -> str:
        """SCF exchange-correlation functional string (alias for ``xc_dh.xc_scf``)."""
        return self.xc_dh.xc_scf

    @property
    def xc_n(self) -> str | None:
        """Non-SCF exchange-correlation functional string, or ``None`` when absent."""
        return self.xc_dh.xc_nscf

    def dump_flags(self, verbose=None):
        """Log all driver settings at INFO level and return self."""
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
        log.info("df_backend = %s", self.df_backend)
        log.info("exxdiv = %s", self.exxdiv)
        log.info("min_gap_ha = %s", self.min_gap_ha)
        log.info("allow_small_gap = %s", self.allow_small_gap)
        log.info("allow_fractional_occ = %s", self.allow_fractional_occ)
        log.info("scf_stabilization = %s", self.scf_stabilization)
        log.info("dispersion = %s", self.xc_dh.dispersion)
        return self

    def reset(self, cell=None):
        """Clear all cached results and optionally replace the cell."""
        if cell is not None:
            if getattr(cell, "spin", 0) != 0:
                raise NotImplementedError(
                    "Open-shell periodic double hybrids are not supported: PySCF "
                    "has no working periodic unrestricted KMP2 "
                    "(pyscf.pbc.mp.kump2.KUMP2.kernel raises NotImplementedError). "
                    "Only closed-shell spin=0 cells are supported."
                )
            self.cell = cell
            self.mol = cell
        self.mf_s = None
        self.mf_n = None
        self.kmp2 = None
        self.reference_safety = None
        self._scf_stabilizer_handle = None
        self.e_scf = None
        self.e_dfa = None
        self.e_pt2 = None
        self.e_corr_os = None
        self.e_corr_ss = None
        self.e_tot = None
        self.e_disp = None
        return self

    def _new_ks(self, xc: str):
        """Build a KRKS object with the driver's df_backend and exxdiv."""
        mf = dft.KRKS(self.cell, kpts=self.kpts, xc=xc)
        mf.exxdiv = self.exxdiv
        if self.df_backend == "gdf":
            return mf.density_fit()
        if self.df_backend == "rsdf":
            return mf.rs_density_fit()
        if self.df_backend == "fft":
            return mf
        raise ValueError("df_backend must be 'gdf', 'rsdf', or 'fft'")

    def run_scf(self, **kwargs):
        """Run the SCF step and store the converged mean-field as ``self.mf_s``."""
        from .scf_stabilizers import configure_periodic_scf

        self.mf_s = self._new_ks(self.xc_dh.xc_scf)
        self._scf_stabilizer_handle = configure_periodic_scf(
            self.mf_s,
            self.scf_stabilization,
        )
        try:
            self.e_scf = self.mf_s.kernel(**kwargs)
        finally:
            if self._scf_stabilizer_handle is not None:
                self._scf_stabilizer_handle.restore()
                self._scf_stabilizer_handle = None
        if not self.mf_s.converged:
            raise RuntimeError("KRDH SCF did not converge")
        return self.mf_s

    def check_reference_safety(self):
        """Check the SCF reference for small gaps and fractional occupations."""
        from .diagnostics import format_reference_safety, reference_safety_report

        if self.mf_s is None:
            self.run_scf()
        report = reference_safety_report(
            self.mf_s,
            min_gap_ha=self.min_gap_ha,
        )
        self.reference_safety = report
        if report["has_fractional_occ"] and not self.allow_fractional_occ:
            raise RuntimeError(format_reference_safety(report))
        if not report["ok"] and not self.allow_small_gap:
            raise RuntimeError(format_reference_safety(report))
        return report

    def energy_dfa(self, **kwargs) -> float:
        """Compute and return the DFA energy (SCF + optional nscf XC re-evaluation)."""
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
        """Evaluate the scaled PT2 correlation contribution via canonical KMP2."""
        if self.xc_dh.requires_lr_pt2:
            raise NotImplementedError(
                "Range-separated double hybrids requiring long-range PT2 are not "
                "supported by the periodic DH driver."
            )
        if not self.xc_dh.eval_pt2:
            self.e_corr_os = 0.0
            self.e_corr_ss = 0.0
            self.e_pt2 = 0.0
            return self.e_pt2
        if self.mf_s is None:
            self.run_scf(**kwargs)
        self.check_reference_safety()
        self.kmp2 = mp.KMP2(self.mf_s, frozen=self.frozen)
        e_corr, _ = self.kmp2.kernel(with_t2=self.with_t2)
        from .pt2_assembly import assemble_pt2_energy

        e_corr_os = getattr(self.kmp2, "e_corr_os", None)
        e_corr_ss = getattr(self.kmp2, "e_corr_ss", None)
        self.e_corr_os = None if e_corr_os is None else float(e_corr_os)
        self.e_corr_ss = None if e_corr_ss is None else float(e_corr_ss)
        self.e_pt2 = assemble_pt2_energy(self.xc_dh, e_corr, e_corr_os, e_corr_ss)
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
        self.e_disp = float(correction(self.cell, self.xc_dh))
        return self.e_disp

    def nuc_grad_method(self):
        raise NotImplementedError(
            "Periodic double-hybrid gradients require KMP2 relaxed-density or "
            "Z-vector machinery that PySCF does not currently expose."
        )

    Gradients = nuc_grad_method

    def polar_method(self):
        raise NotImplementedError(
            "Periodic double-hybrid properties require a separate periodic "
            "response formulation and are not supported by this driver."
        )

    def kernel(self, **kwargs) -> float:
        """Run SCF, the DFA, the scaled PT2 correction, and dispersion."""
        self.check_sanity()
        self.dump_flags()
        self.e_tot = (
            self.energy_dfa(**kwargs)
            + self.energy_pt2()
            + self.energy_dispersion()
        )
        return self.e_tot


KDH = KRDH
