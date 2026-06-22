from __future__ import annotations

from typing import Any

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import dft, mp

from .xc import DoubleHybridFunctional, parse_dh_xc


class KRDH(lib.StreamObject):
    """Restricted periodic double-hybrid DFT driver (k-point sampling).

    Implements the energy as::

        E_DH = E_DFA[xc_scf or xc_nscf] + c_pt2 * E_PT2

    where E_PT2 is evaluated with KMP2 on the SCF orbitals.  Only
    closed-shell (spin=0) cells are supported; PySCF has no working periodic
    unrestricted KMP2.

    Attributes
    ----------
    cell : pyscf.pbc.gto.Cell
        Periodic cell object (also aliased to ``mol`` for StreamObject compat).
    xc_dh : DoubleHybridFunctional
        Parsed double-hybrid functional descriptor.
    kpts : array_like
        k-point mesh used for the SCF and PT2 steps.
    frozen : int or list of int or None
        Frozen-core specification passed to KMP2.
    with_t2 : bool
        Whether to store MP2 amplitudes.
    df_backend : {'gdf', 'rsdf', 'fft'}
        Density-fitting backend for the SCF and PT2 steps.
    exxdiv : str or None
        Divergence treatment for exact exchange (default: 'ewald' Madelung).
    min_gap_ha : float
        Minimum occupied-virtual gap (Ha) below which PT2 is refused unless
        ``allow_small_gap=True``.
    allow_small_gap : bool
        Bypass the small-gap safety guard.
    allow_fractional_occ : bool
        Bypass the fractional-occupation safety guard.
    scf_stabilization : dict or None
        Keyword arguments forwarded to ``configure_periodic_scf``.
    mf_s : KRKS or None
        SCF mean-field object (set after ``run_scf``).
    mf_n : KRKS or None
        Non-self-consistent mean-field object for the nscf exchange-correlation
        step (set after ``energy_dfa``; equals ``mf_s`` when ``xc_nscf`` is None).
    kmp2 : KMP2 or None
        MP2 object (set after ``energy_pt2``).
    reference_safety : dict or None
        Output of ``check_reference_safety`` (set during ``energy_pt2``).
    e_scf, e_dfa, e_pt2, e_tot : float or None
        Energy components populated by the respective ``energy_*`` methods.
    e_corr_os, e_corr_ss : float or None
        Opposite-spin and same-spin MP2 correlation energies when available.
    """

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
    ) -> None:
        """Initialize KRDH for *cell* with the given double-hybrid functional.

        Parameters
        ----------
        cell : pyscf.pbc.gto.Cell
            Periodic cell.  Must have ``spin == 0``; open-shell cells raise
            ``NotImplementedError`` because PySCF has no working KUMP2.
        xc : str or dict or DoubleHybridFunctional
            Double-hybrid functional name, raw parameter dict, or pre-parsed
            ``DoubleHybridFunctional`` descriptor.
        kpts : array_like or None
            k-point mesh.  Defaults to a single Gamma point when ``None``.
        frozen : int or list of int or None
            Frozen orbitals passed directly to ``KMP2``.
        with_t2 : bool
            Store MP2 t2 amplitudes.
        df_backend : {'gdf', 'rsdf', 'fft'}
            Density-fitting back-end for the KRKS and KMP2 objects.
        exxdiv : str or None
            Exchange divergence treatment (default ``'ewald'``).
        min_gap_ha : float
            Gap threshold (Ha) for the reference-safety guard.
        allow_small_gap : bool
            Suppress the small-gap RuntimeError.
        allow_fractional_occ : bool
            Suppress the fractional-occupation RuntimeError.
        scf_stabilization : dict or None
            Options forwarded to ``configure_periodic_scf``.
        """
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
        return self

    def reset(self, cell=None):
        """Clear all cached results and optionally replace the cell.

        All energy attributes and intermediate objects (``mf_s``, ``mf_n``,
        ``kmp2``, ``reference_safety``) are set to ``None``.  If *cell* is
        provided it must have ``spin == 0``.
        """
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
        return self

    def _new_ks(self, xc: str):
        """Build a KRKS object with exxdiv pinned on both xc_scf and xc_nscf so
        the Madelung correction is applied consistently (PySCF scales it by each
        functional's hybrid fraction)."""
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
        """Run the SCF step and store the converged mean-field as ``self.mf_s``.

        Applies any requested SCF stabilization, then asserts convergence.
        Extra keyword arguments are forwarded to ``mf_s.kernel``.

        Returns
        -------
        mf_s : KRKS
            Converged SCF mean-field object.

        Raises
        ------
        RuntimeError
            If the SCF does not converge.
        """
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
        """Check the SCF reference for small gaps and fractional occupations.

        Runs the SCF if not already done, then calls
        ``diagnostics.reference_safety_report``.  The result is stored as
        ``self.reference_safety``.

        Returns
        -------
        report : dict
            Diagnostic report as returned by ``reference_safety_report``.

        Raises
        ------
        RuntimeError
            If fractional occupations are detected and ``allow_fractional_occ``
            is ``False``, or if the gap is below ``min_gap_ha`` and
            ``allow_small_gap`` is ``False``.
        """
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
        """Compute and return the DFA energy (SCF + optional nscf XC re-evaluation).

        When ``xc_nscf`` is set, the SCF density matrix is evaluated with the
        non-self-consistent functional to obtain the DFA energy component.
        Otherwise the SCF total energy is used directly.  The result is stored
        as ``self.e_dfa`` and ``self.mf_n`` is set to the evaluation object.

        Returns
        -------
        e_dfa : float
            DFA energy in Hartree.
        """
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
                "supported by the periodic DH checkpoint."
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

    def nuc_grad_method(self):
        raise NotImplementedError(
            "Periodic double-hybrid gradients require KMP2 relaxed-density or "
            "Z-vector machinery that PySCF does not currently expose."
        )

    Gradients = nuc_grad_method

    def polar_method(self):
        raise NotImplementedError(
            "Periodic double-hybrid properties require a separate periodic "
            "response formulation and are not supported by this checkpoint."
        )

    def kernel(self, **kwargs) -> float:
        """Run SCF, the non-self-consistent DFA, and the scaled PT2 correction."""
        self.check_sanity()
        self.dump_flags()
        self.e_tot = self.energy_dfa(**kwargs) + self.energy_pt2()
        return self.e_tot


KDH = KRDH
