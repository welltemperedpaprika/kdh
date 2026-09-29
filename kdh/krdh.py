"""Restricted periodic double-hybrid DFT driver with k-point sampling."""
import numpy as np
from pyscf.lib import logger
from pyscf.pbc import dft, mp, scf

from .rdfdh import DHBase
from .xc import is_pure_hf

OCC_TOL = 1e-8


class KRDH(DHBase):
    """Restricted (closed-shell) k-point periodic double-hybrid DFT.

    KMP2 is refused on references with an occupied-virtual gap below
    ``min_gap_ha`` or with fractional occupations, unless explicitly allowed.

    Attributes:
        df_backend : str
            'gdf', 'rsdf' or 'fft'.
        min_gap : float or None
            Smallest occupied-virtual gap (Hartree) over all k-points, set by
            :meth:`check_reference`.
    """

    def __init__(self, cell, xc="B2PLYP", *, kpts=None, df_backend="gdf",
                 exxdiv="ewald", min_gap_ha=0.01, allow_small_gap=False,
                 allow_fractional_occ=False, **kwargs):
        super().__init__(cell, xc, **kwargs)
        self.kpts = cell.make_kpts([1, 1, 1]) if kpts is None else kpts
        self.df_backend = df_backend
        self.exxdiv = exxdiv
        self.min_gap_ha = min_gap_ha
        self.allow_small_gap = allow_small_gap
        self.allow_fractional_occ = allow_fractional_occ
        self._keys = set(self.__dict__)

    @property
    def cell(self):
        return self.mol

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info("nkpts = %d", len(self.kpts))
        log.info("df_backend = %s", self.df_backend)
        log.info("exxdiv = %s", self.exxdiv)
        log.info("min_gap_ha = %s", self.min_gap_ha)
        return self

    def reset(self, cell=None):
        self.min_gap = None
        return super().reset(cell)

    def _new_ks(self, xc):
        if is_pure_hf(xc):
            mf = scf.KRHF(self.mol, kpts=self.kpts)
        else:
            mf = dft.KRKS(self.mol, kpts=self.kpts, xc=xc)
        mf.exxdiv = self.exxdiv
        if self.df_backend == "gdf":
            return mf.density_fit()
        if self.df_backend == "rsdf":
            return mf.rs_density_fit()
        if self.df_backend == "fft":
            return mf
        raise ValueError("df_backend must be 'gdf', 'rsdf', or 'fft'")

    def _new_mp2(self):
        return mp.KMP2(self.mf_s, frozen=self.frozen)

    def check_reference(self):
        """Check the SCF reference is gapped and integer-occupied; return the gap."""
        if self.mf_s is None:
            self.run_scf()
        gaps = []
        fractional = False
        for e, occ in zip(self.mf_s.mo_energy, self.mf_s.mo_occ):
            occupied = occ > OCC_TOL
            fractional |= bool(np.any(abs(occ[occupied] - 2) > OCC_TOL))
            if occupied.any() and not occupied.all():
                gaps.append(e[~occupied].min() - e[occupied].max())
        self.min_gap = float(min(gaps)) if gaps else None
        logger.info(self, "min occupied-virtual gap = %s Ha", self.min_gap)

        if fractional and not self.allow_fractional_occ:
            raise RuntimeError(
                "KRDH reference has fractional occupations; "
                "set allow_fractional_occ=True to run KMP2 anyway."
            )
        if (self.min_gap is None or self.min_gap < self.min_gap_ha) and not self.allow_small_gap:
            raise RuntimeError(
                f"KRDH reference gap {self.min_gap} Ha is below min_gap_ha={self.min_gap_ha}; "
                "set allow_small_gap=True to run KMP2 anyway."
            )
        return self.min_gap

    def energy_pt2(self, **kwargs):
        if self.xc_dh.eval_pt2:
            if self.mf_s is None:
                self.run_scf(**kwargs)
            self.check_reference()
        return super().energy_pt2(**kwargs)
