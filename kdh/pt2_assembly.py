"""Assembly of the PT2 contribution to a periodic double-hybrid energy.

Spin-scaled functionals (``c_os != c_ss``) require the opposite-spin and
same-spin KMP2 components. These are exposed by closed-shell
``pyscf.pbc.mp.KMP2`` in PySCF runtimes from 2.3.0 onward; when they are
absent the spin-scaled assembly cannot proceed and a clear error naming the
requirement is raised rather than silently mis-scaling the correlation
energy.
"""
from __future__ import annotations


def assemble_pt2_energy(functional, e_corr, e_corr_os, e_corr_ss) -> float:
    spin_scaled = abs(functional.c_os - functional.c_ss) > 1e-14
    if e_corr_os is None or e_corr_ss is None:
        if spin_scaled:
            raise RuntimeError(
                "spin-scaled periodic double-hybrid PT2 requires KMP2 "
                "e_corr_os/e_corr_ss components; this PySCF runtime does not "
                "expose them (needs PySCF >= 2.3.0 closed-shell KMP2)."
            )
        return functional.c_pt2 * functional.c_os * float(e_corr)
    return functional.c_pt2 * (
        functional.c_os * float(e_corr_os) + functional.c_ss * float(e_corr_ss)
    )
