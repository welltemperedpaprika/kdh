"""PT2 correlation energy assembly for double-hybrid functionals."""
from __future__ import annotations


def assemble_pt2_energy(functional, e_corr, e_corr_os=None, e_corr_ss=None) -> float:
    """Scale and combine MP2 correlation energy components."""
    if not functional.eval_pt2:
        return 0.0
    spin_scaled = abs(functional.c_os - functional.c_ss) > 1e-14
    if e_corr_os is None or e_corr_ss is None:
        if spin_scaled:
            raise RuntimeError(
                "spin-scaled double-hybrid PT2 requires OS/SS MP2 components (e_corr_os and e_corr_ss)."
            )
        return functional.c_pt2 * functional.c_os * float(e_corr)
    return functional.c_pt2 * (
        functional.c_os * float(e_corr_os) + functional.c_ss * float(e_corr_ss)
    )

