"""Reference-safety diagnostics for periodic double-hybrid calculations.

Provides gap analysis and fractional-occupation detection utilities used by
``KRDH.check_reference_safety`` to guard against degenerate references that
would make PT2 energetically meaningless or numerically catastrophic.
"""
from __future__ import annotations

from typing import Any

import numpy as np

HARTREE_TO_EV = 27.211386245988


def _as_kpoint_arrays(values: Any) -> list[np.ndarray]:
    """Normalize orbital data to a list of per-k-point float arrays."""
    if isinstance(values, (list, tuple)):
        return [np.asarray(value, dtype=float) for value in values]

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return [array]
    if array.ndim == 2:
        return [array[k] for k in range(array.shape[0])]
    raise ValueError("orbital data must be a 1D array or k-point sequence")


def _has_fractional_occ(occ: np.ndarray, fractional_tol: float) -> bool:
    """Return True if any occupied orbital has a non-integer occupation."""
    occupied = occ > fractional_tol
    if not np.any(occupied):
        return False
    return bool(np.any(np.abs(occ[occupied] - 2.0) > fractional_tol))


def orbital_gap_report(mo_energy, mo_occ, *, fractional_tol=1e-8) -> dict:
    """Compute the occupied-virtual gap across all k-points."""
    energy_by_kpt = _as_kpoint_arrays(mo_energy)
    occ_by_kpt = _as_kpoint_arrays(mo_occ)
    if len(energy_by_kpt) != len(occ_by_kpt):
        raise ValueError("mo_energy and mo_occ must have the same number of k-points")

    gaps = []
    has_fractional_occ = False
    missing_gap = False
    for energy, occ in zip(energy_by_kpt, occ_by_kpt, strict=True):
        if energy.shape != occ.shape:
            raise ValueError("mo_energy and mo_occ arrays must have matching shapes")
        has_fractional_occ = has_fractional_occ or _has_fractional_occ(
            occ,
            fractional_tol,
        )
        occupied_energy = energy[occ > fractional_tol]
        virtual_energy = energy[occ <= fractional_tol]
        if occupied_energy.size == 0 or virtual_energy.size == 0:
            missing_gap = True
            continue
        gaps.append(float(np.min(virtual_energy) - np.max(occupied_energy)))

    min_gap_ha = min(gaps) if gaps else None
    min_gap_ev = min_gap_ha * HARTREE_TO_EV if min_gap_ha is not None else None
    ok = min_gap_ha is not None
    if missing_gap or min_gap_ha is None:
        reason = "missing occupied or virtual orbitals for gap diagnostic"
    elif has_fractional_occ:
        reason = "fractional occupations detected"
    else:
        reason = "reference has an occupied-virtual gap"

    return {
        "ok": ok,
        "min_gap_ha": min_gap_ha,
        "min_gap_ev": min_gap_ev,
        "has_fractional_occ": has_fractional_occ,
        "n_kpts": len(energy_by_kpt),
        "reason": reason,
    }


def reference_safety_report(
    mf,
    *,
    min_gap_ha=0.01,
    fractional_tol=1e-8,
) -> dict:
    """Assess whether the SCF reference is safe for a periodic DH calculation."""
    report = orbital_gap_report(
        mf.mo_energy,
        mf.mo_occ,
        fractional_tol=fractional_tol,
    )
    gap = report["min_gap_ha"]
    if gap is None:
        report["ok"] = False
        return report
    if gap < min_gap_ha:
        report["ok"] = False
        report["reason"] = (
            f"small-gap reference: minimum gap {gap:.8g} Ha is below "
            f"threshold {min_gap_ha:.8g} Ha"
        )
        return report

    report["ok"] = True
    if report["has_fractional_occ"]:
        report["reason"] = "fractional occupations detected"
    else:
        report["reason"] = (
            f"minimum gap {gap:.8g} Ha is above threshold {min_gap_ha:.8g} Ha"
        )
    return report


def format_reference_safety(report: dict) -> str:
    """Format a reference-safety report dict as a human-readable error string."""
    gap_ha = report["min_gap_ha"]
    gap_ev = report["min_gap_ev"]
    if gap_ha is None:
        gap_text = "minimum gap unavailable"
    else:
        gap_text = f"minimum gap {gap_ha:.8g} Ha ({gap_ev:.6g} eV)"
    fractional = "fractional occupations present" if report[
        "has_fractional_occ"
    ] else "integer occupations"
    return (
        "Periodic DH reference safety check failed or requires override: "
        f"{report['reason']}; {gap_text}; {fractional}; "
        f"{report['n_kpts']} k-point(s)"
    )
