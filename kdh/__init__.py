"""Molecular and k-point periodic double-hybrid DFT for PySCF.

Public API: RDFDH (molecular closed-shell), UDFDH (molecular open-shell),
KRDH / KDH (periodic), DoubleHybridFunctional, parse_dh_xc.
"""
from .xc import DoubleHybridFunctional, parse_dh_xc

__all__ = [
    "DoubleHybridFunctional",
    "KDH",
    "KRDH",
    "RDFDH",
    "UDFDH",
    "parse_dh_xc",
]


def __getattr__(name):
    if name in {"KDH", "KRDH"}:
        from .krdh import KDH, KRDH

        return {"KDH": KDH, "KRDH": KRDH}[name]
    if name == "RDFDH":
        from .rdfdh import RDFDH

        return RDFDH
    if name == "UDFDH":
        from .udfdh import UDFDH

        return UDFDH
    raise AttributeError(f"module 'kdh' has no attribute {name!r}")
