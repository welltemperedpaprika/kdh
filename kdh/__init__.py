"""Molecular and k-point periodic double-hybrid DFT for PySCF.

Public API:
    ``RDFDH``  -- molecular restricted double-hybrid driver
    ``KRDH`` / ``KDH`` -- closed-shell k-point periodic double-hybrid driver
    ``DoubleHybridFunctional`` / ``parse_dh_xc`` -- functional registry

Heavy driver classes import PySCF lazily via ``__getattr__`` so that importing
``kdh`` does not import PySCF until a driver is used.
"""
from .xc import DoubleHybridFunctional, parse_dh_xc

__all__ = ["DoubleHybridFunctional", "KDH", "KRDH", "RDFDH", "parse_dh_xc"]


def __getattr__(name):
    if name in {"KDH", "KRDH"}:
        from .krdh import KDH, KRDH

        return {"KDH": KDH, "KRDH": KRDH}[name]
    if name == "RDFDH":
        from .rdfdh import RDFDH

        return RDFDH
    raise AttributeError(f"module 'kdh' has no attribute {name!r}")
