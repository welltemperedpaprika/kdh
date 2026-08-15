"""Analytic nuclear-gradient subpackage for molecular double hybrids.

Only the restricted, conventional (non-xDH), unscaled-MP2 B2PLYP-family
molecular gradient is implemented; see :mod:`kdh.grad.rdfdh`.
Unsupported cases raise ``NotImplementedError`` from
:meth:`kdh.rdfdh.RDFDH.nuc_grad_method`.
"""
from .rdfdh import Gradients

__all__ = ["Gradients"]
