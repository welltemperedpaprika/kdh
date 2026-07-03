"""Analytic nuclear-gradient subpackage for molecular double hybrids.

Only the restricted, conventional (non-xDH), unscaled-MP2 B2PLYP-family
molecular gradient is implemented; see :mod:`kdh.grad.rdfdh`. Every unsupported
case is refused by :meth:`kdh.rdfdh.RDFDH.nuc_grad_method` with a message
naming the missing response terms.
"""
from .rdfdh import Gradients

__all__ = ["Gradients"]
