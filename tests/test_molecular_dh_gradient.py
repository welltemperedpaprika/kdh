"""Scope/refusal tests for the analytic molecular double-hybrid nuclear gradient.

The analytic gradient covers closed-shell, conventional, unscaled-PT2 double
hybrids (e.g. B2PLYP); the analytic-vs-finite-difference check lives in
tests/test_native_smoke.py.
"""
import numpy as np
import pytest

from kdh.krdh import KRDH
from kdh.rdfdh import RDFDH
from kdh.udfdh import UDFDH


class FakeMol:
    verbose = 0
    stdout = None
    max_memory = 4000
    spin = 0


class OpenShellMol(FakeMol):
    spin = 1


class FakeCell:
    verbose = 0
    stdout = None
    max_memory = 4000
    spin = 0

    def make_kpts(self, mesh):
        return np.zeros((1, 3))


def test_supported_b2plyp_is_dh_owned_not_native_mp2():
    grad = RDFDH(FakeMol(), xc="B2PLYP").nuc_grad_method()
    from pyscf.grad import mp2 as native_mp2_grad

    assert type(grad).__module__.startswith("kdh.grad")
    assert not isinstance(grad, native_mp2_grad.Gradients)


def test_xdh_gradient_refused_naming_nscf():
    with pytest.raises(NotImplementedError, match="xc_nscf"):
        RDFDH(FakeMol(), xc="XYG3").nuc_grad_method()


def test_spin_scaled_gradient_refused_naming_os_ss():
    with pytest.raises(NotImplementedError, match="c_os"):
        RDFDH(FakeMol(), xc="SCSMP2").nuc_grad_method()


def test_dispersion_gradient_refused():
    with pytest.raises(NotImplementedError, match="dispersion"):
        RDFDH(FakeMol(), xc="B2PLYPD3BJ").nuc_grad_method()


def test_df_gradient_refused():
    with pytest.raises(NotImplementedError, match="density-fitted"):
        RDFDH(FakeMol(), xc="B2PLYP", df=True).nuc_grad_method()


def test_frozen_gradient_refused():
    with pytest.raises(NotImplementedError, match="frozen"):
        RDFDH(FakeMol(), xc="B2PLYP", frozen=1).nuc_grad_method()


def test_open_shell_gradient_refused():
    with pytest.raises(NotImplementedError, match="UDFDH"):
        UDFDH(OpenShellMol(), xc="B2PLYP").nuc_grad_method()


def test_periodic_gradient_refused():
    with pytest.raises(NotImplementedError, match="KRDH"):
        KRDH(FakeCell(), xc="B2PLYP").nuc_grad_method()
