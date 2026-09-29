import numpy as np
import pytest

from kdh.numderiv import numerical_nuc_grad, optimize


def _linear_energy(weights):
    def energy(coords):
        return float(np.sum(weights * coords))

    return energy


class LinearDriver:
    """Fake driver whose total energy is linear in the coordinates.

    A linear energy has an exact central finite-difference gradient equal to
    the weight matrix, so the numerics can be checked without PySCF.
    """

    def __init__(self, coords, energy_fn):
        self.coords = np.asarray(coords, dtype=float)
        self._energy_fn = energy_fn

    def kernel(self, dm0=None):
        return self._energy_fn(self.coords)


def test_linear_energy_gradient_is_exact():
    weights = np.array([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]])
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    grad = numerical_nuc_grad(
        lambda c: LinearDriver(c, _linear_energy(weights)),
        coords,
        dm0_reuse=False,
    )
    assert np.allclose(grad, weights, atol=1e-9)


def test_atmlst_restricts_displaced_atoms():
    weights = np.array([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]])
    coords = np.zeros((2, 3))
    grad = numerical_nuc_grad(
        lambda c: LinearDriver(c, _linear_energy(weights)),
        coords,
        atmlst=[1],
        dm0_reuse=False,
    )
    assert np.allclose(grad[0], 0.0)
    assert np.allclose(grad[1], weights[1], atol=1e-9)


def test_rejects_bad_coords_shape():
    with pytest.raises(ValueError, match="natm, 3"):
        numerical_nuc_grad(
            lambda c: LinearDriver(c, _linear_energy(np.zeros((1, 3)))),
            np.zeros(3),
        )


def test_dm_reuse_passes_converged_dm_as_guess():
    received = []

    class DMDriver:
        def __init__(self, coords):
            self.coords = np.asarray(coords, dtype=float)

            class _MF:
                def make_rdm1(self_inner):
                    return "DM_SENTINEL"

            self.mf_s = _MF()

        def kernel(self, dm0=None):
            received.append(dm0)
            return float(np.sum(self.coords))

    numerical_nuc_grad(DMDriver, np.zeros((1, 3)), dm0_reuse=True)
    assert received[0] is None
    assert all(dm == "DM_SENTINEL" for dm in received[1:])
    assert len(received) == 1 + 2 * 3


def test_no_dm_reuse_passes_no_guess():
    received = []

    class DMDriver:
        def __init__(self, coords):
            self.coords = np.asarray(coords, dtype=float)

        def kernel(self, dm0=None):
            received.append(dm0)
            return float(np.sum(self.coords))

    numerical_nuc_grad(DMDriver, np.zeros((1, 3)), dm0_reuse=False)
    assert received == [None] * (2 * 3)


def _geomopt_backend_available():
    try:
        from pyscf.geomopt import geometric_solver  # noqa: F401

        return True
    except ImportError:
        pass
    try:
        from pyscf.geomopt import berny_solver  # noqa: F401

        return True
    except ImportError:
        return False


def test_optimize_without_backend_raises_importerror():
    if _geomopt_backend_available():
        pytest.skip("a geometry-optimizer backend is installed")
    with pytest.raises(ImportError, match="geomeTRIC or pyberny"):
        optimize(lambda coords: None, object())
