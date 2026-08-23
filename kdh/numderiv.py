"""Finite-difference nuclear gradients and molecular geometry optimization."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from pyscf import lib
from pyscf.lib import logger

DEFAULT_STEP = 1.0e-3


def _converged_dm(driver: Any):
    """Return the converged density matrix of driver or None."""
    mf = getattr(driver, "mf_s", None)
    if mf is None:
        mf = driver
    make_rdm1 = getattr(mf, "make_rdm1", None)
    if make_rdm1 is None:
        return None
    return make_rdm1()


def _energy(driver: Any, dm0) -> float:
    """Run driver and return its total energy, optionally seeding SCF with dm0."""
    if dm0 is None:
        return float(driver.kernel())
    return float(driver.kernel(dm0=dm0))


def numerical_nuc_grad(
    driver_factory: Callable[[np.ndarray], Any],
    coords: Any,
    *,
    step: float = DEFAULT_STEP,
    dm0_reuse: bool = True,
    atmlst: Any | None = None,
    verbose: int | None = None,
) -> np.ndarray:
    """Central finite-difference nuclear gradient dE/dR in Hartree/Bohr."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (natm, 3)")
    natm = coords.shape[0]
    if atmlst is None:
        atmlst = range(natm)
    atmlst = [int(i) for i in atmlst]

    base_driver = driver_factory(coords)
    log = logger.new_logger(base_driver, verbose)
    n_eval = 2 * 3 * len(atmlst)
    log.warn(
        "numerical_nuc_grad: central finite differences require %d energy "
        "evaluations (2 x 3N for %d displaced atom(s)); density-fitting "
        "integrals are geometry-dependent and are rebuilt at every "
        "displacement.",
        n_eval,
        len(atmlst),
    )

    if dm0_reuse:
        base_driver.kernel()
        dm0 = _converged_dm(base_driver)
    else:
        dm0 = None

    grad = np.zeros((natm, 3))
    for i in atmlst:
        for x in range(3):
            plus = coords.copy()
            plus[i, x] += step
            minus = coords.copy()
            minus[i, x] -= step
            e_plus = _energy(driver_factory(plus), dm0)
            e_minus = _energy(driver_factory(minus), dm0)
            grad[i, x] = (e_plus - e_minus) / (2.0 * step)
    return grad


class _NumericalGradScanner(lib.GradScanner):
    """Gradient scanner presenting a numerical gradient to pyscf.geomopt."""

    def __init__(self, driver_factory, mol, step, dm0_reuse, verbose):
        self._driver_factory = driver_factory
        self.mol = mol
        self.step = step
        self.dm0_reuse = dm0_reuse
        self.verbose = mol.verbose if verbose is None else verbose
        self.stdout = mol.stdout
        self.base = self
        self._converged = True

    @property
    def converged(self):
        return self._converged

    def __call__(self, mol):
        coords = mol.atom_coords()
        driver = self._driver_factory(coords)
        energy = float(driver.kernel())
        grad = numerical_nuc_grad(
            self._driver_factory,
            coords,
            step=self.step,
            dm0_reuse=self.dm0_reuse,
            verbose=self.verbose,
        )
        self.mol = mol
        return energy, grad


def _import_geomopt():
    try:
        from pyscf.geomopt import geometric_solver as solver

        return solver
    except ImportError:
        pass
    try:
        from pyscf.geomopt import berny_solver as solver

        return solver
    except ImportError:
        pass
    raise ImportError(
        "Geometry optimization requires geomeTRIC or pyberny; neither is "
        "importable. Install one with 'pip install geometric' or 'pip install pyberny'."
    )


def optimize(
    driver_factory: Callable[[np.ndarray], Any],
    mol: Any,
    *,
    step: float = DEFAULT_STEP,
    dm0_reuse: bool = True,
    maxsteps: int = 100,
    verbose: int | None = None,
    **conv_params: Any,
) -> Any:
    """Optimize a molecular geometry using numerical nuclear gradients."""
    solver = _import_geomopt()
    scanner = _NumericalGradScanner(driver_factory, mol, step, dm0_reuse, verbose)
    return solver.optimize(scanner, maxsteps=maxsteps, **conv_params)


def numerical_cell_gradient(*args: Any, **kwargs: Any):
    raise NotImplementedError(
        "Numerical cell/lattice-parameter gradients are not supported for periodic double hybrids."
    )


def numerical_stress(*args: Any, **kwargs: Any):
    raise NotImplementedError(
        "Numerical stress tensors are not supported for periodic double hybrids."
    )

