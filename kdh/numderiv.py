"""Finite-difference nuclear gradients and molecular geometry optimization."""
import numpy as np

DEFAULT_STEP = 1e-3


def _fd_grad(driver_factory, coords, step, dm0, atmlst):
    grad = np.zeros_like(coords)
    for i in atmlst:
        for x in range(3):
            e = []
            for sign in (1, -1):
                displaced = coords.copy()
                displaced[i, x] += sign * step
                e.append(driver_factory(displaced).kernel(dm0=dm0))
            grad[i, x] = (e[0] - e[1]) / (2 * step)
    return grad


def numerical_nuc_grad(driver_factory, coords, *, step=DEFAULT_STEP, dm0_reuse=True,
                       atmlst=None):
    """Central finite-difference nuclear gradient dE/dR in Hartree/Bohr.

    Args:
        driver_factory : callable
            Maps coordinates (natm, 3) in Bohr to a driver such as RDFDH whose
            ``kernel(dm0=...)`` returns the total energy.
        dm0_reuse : bool
            Seed every displaced SCF with the reference density matrix.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (natm, 3)")
    dm0 = None
    if dm0_reuse:
        driver = driver_factory(coords)
        driver.kernel()
        dm0 = driver.mf_s.make_rdm1()
    if atmlst is None:
        atmlst = range(len(coords))
    return _fd_grad(driver_factory, coords, step, dm0, atmlst)


def optimize(driver_factory, mol, *, step=DEFAULT_STEP, dm0_reuse=True, maxsteps=100,
             **conv_params):
    """Optimize a molecular geometry with numerical gradients (geomeTRIC or pyberny)."""
    try:
        from pyscf.geomopt import geometric_solver as solver
    except ImportError:
        try:
            from pyscf.geomopt import berny_solver as solver
        except ImportError:
            raise ImportError(
                "Geometry optimization requires geomeTRIC or pyberny "
                "(pip install geometric)."
            ) from None
    from pyscf.geomopt.addons import as_pyscf_method

    def energy_and_grad(mol):
        coords = mol.atom_coords()
        driver = driver_factory(coords)
        e_tot = driver.kernel()
        dm0 = driver.mf_s.make_rdm1() if dm0_reuse else None
        return e_tot, _fd_grad(driver_factory, coords, step, dm0, range(mol.natm))

    method = as_pyscf_method(mol, energy_and_grad)
    return solver.optimize(method, maxsteps=maxsteps, **conv_params)
