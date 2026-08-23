"""Native (in-process) smoke gates that exercise real PySCF SCF/MP2.

Kept deliberately small: one gamma-point KRDH-vs-manual parity smoke and the
analytic-vs-finite-difference double-hybrid gradient gate. Both need PySCF and
are marked ``native`` so they can be deselected (``-m 'not native'``); they skip
cleanly when PySCF is absent.
"""
import numpy as np
import pytest

pytest.importorskip("pyscf")

pytestmark = pytest.mark.native


def test_krdh_b2plyp_matches_manual_native_kmp2_energy():
    from pyscf.pbc import dft, gto, mp

    from kdh.krdh import KRDH

    cell = gto.Cell()
    cell.atom = "He 0 0 0"
    cell.a = np.eye(3) * 4.0
    cell.basis = "gth-dzvp"
    cell.pseudo = "gth-pade"
    cell.verbose = 0
    cell.build()
    kpts = cell.make_kpts([1, 1, 1])

    xc = "0.53*HF + 0.47*B88, 0.73*LYP"
    wrapped = KRDH(cell, xc="B2PLYP", kpts=kpts, df_backend="gdf")
    e_wrapped = wrapped.kernel()

    mf = dft.KRKS(cell, kpts=kpts, xc=xc).density_fit()
    mf.conv_tol = 1e-8
    mf.kernel()
    mymp = mp.KMP2(mf)
    mymp.kernel(with_t2=False)
    e_manual = mf.e_tot + 0.27 * (mymp.e_corr_os + mymp.e_corr_ss)

    assert abs(e_wrapped - e_manual) < 1e-8
    assert wrapped.reference_safety is not None
    assert wrapped.reference_safety["min_gap_ha"] is not None


def test_b2plyp_analytic_gradient_matches_numderiv():
    from pyscf import gto

    from kdh.numderiv import numerical_nuc_grad
    from kdh.rdfdh import RDFDH

    atom = "O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24"

    def factory(coords):
        mol = gto.M(atom=atom, basis="sto-3g", verbose=0)
        mol.set_geom_(coords, unit="Bohr")
        return RDFDH(
            mol, xc="B2PLYP", conv_tol=1e-12, conv_tol_grad=1e-9, grids_level=5
        )

    mol0 = gto.M(atom=atom, basis="sto-3g", verbose=0)
    coords0 = mol0.atom_coords()

    drv = factory(coords0)
    drv.kernel()
    g_an = drv.nuc_grad_method().kernel()
    g_fd = numerical_nuc_grad(factory, coords0, step=1e-3)

    assert float(np.max(np.abs(g_an - g_fd))) < 1e-6
