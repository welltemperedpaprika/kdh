import numpy as np
import pytest

pytest.importorskip("dftd3")

from pyscf.data.nist import BOHR
from pyscf.pbc import gto

from kdh.dispersion_d3 import correction
from kdh.krdh import KRDH
from kdh.xc import parse_dh_xc

B2PLYP_D3BJ = parse_dh_xc("B2PLYPD3BJ")


def _he_cell(a_ang=4.0):
    cell = gto.Cell()
    cell.atom = f"He 0 0 0; He 0 0 {a_ang / 2}"
    cell.a = np.eye(3) * a_ang
    cell.unit = "Angstrom"
    cell.basis = "gth-dzvp"
    cell.pseudo = "gth-pade"
    cell.verbose = 0
    cell.build()
    return cell


def _diamond_cell():
    cell = gto.Cell()
    cell.atom = "C 0 0 0; C 0.8925 0.8925 0.8925"
    cell.a = np.eye(3) * 3.57
    cell.unit = "Angstrom"
    cell.basis = "gth-dzvp"
    cell.pseudo = "gth-pade"
    cell.verbose = 0
    cell.build()
    return cell


def test_periodic_energy_is_attractive():
    cell = _he_cell()
    e_disp = correction(cell, B2PLYP_D3BJ, cell.make_kpts([1, 1, 1]))
    assert e_disp <= 0.0


def test_periodic_bohr_and_hartree_conventions_pinned():
    from dftd3.interface import DispersionModel, RationalDampingParam

    a_ang = 4.0
    cell = _he_cell(a_ang)
    e_adapter = correction(cell, B2PLYP_D3BJ, cell.make_kpts([1, 1, 1]))

    numbers = np.array([2, 2])
    positions_bohr = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, (a_ang / 2) / BOHR]])
    lattice_bohr = np.eye(3) * (a_ang / BOHR)
    model = DispersionModel(
        numbers,
        positions_bohr,
        lattice=lattice_bohr,
        periodic=np.array([True, True, True]),
    )
    e_ref = float(
        model.get_dispersion(RationalDampingParam(method="b2plyp"), grad=False)[
            "energy"
        ]
    )

    assert abs(e_adapter - e_ref) < 1e-10


def test_pseudopotential_cell_uses_true_atomic_number():
    from dftd3.interface import DispersionModel, RationalDampingParam

    cell = _diamond_cell()
    assert list(cell.atom_charges()) == [4, 4]

    e_adapter = correction(cell, B2PLYP_D3BJ, cell.make_kpts([1, 1, 1]))
    param = RationalDampingParam(method="b2plyp")
    periodic = np.array([True, True, True])

    model_true = DispersionModel(
        np.array([6, 6]),
        cell.atom_coords(),
        lattice=cell.lattice_vectors(),
        periodic=periodic,
    )
    e_true = float(model_true.get_dispersion(param, grad=False)["energy"])

    model_valence = DispersionModel(
        np.array([4, 4]),
        cell.atom_coords(),
        lattice=cell.lattice_vectors(),
        periodic=periodic,
    )
    e_valence = float(model_valence.get_dispersion(param, grad=False)["energy"])

    assert abs(e_adapter - e_true) < 1e-12
    assert abs(e_adapter - e_valence) > 1e-6


def test_kpts_do_not_change_the_dispersion_energy():
    cell = _he_cell()
    e_gamma = correction(cell, B2PLYP_D3BJ, cell.make_kpts([1, 1, 1]))
    e_mesh = correction(cell, B2PLYP_D3BJ, cell.make_kpts([2, 2, 2]))
    e_none = correction(cell, B2PLYP_D3BJ, None)

    assert e_gamma == pytest.approx(e_mesh, abs=0.0, rel=0.0)
    assert e_gamma == pytest.approx(e_none, abs=0.0, rel=0.0)


def test_dilating_the_cell_reduces_the_dispersion_magnitude():
    e_dense = correction(_he_cell(4.0), B2PLYP_D3BJ, None)
    e_dilute = correction(_he_cell(8.0), B2PLYP_D3BJ, None)

    assert e_dense < e_dilute <= 0.0


def test_krdh_energy_dispersion_uses_builtin_adapter():
    cell = _he_cell()
    kpts = cell.make_kpts([1, 1, 1])
    krdh = KRDH(cell, xc="B2PLYPD3BJ", kpts=kpts)
    e_disp = krdh.energy_dispersion()

    assert e_disp == pytest.approx(correction(cell, B2PLYP_D3BJ, kpts))
    assert e_disp <= 0.0
