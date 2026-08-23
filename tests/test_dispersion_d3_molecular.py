import pytest

pytest.importorskip("dftd3")

from pyscf import gto

from kdh.dispersion_d3 import correction
from kdh.rdfdh import RDFDH
from kdh.xc import parse_dh_xc

B2PLYP_D3BJ = parse_dh_xc("B2PLYPD3BJ")


def _ne_dimer():
    return gto.M(atom="Ne 0 0 0; Ne 0 0 3.0", basis="sto-3g", verbose=0)


def _water():
    return gto.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="sto-3g",
        verbose=0,
    )


@pytest.mark.parametrize("mol_factory", [_ne_dimer, _water])
def test_molecular_matches_independent_dftd3_reference(mol_factory):
    from dftd3.pyscf import DFTD3Dispersion

    mol = mol_factory()
    e_adapter = correction(mol, B2PLYP_D3BJ, None)
    e_ref = float(DFTD3Dispersion(mol, xc="b2plyp", version="d3bj").kernel()[0])

    assert abs(e_adapter - e_ref) < 1e-8
    assert e_adapter <= 0.0


def test_b2plyp_library_matches_literature_damping_params():
    mol = _water()
    f_library = parse_dh_xc(
        {
            "name": "B2PLYP-d3-library",
            "xc_scf": "0.53*HF + 0.47*B88, 0.73*LYP",
            "c_pt2": 0.27,
            "dispersion": {"method": "d3bj", "xc": "b2plyp"},
        }
    )
    f_literature = parse_dh_xc(
        {
            "name": "B2PLYP-d3-literature",
            "xc_scf": "0.53*HF + 0.47*B88, 0.73*LYP",
            "c_pt2": 0.27,
            "dispersion": {
                "method": "d3bj",
                "params": {"s6": 0.64, "a1": 0.3065, "s8": 0.9147, "a2": 5.057},
            },
        }
    )

    f_literature_s9 = parse_dh_xc(
        {
            "name": "B2PLYP-d3-literature-s9",
            "xc_scf": "0.53*HF + 0.47*B88, 0.73*LYP",
            "c_pt2": 0.27,
            "dispersion": {
                "method": "d3bj",
                "params": {
                    "s6": 0.64,
                    "a1": 0.3065,
                    "s8": 0.9147,
                    "a2": 5.057,
                    "s9": 0.0,
                },
            },
        }
    )

    e_library = correction(mol, f_library, None)
    e_literature = correction(mol, f_literature, None)
    e_literature_s9 = correction(mol, f_literature_s9, None)

    assert abs(e_library - e_literature) < 1e-12
    assert e_literature == e_literature_s9


def test_unknown_method_is_refused():
    f = parse_dh_xc(
        {
            "name": "bogus",
            "xc_scf": "PBE",
            "c_pt2": 0.5,
            "dispersion": {"method": "d5-imaginary", "xc": "b2plyp"},
        }
    )
    with pytest.raises(ValueError, match="unknown dispersion method"):
        correction(_water(), f, None)


def test_d4_method_is_refused_naming_dftd4():
    f = parse_dh_xc(
        {
            "name": "revdsd-d4",
            "xc_scf": "PBE",
            "c_pt2": 0.5,
            "dispersion": {"method": "d4", "xc": "revdsd-pbep86"},
        }
    )
    with pytest.raises(NotImplementedError, match="dftd4"):
        correction(_water(), f, None)


def test_d3bj_without_params_or_xc_is_refused():
    f = parse_dh_xc(
        {
            "name": "underdetermined",
            "xc_scf": "PBE",
            "c_pt2": 0.5,
            "dispersion": {"method": "d3bj"},
        }
    )
    with pytest.raises(ValueError, match="s6=1.0"):
        correction(_water(), f, None)


def test_unknown_xc_name_is_refused():
    f = parse_dh_xc(
        {
            "name": "no-such-functional",
            "xc_scf": "PBE",
            "c_pt2": 0.5,
            "dispersion": {"method": "d3bj", "xc": "definitely-not-a-functional"},
        }
    )
    with pytest.raises(ValueError, match="no built-in D3"):
        correction(_water(), f, None)


def test_rdfdh_energy_dispersion_uses_builtin_adapter():
    mol = _ne_dimer()
    rdfdh = RDFDH(mol, xc="B2PLYPD3BJ")
    e_disp = rdfdh.energy_dispersion()

    assert e_disp == pytest.approx(correction(mol, B2PLYP_D3BJ, None))
    assert e_disp < 0.0


def test_rdfdh_injected_dispersion_wins_over_metadata():
    mol = _ne_dimer()
    rdfdh = RDFDH(
        mol, xc="B2PLYPD3BJ", dispersion_correction=lambda s, f, k: -0.5
    )
    assert rdfdh.energy_dispersion() == pytest.approx(-0.5)


def test_rdfdh_without_dispersion_metadata_is_zero():
    mol = _ne_dimer()
    rdfdh = RDFDH(mol, xc="B2PLYP")
    assert rdfdh.energy_dispersion() == 0.0


def test_d3zero_library_matches_literature_damping_params():
    mol = _water()
    f_library = parse_dh_xc(
        {
            "name": "B2PLYP-d3zero-library",
            "xc_scf": "0.53*HF + 0.47*B88, 0.73*LYP",
            "c_pt2": 0.27,
            "dispersion": {"method": "d3zero", "xc": "b2plyp"},
        }
    )
    f_literature = parse_dh_xc(
        {
            "name": "B2PLYP-d3zero-literature",
            "xc_scf": "0.53*HF + 0.47*B88, 0.73*LYP",
            "c_pt2": 0.27,
            "dispersion": {
                "method": "d3zero",
                "params": {"s6": 0.64, "rs6": 1.427, "s8": 1.022},
            },
        }
    )

    e_library = correction(mol, f_library, None)
    e_literature = correction(mol, f_literature, None)

    assert abs(e_library - e_literature) < 1e-12
    assert e_library < 0.0


def test_d3zero_differs_from_d3bj():
    mol = _water()
    f_zero = parse_dh_xc(
        {
            "name": "z",
            "xc_scf": "PBE",
            "c_pt2": 0.5,
            "dispersion": {"method": "d3zero", "xc": "b2plyp"},
        }
    )
    e_zero = correction(mol, f_zero, None)
    e_bj = correction(mol, B2PLYP_D3BJ, None)

    assert abs(e_zero - e_bj) > 1e-9
