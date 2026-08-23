import pytest

from kdh.pt2_assembly import assemble_pt2_energy
from kdh.xc import parse_dh_xc


def test_assembles_from_os_ss_components():
    f = parse_dh_xc("B2PLYP")
    e = assemble_pt2_energy(f, e_corr=-1.0, e_corr_os=-0.7, e_corr_ss=-0.3)
    assert e == pytest.approx(0.27 * (-0.7 - 0.3))


def test_spin_scaled_requires_components():
    f = parse_dh_xc("XYGJOS")
    with pytest.raises(RuntimeError, match="e_corr_os"):
        assemble_pt2_energy(f, e_corr=-1.0, e_corr_os=None, e_corr_ss=None)


def test_unscaled_falls_back_to_total_when_components_absent():
    f = parse_dh_xc("HFMP2")
    e = assemble_pt2_energy(f, e_corr=-1.0, e_corr_os=None, e_corr_ss=None)
    assert e == pytest.approx(1.0 * 1.0 * -1.0)
