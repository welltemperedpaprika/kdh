import pytest

from kdh.xc import DoubleHybridFunctional, parse_dh_xc


def test_parse_named_b2plyp():
    spec = parse_dh_xc("B2PLYP")

    assert spec.name == "B2PLYP"
    assert spec.xc_scf == "0.53*HF + 0.47*B88, 0.73*LYP"
    assert spec.xc_nscf is None
    assert spec.c_pt2 == pytest.approx(0.27)
    assert spec.c_os == pytest.approx(1.0)
    assert spec.c_ss == pytest.approx(1.0)
    assert spec.requires_lr_pt2 is False


def test_parse_named_hf_mp2_is_pure_hf_reference_with_full_pt2():
    spec = parse_dh_xc("HF-MP2")

    assert spec.name == "HF-MP2"
    assert spec.xc_scf == "HF"
    assert spec.xc_nscf is None
    assert spec.c_pt2 == pytest.approx(1.0)
    assert spec.c_os == pytest.approx(1.0)
    assert spec.c_ss == pytest.approx(1.0)
    assert spec.eval_pt2


def test_parse_named_xyg3_has_nscf_functional():
    spec = parse_dh_xc("XYG3")

    assert spec.name == "XYG3"
    assert spec.xc_scf == "B3LYPg"
    assert spec.xc_nscf == "-0.014*LDA + 0.8033*HF + 0.2107*B88, 0.6789*LYP"
    assert spec.c_pt2 == pytest.approx(0.3211)


def test_parse_named_xygjos_is_published_parameterization():
    # Zhang, Xu, Jung, Goddard, PNAS 108, 19896 (2011). The detailed
    # literature pinning (with the VWN1RPA/VWN3 evidence) lives in
    # tests/test_dh_functional_literature_coefficients.py.
    spec = parse_dh_xc("XYGJ-OS")

    assert spec.name == "XYGJOS"
    assert spec.xc_scf == "B3LYPg"
    assert spec.xc_nscf == "0.7731*HF + 0.2269*LDA, 0.2309*VWN3 + 0.2754*LYP"
    assert spec.c_pt2 == pytest.approx(0.4364)
    assert spec.c_os == pytest.approx(1.0)
    assert spec.c_ss == pytest.approx(0.0)


def test_parse_custom_dict():
    spec = parse_dh_xc(
        {
            "name": "MY-DH",
            "xc_scf": "0.5*HF + 0.5*PBE, 0.8*PBE",
            "xc_nscf": None,
            "c_pt2": 0.2,
            "c_os": 1.1,
            "c_ss": 0.8,
        }
    )

    assert spec == DoubleHybridFunctional(
        name="MY-DH",
        xc_scf="0.5*HF + 0.5*PBE, 0.8*PBE",
        xc_nscf=None,
        c_pt2=0.2,
        c_os=1.1,
        c_ss=0.8,
    )


def test_unknown_named_functional_fails_clearly():
    with pytest.raises(ValueError, match="Unknown double-hybrid functional"):
        parse_dh_xc("NOT-A-DH")
