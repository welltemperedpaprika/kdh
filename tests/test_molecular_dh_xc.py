import pytest

from kdh.xc import KNOWN_DH_FUNCTIONALS, DoubleHybridFunctional, parse_dh_xc


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


@pytest.mark.parametrize("alias", ["DSD-BLYP-D3(BJ)", "dsd_blyp_d3bj", "Dsd-Blyp-D3Bj"])
def test_name_canonicalization_resolves_to_one_entry(alias):
    assert parse_dh_xc(alias) is KNOWN_DH_FUNCTIONALS["DSDBLYPD3BJ"]


@pytest.mark.parametrize("key", sorted(KNOWN_DH_FUNCTIONALS))
def test_display_name_round_trips(key):
    spec = KNOWN_DH_FUNCTIONALS[key]
    assert parse_dh_xc(spec.name) is spec
