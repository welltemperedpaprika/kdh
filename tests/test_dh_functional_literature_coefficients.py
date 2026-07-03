"""Pin every KNOWN_DH_FUNCTIONALS entry to its literature definition.

Each entry is pinned, with its citation, to the values verified against the
primary literature and at least one independent implementation. A wrong
coefficient in kdh/xc.py would pass a parity check against a manual KRKS+KMP2
path silently (both sides read the same registry), so this table-driven test is
the independent side. Any future registry edit must consciously update it.
"""

import pytest

from kdh.pt2_assembly import assemble_pt2_energy
from kdh.xc import KNOWN_DH_FUNCTIONALS, parse_dh_xc

# Each row: registry key -> (citation, xc_scf, xc_nscf, c_pt2, c_os, c_ss).
# Citations name the defining paper; xc strings are PySCF xc-string syntax
# (exchange before the comma, correlation after; "LDA" = Slater exchange;
# "VWN3" = VWN1-RPA a.k.a. libxc LDA_C_VWN_RPA, NOT VWN5).
LITERATURE_DEFINITIONS = {
    # Not a literature functional: degenerate double hybrid (pure HF
    # reference, full MP2) so HF-based MP2/SCS-MP2/SOS-MP2 protocols run
    # through the same pipeline as the real double hybrids.
    "HFMP2": (
        "local pipeline convenience (HF reference + full MP2)",
        "HF",
        None,
        1.0,
        1.0,
        1.0,
    ),
    # Grimme, J. Chem. Phys. 124, 034108 (2006), doi:10.1063/1.2148954:
    # 53% HF + 47% B88 exchange, 73% LYP correlation, c_pt2 = 0.27.
    "B2PLYP": (
        "Grimme, J. Chem. Phys. 124, 034108 (2006)",
        "0.53*HF + 0.47*B88, 0.73*LYP",
        None,
        0.27,
        1.0,
        1.0,
    ),
    # Same functional coefficients as B2PLYP (Grimme, J. Chem. Phys. 124,
    # 034108 (2006)) plus mandatory D3(BJ) dispersion. Damping parameters
    # s6=0.64, a1=0.3065, s8=0.9147, a2=5.057 from Grimme, Ehrlich, Goerigk,
    # J. Comput. Chem. 32, 1456 (2011), doi:10.1002/jcc.21759, sourced at
    # runtime from the dftd3 library database via dispersion metadata
    # {"method": "d3bj", "xc": "b2plyp"} and pinned library-vs-literature in
    # tests/test_dispersion_d3_molecular.py.
    "B2PLYPD3BJ": (
        "Grimme, J. Chem. Phys. 124, 034108 (2006); "
        "Grimme, Ehrlich, Goerigk, J. Comput. Chem. 32, 1456 (2011)",
        "0.53*HF + 0.47*B88, 0.73*LYP",
        None,
        0.27,
        1.0,
        1.0,
    ),
    # Bremond, Adamo, J. Chem. Phys. 135, 024106 (2011),
    # doi:10.1063/1.3604569: PBE0-DH, 50% HF exchange, c_pt2 = 1/8,
    # correlation (1 - 1/8) * PBE.
    "PBE0DH": (
        "Bremond, Adamo, J. Chem. Phys. 135, 024106 (2011)",
        "0.50*HF + 0.50*PBE, 0.875*PBE",
        None,
        0.125,
        1.0,
        1.0,
    ),
    # Bremond, Sancho-Garcia, Perez-Jimenez, Adamo, J. Chem. Phys. 141,
    # 031101 (2014), doi:10.1063/1.4890314: PBE-QIDH, HF fraction
    # 3^(-1/3) = 0.693361, c_pt2 = 1/3, correlation 2/3 * PBE.
    "PBE0QIDH": (
        "Bremond, Sancho-Garcia, Perez-Jimenez, Adamo, "
        "J. Chem. Phys. 141, 031101 (2014)",
        "0.693361*HF + 0.306639*PBE, 0.666667*PBE",
        None,
        0.333333,
        1.0,
        1.0,
    ),
    # Chai, Mao, Chem. Phys. Lett. 538, 121 (2012),
    # doi:10.1016/j.cplett.2012.04.045: PBE0-2, HF fraction
    # 2^(-1/3) = 0.793701, c_pt2 = 1/2, correlation 1/2 * PBE.
    "PBE02": (
        "Chai, Mao, Chem. Phys. Lett. 538, 121 (2012)",
        "0.793701*HF + 0.206299*PBE, 0.50*PBE",
        None,
        0.50,
        1.0,
        1.0,
    ),
    # Zhang, Xu, Goddard, PNAS 106, 4963 (2009),
    # doi:10.1073/pnas.0901093106: XYG3 on B3LYP(Gaussian/VWN-RPA) orbitals,
    # exchange 0.8033*HF - 0.0140*Slater + 0.2107*B88, correlation
    # 0.6789*LYP, full-spin c_pt2 = 0.3211. Matches the ajz34/dh extension.
    "XYG3": (
        "Zhang, Xu, Goddard, PNAS 106, 4963 (2009)",
        "B3LYPg",
        "-0.014*LDA + 0.8033*HF + 0.2107*B88, 0.6789*LYP",
        0.3211,
        1.0,
        1.0,
    ),
    # Zhang, Xu, Jung, Goddard, PNAS 108, 19896 (2011),
    # doi:10.1073/pnas.1115123108: XYGJ-OS on B3LYP(Gaussian) orbitals,
    # exchange 0.7731*HF + 0.2269*Slater, correlation 0.2309*VWN1RPA
    # (PySCF token VWN3) + 0.2754*LYP, opposite-spin-only PT2 with
    # c_pt2 = 0.4364. Verified against the Q-Chem manual and the ajz34/dh
    # extension ("0.7731*HF + 0.2269*LDA, 0.2309*VWN3 + 0.2754*LYP",
    # c_pt2 = 0.4364, c_os = 1, c_ss = 0).
    "XYGJOS": (
        "Zhang, Xu, Jung, Goddard, PNAS 108, 19896 (2011)",
        "B3LYPg",
        "0.7731*HF + 0.2269*LDA, 0.2309*VWN3 + 0.2754*LYP",
        0.4364,
        1.0,
        0.0,
    ),
    # Karton, Tarnopolsky, Lamere, Schatz, Martin, J. Phys. Chem. A 112,
    # 12868 (2008), doi:10.1021/jp801805p: B2GP-PLYP, 65% HF + 35% B88
    # exchange, 64% LYP correlation, c_pt2 = 0.36. Dispersion is the D3(BJ)
    # form (database method "b2gpplyp"); coefficients confirmed by Psi4.
    "B2GPPLYPD3BJ": (
        "Karton, Tarnopolsky, Lamere, Schatz, Martin, "
        "J. Phys. Chem. A 112, 12868 (2008)",
        "0.65*HF + 0.35*B88, 0.64*LYP",
        None,
        0.36,
        1.0,
        1.0,
    ),
    # Schwabe, Grimme, Phys. Chem. Chem. Phys. 8, 4398 (2006),
    # doi:10.1039/b608478h: mPW2PLYP, 55% HF + 45% mPW91 exchange, 75% LYP
    # correlation, c_pt2 = 0.25. MPW91 = libxc GGA_X_MPW91 (Adamo-Barone).
    "MPW2PLYPD3BJ": (
        "Schwabe, Grimme, Phys. Chem. Chem. Phys. 8, 4398 (2006)",
        "0.55*HF + 0.45*MPW91, 0.75*LYP",
        None,
        0.25,
        1.0,
        1.0,
    ),
    # Kozuch, Martin, J. Comput. Chem. 34, 2327 (2013),
    # doi:10.1002/jcc.23391: DSD-BLYP (2013 revision), 71% HF + 29% B88
    # exchange, 54% LYP correlation, SCS PT2 with c_os = 0.47, c_ss = 0.40
    # (c_pt2 = 1.0). D3(BJ) damping is explicit (a2 = 5.4), NOT the db
    # dsdblyp 2010-vintage row. Confirmed by Psi4 and revDSD Table 3.
    "DSDBLYPD3BJ": (
        "Kozuch, Martin, J. Comput. Chem. 34, 2327 (2013)",
        "0.71*HF + 0.29*B88, 0.54*LYP",
        None,
        1.0,
        0.47,
        0.40,
    ),
    # Kozuch, Martin, J. Comput. Chem. 34, 2327 (2013),
    # doi:10.1002/jcc.23391: DSD-PBEP86 (2013), 69% HF + 31% PBE exchange,
    # 44% P86 correlation, SCS PT2 with c_os = 0.52, c_ss = 0.22
    # (c_pt2 = 1.0). D3(BJ) database row dsdpbep86 matches the paper.
    "DSDPBEP86D3BJ": (
        "Kozuch, Martin, J. Comput. Chem. 34, 2327 (2013)",
        "0.69*HF + 0.31*PBE, 0.44*P86",
        None,
        1.0,
        0.52,
        0.22,
    ),
    # Santra, Sylvetsky, Martin, J. Phys. Chem. A 123, 5129 (2019),
    # doi:10.1021/acs.jpca.9b03157, Table 3: revDSD-PBEP86-D3(BJ), 69% HF +
    # 31% PBE exchange, 42.96% P86 correlation, SCS PT2 with c_os = 0.5785,
    # c_ss = 0.0799 (c_pt2 = 1.0). Database row revdsdpbep86 matches Table 3.
    "REVDSDPBEP86D3BJ": (
        "Santra, Sylvetsky, Martin, J. Phys. Chem. A 123, 5129 (2019), Table 3",
        "0.69*HF + 0.31*PBE, 0.4296*P86",
        None,
        1.0,
        0.5785,
        0.0799,
    ),
    # Santra, Sylvetsky, Martin, J. Phys. Chem. A 123, 5129 (2019),
    # doi:10.1021/acs.jpca.9b03157, Table 3: revDSD-BLYP-D3(BJ), 71% HF +
    # 29% B88 exchange, 53.13% LYP correlation, SCS PT2 with c_os = 0.5477,
    # c_ss = 0.1979 (c_pt2 = 1.0). D3(BJ) damping is explicit (not in db).
    "REVDSDBLYPD3BJ": (
        "Santra, Sylvetsky, Martin, J. Phys. Chem. A 123, 5129 (2019), Table 3",
        "0.71*HF + 0.29*B88, 0.5313*LYP",
        None,
        1.0,
        0.5477,
        0.1979,
    ),
    # Grimme, J. Chem. Phys. 118, 9095 (2003), doi:10.1063/1.1569242:
    # SCS-MP2 on a pure HF reference, c_os = 6/5 = 1.2, c_ss = 1/3
    # (c_pt2 = 1.0). No dispersion.
    "SCSMP2": (
        "Grimme, J. Chem. Phys. 118, 9095 (2003)",
        "HF",
        None,
        1.0,
        1.2,
        1.0 / 3.0,
    ),
    # Jung, Lochan, Dutoi, Head-Gordon, J. Chem. Phys. 121, 9793 (2004),
    # doi:10.1063/1.1809602: SOS-MP2 on a pure HF reference, c_os = 1.3,
    # c_ss = 0.0 (c_pt2 = 1.0). No dispersion.
    "SOSMP2": (
        "Jung, Lochan, Dutoi, Head-Gordon, J. Chem. Phys. 121, 9793 (2004)",
        "HF",
        None,
        1.0,
        1.3,
        0.0,
    ),
}


def test_every_registry_entry_has_a_literature_row_and_vice_versa():
    assert set(KNOWN_DH_FUNCTIONALS) == set(LITERATURE_DEFINITIONS), (
        "Registry and literature-pinning table diverged. Any new registry "
        "entry needs a verified literature row here (with citation), and "
        "any removal must drop its row."
    )


@pytest.mark.parametrize("key", sorted(LITERATURE_DEFINITIONS))
def test_registry_entry_matches_literature(key):
    citation, xc_scf, xc_nscf, c_pt2, c_os, c_ss = LITERATURE_DEFINITIONS[key]
    spec = KNOWN_DH_FUNCTIONALS[key]

    context = f"{key} ({citation})"
    assert spec.xc_scf == xc_scf, context
    assert spec.xc_nscf == xc_nscf, context
    assert spec.c_pt2 == pytest.approx(c_pt2, abs=1e-12), context
    assert spec.c_os == pytest.approx(c_os, abs=1e-12), context
    assert spec.c_ss == pytest.approx(c_ss, abs=1e-12), context


def test_b2plypd3bj_carries_the_d3bj_dispersion_metadata():
    """B2PLYP-D3(BJ) must name the library damping-database method.

    The dispersion metadata {"method": "d3bj", "xc": "b2plyp"} routes
    kdh.dispersion_d3 to the dftd3 library's built-in damping parameters for
    B2PLYP (Grimme, Ehrlich, Goerigk, J. Comput. Chem. 32, 1456 (2011):
    s6=0.64, a1=0.3065, s8=0.9147, a2=5.057); the numeric library-vs-literature
    pin lives in tests/test_dispersion_d3_molecular.py.
    """
    spec = KNOWN_DH_FUNCTIONALS["B2PLYPD3BJ"]
    assert dict(spec.dispersion) == {"method": "d3bj", "xc": "b2plyp"}
    bare = KNOWN_DH_FUNCTIONALS["B2PLYP"]
    assert not bare.dispersion
    assert spec.xc_scf == bare.xc_scf
    assert spec.c_pt2 == bare.c_pt2


def test_xygjos_is_not_the_old_spin_truncated_xyg3():
    """Regression guard: XYGJ-OS must not be a spin-truncated XYG3.

    A prior bug reused XYG3's xc_nscf with c_pt2 = 0.3211 and c_ss = 0, i.e.
    "XYG3 with same-spin PT2 dropped" -- a different functional from the
    published XYGJ-OS parameterization.
    """
    xyg3 = KNOWN_DH_FUNCTIONALS["XYG3"]
    xygjos = KNOWN_DH_FUNCTIONALS["XYGJOS"]

    assert xygjos.xc_nscf != xyg3.xc_nscf
    assert xygjos.c_pt2 != pytest.approx(xyg3.c_pt2)
    assert "VWN3" in xygjos.xc_nscf


# Dispersion metadata pinned per entry. db-lookup entries carry an "xc" method
# name resolved against the dftd3 database; explicit-params entries carry the
# full {s6, a1, s8, a2} set from the defining paper and MUST NOT use a name
# lookup (their db rows are absent or a wrong vintage).
DISPERSION_METADATA = {
    "B2PLYPD3BJ": {"method": "d3bj", "xc": "b2plyp"},
    "B2GPPLYPD3BJ": {"method": "d3bj", "xc": "b2gpplyp"},
    "MPW2PLYPD3BJ": {"method": "d3bj", "xc": "mpw2plyp"},
    "DSDPBEP86D3BJ": {"method": "d3bj", "xc": "dsdpbep86"},
    "REVDSDPBEP86D3BJ": {"method": "d3bj", "xc": "revdsdpbep86"},
    "DSDBLYPD3BJ": {
        "method": "d3bj",
        "params": {"s6": 0.57, "a1": 0.0, "s8": 0.0, "a2": 5.4},
    },
    "REVDSDBLYPD3BJ": {
        "method": "d3bj",
        "params": {"s6": 0.5451, "a1": 0.0, "s8": 0.0, "a2": 5.2},
    },
}

# db-lookup entries: expected D3(BJ) damping set from the provenance record,
# used as a drift guard against the installed dftd3 method-name database.
DB_DAMPING = {
    "b2gpplyp": {"s6": 0.56, "a1": 0.0, "s8": 0.2597, "a2": 6.3332},
    "mpw2plyp": {"s6": 0.66, "a1": 0.4105, "s8": 0.6223, "a2": 5.0136},
    "dsdpbep86": {"s6": 0.48, "a1": 0.0, "s8": 0.0, "a2": 5.6},
    "revdsdpbep86": {"s6": 0.4377, "a1": 0.0, "s8": 0.0, "a2": 5.5},
}

# Entries that must carry no dispersion metadata at all.
DISPERSIONLESS_ENTRIES = ("HFMP2", "B2PLYP", "SCSMP2", "SOSMP2")


def _water():
    from pyscf import gto

    return gto.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="sto-3g",
        verbose=0,
    )


@pytest.mark.parametrize("key", sorted(DISPERSION_METADATA))
def test_dispersion_metadata_matches_provenance(key):
    spec = KNOWN_DH_FUNCTIONALS[key]
    assert dict(spec.dispersion) == DISPERSION_METADATA[key]


@pytest.mark.parametrize("key", DISPERSIONLESS_ENTRIES)
def test_scs_and_hf_entries_carry_no_dispersion(key):
    assert not KNOWN_DH_FUNCTIONALS[key].dispersion


@pytest.mark.parametrize("dbname", sorted(DB_DAMPING))
def test_library_db_lookup_matches_provenance_damping(dbname):
    """Drift guard: dftd3 method-name lookup must equal the provenance damping.

    RationalDampingParam does not expose its parameters, so the check is by
    dispersion energy on a fixed molecule: the db-lookup damping and an
    explicit-params set built from the provenance record must agree bit-for-bit.
    """
    pytest.importorskip("dftd3")
    from kdh.dispersion_d3 import correction

    mol = _water()
    f_db = parse_dh_xc(
        {
            "name": f"{dbname}-db",
            "xc_scf": "PBE",
            "c_pt2": 1.0,
            "dispersion": {"method": "d3bj", "xc": dbname},
        }
    )
    f_lit = parse_dh_xc(
        {
            "name": f"{dbname}-lit",
            "xc_scf": "PBE",
            "c_pt2": 1.0,
            "dispersion": {"method": "d3bj", "params": DB_DAMPING[dbname]},
        }
    )
    assert correction(mol, f_db, None) == correction(mol, f_lit, None)


@pytest.mark.parametrize("key", ["DSDBLYPD3BJ", "REVDSDBLYPD3BJ"])
def test_explicit_params_entries_do_not_use_a_name_lookup(key):
    """DSD-BLYP / revDSD-BLYP carry the full explicit damping set, no db name.

    Their dftd3 db rows are either a wrong vintage (dsdblyp) or absent
    (revdsdblyp), so name-lookup must not be the code path.
    """
    spec = KNOWN_DH_FUNCTIONALS[key]
    meta = dict(spec.dispersion)
    assert meta["method"] == "d3bj"
    assert "xc" not in meta
    assert set(meta["params"]) == {"s6", "a1", "s8", "a2"}
    assert dict(meta["params"]) == DISPERSION_METADATA[key]["params"]


def test_dsd_blyp_vintage_differs_from_the_2010_db_row():
    """Guard the DSD-BLYP vintage trap.

    The entry pins the 2013 revision a2 = 5.4; the dftd3 db `dsdblyp` row is
    the 2010-vintage pairing. They must give different dispersion energies, so
    a stray switch to the db name-lookup would be caught.
    """
    spec = KNOWN_DH_FUNCTIONALS["DSDBLYPD3BJ"]
    assert dict(spec.dispersion)["params"]["a2"] == 5.4

    pytest.importorskip("dftd3")
    from kdh.dispersion_d3 import correction

    mol = _water()
    f_db_2010 = parse_dh_xc(
        {
            "name": "dsdblyp-db-2010",
            "xc_scf": "0.71*HF + 0.29*B88, 0.54*LYP",
            "c_pt2": 1.0,
            "dispersion": {"method": "d3bj", "xc": "dsdblyp"},
        }
    )
    e_entry = correction(mol, spec, None)
    e_db_2010 = correction(mol, f_db_2010, None)
    assert abs(e_entry - e_db_2010) > 1e-6


@pytest.mark.parametrize(
    "name,c_os,c_ss",
    [("SCS-MP2", 1.2, 1.0 / 3.0), ("SOS-MP2", 1.3, 0.0)],
)
def test_scs_sos_mp2_round_trip(name, c_os, c_ss):
    spec = parse_dh_xc(name)
    assert spec.name == name
    assert spec.xc_scf == "HF"
    assert spec.xc_nscf is None
    assert spec.c_pt2 == pytest.approx(1.0, abs=1e-12)
    assert spec.c_os == pytest.approx(c_os, abs=1e-12)
    assert spec.c_ss == pytest.approx(c_ss, abs=1e-12)
    assert not spec.dispersion


class _FakeMP2:
    """Minimal MP2 result carrying the OS/SS correlation components."""

    def __init__(self, e_corr_os, e_corr_ss):
        self.e_corr_os = e_corr_os
        self.e_corr_ss = e_corr_ss
        self.e_corr = e_corr_os + e_corr_ss


def test_scs_mp2_assembles_from_os_ss_components_end_to_end():
    spec = parse_dh_xc("SCS-MP2")
    mp2 = _FakeMP2(e_corr_os=-0.30, e_corr_ss=-0.10)

    e_pt2 = assemble_pt2_energy(
        spec, mp2.e_corr, mp2.e_corr_os, mp2.e_corr_ss
    )
    expected = spec.c_pt2 * (
        spec.c_os * mp2.e_corr_os + spec.c_ss * mp2.e_corr_ss
    )
    assert e_pt2 == pytest.approx(expected, abs=1e-14)
    assert e_pt2 == pytest.approx(1.2 * -0.30 + (1.0 / 3.0) * -0.10, abs=1e-14)


@pytest.mark.parametrize(
    "alias",
    ["DSD-BLYP-D3BJ", "dsd_blyp_d3bj", "DSDBLYPD3BJ", "Dsd-Blyp-D3Bj"],
)
def test_name_canonicalization_resolves_to_one_entry(alias):
    assert parse_dh_xc(alias) is KNOWN_DH_FUNCTIONALS["DSDBLYPD3BJ"]
