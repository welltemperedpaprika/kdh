"""Pin the functional registry to independently transcribed literature values."""

import pytest

from kdh.xc import KNOWN_DH_FUNCTIONALS, parse_dh_xc

# Each row: registry key -> (citation, xc_scf, xc_nscf, c_pt2, c_os, c_ss).
# Citations name the defining paper; xc strings are PySCF xc-string syntax
# (exchange before the comma, correlation after; "LDA" = Slater exchange;
# "VWN3" = VWN1-RPA a.k.a. libxc LDA_C_VWN_RPA, NOT VWN5).
LITERATURE_DEFINITIONS = {
    # Not a literature functional: HF reference + full MP2.
    "HFMP2": (
        "HF + MP2",
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
    # B2PLYP plus D3(BJ) (Grimme, Ehrlich, Goerigk, J. Comput. Chem. 32, 1456
    # (2011), doi:10.1002/jcc.21759); damping pinned in
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


# Dispersion metadata pinned per entry. db-lookup entries carry an "xc" method
# name resolved against the dftd3 database; explicit-params entries carry the
# full {s6, a1, s8, a2} set from the defining paper rather than a database name
# whose row is absent or corresponds to a different parameterization.
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


def _water():
    from pyscf import gto

    return gto.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="sto-3g",
        verbose=0,
    )


@pytest.mark.parametrize("key", sorted(KNOWN_DH_FUNCTIONALS))
def test_dispersion_metadata_matches_provenance(key):
    assert KNOWN_DH_FUNCTIONALS[key].dispersion == DISPERSION_METADATA.get(key, {})


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
    assert correction(mol, f_db) == correction(mol, f_lit)
