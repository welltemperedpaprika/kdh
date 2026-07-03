"""Double-hybrid functional registry and parser.

XYGJ-OS uses VWN3 (VWN1-RPA) for its LDA correlation, not VWN5; see
Zhang et al., PNAS 108, 19896 (2011).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


def _empty_mapping() -> Mapping[str, Any]:
    return MappingProxyType({})


@dataclass(frozen=True)
class DoubleHybridFunctional:
    name: str
    xc_scf: str
    xc_nscf: str | None
    c_pt2: float
    c_os: float = 1.0
    c_ss: float = 1.0
    dispersion: Mapping[str, Any] = field(default_factory=_empty_mapping)
    requires_lr_pt2: bool = False

    @property
    def eval_pt2(self) -> bool:
        return abs(self.c_pt2) > 1e-14 and (
            abs(self.c_os) > 1e-14 or abs(self.c_ss) > 1e-14
        )


KNOWN_DH_FUNCTIONALS: dict[str, DoubleHybridFunctional] = {
    "HFMP2": DoubleHybridFunctional(
        name="HF-MP2",
        xc_scf="HF",
        xc_nscf=None,
        c_pt2=1.0,
    ),
    "B2PLYP": DoubleHybridFunctional(
        name="B2PLYP",
        xc_scf="0.53*HF + 0.47*B88, 0.73*LYP",
        xc_nscf=None,
        c_pt2=0.27,
    ),
    # Grimme, J. Chem. Phys. 124, 034108 (2006); D3(BJ) damping from the dftd3
    # database via the "b2plyp" method name (Grimme, Ehrlich, Goerigk,
    # J. Comput. Chem. 32, 1456 (2011)).
    "B2PLYPD3BJ": DoubleHybridFunctional(
        name="B2PLYP-D3(BJ)",
        xc_scf="0.53*HF + 0.47*B88, 0.73*LYP",
        xc_nscf=None,
        c_pt2=0.27,
        dispersion=MappingProxyType({"method": "d3bj", "xc": "b2plyp"}),
    ),
    # Karton et al., J. Phys. Chem. A 112, 12868 (2008); D3(BJ) from the dftd3
    # "b2gpplyp" database row.
    "B2GPPLYPD3BJ": DoubleHybridFunctional(
        name="B2GP-PLYP-D3(BJ)",
        xc_scf="0.65*HF + 0.35*B88, 0.64*LYP",
        xc_nscf=None,
        c_pt2=0.36,
        dispersion=MappingProxyType({"method": "d3bj", "xc": "b2gpplyp"}),
    ),
    # Schwabe, Grimme, Phys. Chem. Chem. Phys. 8, 4398 (2006); MPW91 is libxc
    # GGA_X_MPW91. D3(BJ) from the dftd3 "mpw2plyp" database row.
    "MPW2PLYPD3BJ": DoubleHybridFunctional(
        name="mPW2PLYP-D3(BJ)",
        xc_scf="0.55*HF + 0.45*MPW91, 0.75*LYP",
        xc_nscf=None,
        c_pt2=0.25,
        dispersion=MappingProxyType({"method": "d3bj", "xc": "mpw2plyp"}),
    ),
    # Kozuch, Martin, J. Comput. Chem. 34, 2327 (2013); SCS PT2 in c_os/c_ss.
    # D3(BJ) damping is EXPLICIT (a2=5.4): the dftd3 "dsdblyp" row is the 2010
    # vintage (wrong pairing) and would mis-damp this 2013 functional. Explicit
    # params default s9=0.0 (two-body), matching the database convention.
    "DSDBLYPD3BJ": DoubleHybridFunctional(
        name="DSD-BLYP-D3(BJ)",
        xc_scf="0.71*HF + 0.29*B88, 0.54*LYP",
        xc_nscf=None,
        c_pt2=1.0,
        c_os=0.47,
        c_ss=0.40,
        dispersion=MappingProxyType(
            {
                "method": "d3bj",
                "params": {"s6": 0.57, "a1": 0.0, "s8": 0.0, "a2": 5.4},
            }
        ),
    ),
    # Kozuch, Martin, J. Comput. Chem. 34, 2327 (2013); the dftd3 "dsdpbep86"
    # database row matches the 2013 paper.
    "DSDPBEP86D3BJ": DoubleHybridFunctional(
        name="DSD-PBEP86-D3(BJ)",
        xc_scf="0.69*HF + 0.31*PBE, 0.44*P86",
        xc_nscf=None,
        c_pt2=1.0,
        c_os=0.52,
        c_ss=0.22,
        dispersion=MappingProxyType({"method": "d3bj", "xc": "dsdpbep86"}),
    ),
    # Santra, Sylvetsky, Martin, J. Phys. Chem. A 123, 5129 (2019), Table 3;
    # D3(BJ) from the dftd3 "revdsdpbep86" database row.
    "REVDSDPBEP86D3BJ": DoubleHybridFunctional(
        name="revDSD-PBEP86-D3(BJ)",
        xc_scf="0.69*HF + 0.31*PBE, 0.4296*P86",
        xc_nscf=None,
        c_pt2=1.0,
        c_os=0.5785,
        c_ss=0.0799,
        dispersion=MappingProxyType({"method": "d3bj", "xc": "revdsdpbep86"}),
    ),
    # Santra, Sylvetsky, Martin, J. Phys. Chem. A 123, 5129 (2019), Table 3;
    # D3(BJ) damping is EXPLICIT (this functional is absent from the dftd3
    # database). Explicit params default s9=0.0 (two-body).
    "REVDSDBLYPD3BJ": DoubleHybridFunctional(
        name="revDSD-BLYP-D3(BJ)",
        xc_scf="0.71*HF + 0.29*B88, 0.5313*LYP",
        xc_nscf=None,
        c_pt2=1.0,
        c_os=0.5477,
        c_ss=0.1979,
        dispersion=MappingProxyType(
            {
                "method": "d3bj",
                "params": {"s6": 0.5451, "a1": 0.0, "s8": 0.0, "a2": 5.2},
            }
        ),
    ),
    # Grimme, J. Chem. Phys. 118, 9095 (2003): SCS-MP2 on a HF reference,
    # c_os = 6/5, c_ss = 1/3. No dispersion.
    "SCSMP2": DoubleHybridFunctional(
        name="SCS-MP2",
        xc_scf="HF",
        xc_nscf=None,
        c_pt2=1.0,
        c_os=1.2,
        c_ss=1.0 / 3.0,
    ),
    # Jung, Lochan, Dutoi, Head-Gordon, J. Chem. Phys. 121, 9793 (2004):
    # SOS-MP2 on a HF reference, c_os = 1.3, c_ss = 0.0. No dispersion.
    "SOSMP2": DoubleHybridFunctional(
        name="SOS-MP2",
        xc_scf="HF",
        xc_nscf=None,
        c_pt2=1.0,
        c_os=1.3,
        c_ss=0.0,
    ),
    "PBE0DH": DoubleHybridFunctional(
        name="PBE0DH",
        xc_scf="0.50*HF + 0.50*PBE, 0.875*PBE",
        xc_nscf=None,
        c_pt2=0.125,
    ),
    "PBE0QIDH": DoubleHybridFunctional(
        name="PBE0QIDH",
        xc_scf="0.693361*HF + 0.306639*PBE, 0.666667*PBE",
        xc_nscf=None,
        c_pt2=0.333333,
    ),
    "PBE02": DoubleHybridFunctional(
        name="PBE02",
        xc_scf="0.793701*HF + 0.206299*PBE, 0.50*PBE",
        xc_nscf=None,
        c_pt2=0.50,
    ),
    "XYG3": DoubleHybridFunctional(
        name="XYG3",
        xc_scf="B3LYPg",
        xc_nscf="-0.014*LDA + 0.8033*HF + 0.2107*B88, 0.6789*LYP",
        c_pt2=0.3211,
    ),
    "XYGJOS": DoubleHybridFunctional(
        name="XYGJOS",
        xc_scf="B3LYPg",
        xc_nscf="0.7731*HF + 0.2269*LDA, 0.2309*VWN3 + 0.2754*LYP",
        c_pt2=0.4364,
        c_os=1.0,
        c_ss=0.0,
    ),
}


def _canonical_name(name: str) -> str:
    return name.replace("-", "").replace("_", "").upper()


def parse_dh_xc(
    xc: str | Mapping[str, Any] | DoubleHybridFunctional,
) -> DoubleHybridFunctional:
    if isinstance(xc, DoubleHybridFunctional):
        return xc

    if isinstance(xc, str):
        key = _canonical_name(xc)
        try:
            return KNOWN_DH_FUNCTIONALS[key]
        except KeyError as err:
            known = ", ".join(sorted(KNOWN_DH_FUNCTIONALS))
            raise ValueError(
                f"Unknown double-hybrid functional {xc!r}. Known functionals: {known}"
            ) from err

    if not isinstance(xc, Mapping):
        raise TypeError(
            "Double-hybrid functional must be a name, mapping, or "
            "DoubleHybridFunctional"
        )

    required = {"xc_scf", "c_pt2"}
    missing = sorted(required - set(xc))
    if missing:
        raise ValueError(f"Custom double-hybrid functional missing keys: {missing}")

    return DoubleHybridFunctional(
        name=str(xc.get("name", "CUSTOM-DH")),
        xc_scf=str(xc["xc_scf"]),
        xc_nscf=xc.get("xc_nscf"),
        c_pt2=float(xc["c_pt2"]),
        c_os=float(xc.get("c_os", 1.0)),
        c_ss=float(xc.get("c_ss", 1.0)),
        dispersion=xc.get("dispersion", _empty_mapping()),
        requires_lr_pt2=bool(xc.get("requires_lr_pt2", False)),
    )
