"""Double-hybrid functional registry and parser."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DoubleHybridFunctional:
    """Double-hybrid functional specification.

    E = E_xc_nscf[D(xc_scf)] + c_pt2 * (c_os * E_os + c_ss * E_ss), where the
    orbitals come from an SCF with ``xc_scf``. ``xc_nscf=None`` means the
    conventional (self-consistent) form, xc_nscf == xc_scf.
    """
    name: str
    xc_scf: str
    xc_nscf: str | None = None
    c_pt2: float = 0.0
    c_os: float = 1.0
    c_ss: float = 1.0
    dispersion: dict[str, Any] = field(default_factory=dict)

    @property
    def eval_pt2(self) -> bool:
        return self.c_pt2 != 0 and (self.c_os != 0 or self.c_ss != 0)


def is_pure_hf(xc: str) -> bool:
    return xc.strip().upper() == "HF"


KNOWN_DH_FUNCTIONALS: dict[str, DoubleHybridFunctional] = {
    "HFMP2": DoubleHybridFunctional(
        name="HF-MP2",
        xc_scf="HF",
        c_pt2=1.0,
    ),
    "B2PLYP": DoubleHybridFunctional(
        name="B2PLYP",
        xc_scf="0.53*HF + 0.47*B88, 0.73*LYP",
        c_pt2=0.27,
    ),
    "B2PLYPD3BJ": DoubleHybridFunctional(
        name="B2PLYP-D3(BJ)",
        xc_scf="0.53*HF + 0.47*B88, 0.73*LYP",
        c_pt2=0.27,
        dispersion={"method": "d3bj", "xc": "b2plyp"},
    ),
    "B2GPPLYPD3BJ": DoubleHybridFunctional(
        name="B2GP-PLYP-D3(BJ)",
        xc_scf="0.65*HF + 0.35*B88, 0.64*LYP",
        c_pt2=0.36,
        dispersion={"method": "d3bj", "xc": "b2gpplyp"},
    ),
    "MPW2PLYPD3BJ": DoubleHybridFunctional(
        name="mPW2PLYP-D3(BJ)",
        xc_scf="0.55*HF + 0.45*MPW91, 0.75*LYP",
        c_pt2=0.25,
        dispersion={"method": "d3bj", "xc": "mpw2plyp"},
    ),
    "DSDBLYPD3BJ": DoubleHybridFunctional(
        name="DSD-BLYP-D3(BJ)",
        xc_scf="0.71*HF + 0.29*B88, 0.54*LYP",
        c_pt2=1.0,
        c_os=0.47,
        c_ss=0.40,
        dispersion={
            "method": "d3bj",
            "params": {"s6": 0.57, "a1": 0.0, "s8": 0.0, "a2": 5.4},
        },
    ),
    "DSDPBEP86D3BJ": DoubleHybridFunctional(
        name="DSD-PBEP86-D3(BJ)",
        xc_scf="0.69*HF + 0.31*PBE, 0.44*P86",
        c_pt2=1.0,
        c_os=0.52,
        c_ss=0.22,
        dispersion={"method": "d3bj", "xc": "dsdpbep86"},
    ),
    "REVDSDPBEP86D3BJ": DoubleHybridFunctional(
        name="revDSD-PBEP86-D3(BJ)",
        xc_scf="0.69*HF + 0.31*PBE, 0.4296*P86",
        c_pt2=1.0,
        c_os=0.5785,
        c_ss=0.0799,
        dispersion={"method": "d3bj", "xc": "revdsdpbep86"},
    ),
    "REVDSDBLYPD3BJ": DoubleHybridFunctional(
        name="revDSD-BLYP-D3(BJ)",
        xc_scf="0.71*HF + 0.29*B88, 0.5313*LYP",
        c_pt2=1.0,
        c_os=0.5477,
        c_ss=0.1979,
        dispersion={
            "method": "d3bj",
            "params": {"s6": 0.5451, "a1": 0.0, "s8": 0.0, "a2": 5.2},
        },
    ),
    "SCSMP2": DoubleHybridFunctional(
        name="SCS-MP2",
        xc_scf="HF",
        c_pt2=1.0,
        c_os=1.2,
        c_ss=1.0 / 3.0,
    ),
    "SOSMP2": DoubleHybridFunctional(
        name="SOS-MP2",
        xc_scf="HF",
        c_pt2=1.0,
        c_os=1.3,
        c_ss=0.0,
    ),
    "PBE0DH": DoubleHybridFunctional(
        name="PBE0DH",
        xc_scf="0.50*HF + 0.50*PBE, 0.875*PBE",
        c_pt2=0.125,
    ),
    "PBE0QIDH": DoubleHybridFunctional(
        name="PBE0QIDH",
        xc_scf="0.693361*HF + 0.306639*PBE, 0.666667*PBE",
        c_pt2=0.333333,
    ),
    "PBE02": DoubleHybridFunctional(
        name="PBE02",
        xc_scf="0.793701*HF + 0.206299*PBE, 0.50*PBE",
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


def parse_dh_xc(xc: str | dict[str, Any] | DoubleHybridFunctional) -> DoubleHybridFunctional:
    """Resolve a registry name, a dict of DoubleHybridFunctional fields, or a
    DoubleHybridFunctional. Names ignore case, '-', '_' and parentheses."""
    if isinstance(xc, DoubleHybridFunctional):
        return xc
    if isinstance(xc, dict):
        return DoubleHybridFunctional(**{"name": "CUSTOM-DH", **xc})
    if isinstance(xc, str):
        key = "".join(c for c in xc.upper() if c not in "-_()")
        if key not in KNOWN_DH_FUNCTIONALS:
            raise ValueError(
                f"Unknown double-hybrid functional {xc!r}. "
                f"Known functionals: {', '.join(sorted(KNOWN_DH_FUNCTIONALS))}"
            )
        return KNOWN_DH_FUNCTIONALS[key]
    raise TypeError("xc must be a functional name, a dict, or a DoubleHybridFunctional")
