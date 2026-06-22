"""Double-hybrid functional registry and parser.

Functional provenance
---------------------

HF-MP2
~~~~~~
The ``HFMP2`` entry is a degenerate double hybrid: a pure HF SCF reference with
full (unscaled) MP2 correlation (c_pt2 = 1.0, c_os = c_ss = 1.0, no DFT
exchange or correlation).  It is included so that HF-based MP2/SCS-MP2/SOS-MP2
literature protocols can be replicated through the same scan/summary pipeline as
the real double hybrids without any special-casing.

XYGJ-OS
~~~~~~~
Zhang, Xu, Jung, Goddard, *PNAS* **108**, 19896 (2011),
doi:10.1073/pnas.1115123108.

Exchange: 0.7731 * HF + 0.2269 * Slater (LDA).
Correlation: 0.2309 * VWN(RPA) + 0.2754 * LYP.
Opposite-spin-only PT2: c_pt2 = 0.4364, c_os = 1.0, c_ss = 0.0, evaluated on
B3LYP (Gaussian variant, ``B3LYPg``) orbitals (xc_scf = ``"B3LYPg"``).

The paper's LDA correlation is the VWN1-RPA parameterization, whose PySCF token
is ``"VWN3"`` (alias ``"VWNRPA"``, libxc ``LDA_C_VWN_RPA``).  Plain ``"VWN"``
would select VWN5 and is wrong here.

Cross-checked against the ajz34/dh PySCF extension (``pyscf/dh/dhutil.py``) and
the Q-Chem manual (section "Exchange-Correlation Functionals": 77.31% HF +
22.69% Slater exchange, 23.09% VWN1RPA + 27.54% LYP + 43.64% OS-MP2
correlation).

Pre-2026-06-11 erratum: before 2026-06-11 this entry wrongly reused XYG3's
``xc_nscf`` with ``c_pt2 = 0.3211`` (a spin-truncated XYG3, not the published
XYGJ-OS); rows harvested under that definition are annotated in
``docs/periodic-dh-scs-story-and-xdh-track.md``.
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
