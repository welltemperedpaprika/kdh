# kdh

[![Tests](https://github.com/welltemperedpaprika/kdh/actions/workflows/tests.yml/badge.svg)](https://github.com/welltemperedpaprika/kdh/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Molecular and k-point periodic double-hybrid DFT for [PySCF](https://github.com/pyscf/pyscf).

- **`RDFDH`**: closed-shell molecular double hybrids, conventional and xDH (e.g. XYG3), optional density fitting (`df=True`), and analytic nuclear gradients for conventional unscaled functionals (e.g. B2PLYP).
- **`UDFDH`**: open-shell molecular double hybrids.
- **`KRDH`**: closed-shell periodic double hybrids with k-point sampling. KMP2 is refused on small-gap or fractionally occupied references unless explicitly allowed.
- **`numerical_nuc_grad`, `optimize`**: finite-difference gradients and geometry optimization for any functional (requires geomeTRIC or pyberny).

Functionals with a `-D3(BJ)` suffix add the D3 dispersion correction automatically (requires `dftd3`).

## Supported Functionals

- Conventional: `B2PLYP`, `PBE0-DH`, `PBE0-QIDH`, `PBE0-2`, `HF-MP2`, `SCS-MP2`, `SOS-MP2`
- xDH (non-self-consistent): `XYG3`, `XYGJ-OS`
- With D3(BJ): `B2PLYP-D3(BJ)`, `B2GP-PLYP-D3(BJ)`, `mPW2PLYP-D3(BJ)`, `DSD-BLYP-D3(BJ)`, `DSD-PBEP86-D3(BJ)`, `revDSD-PBEP86-D3(BJ)`, `revDSD-BLYP-D3(BJ)`
- Custom: pass a dict of `DoubleHybridFunctional` fields, e.g. `{"xc_scf": "0.5*HF + 0.5*PBE, 0.8*PBE", "c_pt2": 0.2}`

Names ignore case, `-`, `_` and parentheses.

## Installation

```bash
pip install -e .                # Core
pip install -e ".[dispersion]"  # With D3 dispersion support
```

## Quick Start

```python
from pyscf import gto
from kdh import RDFDH

mol = gto.M(atom="O 0 0 0; H 0 0 0.96; H 0.93 0 -0.26", basis="def2-SVP")
dh = RDFDH(mol, xc="B2PLYP")
e_tot = dh.kernel()
g = dh.Gradients().kernel()
```

```python
from pyscf.pbc import gto
from kdh import KRDH

cell = gto.M(
    atom="Si 0 0 0; Si 1.3575 1.3575 1.3575",
    a=[[0, 2.715, 2.715], [2.715, 0, 2.715], [2.715, 2.715, 0]],
    basis="gth-dzvp",
    pseudo="gth-pade",
)
dh = KRDH(cell, xc="B2PLYP", kpts=cell.make_kpts([2, 2, 2]))
e_tot = dh.kernel()
```

More in [`examples/`](examples).

## Testing

```bash
pytest                  # all tests
pytest -m "not native"  # skip the end-to-end PySCF runs
```

## License

Apache-2.0.
