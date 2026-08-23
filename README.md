# kdh

[![Tests](https://github.com/welltemperedpaprika/kdh/actions/workflows/tests.yml/badge.svg)](https://github.com/welltemperedpaprika/kdh/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Molecular and k-point periodic double-hybrid DFT for [PySCF](https://github.com/pyscf/pyscf).

`kdh` provides double-hybrid DFT drivers for molecular and periodic calculations:

- **`RDFDH`** — Molecular closed-shell (restricted) double-hybrid driver with support for standard and non-self-consistent (xDH) functionals, optional density fitting (`df=True`), and analytic nuclear gradients for conventional double hybrids (e.g. B2PLYP).
- **`UDFDH`** — Molecular open-shell (unrestricted) double-hybrid driver.
- **`KRDH` / `KDH`** — Periodic restricted double-hybrid driver with k-point sampling, opposite-spin / same-spin (OS/SS) spin-scaling, and reference stability checks.

Dispersion corrections (D3(BJ) and D3(0)) are automatically applied when selecting dispersion-enabled functionals (requires `dftd3`).

## Supported Functionals

The functional registry includes:
- Conventional: `B2PLYP`, `B2GP-PLYP`, `mPW2PLYP`, `PBE0-DH`, `PBE0-QIDH`, `PBE0-2`, `HF-MP2`, `SCS-MP2`, `SOS-MP2`
- Non-self-consistent (xDH): `XYG3`, `XYGJ-OS`
- Dispersion-corrected: `B2PLYP-D3(BJ)`, `B2GP-PLYP-D3(BJ)`, `mPW2PLYP-D3(BJ)`, `DSD-BLYP-D3(BJ)`, `DSD-PBEP86-D3(BJ)`, `revDSD-PBEP86-D3(BJ)`, `revDSD-BLYP-D3(BJ)`
- Custom functionals via specification dictionaries.

## Installation

```bash
pip install -e .              # Core
pip install -e ".[dispersion]"  # With D3 dispersion support
```

## Quick Start

### Molecular Double-Hybrid

```python
from pyscf import gto
from kdh import RDFDH

mol = gto.M(atom="O 0 0 0; H 0 0 0.96; H 0.93 0 -0.26", basis="def2-SVP")
dh = RDFDH(mol, xc="B2PLYP")
e_tot = dh.kernel()
print(f"B2PLYP energy: {e_tot:.8f} Ha")

# Analytic gradient
g = dh.Gradients().kernel()
```

### Periodic Double-Hybrid (k-points)

```python
import numpy as np
from pyscf.pbc import gto
from kdh import KRDH

cell = gto.Cell()
cell.atom = "Si 0 0 0; Si 1.3575 1.3575 1.3575"
cell.a = np.eye(3) * 5.43
cell.basis = "gth-dzvp"
cell.pseudo = "gth-pade"
cell.build()

kpts = cell.make_kpts([2, 2, 2])
dh = KRDH(cell, xc="B2PLYP", kpts=kpts)
e_tot = dh.kernel()
print(f"Periodic B2PLYP energy: {e_tot:.8f} Ha")
```

## Testing

```bash
pytest
```

## License

Apache-2.0.

