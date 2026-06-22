# kdh

Molecular and k-point periodic double-hybrid DFT for [PySCF](https://github.com/pyscf/pyscf).

`kdh` provides closed-shell double-hybrid drivers that combine a
non-self-consistent (xDH) exchange–correlation flow with an MP2 / KMP2
perturbative correlation correction:

- **`KRDH`** — k-point periodic driver built on `pyscf.pbc.mp.KMP2`, with
  opposite-spin / same-spin (OS/SS) spin scaling, the `xc_scf -> xc_nscf`
  non-self-consistent flow, reference safety gates (gap / fractional
  occupation), SCF stabilizers, and an optional dftd3-backed D3(BJ)
  dispersion correction.
- **`RDFDH`** — molecular driver for the same functional family.

A functional registry (`B2PLYP`, `XYG3`, `XYGJ-OS`, `PBE0-DH`, `PBE0-QIDH`,
`PBE0-2`, `B2PLYP-D3BJ`, …) maps names to exchange, correlation, and PT2
coefficients.

## Install

```bash
pip install -e .              # core
pip install -e .[dispersion]  # + D3(BJ) dispersion (needs dftd3)
```

## Quick start

```python
import numpy as np
from pyscf.pbc import gto
from kdh import KRDH

a = 5.43
fcc = [(0,0,0),(0,0.5,0.5),(0.5,0,0.5),(0.5,0.5,0)]
dia = [(0,0,0),(0.25,0.25,0.25)]
frac = np.array([[b[i]+d[i] for i in range(3)] for b in fcc for d in dia]) % 1.0
cell = gto.Cell()
cell.atom = [("Si", tuple(x)) for x in frac @ (np.eye(3)*a)]
cell.a = np.eye(3)*a
cell.unit = "A"; cell.basis = "gth-dzvp"; cell.pseudo = "gth-pade"
cell.build()

dh = KRDH(cell, xc="B2PLYP", kpts=cell.make_kpts([2, 2, 2]))
print(dh.kernel())   # total double-hybrid energy
```

See `examples/` for molecular, periodic, parity, and dispersion demos.

## Scope

Closed-shell, energy-only. Open-shell and analytic gradients raise a clear
`NotImplementedError`. Dispersion is the additive D3(BJ) correction (the
electronic functional is unchanged).

## License

Apache-2.0.
