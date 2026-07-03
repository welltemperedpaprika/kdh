# kdh

Molecular and k-point periodic double-hybrid DFT for [PySCF](https://github.com/pyscf/pyscf).

`kdh` provides double-hybrid drivers that combine a non-self-consistent (xDH)
exchange–correlation flow with an MP2 / KMP2 perturbative correlation
correction:

- **`KRDH`** — k-point periodic driver built on `pyscf.pbc.mp.KMP2`, with
  opposite-spin / same-spin (OS/SS) spin scaling, the `xc_scf -> xc_nscf`
  non-self-consistent flow, reference safety gates (gap / fractional
  occupation), SCF stabilizers, and dispersion.
- **`RDFDH`** — molecular closed-shell driver for the same functional family,
  with an opt-in density-fitting path (`df=True`) and an analytic nuclear
  gradient for conventional, unscaled-MP2 functionals (e.g. B2PLYP).
- **`UDFDH`** — molecular open-shell (unrestricted) driver (any spin).

**Dispersion** is a built-in, additive `dftd3`-backed D3(BJ) / D3(0) correction,
resolved from functional metadata (no external callable needed); the electronic
functional is unchanged. **Gradients**: finite-difference `numerical_nuc_grad`
+ `optimize` for any driver, plus the closed-shell analytic double-hybrid
gradient above.

A functional registry maps names to exchange, correlation, and PT2 coefficients:
`B2PLYP`, `XYG3`, `XYGJ-OS`, `PBE0-DH`, `PBE0-QIDH`, `PBE0-2`, the dispersion-
corrected `B2PLYP-D3BJ`, `B2GP-PLYP-D3BJ`, `mPW2PLYP-D3BJ`, `DSD-BLYP-D3BJ`,
`DSD-PBEP86-D3BJ`, `revDSD-PBEP86-D3BJ`, `revDSD-BLYP-D3BJ`, and the HF-reference
`SCS-MP2` / `SOS-MP2`, among others.

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

See `examples/` for molecular, periodic, parity, dispersion, open-shell, and
geometry-optimization demos.

## Scope

Molecular closed-shell (`RDFDH`), molecular open-shell (`UDFDH`), and k-point
periodic closed-shell (`KRDH`). Periodic open-shell and periodic analytic
gradients raise a clear `NotImplementedError` (PySCF has no working periodic
unrestricted KMP2 / KMP2 relaxed density); use the finite-difference gradient
for periodic forces. The analytic gradient is closed-shell, conventional,
unscaled-MP2 only; every other case is refused with a message naming the
missing response terms.

## License

Apache-2.0.
