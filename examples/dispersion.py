"""B2PLYP-D3(BJ) dispersion correction via dftd3 (Si diamond)."""
from kdh import KRDH
from examples.periodic_energy import silicon_cell


def main():
    cell = silicon_cell()
    kpts = cell.make_kpts([2, 2, 2])
    plain = KRDH(cell, xc="B2PLYP", kpts=kpts).kernel()
    corrected = KRDH(cell, xc="B2PLYP-D3BJ", kpts=kpts)
    e = corrected.kernel()
    print(f"B2PLYP:        {plain:.8f} Ha")
    print(f"B2PLYP-D3(BJ): {e:.8f} Ha")
    print(f"E_disp:        {corrected.e_disp:.8f} Ha")


if __name__ == "__main__":
    main()
