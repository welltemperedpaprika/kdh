"""KRDH reproduces a manual KRKS + KMP2 double-hybrid assembly."""
from pyscf.pbc import dft, mp

from kdh import KRDH
from kdh.xc import parse_dh_xc
from examples.periodic_energy import silicon_cell


def main():
    cell = silicon_cell()
    kpts = cell.make_kpts([2, 2, 2])
    e_kdh = KRDH(cell, xc="B2PLYP", kpts=kpts).kernel()

    spec = parse_dh_xc("B2PLYP")
    mf = dft.KRKS(cell, kpts=kpts, xc=spec.xc_scf).density_fit()
    mf.kernel()
    mmp = mp.KMP2(mf)
    mmp.kernel(with_t2=False)
    e_manual = mf.e_tot + spec.c_pt2 * (mmp.e_corr_os + mmp.e_corr_ss)

    delta = e_kdh - e_manual
    print(f"KRDH:   {e_kdh:.10f} Ha")
    print(f"manual: {e_manual:.10f} Ha")
    print(f"delta:  {delta:.2e} Ha")
    assert abs(delta) < 1e-9, delta
    print("parity OK")


if __name__ == "__main__":
    main()
