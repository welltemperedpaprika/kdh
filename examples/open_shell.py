"""Open-shell double-hybrid energy with UDFDH (triplet O atom)."""
from pyscf import gto

from kdh import UDFDH


def main():
    mol = gto.M(atom="O 0 0 0", spin=2, basis="cc-pVDZ", verbose=0)
    dh = UDFDH(mol, xc="B2PLYP")
    e = dh.kernel()
    print(f"O atom (triplet) B2PLYP total energy: {e:.8f} Ha")
    print(f"  OS correlation: {dh.e_corr_os:.8f} Ha")
    print(f"  SS correlation: {dh.e_corr_ss:.8f} Ha")


if __name__ == "__main__":
    main()
