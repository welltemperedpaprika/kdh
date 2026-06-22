"""Molecular double-hybrid energy with RDFDH."""
from pyscf import gto

from kdh import RDFDH


def main():
    mol = gto.M(atom="O 0 0 0; H 0 0 0.96; H 0.93 0 -0.26", basis="def2-SVP")
    e = RDFDH(mol, xc="B2PLYP").kernel()
    print(f"water B2PLYP total energy: {e:.8f} Ha")


if __name__ == "__main__":
    main()
