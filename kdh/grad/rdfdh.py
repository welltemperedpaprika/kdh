"""Analytic nuclear gradient for conventional molecular double hybrids.

Differentiates ``E = E_KS[xc_scf] + c_pt2 * E_MP2`` for the first-target case
only: restricted (RKS) reference, ``xc_nscf is None``, full-range, and unscaled
MP2 (``c_os == c_ss``, e.g. B2PLYP). The gradient is

    dE/dx = dE_KS/dx + lam * dE_corr/dx,   lam = c_pt2 * c_os,

with ``dE_KS/dx`` the native variational RKS gradient (grid response included)
and ``dE_corr/dx`` the correlation-only MP2 gradient assembled here.

This is *not* ``mp.MP2(mf_ks).nuc_grad_method()``: ``pyscf.grad.mp2`` assumes an
RHF reference, so its Z-vector solve and reference-Fock derivative use the
bare-HF operator. For a KS-hybrid reference the two reference-dependent
operators are replaced by their Kohn-Sham generalizations:

1. Z-vector / Lagrangian response uses ``mf_s.gen_response(singlet=None,
   hermi=1)`` -- the KS orbital Hessian ``J + fxc + c_x K`` -- not the bare-HF
   ``get_veff``.
2. The reference-Fock nuclear derivative uses the Kohn-Sham
   ``pyscf.hessian.rks.Hessian.make_h1`` skeleton (``hcore + J + c_x K + Vxc``,
   including the XC-kernel geometric response), with the one-electron part split
   off. That ``make_h1`` skeleton neglects the grid-weight (quadrature-point)
   response of the XC term -- a small approximation, well below the validation
   thresholds. This is the Neese-Schwabe-Grimme (J. Chem. Phys. 126, 124115,
   2007) assembly.

Validation: the assembled B2PLYP gradient matches
:func:`kdh.numderiv.numerical_nuc_grad` to ~4.0e-7 Ha/bohr (residual is FD
step truncation), and in the pure-HF limit ``rhf_grad + dE_corr`` reproduces
``pyscf.grad.mp2`` to ~1e-15 Ha/bohr (an analytic reference, no FD noise).
"""
from __future__ import annotations

from functools import reduce

import numpy as np
from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.grad.mp2 import _shell_prange
from pyscf.lib import logger
from pyscf.mp import mp2
from pyscf.scf import cphf

einsum = lib.einsum


def _make_h1_veff_deriv(mf_s, hcore_deriv):
    """Kohn-Sham reference-Fock nuclear derivative, minus the core term.

    Returns a ``(natm, 3, nao, nao)`` array whose atom-``A`` block is
    ``d/dR_A (J[D] + c_x K[D] + Vxc[D])`` at fixed reference density ``D``,
    built from ``Hessian.make_h1`` (``d/dR_A (hcore + J + c_x K + Vxc)``) with
    the one-electron ``hcore`` derivative subtracted so the caller can weight
    the two-particle cross term itself.
    """
    hess = mf_s.Hessian()
    h1ao = np.asarray(hess.make_h1(mf_s.mo_coeff, mf_s.mo_occ))
    veff_deriv = np.empty_like(h1ao)
    for ia in range(mf_s.mol.natm):
        veff_deriv[ia] = h1ao[ia] - hcore_deriv(ia)
    return veff_deriv


def grad_elec_correlation(mf_s, lam, atmlst=None, max_memory=2000,
                          cphf_max_cycle=50, cphf_tol=1e-9, verbose=logger.INFO):
    """Correlation-only MP2 nuclear gradient on a Kohn-Sham reference.

    Assembles ``lam * dE_MP2/dx`` (electronic part only) for the unscaled MP2
    correlation on the converged reference ``mf_s`` (``RKS``, or ``RHF`` for the
    pure-HF validation path), with the orbital response using the Kohn-Sham
    orbital Hessian. ``lam = c_pt2 * c_os`` (== ``c_pt2 * c_ss``). Add the pure
    KS/HF SCF gradient separately.
    """
    log = logger.new_logger(mf_s, verbose)
    mol = mf_s.mol
    mo_coeff = np.asarray(mf_s.mo_coeff)
    mo_energy = np.asarray(mf_s.mo_energy)
    mo_occ = np.asarray(mf_s.mo_occ)
    nao, nmo = mo_coeff.shape
    nocc = int((mo_occ > 0).sum())
    nvir = nmo - nocc
    orbo = mo_coeff[:, :nocc]
    orbv = mo_coeff[:, nocc:]

    pt = mp2.MP2(mf_s)
    pt.verbose = 0
    pt.kernel(with_t2=True)
    t2 = pt.t2

    doo, dvv = mp2._gamma1_intermediates(pt, t2)
    doo = doo * lam
    dvv = dvv * lam

    part_dm2 = _ao2mo.nr_e2(t2.reshape(nocc**2, nvir**2),
                            np.asarray(orbv.T, order='F'), (0, nao, 0, nao),
                            's1', 's1').reshape(nocc, nocc, nao, nao)
    part_dm2 = (part_dm2.transpose(0, 2, 3, 1) * 4
                - part_dm2.transpose(0, 3, 2, 1) * 2) * lam

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    diagidx = np.arange(nao)
    diagidx = diagidx * (diagidx + 1) // 2 + diagidx
    de = np.zeros((len(atmlst), 3))
    Imat = np.zeros((nao, nao))

    blksize = max(1, int(max(1, max_memory) * .9e6 / 8 / (nao**3 * 2.5)))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
        ip1 = p0
        for b0, b1, nf in _shell_prange(mol, shl0, shl1, blksize):
            ip0, ip1 = ip1, ip1 + nf
            dm2buf = einsum('pi,iqrj->pqrj', orbo[ip0:ip1], part_dm2)
            dm2buf += einsum('qi,iprj->pqrj', orbo, part_dm2[:, ip0:ip1])
            dm2buf = einsum('pqrj,sj->pqrs', dm2buf, orbo)
            dm2buf = dm2buf + dm2buf.transpose(0, 1, 3, 2)
            dm2buf = lib.pack_tril(dm2buf.reshape(-1, nao, nao)).reshape(nf, nao, -1)
            dm2buf[:, :, diagidx] *= .5

            shls_slice = (b0, b1, 0, mol.nbas, 0, mol.nbas, 0, mol.nbas)
            eri0 = mol.intor('int2e', aosym='s2kl', shls_slice=shls_slice)
            Imat += einsum('ipx,iqx->pq', eri0.reshape(nf, nao, -1), dm2buf)
            eri0 = None

            eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                             shls_slice=shls_slice).reshape(3, nf, nao, -1)
            de[k] -= einsum('xijk,ijk->x', eri1, dm2buf) * 2
            eri1 = dm2buf = None

    Imat = reduce(np.dot, (mo_coeff.T, Imat, mf_s.get_ovlp(), mo_coeff)) * -1

    dm1mo = np.zeros((nmo, nmo))
    dm1mo[:nocc, :nocc] = doo + doo.T
    dm1mo[nocc:, nocc:] = dvv + dvv.T

    vresp = mf_s.gen_response(singlet=None, hermi=1)

    dm1_ao = reduce(np.dot, (mo_coeff, dm1mo, mo_coeff.T))
    vhf = vresp(dm1_ao) * 2
    Xvo = reduce(np.dot, (orbv.T, vhf, orbo))
    Xvo += Imat[:nocc, nocc:].T - Imat[nocc:, :nocc]

    def fvind(x):
        x = x.reshape(nvir, nocc)
        dm = reduce(np.dot, (orbv, x, orbo.T))
        v = vresp(dm + dm.T)
        v = reduce(np.dot, (orbv.T, v, orbo))
        return v * 2

    dvo = cphf.solve(fvind, mo_energy, mo_occ, Xvo,
                     max_cycle=cphf_max_cycle, tol=cphf_tol)[0]
    dm1mo[nocc:, :nocc] = dvo
    dm1mo[:nocc, nocc:] = dvo.T
    log.timer_debug1('correlation Z-vector (KS response)')

    Imat[nocc:, :nocc] = Imat[:nocc, nocc:].T
    im1 = reduce(np.dot, (mo_coeff, Imat, mo_coeff.T))

    mf_grad = mf_s.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    zeta = lib.direct_sum('i+j->ij', mo_energy, mo_energy) * .5
    zeta[nocc:, :nocc] = mo_energy[:nocc]
    zeta[:nocc, nocc:] = mo_energy[:nocc].reshape(-1, 1)
    zeta = reduce(np.dot, (mo_coeff, zeta * dm1mo, mo_coeff.T))

    dm1 = reduce(np.dot, (mo_coeff, dm1mo, mo_coeff.T))
    p1 = np.dot(orbo, orbo.T)
    vhf_s1occ = reduce(np.dot, (p1, vresp(dm1 + dm1.T), p1))

    veff_deriv = _make_h1_veff_deriv(mf_s, hcore_deriv)

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1a = offsetdic[ia]
        de[k] += einsum('xij,ij->x', s1[:, p0:p1a], im1[p0:p1a])
        de[k] += einsum('xji,ij->x', s1[:, p0:p1a], im1[:, p0:p1a])

        h1ao = hcore_deriv(ia)
        de[k] += einsum('xij,ji->x', h1ao, dm1)

        de[k] -= einsum('xij,ij->x', s1[:, p0:p1a], zeta[p0:p1a])
        de[k] -= einsum('xji,ij->x', s1[:, p0:p1a], zeta[:, p0:p1a])

        de[k] -= einsum('xij,ij->x', s1[:, p0:p1a], vhf_s1occ[p0:p1a]) * 2
        de[k] += einsum('xij,ij->x', veff_deriv[ia], dm1)

    return de


class Gradients(lib.StreamObject):
    """Analytic nuclear gradient object for a supported :class:`~kdh.rdfdh.RDFDH`.

    Produced by :meth:`kdh.rdfdh.RDFDH.nuc_grad_method`, which performs the scope
    checks. The supported case is closed-shell, conventional (``xc_nscf is
    None``), full-range, unscaled-MP2 (``c_os == c_ss``) double hybrids without
    dispersion, frozen core, or density fitting.
    """

    def __init__(self, base):
        self.base = base
        self.mol = base.mol
        self.stdout = getattr(base.mol, "stdout", None)
        self.verbose = getattr(base.mol, "verbose", 0)
        self.max_memory = getattr(base.mol, "max_memory", 4000)
        self.cphf_max_cycle = 50
        self.cphf_tol = 1e-9
        self.grid_response = True
        self.atmlst = None
        self.de = None

    def kernel(self, atmlst=None):
        """Compute and return the total nuclear gradient (natm, 3) in Ha/bohr."""
        base = self.base
        if base.e_tot is None:
            base.kernel()
        mf_s = base.mf_s
        if not mf_s.converged:
            raise RuntimeError("RDFDH SCF is not converged; cannot differentiate")

        if atmlst is None:
            atmlst = range(self.mol.natm)
        atmlst = list(atmlst)

        lam = base.xc_dh.c_pt2 * base.xc_dh.c_os
        log = logger.new_logger(self, self.verbose)

        scf_grad = mf_s.nuc_grad_method()
        if hasattr(scf_grad, "grid_response"):
            scf_grad.grid_response = self.grid_response
        de_scf = scf_grad.kernel(atmlst=atmlst)

        de_corr = grad_elec_correlation(
            mf_s, lam, atmlst=atmlst, max_memory=self.max_memory,
            cphf_max_cycle=self.cphf_max_cycle, cphf_tol=self.cphf_tol,
            verbose=self.verbose,
        )

        self.de = de_scf + de_corr
        if self.mol.symmetry:
            self.de = scf_grad.symmetrize(self.de, atmlst)
        log.timer_debug1("RDFDH analytic gradient")
        self.atmlst = atmlst
        return self.de

    grad = kernel

    def as_scanner(self):
        raise NotImplementedError(
            "RDFDH analytic-gradient scanner is not implemented; use "
            "kdh.numderiv.optimize for geometry optimization."
        )
