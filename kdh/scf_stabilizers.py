from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


def _copy_density(dm):
    if isinstance(dm, (list, tuple)):
        return [np.array(block, copy=True) for block in dm]
    return np.array(dm, copy=True)


def mix_density_update(dm_old, dm_new, *, beta: float):
    """Return dm_old + beta * (dm_new - dm_old)."""
    if not (0.0 < beta <= 1.0):
        raise ValueError("density damping requires 0 < beta <= 1")
    if isinstance(dm_old, (list, tuple)):
        return [
            old + beta * (new - old)
            for old, new in zip(dm_old, dm_new, strict=True)
        ]
    return dm_old + beta * (dm_new - dm_old)


@dataclass
class DampedDensityHandle:
    mf: Any
    original_get_fock: Callable
    original_make_rdm1: Callable
    original_pre_kernel: Callable | None
    beta: float
    start_cycle: int
    last_dm: Any = None
    active_cycle: int | None = None

    def restore(self) -> None:
        self.mf.get_fock = self.original_get_fock
        self.mf.make_rdm1 = self.original_make_rdm1
        if self.original_pre_kernel is not None:
            self.mf.pre_kernel = self.original_pre_kernel


@dataclass(frozen=True)
class SCFStabilizationSettings:
    """Settings that harden a periodic SCF calculation against convergence failure.

    Attributes
    ----------
    max_cycle : int or None
        Override the SCF maximum iteration count.
    diis_space : int or None
        Override the DIIS history size.
    level_shift : float or None
        Virtual-orbital level shift in Hartree, passed to ``mf.level_shift``.
    fock_damping : float or None
        Fock-matrix damping coefficient passed to ``mf.damp``.
    density_mixing_beta : float or None
        Step size for the density-matrix damped-update mixer (0 < beta <= 1).
        Values below 1.0 activate :func:`install_damped_density_mixer`.
    density_mixing_start_cycle : int
        SCF cycle (0-indexed) at which density mixing begins. Default 1.
    diis_damp : float or None
        HZ's damped-DIIS parameter (PySCF PR #2053, commit 559202325). The
        Fock entering the DIIS history becomes ``f*(1-damp) + f_prev*damp``,
        so damping stays active *during* DIIS instead of only before
        ``diis_start_cycle``. This is the analog of QE ``mixing_beta`` /
        FHI-aims ``charge_mix_param``. 0 disables; values near 1 mean heavy
        damping (tiny steps).
    lindep_threshold : float or None
        Project out near-null overlap eigenvectors (eigenvalue < threshold)
        per k-point via ``pyscf.scf.addons.remove_linear_dep_`` (canonical
        orthogonalization; ``khf.eig`` dispatches to the overridden ``_eigh``
        for every k). The same value must be passed as the activation trigger
        ``lindep``: the PySCF default trigger (``cond(S) > 1e10``) silently
        no-ops on the ``cond ~ 1e9`` compressed-volume cases this exists for.
    """

    max_cycle: int | None = None
    diis_space: int | None = None
    level_shift: float | None = None
    fock_damping: float | None = None
    density_mixing_beta: float | None = None
    density_mixing_start_cycle: int = 1
    diis_damp: float | None = None
    lindep_threshold: float | None = None


def install_damped_density_mixer(mf, *, beta: float, start_cycle: int = 1):
    if not (0.0 < beta <= 1.0):
        raise ValueError("density damping requires 0 < beta <= 1")
    if start_cycle < 0:
        raise ValueError("start_cycle must be non-negative")

    original_get_fock = mf.get_fock
    original_make_rdm1 = mf.make_rdm1
    original_pre_kernel = getattr(mf, "pre_kernel", None)
    handle = DampedDensityHandle(
        mf=mf,
        original_get_fock=original_get_fock,
        original_make_rdm1=original_make_rdm1,
        original_pre_kernel=original_pre_kernel,
        beta=beta,
        start_cycle=start_cycle,
    )

    def pre_kernel(envs):
        if "dm" in envs:
            handle.last_dm = _copy_density(envs["dm"])
        if callable(original_pre_kernel):
            original_pre_kernel(envs)

    def get_fock(*args, **kwargs):
        cycle = kwargs.get("cycle", args[4] if len(args) >= 5 else -1)
        handle.active_cycle = cycle if cycle is not None and cycle >= 0 else None
        return original_get_fock(*args, **kwargs)

    def make_rdm1(*args, **kwargs):
        dm_new = original_make_rdm1(*args, **kwargs)
        if handle.active_cycle is None or handle.active_cycle < start_cycle:
            handle.last_dm = _copy_density(dm_new)
            return dm_new
        if handle.last_dm is None:
            handle.last_dm = _copy_density(dm_new)
            return dm_new
        dm_mixed = mix_density_update(handle.last_dm, dm_new, beta=beta)
        handle.last_dm = _copy_density(dm_mixed)
        return dm_mixed

    mf.pre_kernel = pre_kernel
    mf.get_fock = get_fock
    mf.make_rdm1 = make_rdm1
    return handle


def configure_periodic_scf(mf, settings: SCFStabilizationSettings | None):
    if settings is None:
        return None
    if settings.max_cycle is not None:
        mf.max_cycle = settings.max_cycle
    if settings.diis_space is not None:
        mf.diis_space = settings.diis_space
    if settings.level_shift is not None:
        mf.level_shift = settings.level_shift
    if settings.fock_damping is not None:
        mf.damp = settings.fock_damping
    if settings.diis_damp is not None:
        if not (0.0 <= settings.diis_damp < 1.0):
            raise ValueError("diis_damp requires 0 <= damp < 1")
        if not hasattr(type(mf), "diis_damp"):
            raise RuntimeError(
                "this PySCF runtime does not support diis_damp "
                "(needs PySCF PR #2053 / commit 559202325)"
            )
        mf.diis_damp = settings.diis_damp
    if settings.lindep_threshold is not None:
        if not (0.0 < settings.lindep_threshold < 1e-3):
            raise ValueError("lindep_threshold requires 0 < threshold < 1e-3")
        from pyscf.scf import addons

        addons.remove_linear_dep_(
            mf,
            threshold=settings.lindep_threshold,
            lindep=settings.lindep_threshold,
        )
    if settings.density_mixing_beta is not None:
        if not (0.0 < settings.density_mixing_beta <= 1.0):
            raise ValueError("density damping requires 0 < beta <= 1")
    if settings.density_mixing_beta is not None and settings.density_mixing_beta < 1.0:
        return install_damped_density_mixer(
            mf,
            beta=settings.density_mixing_beta,
            start_cycle=settings.density_mixing_start_cycle,
        )
    return None
