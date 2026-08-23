"""SCF stabilization utilities for periodic calculations."""
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
    """Settings to stabilize periodic SCF convergence."""
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


def configure_periodic_scf(mf, settings: Any | None):
    if settings is None:
        return None
    max_cycle = getattr(settings, "max_cycle", None)
    if max_cycle is not None:
        mf.max_cycle = max_cycle
    diis_space = getattr(settings, "diis_space", None)
    if diis_space is not None:
        mf.diis_space = diis_space
    level_shift = getattr(settings, "level_shift", None)
    if level_shift is not None:
        mf.level_shift = level_shift
    fock_damping = getattr(settings, "fock_damping", None)
    if fock_damping is not None:
        mf.damp = fock_damping
    diis_damp = getattr(settings, "diis_damp", None)
    if diis_damp is not None:
        if not (0.0 <= diis_damp < 1.0):
            raise ValueError("diis_damp requires 0 <= damp < 1")
        if not hasattr(type(mf), "diis_damp"):
            raise RuntimeError("this PySCF build does not support diis_damp")
        mf.diis_damp = diis_damp
    lindep_threshold = getattr(settings, "lindep_threshold", None)
    if lindep_threshold is not None:
        if not (0.0 < lindep_threshold < 1e-3):
            raise ValueError("lindep_threshold requires 0 < threshold < 1e-3")
        from pyscf.scf import addons

        addons.remove_linear_dep_(
            mf,
            threshold=lindep_threshold,
            lindep=lindep_threshold,
        )
    density_mixing_beta = getattr(settings, "density_mixing_beta", None)
    if density_mixing_beta is not None:
        if not (0.0 < density_mixing_beta <= 1.0):
            raise ValueError("density damping requires 0 < beta <= 1")
        if density_mixing_beta < 1.0:
            start_cycle = getattr(settings, "density_mixing_start_cycle", 1)
            return install_damped_density_mixer(
                mf,
                beta=density_mixing_beta,
                start_cycle=start_cycle,
            )
    return None


