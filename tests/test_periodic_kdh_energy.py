import numpy as np
import pytest

from kdh.krdh import KRDH


class FakeCell:
    verbose = 0
    stdout = None
    max_memory = 4000
    spin = 0

    def make_kpts(self, mesh):
        assert mesh == [1, 1, 1]
        return np.zeros((1, 3))


class OpenShellCell(FakeCell):
    spin = 1


class FakeKRKS:
    instances = []

    def __init__(self, cell, kpts=None, xc=None):
        self.cell = cell
        self.mol = cell
        self.kpts = kpts
        self.xc = xc
        self.e_tot = 0.0
        self.converged = False
        self.max_memory = cell.max_memory
        self.grids = object()
        self.mo_coeff = ["coeff"]
        self.mo_occ = [np.array([2.0, 2.0, 0.0, 0.0])]
        self.mo_energy = [np.array([-1.0, -0.5, 0.2, 0.7])]
        self.dm = object()
        self.energy_tot_dm = None
        self.df_backend = None
        self.with_df = None
        FakeKRKS.instances.append(self)

    def density_fit(self):
        self.df_backend = "gdf"
        self.with_df = object()
        return self

    def rs_density_fit(self):
        self.df_backend = "rsdf"
        self.with_df = object()
        return self

    def kernel(self, **kwargs):
        self.kernel_kwargs = kwargs
        self.converged = True
        self.e_tot = -5.0
        return self.e_tot

    def make_rdm1(self):
        return self.dm

    def energy_tot(self, dm=None):
        self.energy_tot_dm = dm
        return -4.75


class FakeKRHF(FakeKRKS):
    instances = []

    def __init__(self, cell, kpts=None):
        super().__init__(cell, kpts=kpts, xc="HF")
        FakeKRHF.instances.append(self)


class FakeKMP2:
    instances = []

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        self.mf = mf
        self.frozen = frozen
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.e_corr_os = None
        self.e_corr_ss = None
        FakeKMP2.instances.append(self)

    def kernel(self, with_t2=False):
        self.with_t2 = with_t2
        self.e_corr_os = -0.20
        self.e_corr_ss = -0.05
        return self.e_corr_os + self.e_corr_ss, None


def test_xyg3_uses_xc_scf_for_orbitals_and_xc_nscf_for_dfa(monkeypatch):
    FakeKRKS.instances = []
    FakeKMP2.instances = []
    monkeypatch.setattr("kdh.krdh.dft.KRKS", FakeKRKS)
    monkeypatch.setattr("kdh.krdh.mp.KMP2", FakeKMP2)

    mf = KRDH(FakeCell(), xc="XYG3")

    assert mf.kernel() == pytest.approx(-4.75 + 0.3211 * (-0.20 - 0.05))
    assert FakeKRKS.instances[0].xc == "B3LYPg"
    assert (
        FakeKRKS.instances[1].xc
        == "-0.014*LDA + 0.8033*HF + 0.2107*B88, 0.6789*LYP"
    )
    assert FakeKRKS.instances[1].energy_tot_dm is FakeKRKS.instances[0].dm


def test_spin_scaled_pt2_uses_kmp2_os_ss_components(monkeypatch):
    FakeKRKS.instances = []
    FakeKMP2.instances = []
    monkeypatch.setattr("kdh.krdh.dft.KRKS", FakeKRKS)
    monkeypatch.setattr("kdh.krdh.mp.KMP2", FakeKMP2)

    mf = KRDH(
        FakeCell(),
        xc={
            "name": "CUSTOM-PBC-DH",
            "xc_scf": "PBE",
            "c_pt2": 0.5,
            "c_os": 1.2,
            "c_ss": 0.3,
        },
        with_t2=True,
    )

    assert mf.kernel() == pytest.approx(-5.0 + 0.5 * (1.2 * -0.20 + 0.3 * -0.05))
    assert FakeKMP2.instances[0].with_t2 is True
    assert mf.e_corr_os == pytest.approx(-0.20)
    assert mf.e_corr_ss == pytest.approx(-0.05)


def test_hfmp2_routes_scf_through_krhf_not_krks(monkeypatch):
    FakeKRKS.instances = []
    FakeKRHF.instances = []
    FakeKMP2.instances = []
    monkeypatch.setattr("kdh.krdh.dft.KRKS", FakeKRKS)
    monkeypatch.setattr("kdh.krdh.scf.KRHF", FakeKRHF)
    monkeypatch.setattr("kdh.krdh.mp.KMP2", FakeKMP2)

    mf = KRDH(FakeCell(), xc="HFMP2")

    assert mf.kernel() == pytest.approx(-5.0 + (-0.20 - 0.05))
    assert len(FakeKRKS.instances) == 1
    assert isinstance(FakeKRKS.instances[0], FakeKRHF)
    assert len(FakeKRHF.instances) == 1


def test_xdh_energy_dfa_shares_scf_density_fitting_object(monkeypatch):
    FakeKRKS.instances = []
    FakeKMP2.instances = []
    monkeypatch.setattr("kdh.krdh.dft.KRKS", FakeKRKS)
    monkeypatch.setattr("kdh.krdh.mp.KMP2", FakeKMP2)

    mf = KRDH(FakeCell(), xc="XYG3")
    mf.energy_dfa()

    assert mf.mf_n is not mf.mf_s
    assert mf.mf_s.with_df is not None
    assert mf.mf_n.with_df is mf.mf_s.with_df


def test_open_shell_cell_fails_clearly():
    with pytest.raises(NotImplementedError, match="closed-shell"):
        KRDH(OpenShellCell(), xc="B2PLYP")


def test_periodic_derivatives_fail_clearly():
    mf = KRDH(FakeCell(), xc="B2PLYP")

    with pytest.raises(NotImplementedError, match="Periodic double-hybrid gradients"):
        mf.nuc_grad_method()

    with pytest.raises(NotImplementedError, match="Periodic double-hybrid properties"):
        mf.polar_method()


def test_krdh_restores_scf_stabilizer_after_kernel(monkeypatch):
    calls = []

    class FakeHandle:
        def restore(self):
            calls.append("restore")

    def fake_configure(mf, settings):
        calls.append(("configure", mf, settings))
        return FakeHandle()

    monkeypatch.setattr("kdh.krdh.dft.KRKS", FakeKRKS)
    monkeypatch.setattr(
        "kdh.scf_stabilizers.configure_periodic_scf",
        fake_configure,
    )

    settings = object()
    mf = KRDH(FakeCell(), xc="B2PLYP", scf_stabilization=settings)

    assert mf.run_scf() is FakeKRKS.instances[-1]
    assert calls[0] == ("configure", FakeKRKS.instances[-1], settings)
    assert calls[1] == "restore"
    assert mf._scf_stabilizer_handle is None


def test_krdh_restores_scf_stabilizer_when_kernel_raises(monkeypatch):
    calls = []

    class RaisingKRKS(FakeKRKS):
        def kernel(self, **kwargs):
            raise ValueError("kernel failed")

    class FakeHandle:
        def restore(self):
            calls.append("restore")

    monkeypatch.setattr("kdh.krdh.dft.KRKS", RaisingKRKS)
    monkeypatch.setattr(
        "kdh.scf_stabilizers.configure_periodic_scf",
        lambda mf, settings: FakeHandle(),
    )

    mf = KRDH(FakeCell(), xc="B2PLYP", scf_stabilization=object())

    with pytest.raises(ValueError, match="kernel failed"):
        mf.run_scf()
    assert calls == ["restore"]
    assert mf._scf_stabilizer_handle is None


def test_krdh_still_raises_when_stabilized_scf_does_not_converge(monkeypatch):
    class UnconvergedKRKS(FakeKRKS):
        def kernel(self, **kwargs):
            self.converged = False
            self.e_tot = -5.0
            return self.e_tot

    monkeypatch.setattr("kdh.krdh.dft.KRKS", UnconvergedKRKS)
    monkeypatch.setattr(
        "kdh.scf_stabilizers.configure_periodic_scf",
        lambda mf, settings: None,
    )

    mf = KRDH(FakeCell(), xc="B2PLYP", scf_stabilization=object())

    with pytest.raises(RuntimeError, match="KRDH SCF did not converge"):
        mf.run_scf()
