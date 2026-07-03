import pytest

from kdh.udfdh import UDFDH


class FakeMol:
    verbose = 0
    stdout = None
    max_memory = 4000
    spin = 1


class ClosedShellMol(FakeMol):
    spin = 0


class FakeUKS:
    instances = []

    def __init__(self, mol, xc=None):
        self.mol = mol
        self.xc = xc
        self.e_tot = 0.0
        self.converged = False
        self.dm = object()
        self.energy_tot_dm = None
        self.grids = object()
        FakeUKS.instances.append(self)

    def kernel(self):
        self.converged = True
        self.e_tot = -10.0
        return self.e_tot

    def make_rdm1(self):
        return self.dm

    def energy_tot(self, dm=None):
        self.energy_tot_dm = dm
        return -9.5


class FakeUHF(FakeUKS):
    instances = []

    def __init__(self, mol):
        super().__init__(mol, xc="HF")
        FakeUHF.instances.append(self)


class FakeUMP2:
    def __init__(self, mf, frozen=None):
        self.mf = mf
        self.frozen = frozen
        self.e_corr = None
        self.e_corr_os = None
        self.e_corr_ss = None

    def kernel(self, with_t2=False):
        self.e_corr_os = -0.30
        self.e_corr_ss = -0.10
        self.e_corr = self.e_corr_os + self.e_corr_ss
        return self.e_corr, None


def test_open_shell_molecule_is_accepted():
    mf = UDFDH(FakeMol(), xc="B2PLYP")
    assert mf.mol.spin == 1


def test_b2plyp_uses_scf_energy_and_scaled_os_ss(monkeypatch):
    FakeUKS.instances = []
    monkeypatch.setattr("kdh.udfdh.dft.UKS", FakeUKS)
    monkeypatch.setattr("kdh.udfdh.mp.UMP2", FakeUMP2)

    mf = UDFDH(FakeMol(), xc="B2PLYP")

    assert mf.kernel() == pytest.approx(-10.0 + 0.27 * (-0.30 - 0.10))
    assert FakeUKS.instances[0].xc == "0.53*HF + 0.47*B88, 0.73*LYP"
    assert mf.e_corr_os == pytest.approx(-0.30)
    assert mf.e_corr_ss == pytest.approx(-0.10)


def test_xyg3_uses_nscf_energy(monkeypatch):
    FakeUKS.instances = []
    monkeypatch.setattr("kdh.udfdh.dft.UKS", FakeUKS)
    monkeypatch.setattr("kdh.udfdh.mp.UMP2", FakeUMP2)

    mf = UDFDH(FakeMol(), xc="XYG3")

    assert mf.kernel() == pytest.approx(-9.5 + 0.3211 * (-0.30 - 0.10))
    assert FakeUKS.instances[0].xc == "B3LYPg"
    assert (
        FakeUKS.instances[1].xc
        == "-0.014*LDA + 0.8033*HF + 0.2107*B88, 0.6789*LYP"
    )
    assert FakeUKS.instances[1].grids is FakeUKS.instances[0].grids
    assert FakeUKS.instances[1].energy_tot_dm is FakeUKS.instances[0].dm


def test_spin_scaled_custom_functional_routes_os_ss(monkeypatch):
    FakeUKS.instances = []
    FakeUHF.instances = []
    monkeypatch.setattr("kdh.udfdh.dft.UKS", FakeUKS)
    monkeypatch.setattr("kdh.udfdh.scf.UHF", FakeUHF)
    monkeypatch.setattr("kdh.udfdh.mp.UMP2", FakeUMP2)

    mf = UDFDH(
        FakeMol(),
        xc={
            "xc_scf": "HF",
            "c_pt2": 0.5,
            "c_os": 1.3,
            "c_ss": 0.2,
        },
    )

    expected_pt2 = 0.5 * (1.3 * (-0.30) + 0.2 * (-0.10))
    assert mf.kernel() == pytest.approx(-10.0 + expected_pt2)
    assert mf.e_pt2 == pytest.approx(expected_pt2)


def test_spin_scaled_requires_mp2_spin_components(monkeypatch):
    class FakeUMP2MissingSpinComponents:
        def __init__(self, mf, frozen=None):
            self.mf = mf
            self.frozen = frozen
            self.e_corr = None

        def kernel(self, with_t2=False):
            self.e_corr = -0.40
            return self.e_corr, None

    FakeUHF.instances = []
    monkeypatch.setattr("kdh.udfdh.scf.UHF", FakeUHF)
    monkeypatch.setattr("kdh.udfdh.mp.UMP2", FakeUMP2MissingSpinComponents)

    mf = UDFDH(
        FakeMol(),
        xc={
            "xc_scf": "HF",
            "c_pt2": 0.2,
            "c_os": 1.0,
            "c_ss": 0.0,
        },
    )

    with pytest.raises(RuntimeError, match="spin-scaled"):
        mf.kernel()


def test_lr_pt2_functional_fails_clearly():
    mf = UDFDH(
        FakeMol(),
        xc={
            "xc_scf": "HF",
            "c_pt2": 1.0,
            "requires_lr_pt2": True,
        },
    )

    with pytest.raises(NotImplementedError, match="long-range PT2"):
        mf.kernel()


def test_hfmp2_routes_scf_through_uhf_not_uks(monkeypatch):
    FakeUKS.instances = []
    FakeUHF.instances = []
    monkeypatch.setattr("kdh.udfdh.dft.UKS", FakeUKS)
    monkeypatch.setattr("kdh.udfdh.scf.UHF", FakeUHF)
    monkeypatch.setattr("kdh.udfdh.mp.UMP2", FakeUMP2)

    mf = UDFDH(FakeMol(), xc="HFMP2")

    assert mf.kernel() == pytest.approx(-10.0 + (-0.30 - 0.10))
    assert len(FakeUKS.instances) == 1
    assert isinstance(FakeUKS.instances[0], FakeUHF)
    assert len(FakeUHF.instances) == 1


def test_gradients_fail_clearly():
    mf = UDFDH(FakeMol(), xc="B2PLYP")

    with pytest.raises(NotImplementedError, match="open-shell"):
        mf.nuc_grad_method()
