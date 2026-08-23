import pytest

from kdh.rdfdh import RDFDH


class FakeMol:
    verbose = 0
    stdout = None
    max_memory = 4000
    spin = 0


class OpenShellMol(FakeMol):
    spin = 1


class FakeRKS:
    instances = []

    def __init__(self, mol, xc=None):
        self.mol = mol
        self.xc = xc
        self.e_tot = 0.0
        self.converged = False
        self.dm = object()
        self.energy_tot_dm = None
        self.grids = object()
        FakeRKS.instances.append(self)

    def kernel(self):
        self.converged = True
        self.e_tot = -10.0
        return self.e_tot

    def make_rdm1(self):
        return self.dm

    def energy_tot(self, dm=None):
        self.energy_tot_dm = dm
        return -9.5

    def density_fit(self):
        self.density_fitted = True
        return self


class FakeRHF(FakeRKS):
    instances = []

    def __init__(self, mol):
        super().__init__(mol, xc="HF")
        FakeRHF.instances.append(self)


class FakeMP2:
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


def test_b2plyp_uses_scf_energy_and_scaled_os_ss(monkeypatch):
    FakeRKS.instances = []
    monkeypatch.setattr("kdh.rdfdh.dft.RKS", FakeRKS)
    monkeypatch.setattr("kdh.rdfdh.mp.MP2", FakeMP2)

    mf = RDFDH(FakeMol(), xc="B2PLYP")

    assert mf.kernel() == pytest.approx(-10.0 + 0.27 * (-0.30 - 0.10))
    assert FakeRKS.instances[0].xc == "0.53*HF + 0.47*B88, 0.73*LYP"
    assert mf.e_corr_os == pytest.approx(-0.30)
    assert mf.e_corr_ss == pytest.approx(-0.10)


def test_xyg3_uses_nscf_energy(monkeypatch):
    FakeRKS.instances = []
    monkeypatch.setattr("kdh.rdfdh.dft.RKS", FakeRKS)
    monkeypatch.setattr("kdh.rdfdh.mp.MP2", FakeMP2)

    mf = RDFDH(FakeMol(), xc="XYG3")

    assert mf.kernel() == pytest.approx(-9.5 + 0.3211 * (-0.30 - 0.10))
    assert FakeRKS.instances[0].xc == "B3LYPg"
    assert (
        FakeRKS.instances[1].xc
        == "-0.014*LDA + 0.8033*HF + 0.2107*B88, 0.6789*LYP"
    )
    assert FakeRKS.instances[1].grids is FakeRKS.instances[0].grids
    assert FakeRKS.instances[1].energy_tot_dm is FakeRKS.instances[0].dm


def test_hfmp2_routes_scf_through_rhf_not_rks(monkeypatch):
    FakeRKS.instances = []
    FakeRHF.instances = []
    monkeypatch.setattr("kdh.rdfdh.dft.RKS", FakeRKS)
    monkeypatch.setattr("kdh.rdfdh.scf.RHF", FakeRHF)
    monkeypatch.setattr("kdh.rdfdh.mp.MP2", FakeMP2)

    mf = RDFDH(FakeMol(), xc="HFMP2")

    assert mf.kernel() == pytest.approx(-10.0 + (-0.30 - 0.10))
    assert len(FakeRKS.instances) == 1
    assert isinstance(FakeRKS.instances[0], FakeRHF)
    assert len(FakeRHF.instances) == 1


def test_open_shell_molecule_fails_clearly():
    with pytest.raises(NotImplementedError, match="closed-shell"):
        RDFDH(OpenShellMol(), xc="B2PLYP")


def test_zero_pt2_custom_functional_skips_mp2(monkeypatch):
    FakeRKS.instances = []
    FakeRHF.instances = []
    monkeypatch.setattr("kdh.rdfdh.dft.RKS", FakeRKS)
    monkeypatch.setattr("kdh.rdfdh.scf.RHF", FakeRHF)

    def fail_mp2(*args, **kwargs):
        raise AssertionError("MP2 should not run for zero-PT2 functionals")

    monkeypatch.setattr("kdh.rdfdh.mp.MP2", fail_mp2)

    mf = RDFDH(
        FakeMol(),
        xc={
            "xc_scf": "HF",
            "c_pt2": 0.0,
        },
    )

    assert mf.kernel() == pytest.approx(-10.0)
    assert len(FakeRHF.instances) == 1
    assert mf.e_pt2 == pytest.approx(0.0)
    assert mf.e_corr_os == pytest.approx(0.0)
    assert mf.e_corr_ss == pytest.approx(0.0)


def test_scaled_os_ss_requires_mp2_spin_components(monkeypatch):
    class FakeMP2MissingSpinComponents:
        def __init__(self, mf, frozen=None):
            self.mf = mf
            self.frozen = frozen
            self.e_corr = None

        def kernel(self, with_t2=False):
            self.e_corr = -0.40
            return self.e_corr, None

    FakeRKS.instances = []
    FakeRHF.instances = []
    monkeypatch.setattr("kdh.rdfdh.dft.RKS", FakeRKS)
    monkeypatch.setattr("kdh.rdfdh.scf.RHF", FakeRHF)
    monkeypatch.setattr("kdh.rdfdh.mp.MP2", FakeMP2MissingSpinComponents)

    mf = RDFDH(
        FakeMol(),
        xc={
            "xc_scf": "HF",
            "c_pt2": 0.2,
            "c_os": 1.0,
            "c_ss": 0.0,
        },
    )

    with pytest.raises(RuntimeError, match="OS/SS MP2 components"):
        mf.kernel()


def test_lr_pt2_functional_fails_clearly():
    mf = RDFDH(
        FakeMol(),
        xc={
            "xc_scf": "HF",
            "c_pt2": 1.0,
            "requires_lr_pt2": True,
        },
    )

    with pytest.raises(NotImplementedError, match="long-range PT2"):
        mf.kernel()


def test_b2plyp_gradient_is_supported_dh_owned():
    mf = RDFDH(FakeMol(), xc="B2PLYP")

    grad = mf.nuc_grad_method()
    assert type(grad).__module__.startswith("kdh.grad")
    assert type(grad).__name__ == "Gradients"


def test_xdh_gradient_refused_naming_nscf():
    mf = RDFDH(FakeMol(), xc="XYG3")

    with pytest.raises(NotImplementedError, match="xc_nscf"):
        mf.nuc_grad_method()


def test_df_default_off_does_not_density_fit(monkeypatch):
    FakeRKS.instances = []
    monkeypatch.setattr("kdh.rdfdh.dft.RKS", FakeRKS)
    monkeypatch.setattr("kdh.rdfdh.mp.MP2", FakeMP2)

    mf = RDFDH(FakeMol(), xc="B2PLYP")
    mf.kernel()

    assert getattr(mf.mf_s, "density_fitted", False) is False


def test_df_flag_density_fits_scf(monkeypatch):
    FakeRKS.instances = []
    monkeypatch.setattr("kdh.rdfdh.dft.RKS", FakeRKS)
    monkeypatch.setattr("kdh.rdfdh.mp.MP2", FakeMP2)

    mf = RDFDH(FakeMol(), xc="B2PLYP", df=True)
    mf.kernel()

    assert mf.mf_s.density_fitted is True
