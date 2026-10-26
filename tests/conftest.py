import pytest

lhapdf = pytest.importorskip("lhapdf")
lhapdf.setVerbosity(0)

# Asymmetric Hessian, discontinuous at the charm and bottom thresholds (3 subgrids)
ASYMMETRIC_SET = "MSHT20nnlo_as118"
# Symmetric Hessian with alpha_s members
SYMMETRIC_SET = "NNPDF40_nnlo_hessian_pdfas"


def require_set(name):
    if lhapdf.findFile(f"{name}/{name}.info") is None:
        pytest.skip(f"PDF set {name} not installed")
    return name


@pytest.fixture
def asymmetric_set():
    return require_set(ASYMMETRIC_SET)


@pytest.fixture
def symmetric_set():
    return require_set(SYMMETRIC_SET)


@pytest.fixture
def lhapdf_tmp_path(tmp_path):
    """A temporary folder added to the LHAPDF search paths."""
    lhapdf.pathsPrepend(str(tmp_path))
    return tmp_path
