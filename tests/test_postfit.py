import lhapdf
import numpy as np
import pytest

from postfit_pdf_grids import PostfitPdf, read_covariance_file, write_covariance_file
from postfit_pdf_grids.cli import make_grids
from postfit_pdf_grids.evaluate import members_xfxq

FLAVOURS = ("g", "u", "dbar", "c", "uv")
# Includes Q just below the MSHT20 charm (1.4 GeV) and bottom (4.75 GeV) thresholds
X, Q = (
    a.ravel() for a in np.meshgrid([1e-3, 0.01, 0.1, 0.5], [1.39, 2.0, 4.7, 10.0, 91.0])
)


def hessian_diffs(set_name, flavour, x=X, q=Q):
    """Members - central evaluated with LHAPDF, shape (n_hessian, n_points)."""
    values = members_xfxq(set_name, flavour, x, q)
    return values[0], values[1:] - values[0]


@pytest.mark.parametrize("method", ["average", "quadratic"])
def test_unconstrained_asymmetric(asymmetric_set, lhapdf_tmp_path, method):
    """With unit covariance and zero pulls the postfit set reproduces the symmetrized prefit."""
    n = 64 if method == "quadratic" else 32
    postfit = PostfitPdf(asymmetric_set, np.eye(n), np.zeros(n), symmetrization=method)
    set_name = f"test_unconstrained_{method}"
    postfit.write(lhapdf_tmp_path, set_name, 999000)

    new_set = lhapdf.getPDFSet(set_name)
    assert new_set.errorType == "symmhessian"
    assert new_set.size == n + 1

    for flavour in FLAVOURS:
        central, diffs = hessian_diffs(asymmetric_set, flavour)
        down, up = diffs[::2], diffs[1::2]
        expected = np.sum((0.5 * (up - down)) ** 2, axis=0)
        if method == "quadratic":
            expected += np.sum((0.5 * np.sqrt(3) * (up + down)) ** 2, axis=0)

        new_central, new_diffs = hessian_diffs(set_name, flavour)
        np.testing.assert_allclose(new_central, central, rtol=1e-7, atol=1e-12)
        np.testing.assert_allclose(
            np.sum(new_diffs**2, axis=0), expected, rtol=1e-5, atol=1e-14
        )


def test_constrained_symmetric(symmetric_set, lhapdf_tmp_path):
    """Postfit central = central + D pulls and postfit variance = D C D^T."""
    n = lhapdf.getPDFSet(symmetric_set).errorInfo.nmemCore
    rng = np.random.default_rng(1)
    a = rng.normal(size=(n, n)) / np.sqrt(n)
    covariance = a @ a.T
    pulls = rng.normal(scale=0.3, size=n)
    scale = 1.5

    set_name = "test_constrained_symmetric"
    PostfitPdf(symmetric_set, covariance, pulls, scale=scale).write(
        lhapdf_tmp_path, set_name, 999001
    )

    # Extrapolation below the grid is not linear in the knot values
    in_grid = Q >= float(lhapdf.getPDFSet(symmetric_set).get_entry("QMin"))
    x, q = X[in_grid], Q[in_grid]
    for flavour in FLAVOURS:
        central, diffs = hessian_diffs(symmetric_set, flavour, x, q)
        diffs = diffs * scale
        new_central, new_diffs = hessian_diffs(set_name, flavour, x, q)

        np.testing.assert_allclose(
            new_central, central + pulls @ diffs, rtol=1e-6, atol=1e-10
        )
        np.testing.assert_allclose(
            np.sum(new_diffs**2, axis=0),
            np.einsum("ix,ij,jx->x", diffs, covariance, diffs),
            rtol=1e-5,
        )


def test_covariance_file_and_cli(asymmetric_set, lhapdf_tmp_path):
    n = 32
    path = lhapdf_tmp_path / "cov.hdf5"
    labels = [f"pdf{i}" for i in range(1, n + 1)]
    write_covariance_file(
        path,
        0.5 * np.eye(n),
        np.full(n, 0.1),
        labels,
        asymmetric_set,
        pdf_scale=2.0,
        pdf_symm="average",
        command="test",
    )
    content = read_covariance_file(path)
    assert content["pdf_symm"] == "average"
    assert content["pdf_scale"] == 2.0
    assert list(content["labels"]) == labels
    assert content["metadata"] == {"command": "test"}

    make_grids(
        ["-f", str(path), "-o", str(lhapdf_tmp_path), "-i", "999002", "-l", "cli"]
    )
    new_set = lhapdf.getPDFSet(f"{asymmetric_set}_cli_uncx2p0")
    assert new_set.size == n + 1
    assert "Produced by the command" in new_set.description


def test_symmetric_file_without_symmetrization(symmetric_set, tmp_path):
    n = lhapdf.getPDFSet(symmetric_set).errorInfo.nmemCore
    path = tmp_path / "cov.hdf5"
    write_covariance_file(
        path, np.eye(n), np.zeros(n), [f"pdf{i}" for i in range(n)], symmetric_set
    )
    assert read_covariance_file(path)["pdf_symm"] is None
    assert PostfitPdf.from_file(path).symmetrization is None


def test_shape_validation(asymmetric_set):
    with pytest.raises(ValueError):
        PostfitPdf(asymmetric_set, np.eye(64), np.zeros(64), symmetrization="average")
    with pytest.raises(ValueError):
        PostfitPdf(asymmetric_set, np.eye(32), np.zeros(32))
