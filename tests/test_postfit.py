import h5py
import lhapdf
import numpy as np
import pytest

from postfit_pdf_grids import PostfitPdf, read_covariance_file, write_covariance_file
from postfit_pdf_grids.cli import make_grids
from postfit_pdf_grids.evaluate import members_xfxq
from postfit_pdf_grids.weights import asymmetric_weights, symmetric_weights

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
    postfit = PostfitPdf(
        asymmetric_set, np.eye(n), np.zeros(n), asymmetric_weights(64, method)
    )
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
    PostfitPdf(
        symmetric_set, covariance, pulls, symmetric_weights(n), scale=scale
    ).write(lhapdf_tmp_path, set_name, 999001)

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
        asymmetric_weights(64, "average"),
        asymmetric_set,
        pdf_scale=2.0,
        command="test",
    )
    content = read_covariance_file(path)
    assert content["pdf_scale"] == 2.0
    assert content["member_weights"].shape == (n, 64)
    assert list(content["labels"]) == labels
    assert content["metadata"] == {"command": "test"}

    make_grids(
        ["-f", str(path), "-o", str(lhapdf_tmp_path), "-i", "999002", "-l", "cli"]
    )
    new_set = lhapdf.getPDFSet(f"{asymmetric_set}_cli_uncx2p0")
    assert new_set.size == n + 1
    assert "Produced by the command" in new_set.description


def test_old_format_rejected(tmp_path):
    path = tmp_path / "old.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("covariance", data=np.eye(2))
        f.attrs["pdf_name"] = "test"
    with pytest.raises(ValueError, match="format version"):
        read_covariance_file(path)


def test_shape_validation(asymmetric_set):
    weights = asymmetric_weights(64, "average")
    with pytest.raises(ValueError):
        PostfitPdf(asymmetric_set, np.eye(64), np.zeros(64), weights)
    with pytest.raises(ValueError, match="has 64"):
        PostfitPdf(asymmetric_set, np.eye(32), np.zeros(32), np.zeros((32, 70)))
