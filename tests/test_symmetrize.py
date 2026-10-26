import numpy as np
import pytest

from postfit_pdf_grids.symmetrize import n_parameters, symmetrize_hessian

DOWN = np.array([[-1.0, -2.0], [-3.0, 0.5]])
UP = np.array([[2.0, 1.0], [1.0, 1.5]])


def interleave(down, up):
    diffs = np.empty((down.shape[0], 2 * down.shape[1]))
    diffs[:, ::2] = down
    diffs[:, 1::2] = up
    return diffs


def test_average():
    symm = symmetrize_hessian(interleave(DOWN, UP), "average")
    np.testing.assert_allclose(symm, 0.5 * (UP - DOWN))


def test_quadratic():
    symm = symmetrize_hessian(interleave(DOWN, UP), "quadratic")
    np.testing.assert_allclose(symm[:, ::2], 0.5 * (UP - DOWN))
    np.testing.assert_allclose(symm[:, 1::2], 0.5 * np.sqrt(3) * (UP + DOWN))


def test_input_not_modified():
    diffs = interleave(DOWN, UP)
    original = diffs.copy()
    symmetrize_hessian(diffs, "quadratic")
    np.testing.assert_array_equal(diffs, original)


def test_errors():
    with pytest.raises(ValueError):
        symmetrize_hessian(np.zeros((2, 3)), "average")
    with pytest.raises(ValueError):
        symmetrize_hessian(np.zeros((2, 4)), "other")


def test_n_parameters():
    assert n_parameters(50, symmetric=True) == 50
    assert n_parameters(64, symmetric=False, method="average") == 32
    assert n_parameters(64, symmetric=False, method="quadratic") == 64
    with pytest.raises(ValueError):
        n_parameters(64, symmetric=False)
