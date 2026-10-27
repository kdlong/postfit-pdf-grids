import numpy as np
import pytest

from postfit_pdf_grids.weights import asymmetric_weights, symmetric_weights

DOWN = np.array([[-1.0, -2.0], [-3.0, 0.5]])
UP = np.array([[2.0, 1.0], [1.0, 1.5]])
# Member differences (points, members), members ordered (down_1, up_1, down_2, up_2)
DIFFS = np.stack([DOWN[:, 0], UP[:, 0], DOWN[:, 1], UP[:, 1]], axis=-1)


def test_symmetric():
    np.testing.assert_array_equal(DIFFS @ symmetric_weights(4).T, DIFFS)


def test_average():
    templates = DIFFS @ asymmetric_weights(4, "average").T
    np.testing.assert_allclose(templates, 0.5 * (UP - DOWN))


def test_quadratic():
    templates = DIFFS @ asymmetric_weights(4, "quadratic").T
    np.testing.assert_allclose(templates[:, ::2], 0.5 * (UP - DOWN))
    np.testing.assert_allclose(templates[:, 1::2], 0.5 * np.sqrt(3) * (UP + DOWN))


def test_errors():
    with pytest.raises(ValueError):
        asymmetric_weights(3, "average")
    with pytest.raises(ValueError):
        asymmetric_weights(4, "other")
