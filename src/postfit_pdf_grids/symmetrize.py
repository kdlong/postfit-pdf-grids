"""
Symmetrization of asymmetric Hessian PDF uncertainties.

The input columns are the differences member_i - central for members 1..2n, where
members (2k-1, 2k) form pair k and are treated as (down, up).
"""

import numpy as np

METHODS = ("quadratic", "average")


def symmetrize_hessian(diffs, method):
    """
    Symmetrize asymmetric Hessian differences, shape (..., 2n).

    average:   n columns, avg_k = (up_k - down_k) / 2
    quadratic: 2n columns ordered (avg_1, diff_1, avg_2, diff_2, ...),
               with diff_k = sqrt(3) / 2 * (up_k + down_k)
    """
    if method not in METHODS:
        raise ValueError(f"Unknown symmetrization {method}, expected one of {METHODS}")
    if diffs.shape[-1] % 2:
        raise ValueError(
            f"Expected an even number of asymmetric variations, found {diffs.shape[-1]}"
        )

    down = diffs[..., ::2]
    up = diffs[..., 1::2]
    avg = 0.5 * (up - down)

    if method == "average":
        return avg

    symm = np.empty_like(diffs)
    symm[..., ::2] = avg
    symm[..., 1::2] = 0.5 * np.sqrt(3) * (up + down)
    return symm


def n_parameters(n_hessian, symmetric, method=None):
    """Number of fit parameters (covariance dimension) for a Hessian set with n_hessian members."""
    if symmetric:
        return n_hessian
    if method == "average":
        return n_hessian // 2
    if method == "quadratic":
        return n_hessian
    raise ValueError(
        f"Asymmetric Hessian sets need a symmetrization method from {METHODS}, got {method}"
    )
