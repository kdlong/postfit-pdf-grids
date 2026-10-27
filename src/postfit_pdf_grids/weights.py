"""
Member weights for common ways of turning Hessian members into fit parameters.

A fit framework writing the postfit file should build the weights matching its own
templates; these cover the usual cases. For asymmetric Hessian sets, members
(2k-1, 2k) are treated as (down_k, up_k).
"""

import numpy as np

SYMMETRIZATIONS = ("average", "quadratic")


def symmetric_weights(n_members):
    """Parameter k has template f_k - f_0."""
    return np.eye(n_members)


def asymmetric_weights(n_members, symmetrization):
    """
    average:   n_members / 2 parameters avg_k = (up_k - down_k) / 2
    quadratic: n_members parameters ordered (avg_1, diff_1, avg_2, diff_2, ...),
               with diff_k = sqrt(3) / 2 (up_k + down_k)
    """
    if symmetrization not in SYMMETRIZATIONS:
        raise ValueError(
            f"Unknown symmetrization {symmetrization}, expected one of {SYMMETRIZATIONS}"
        )
    if n_members % 2:
        raise ValueError(
            f"Expected an even number of asymmetric members, found {n_members}"
        )

    n_pairs = n_members // 2
    pairs = np.arange(n_pairs)
    avg = np.zeros((n_pairs, n_members))
    avg[pairs, 2 * pairs] = -0.5
    avg[pairs, 2 * pairs + 1] = 0.5
    if symmetrization == "average":
        return avg

    diff = np.zeros((n_pairs, n_members))
    diff[pairs, 2 * pairs] = 0.5 * np.sqrt(3)
    diff[pairs, 2 * pairs + 1] = 0.5 * np.sqrt(3)
    weights = np.empty((n_members, n_members))
    weights[::2] = avg
    weights[1::2] = diff
    return weights
