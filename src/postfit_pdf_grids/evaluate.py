"""Evaluation of PDFs and parton combinations with LHAPDF."""

import lhapdf
import numpy as np

PIDS = {
    "g": 21,
    "d": 1,
    "dbar": -1,
    "u": 2,
    "ubar": -2,
    "s": 3,
    "sbar": -3,
    "c": 4,
    "cbar": -4,
    "b": 5,
    "bbar": -5,
}

COMBINATIONS = ("uv", "dv", "rs")


def xfxq(pdf, flavour, x, q):
    """
    x*f(x, Q) for an LHAPDF PDF member, at arrays x and Q (broadcast together).
    flavour is a PDG ID, a name in PIDS, or one of the combinations
    uv = u - ubar, dv = d - dbar, rs = (s + sbar) / (ubar + dbar).
    """
    x, q = np.broadcast_arrays(np.atleast_1d(x), np.atleast_1d(q))

    def pid_values(pid):
        return np.array(pdf.xfxQ(pid, x.ravel(), q.ravel())).reshape(x.shape)

    if flavour == "uv":
        return pid_values(2) - pid_values(-2)
    if flavour == "dv":
        return pid_values(1) - pid_values(-1)
    if flavour == "rs":
        num = pid_values(3) + pid_values(-3)
        denom = pid_values(-1) + pid_values(-2)
        return np.divide(num, denom, out=np.zeros_like(num), where=denom != 0)
    return pid_values(PIDS.get(flavour, flavour))


def members_xfxq(set_name, flavour, x, q):
    """xfxq for the central and Hessian members of a set (alpha_s members excluded), shape (n_members, ...)."""
    pdf_set = lhapdf.getPDFSet(set_name)
    members = pdf_set.mkPDFs()[: pdf_set.errorInfo.nmemCore + 1]
    return np.stack([xfxq(m, flavour, x, q) for m in members])
