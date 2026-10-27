import logging
import os

import lhapdf
import numpy as np

from postfit_pdf_grids import lhagrid
from postfit_pdf_grids.covariance import read_covariance_file

logger = logging.getLogger(__name__)

# 1 sigma, the confidence level of the postfit covariance
CONF_LEVEL_1SIGMA = 68.26894921370858


class PostfitPdf:
    """
    A PDF set constrained by a postfit covariance and pulls of parameters defined
    as linear combinations of its Hessian members.

    M = scale * D @ W^T, with D the member differences f_j - f_0 on the grid knots
    Postfit central = central + M @ pulls
    Postfit members = postfit central + M @ V sqrt(lambda), for C = V diag(lambda) V^T
    """

    def __init__(
        self, pdf_name, covariance, pulls, member_weights, labels=None, scale=1.0
    ):
        self.pdf_name = pdf_name
        self.scale = scale
        self.covariance = np.asarray(covariance)
        self.pulls = np.asarray(pulls)
        self.member_weights = np.asarray(member_weights)
        self.labels = None if labels is None else np.asarray(labels)

        n = len(self.pulls)
        if (
            self.covariance.shape != (n, n)
            or self.member_weights.ndim != 2
            or self.member_weights.shape[0] != n
        ):
            raise ValueError(
                f"Inconsistent shapes: covariance {self.covariance.shape}, pulls {self.pulls.shape}, member_weights {self.member_weights.shape}"
            )
        if self.labels is not None and self.labels.shape != (n,):
            raise ValueError(f"Expected {n} labels, found {self.labels.shape}")

        pdf_set = lhapdf.getPDFSet(pdf_name)
        n_members = self.member_weights.shape[1]
        if n_members > pdf_set.size - 1:
            raise ValueError(
                f"member_weights use {n_members} members but {pdf_name} has {pdf_set.size - 1}"
            )
        if n_members != pdf_set.errorInfo.nmemCore:
            logger.warning(
                f"member_weights use {n_members} members, {pdf_name} has {pdf_set.errorInfo.nmemCore} core error members"
            )

    @classmethod
    def from_file(cls, path):
        content = read_covariance_file(path)
        return cls(
            content["pdf_name"],
            content["covariance"],
            content["pulls"],
            content["member_weights"],
            labels=content["labels"],
            scale=content["pdf_scale"],
        )

    def hessian_matrix(self):
        """
        Read the prefit set and return (subgrids, central, M), with M of shape
        (n_knots, n_parameters).
        """
        n_members = self.member_weights.shape[1]
        subgrids, values = lhagrid.read_members(self.pdf_name, range(n_members + 1))
        central = values[:, 0]
        diffs = values[:, 1:] - central[:, None]
        return subgrids, central, self.scale * diffs @ self.member_weights.T

    def postfit_members(self):
        """Returns (subgrids, postfit central, postfit Hessian members of shape (n_knots, n_parameters))."""
        subgrids, central, matrix = self.hessian_matrix()
        eigvals, eigvecs = np.linalg.eigh(self.covariance)
        if np.any(eigvals < 0):
            logger.warning(
                f"Clipping {np.count_nonzero(eigvals < 0)} negative covariance eigenvalues to 0 (most negative {eigvals.min():.3g})"
            )
        rotation = eigvecs * np.sqrt(np.maximum(eigvals, 0))

        new_central = central + matrix @ self.pulls
        members = new_central[:, None] + matrix @ rotation
        return subgrids, new_central, members

    def write(self, outfolder, set_name, lhaid, description=None):
        """Write the postfit set as a symmetric Hessian LHAPDF set in outfolder/set_name."""
        subgrids, central, members = self.postfit_members()

        outdir = os.path.join(outfolder, set_name)
        os.makedirs(outdir, exist_ok=True)
        prefix = os.path.join(outdir, set_name)

        if description is None:
            description = f"{self.pdf_name} constrained by a postfit covariance, prefit uncertainties scaled by {self.scale}"
        lhagrid.write_info(
            self.pdf_name,
            f"{prefix}.info",
            {
                "SetDesc": '"{}"'.format(description.replace('"', "'")),
                "SetIndex": str(lhaid),
                "NumMembers": str(members.shape[-1] + 1),
                "ErrorType": "symmhessian",
                "ErrorConfLevel": str(CONF_LEVEL_1SIGMA),
            },
        )
        lhagrid.write_member(
            lhagrid.member_path(prefix, 0), subgrids, central, "central"
        )
        for i in range(members.shape[-1]):
            lhagrid.write_member(
                lhagrid.member_path(prefix, i + 1), subgrids, members[:, i], "error"
            )
        logger.info(f"Wrote {members.shape[-1] + 1} members to {outdir}")
        return outdir
