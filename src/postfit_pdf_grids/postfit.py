import logging
import os

import lhapdf
import numpy as np

from postfit_pdf_grids import lhagrid, symmetrize
from postfit_pdf_grids.covariance import read_covariance_file

logger = logging.getLogger(__name__)

# 1 sigma, the confidence level of the postfit covariance
CONF_LEVEL_1SIGMA = 68.26894921370858


class PostfitPdf:
    """
    A Hessian PDF set constrained by a postfit covariance and pulls of its parameters.

    Postfit central = central + M @ pulls
    Postfit members = postfit central + M @ V sqrt(lambda), for C = V diag(lambda) V^T
    where M holds the (scaled, symmetrized) Hessian differences in parameter order.
    """

    def __init__(
        self, pdf_name, covariance, pulls, labels=None, scale=1.0, symmetrization=None
    ):
        error_info = lhapdf.getPDFSet(pdf_name).errorInfo
        if error_info.coreType not in ("hessian", "symmhessian"):
            raise ValueError(
                f"Unsupported PDF error type {error_info.coreType} for {pdf_name}, only Hessian sets are supported"
            )
        self.pdf_name = pdf_name
        self.symmetric = error_info.coreType == "symmhessian"
        self.n_hessian = error_info.nmemCore
        self.scale = scale
        self.symmetrization = None if self.symmetric else symmetrization

        n = symmetrize.n_parameters(self.n_hessian, self.symmetric, symmetrization)
        self.covariance = np.asarray(covariance)
        self.pulls = np.asarray(pulls)
        self.labels = None if labels is None else np.asarray(labels)
        if self.covariance.shape != (n, n) or self.pulls.shape != (n,):
            raise ValueError(
                f"Expected a ({n}, {n}) covariance and {n} pulls for {pdf_name}, found {self.covariance.shape} and {self.pulls.shape}"
            )
        if self.labels is not None and self.labels.shape != (n,):
            raise ValueError(f"Expected {n} labels, found {self.labels.shape}")

    @classmethod
    def from_file(cls, path, symmetrization=None):
        """Load from a covariance file; symmetrization overrides the stored method."""
        content = read_covariance_file(path)
        return cls(
            content["pdf_name"],
            content["covariance"],
            content["pulls"],
            labels=content["labels"],
            scale=content["pdf_scale"],
            symmetrization=symmetrization or content["pdf_symm"],
        )

    def hessian_matrix(self):
        """
        Read the prefit set and return (subgrids, central, M), with M of shape
        (n_knots, n_parameters).
        """
        subgrids, values = lhagrid.read_members(
            self.pdf_name, range(self.n_hessian + 1)
        )
        central = values[:, 0]
        matrix = (values[:, 1:] - central[:, None]) * self.scale
        if not self.symmetric:
            logger.info(f"Applying {self.symmetrization} symmetrization")
            matrix = symmetrize.symmetrize_hessian(matrix, self.symmetrization)
        return subgrids, central, matrix

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
