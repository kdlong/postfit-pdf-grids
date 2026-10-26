"""
The postfit covariance file: the interface between a fit and grid making.

Datasets
  covariance  (n, n)  postfit covariance of the PDF parameters
  pulls       (n,)    postfit values of the PDF parameters
  labels      (n,)    parameter names, for bookkeeping only
Attributes
  pdf_name    LHAPDF name of the prefit set
  pdf_scale   factor the prefit Hessian uncertainties were scaled by in the fit
  pdf_symm    symmetrization method for asymmetric sets ("quadratic" or "average"),
              absent for symmetric sets
Any further attributes (e.g. provenance) are kept as metadata.

The parameters must be in eigenvector order: (1, ..., n) for symmetric sets and for
"average", and (avg_1, diff_1, avg_2, diff_2, ...) for "quadratic".
"""

import h5py
import numpy as np

_STANDARD_ATTRS = ("pdf_name", "pdf_scale", "pdf_symm")


def write_covariance_file(
    path, covariance, pulls, labels, pdf_name, pdf_scale=1.0, pdf_symm=None, **metadata
):
    with h5py.File(path, "w") as f:
        f.create_dataset("covariance", data=covariance)
        f.create_dataset("pulls", data=pulls)
        f.create_dataset(
            "labels", data=np.array(labels, dtype=h5py.special_dtype(vlen=str))
        )
        f.attrs["pdf_name"] = pdf_name
        f.attrs["pdf_scale"] = pdf_scale
        if pdf_symm is not None:
            f.attrs["pdf_symm"] = pdf_symm
        for key, value in metadata.items():
            f.attrs[key] = value


def read_covariance_file(path):
    """Returns a dict with the datasets and standard attributes, plus a 'metadata' dict."""
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        result = {
            "covariance": f["covariance"][:],
            "pulls": f["pulls"][:],
            "labels": f["labels"][:].astype(str),
        }

    # Older files store an empty string for symmetric sets
    pdf_symm = attrs.pop("pdf_symm", None)
    result["pdf_symm"] = str(pdf_symm) if pdf_symm else None
    result["pdf_name"] = str(attrs.pop("pdf_name"))
    result["pdf_scale"] = float(attrs.pop("pdf_scale", 1.0))
    result["metadata"] = attrs
    return result
