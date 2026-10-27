"""
The postfit PDF file: the interface between a fit and grid making. It is
self-describing, so no conventions of the fit framework are needed to read it.

Datasets
  covariance      (n, n)  postfit covariance of the PDF parameters
  pulls           (n,)    postfit values of the PDF parameters
  labels          (n,)    parameter names, for bookkeeping only
  member_weights  (n, m)  template of parameter i is pdf_scale * sum_j W_ij (f_j - f_0),
                          with f_j LHAPDF member j = 1..m of the set and f_0 the central
Attributes
  format_version  FORMAT_VERSION
  pdf_name        LHAPDF name of the prefit set
  pdf_scale       factor applied to all templates (e.g. a prefit uncertainty inflation)
Any further attributes (e.g. provenance) are kept as metadata.
"""

import h5py
import numpy as np

FORMAT_VERSION = 1

_STANDARD_ATTRS = ("format_version", "pdf_name", "pdf_scale")


def write_covariance_file(
    path,
    covariance,
    pulls,
    labels,
    member_weights,
    pdf_name,
    pdf_scale=1.0,
    **metadata,
):
    covariance = np.asarray(covariance)
    pulls = np.asarray(pulls)
    member_weights = np.asarray(member_weights)
    n = len(pulls)
    if (
        covariance.shape != (n, n)
        or len(labels) != n
        or member_weights.ndim != 2
        or member_weights.shape[0] != n
    ):
        raise ValueError(
            f"Inconsistent shapes: covariance {covariance.shape}, pulls {pulls.shape}, labels {len(labels)}, member_weights {member_weights.shape}"
        )

    with h5py.File(path, "w") as f:
        f.create_dataset("covariance", data=covariance)
        f.create_dataset("pulls", data=pulls)
        f.create_dataset(
            "labels", data=np.array(labels, dtype=h5py.special_dtype(vlen=str))
        )
        f.create_dataset("member_weights", data=member_weights)
        f.attrs["format_version"] = FORMAT_VERSION
        f.attrs["pdf_name"] = pdf_name
        f.attrs["pdf_scale"] = pdf_scale
        for key, value in metadata.items():
            f.attrs[key] = value


def read_covariance_file(path):
    """Returns a dict with the datasets and standard attributes, plus a 'metadata' dict."""
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        version = attrs.pop("format_version", None)
        if version != FORMAT_VERSION:
            raise ValueError(
                f"{path} has format version {version}, expected {FORMAT_VERSION}. "
                "Files from before format version 1 have no member weights and need to be re-exported."
            )
        result = {
            "covariance": f["covariance"][:],
            "pulls": f["pulls"][:],
            "labels": f["labels"][:].astype(str),
            "member_weights": f["member_weights"][:],
        }

    result["pdf_name"] = str(attrs.pop("pdf_name"))
    result["pdf_scale"] = float(attrs.pop("pdf_scale"))
    result["metadata"] = attrs
    return result
