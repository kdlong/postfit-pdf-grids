# postfit-pdf-grids

Build LHAPDF grids for a Hessian PDF set constrained by a fit. Given the postfit
covariance and pulls of the PDF nuisance parameters, it writes a new symmetric Hessian
LHAPDF set whose central member and uncertainties reflect the fit.

With $D$ the member differences $f_j - f_0$ on the grid knots, $W$ the member weights
defining the fit parameters, $s$ the scale, $p$ the pulls and $C = V \Lambda V^T$ the
covariance, the parameter templates are $M = s\,D W^T$ and

- postfit central: $f_0 + M p$
- postfit members: $f_0 + M p + M V \sqrt{\Lambda}$

## Installation

```bash
pip install -r requirements.txt
```

This installs the package in editable mode with its dependencies and pytest.
LHAPDF with its Python bindings must be installed separately.

## Input: the postfit PDF file

An HDF5 file written by the fit framework (in WRemnants by
`scripts/pdf/write_postfit_pdf_covariance.py`), or with
`postfit_pdf_grids.write_covariance_file`. The file is self-describing: how each
fit parameter maps onto the PDF members is stored in it, so no conventions of the fit
framework are needed here.

| Name | Kind | Content |
|---|---|---|
| `covariance` | dataset `(n, n)` | postfit covariance of the PDF parameters |
| `pulls` | dataset `(n,)` | postfit parameter values |
| `labels` | dataset `(n,)` | parameter names (bookkeeping only) |
| `member_weights` | dataset `(n, m)` | $W$: the template of parameter $i$ is $s \sum_j W_{ij} (f_j - f_0)$ for members $j = 1..m$ |
| `format_version` | attribute | currently 1; files with another version are rejected |
| `pdf_name` | attribute | LHAPDF name of the prefit set |
| `pdf_scale` | attribute | $s$, applied to all templates (e.g. a prefit uncertainty inflation) |

Extra attributes (e.g. provenance) are kept as metadata.

`postfit_pdf_grids.weights` builds $W$ for the common cases. For asymmetric sets
members $(2k-1, 2k)$ are taken as $(\mathrm{down}_k, \mathrm{up}_k)$:

| Case | Parameters | Template |
|---|---|---|
| `symmetric_weights` | $1, \dots, m$ | $f_k - f_0$ |
| `asymmetric_weights(m, "average")` | $\mathrm{avg}_1, \dots, \mathrm{avg}_{m/2}$ | $(\mathrm{up}_k - \mathrm{down}_k)/2$ |
| `asymmetric_weights(m, "quadratic")` | $\mathrm{avg}_1, \mathrm{diff}_1, \mathrm{avg}_2, \dots$ | $\mathrm{diff}_k = \tfrac{\sqrt{3}}{2}(\mathrm{up}_k + \mathrm{down}_k)$ |

## Usage

```bash
make-postfit-pdf-grids -f postfit_cov.hdf5 -o grids/ -i 900000 -l myfit
print-postfit-pdf-covariance postfit_cov.hdf5
```

The set is named `<pdf>_<label>_<scale>` unless `--setName` is given. Point LHAPDF at
the output folder (`LHAPDF_DATA_PATH` or `lhapdf.pathsPrepend`) to use it.

From Python:

```python
from postfit_pdf_grids import PostfitPdf
from postfit_pdf_grids.weights import asymmetric_weights

postfit = PostfitPdf(
    "MSHT20nnlo_as118", covariance, pulls, asymmetric_weights(64, "average")
)
postfit.write("grids/", "MSHT20nnlo_as118_myfit", lhaid=900000)
```

`postfit_pdf_grids.evaluate` has helpers to evaluate PDFs and combinations such as
valence distributions for all members of a set.

## Implementation notes

LHAPDF is used to locate sets, read their metadata and evaluate PDFs. Its Python API
does not expose the grid knots or write grids, so `lhagrid.py` reads and writes the
`lhagrid1` member files directly. Knot values are read from the files rather than by
evaluating the PDF at the knots: at a boundary between subgrids LHAPDF returns the
upper-subgrid value, which is wrong for the lower subgrid when the PDF is
discontinuous at a heavy-quark threshold (e.g. MSHT20 at NNLO).

## Tests

```bash
pytest
```

The tests use `MSHT20nnlo_as118` and `NNPDF40_nnlo_hessian_pdfas` and are skipped if
these sets are not installed.
