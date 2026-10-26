# postfit-pdf-grids

Build LHAPDF grids for a Hessian PDF set constrained by a fit. Given the postfit
covariance and pulls of the PDF nuisance parameters, it writes a new symmetric Hessian
LHAPDF set whose central member and uncertainties reflect the fit.

With $M$ the matrix of (scaled, symmetrized) Hessian differences $f_i - f_0$ on the
grid knots, $p$ the pulls and $C = V \Lambda V^T$ the covariance:

- postfit central: $f_0 + M p$
- postfit members: $f_0 + M p + M V \sqrt{\Lambda}$

## Installation

```bash
pip install -e .
```

LHAPDF with its Python bindings must be installed separately.

## Input: the postfit covariance file

An HDF5 file, written with `postfit_pdf_grids.write_covariance_file`:

| Name | Kind | Content |
|---|---|---|
| `covariance` | dataset `(n, n)` | postfit covariance of the PDF parameters |
| `pulls` | dataset `(n,)` | postfit parameter values |
| `labels` | dataset `(n,)` | parameter names (bookkeeping only) |
| `pdf_name` | attribute | LHAPDF name of the prefit set |
| `pdf_scale` | attribute | factor the prefit Hessian uncertainties were scaled by in the fit |
| `pdf_symm` | attribute | `quadratic` or `average` for asymmetric sets; absent for symmetric sets |

Extra attributes (e.g. provenance) are kept as metadata.

**Parameter order.** The parameters must follow the Hessian eigenvector order of the
set. Asymmetric members $(2k-1, 2k)$ are treated as $(\mathrm{down}_k, \mathrm{up}_k)$
and symmetrized as:

| Set / method | Parameters | Definition |
|---|---|---|
| symmetric Hessian | $1, \dots, n$ | $f_k - f_0$ |
| `average` | $\mathrm{avg}_1, \dots, \mathrm{avg}_n$ | $(\mathrm{up}_k - \mathrm{down}_k)/2$ |
| `quadratic` | $\mathrm{avg}_1, \mathrm{diff}_1, \mathrm{avg}_2, \dots$ | $\mathrm{diff}_k = \tfrac{\sqrt{3}}{2}(\mathrm{up}_k + \mathrm{down}_k)$ |

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

postfit = PostfitPdf(
    "MSHT20nnlo_as118", covariance, pulls, scale=1.0, symmetrization="average"
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
