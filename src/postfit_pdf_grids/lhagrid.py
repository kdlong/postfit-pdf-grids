"""
Reading and writing LHAPDF ``lhagrid1`` member files.

LHAPDF's Python API exposes neither the interpolation knots nor a writer, so this
module handles the file format itself. Everything else (locating sets, metadata,
evaluation) goes through LHAPDF.

Grid values are read from the member files rather than by evaluating the PDF at the
knots: at a boundary between subgrids, LHAPDF returns the value of the upper subgrid,
which differs from the stored lower-subgrid value for PDFs that are discontinuous at
heavy-quark thresholds.
"""

import re
from dataclasses import dataclass

import lhapdf
import numpy as np

SEPARATOR = "---"


@dataclass(frozen=True)
class Subgrid:
    """Knot layout of one subgrid, with the knot lines kept verbatim from the source file."""

    x_line: str
    q_line: str
    flavour_line: str

    @property
    def shape(self):
        """(n_x, n_Q, n_flavours); values are stored with x outermost."""
        return tuple(
            len(line.split()) for line in (self.x_line, self.q_line, self.flavour_line)
        )

    @property
    def size(self):
        return int(np.prod(self.shape))

    def same_knots(self, other):
        """Same knots as other, regardless of number formatting in the files."""
        return all(
            np.array_equal(np.array(a.split(), float), np.array(b.split(), float))
            for a, b in zip(
                (self.x_line, self.q_line, self.flavour_line),
                (other.x_line, other.q_line, other.flavour_line),
            )
        )


def set_path(set_name):
    """Path prefix of an installed set, e.g. ``/path/to/NAME/NAME``."""
    info = lhapdf.findFile(f"{set_name}/{set_name}.info")
    if info is None:
        raise FileNotFoundError(f"PDF set {set_name} not found in the LHAPDF paths")
    return info[: -len(".info")]


def member_path(prefix, member):
    return f"{prefix}_{member:04d}.dat"


def read_member(path):
    """
    Read one member file.
    Returns (subgrids, values) with values flattened in file order.
    """
    with open(path) as f:
        # The header before the first separator is not needed
        blocks = _split_blocks(f.read())[1:]

    subgrids = []
    values = []
    for block in blocks:
        lines = block.strip("\n").splitlines()
        if not lines:
            continue
        subgrid = Subgrid(*(line.strip() for line in lines[:3]))
        vals = np.array(" ".join(lines[3:]).split(), dtype=float)
        if vals.size != subgrid.size:
            raise ValueError(
                f"Subgrid {len(subgrids)} of {path} has {vals.size} values, expected {subgrid.size}"
            )
        subgrids.append(subgrid)
        values.append(vals)
    return tuple(subgrids), np.concatenate(values)


def read_members(set_name, members):
    """
    Read several members of an installed set.
    Returns (subgrids, values) with values of shape (n_knots, len(members)).
    """
    prefix = set_path(set_name)
    subgrids = None
    columns = []
    for member in members:
        member_subgrids, vals = read_member(member_path(prefix, member))
        if subgrids is None:
            subgrids = member_subgrids
        elif len(member_subgrids) != len(subgrids) or not all(
            a.same_knots(b) for a, b in zip(member_subgrids, subgrids)
        ):
            raise ValueError(
                f"Member {member} of {set_name} has a different grid layout from member {members[0]}"
            )
        columns.append(vals)
    return subgrids, np.stack(columns, axis=-1)


def write_member(path, subgrids, values, pdf_type):
    """Write one member file; values are flattened in the order returned by read_member."""
    if values.size != sum(s.size for s in subgrids):
        raise ValueError("Number of values does not match the subgrid layout")

    with open(path, "w") as f:
        f.write(f"PdfType: {pdf_type}\nFormat: lhagrid1\n{SEPARATOR}\n")
        offset = 0
        for subgrid in subgrids:
            n_x, n_q, n_flavours = subgrid.shape
            block = values[offset : offset + subgrid.size].reshape(
                n_x * n_q, n_flavours
            )
            offset += subgrid.size
            f.write(f"{subgrid.x_line}\n{subgrid.q_line}\n{subgrid.flavour_line}\n")
            np.savetxt(f, block, fmt="%.8e")
            f.write(f"{SEPARATOR}\n")


def write_info(source_set, path, updates):
    """
    Write a .info file copied from source_set with the top-level keys in updates
    replaced (or appended if absent). Values in updates are written verbatim as YAML.
    """
    with open(f"{set_path(source_set)}.info") as f:
        lines = f.read().splitlines()

    remaining = dict(updates)
    out = []
    skipping = False
    for line in lines:
        match = re.match(r"^(\w+)\s*:", line)
        if match:
            skipping = match.group(1) in remaining
            if skipping:
                key = match.group(1)
                out.append(f"{key}: {remaining.pop(key)}")
                continue
        elif skipping:
            # Continuation line of a replaced multi-line value
            continue
        out.append(line)
    out.extend(f"{key}: {value}" for key, value in remaining.items())

    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def _split_blocks(text):
    return re.split(rf"^\s*{SEPARATOR}\s*$", text, flags=re.MULTILINE)
