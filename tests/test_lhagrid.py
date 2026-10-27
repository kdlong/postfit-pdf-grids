import lhapdf
import numpy as np

from postfit_pdf_grids import lhagrid


def knot_coordinates(subgrids):
    """Per-knot (subgrid index, x, Q, pid) in file order."""
    coords = []
    for i, subgrid in enumerate(subgrids):
        x = np.array(subgrid.x_line.split(), dtype=float)
        q = np.array(subgrid.q_line.split(), dtype=float)
        pids = np.array(subgrid.flavour_line.split(), dtype=int)
        grid = np.meshgrid([i], x, q, pids, indexing="ij")
        coords.append(np.stack([g.ravel() for g in grid], axis=-1))
    return np.concatenate(coords)


def test_values_match_lhapdf(asymmetric_set):
    subgrids, values = lhagrid.read_member(
        lhagrid.member_path(lhagrid.set_path(asymmetric_set), 0)
    )
    assert len(subgrids) == 3

    coords = knot_coordinates(subgrids)
    pdf = lhapdf.mkPDF(asymmetric_set, 0)
    evaluated = np.array(
        [pdf.xfxQ(int(pid), x, q) for _, x, q, pid in coords], dtype=float
    )

    # LHAPDF uses the upper subgrid at a boundary, so the stored values at the top
    # edge of the lower subgrids differ for this set, which is discontinuous there
    is_edge = np.zeros(len(coords), dtype=bool)
    for i in range(len(subgrids) - 1):
        in_subgrid = coords[:, 0] == i
        is_edge |= in_subgrid & (coords[:, 2] == coords[in_subgrid, 2].max())

    np.testing.assert_allclose(
        evaluated[~is_edge], values[~is_edge], rtol=1e-10, atol=1e-18
    )
    assert not np.allclose(evaluated[is_edge], values[is_edge], rtol=1e-6)


def test_read_members(asymmetric_set):
    subgrids, values = lhagrid.read_members(asymmetric_set, [0, 1, 2])
    _, member2 = lhagrid.read_member(
        lhagrid.member_path(lhagrid.set_path(asymmetric_set), 2)
    )
    assert values.shape == (sum(s.size for s in subgrids), 3)
    np.testing.assert_array_equal(values[:, 2], member2)


def test_read_members_formatting(symmetric_set):
    """Members of this set write the knots with different exponent formatting."""
    subgrids, values = lhagrid.read_members(symmetric_set, [0, 1])
    assert values.shape == (sum(s.size for s in subgrids), 2)


def test_write_roundtrip(asymmetric_set, tmp_path):
    subgrids, values = lhagrid.read_member(
        lhagrid.member_path(lhagrid.set_path(asymmetric_set), 0)
    )
    path = tmp_path / "member_0000.dat"
    lhagrid.write_member(path, subgrids, values, "central")

    new_subgrids, new_values = lhagrid.read_member(path)
    assert new_subgrids == subgrids
    np.testing.assert_allclose(new_values, values, rtol=1e-8, atol=0)


def test_write_info(asymmetric_set, tmp_path):
    path = tmp_path / "test.info"
    lhagrid.write_info(
        asymmetric_set,
        path,
        {"SetDesc": '"new description"', "NumMembers": "3", "NewKey": "1"},
    )
    lines = path.read_text().splitlines()
    assert 'SetDesc: "new description"' in lines
    assert "NumMembers: 3" in lines
    assert lines[-1] == "NewKey: 1"
    assert sum(line.startswith("SetDesc") for line in lines) == 1
    # Other keys are untouched
    source = open(f"{lhagrid.set_path(asymmetric_set)}.info").read().splitlines()
    assert [l for l in source if l.startswith("Flavors")] == [
        l for l in lines if l.startswith("Flavors")
    ]
