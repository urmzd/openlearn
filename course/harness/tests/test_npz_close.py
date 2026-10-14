"""Milestone numeric archives must preserve geometry, values, and key identity."""

import struct
import zipfile
from types import SimpleNamespace

import pytest

from olcourse import HarnessError
from olcourse.matchers import m_npz_close


def archive(path, shape, values, key="mel"):
    header = repr({"descr": "<f8", "fortran_order": False, "shape": shape}).encode()
    raw = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header
    raw += struct.pack(f"<{len(values)}d", *values)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(key + ".npy", raw)


def test_numeric_archive_geometry_and_errors(tmp_path):
    ref, got = tmp_path / "ref.npz", tmp_path / "got.npz"
    archive(ref, (2, 2), [1, 2, 3, 4])
    mc = SimpleNamespace(
        run=SimpleNamespace(course_file=lambda _: ref, learner=tmp_path),
        lookup=lambda _: "",
        step_name="mel",
    )
    expect = {"file": str(ref), "output": str(got), "atol": 0.01}
    archive(got, (2, 2), [1.005, 2, 3, 4])
    assert m_npz_close(expect, None, mc).status == "pass"
    for shape, values, key in [
        ((4,), [1, 2, 3, 4], "mel"),
        ((2, 2), [1.1, 2, 3, 4], "mel"),
        ((2, 2), [float("nan"), 2, 3, 4], "mel"),
        ((2, 2), [1, 2, 3, 4], "wrong"),
    ]:
        archive(got, shape, values, key)
        assert m_npz_close(expect, None, mc).status == "fail"
    got.write_bytes(b"not an archive")
    assert m_npz_close(expect, None, mc).status == "fail"
    with pytest.raises(HarnessError, match="tolerances"):
        m_npz_close({**expect, "atol": -1}, None, mc)
