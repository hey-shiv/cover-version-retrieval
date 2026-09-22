"""Metadata / H5 loading and validation, and selective ZIP member extraction."""

import json
import zipfile

import h5py
import numpy as np
import pytest

from cover_retrieval.data.datacos import (
    DataValidationError,
    feature_path,
    load_hpcp,
    load_metadata,
    load_validated_hpcp,
    validate_hpcp,
    work_index,
)
from cover_retrieval.data.remote_zip import file_range_reader, list_members, read_member


def _write_h5(path, hpcp, wid="W_1", pid="P_1"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("hpcp", data=hpcp)
        handle.attrs["label"] = np.bytes_(wid)
        handle.attrs["track_id"] = np.bytes_(pid)


def test_load_metadata_valid(tmp_path):
    path = tmp_path / "meta.json"
    path.write_text(
        json.dumps(
            {
                "W_2": {
                    "P_9": {"work_id": "W_2", "perf_id": "P_9", "perf_title": "x"},
                    "P_3": {"work_id": "W_2", "perf_id": "P_3"},
                }
            }
        )
    )
    metadata = load_metadata(path)
    assert work_index(metadata) == {"W_2": ["P_3", "P_9"]}


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        json.dumps([]),
        json.dumps({}),
        json.dumps({"X_1": {"P_1": {}}}),
        json.dumps({"W_1": {"Q_1": {}}}),
        json.dumps({"W_1": {"P_1": {"work_id": "W_2"}}}),
        json.dumps({"W_1": {}}),
    ],
)
def test_load_metadata_rejects_malformed(tmp_path, content):
    path = tmp_path / "meta.json"
    path.write_text(content)
    with pytest.raises(DataValidationError):
        load_metadata(path)


def test_work_index_exposes_identifiers_only(tmp_path):
    path = tmp_path / "meta.json"
    path.write_text(json.dumps({"W_1": {"P_1": {"perf_artist": "someone", "release_year": "1999"}}}))
    index = work_index(load_metadata(path))
    assert index == {"W_1": ["P_1"]}


def test_feature_path_layout(tmp_path):
    assert feature_path(tmp_path, "W_1", "P_2") == tmp_path / "W_1_hpcp" / "P_2_hpcp.h5"


def test_load_hpcp_reads_deepdish_layout(tmp_path):
    hpcp = np.random.default_rng(0).random((150, 12)).astype(np.float32)
    path = tmp_path / "x.h5"
    _write_h5(path, hpcp, "W_5", "P_6")
    record = load_hpcp(path)
    np.testing.assert_array_equal(record.hpcp, hpcp)
    assert (record.label, record.track_id) == ("W_5", "P_6")


def test_load_validated_orients_and_checks_labels(tmp_path):
    hpcp = np.random.default_rng(0).random((150, 12)).astype(np.float32)
    path = tmp_path / "x.h5"
    _write_h5(path, hpcp, "W_5", "P_6")
    out, result = load_validated_hpcp(path, "W_5", "P_6", min_frames=96)
    assert out.shape == (12, 150) and result.valid
    with pytest.raises(DataValidationError, match="label"):
        load_validated_hpcp(path, "W_7", "P_6")
    with pytest.raises(DataValidationError, match="track_id"):
        load_validated_hpcp(path, "W_5", "P_8")


def test_missing_and_corrupt_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_hpcp(tmp_path / "absent.h5")
    bad = tmp_path / "bad.h5"
    bad.write_bytes(b"definitely not hdf5")
    with pytest.raises(DataValidationError):
        load_hpcp(bad)


def test_validate_hpcp_rules():
    good = np.random.default_rng(1).random((200, 12))
    assert validate_hpcp(good, min_frames=96).valid
    assert not validate_hpcp(np.zeros((200, 11))).valid  # wrong shape
    assert not validate_hpcp(np.zeros((200, 12))).valid  # silent
    assert not validate_hpcp(good[:50], min_frames=96).valid  # too short
    with_nan = good.copy()
    with_nan[3, 2] = np.nan
    rejected = validate_hpcp(with_nan, nonfinite_policy="reject")
    assert not rejected.valid and rejected.n_nonfinite == 1
    zeroed = validate_hpcp(with_nan, nonfinite_policy="zero")
    assert zeroed.valid and zeroed.n_nonfinite == 1


def test_nonfinite_zero_policy_replaces_values(tmp_path):
    hpcp = np.random.default_rng(0).random((120, 12)).astype(np.float32)
    hpcp[10, 3] = np.inf
    path = tmp_path / "x.h5"
    _write_h5(path, hpcp)
    with pytest.raises(DataValidationError):
        load_validated_hpcp(path, "W_1", "P_1", nonfinite_policy="reject")
    out, result = load_validated_hpcp(path, "W_1", "P_1", nonfinite_policy="zero")
    assert np.isfinite(out).all() and out[3, 10] == 0.0 and result.n_nonfinite == 1


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_remote_zip_member_extraction(tmp_path, compression):
    archive = tmp_path / "a.zip"
    payloads = {
        f"root/W_{i}_hpcp/P_{i}_hpcp.h5": np.random.default_rng(i).bytes(5000 + 37 * i) for i in range(5)
    }
    with zipfile.ZipFile(archive, "w", compression=compression) as z:
        for name, data in payloads.items():
            z.writestr(name, data)
    reader, size = file_range_reader(archive)
    members = list_members(reader, size)
    assert set(members) == set(payloads)
    for name, data in payloads.items():
        assert read_member(reader, members[name]) == data


def test_remote_zip_detects_corruption(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("x.bin", b"a" * 1000)
    reader, size = file_range_reader(archive)
    member = list_members(reader, size)["x.bin"]

    def corrupt(start, length):
        data = bytearray(reader(start, length))
        data[-1] ^= 0xFF
        return bytes(data)

    with pytest.raises(ValueError, match="CRC"):
        read_member(corrupt, member, slack=0)
