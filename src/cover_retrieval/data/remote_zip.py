"""Selective extraction of members from a remote ZIP archive via HTTP Range requests.

Da-TACOS feature archives are single multi-gigabyte ZIP files on Zenodo. A ZIP's
central directory sits at the end of the archive, so the list of members and their
byte offsets can be read with a few small range requests, after which any member
can be fetched on its own. On slow links this lets the project fetch exactly the
recordings a manifest needs instead of the whole archive.

The byte source is abstracted (``RangeReader``) so the logic is unit-testable
against local files.
"""

from __future__ import annotations

import http.client
import io
import struct
import time
import urllib.request
import zipfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

RangeReader = Callable[[int, int], bytes]
"""``reader(start, length) -> bytes`` returning exactly ``length`` bytes."""

_LOCAL_HEADER = struct.Struct("<4sHHHHHIIIHH")
_LOCAL_SIGNATURE = b"PK\x03\x04"


class _RangeFile(io.RawIOBase):
    """Minimal seekable read-only file object backed by a ``RangeReader``."""

    def __init__(self, reader: RangeReader, size: int) -> None:
        self._reader = reader
        self._size = size
        self._pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._size - self._pos
        size = max(0, min(size, self._size - self._pos))
        if size == 0:
            return b""
        data = self._reader(self._pos, size)
        self._pos += len(data)
        return data

    def readinto(self, buffer) -> int:  # type: ignore[no-untyped-def]
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


@dataclass(frozen=True)
class MemberInfo:
    name: str
    header_offset: int
    compress_size: int
    file_size: int
    compress_type: int
    crc: int


def list_members(reader: RangeReader, size: int) -> dict[str, MemberInfo]:
    """Read the ZIP central directory (ZIP64 aware) and index members by name."""
    with zipfile.ZipFile(io.BufferedReader(_RangeFile(reader, size), buffer_size=1 << 20)) as archive:
        return {
            info.filename: MemberInfo(
                info.filename,
                info.header_offset,
                info.compress_size,
                info.file_size,
                info.compress_type,
                info.CRC,
            )
            for info in archive.infolist()
            if not info.is_dir()
        }


def read_member(reader: RangeReader, member: MemberInfo, slack: int = 1024) -> bytes:
    """Fetch and decompress one member using a single range request."""
    blob = reader(member.header_offset, _LOCAL_HEADER.size + slack + member.compress_size)
    fields = _LOCAL_HEADER.unpack_from(blob)
    if fields[0] != _LOCAL_SIGNATURE:
        raise ValueError(f"bad local header signature for {member.name}")
    name_len, extra_len = fields[9], fields[10]
    start = _LOCAL_HEADER.size + name_len + extra_len
    if start + member.compress_size > len(blob):  # unusually large local extra field
        blob = reader(member.header_offset, start + member.compress_size)
    payload = blob[start : start + member.compress_size]
    if member.compress_type == zipfile.ZIP_STORED:
        data = payload
    elif member.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(payload, -15)
    else:
        raise ValueError(f"unsupported compression {member.compress_type} for {member.name}")
    if len(data) != member.file_size or zlib.crc32(data) != member.crc:
        raise ValueError(f"size/CRC mismatch for {member.name}")
    return data


def http_range_reader(url: str, retries: int = 30, timeout: float = 120.0) -> tuple[RangeReader, int]:
    """Create a ``RangeReader`` for ``url`` and return it with the remote size."""
    head = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(head, timeout=timeout) as response:
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or "/" not in content_range:
            raise RuntimeError(f"server does not support range requests for {url}")
        size = int(content_range.rsplit("/", 1)[1])

    def reader(start: int, length: int) -> bytes:
        end = min(start + length, size) - 1
        last_error = "short read"
        for attempt in range(1, retries + 1):
            try:
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    data = response.read()
                if len(data) == end - start + 1:
                    return data
                last_error = f"short read ({len(data)} bytes)"
            except (OSError, TimeoutError, http.client.HTTPException) as error:
                last_error = repr(error)
            time.sleep(min(60.0, 2.0 ** min(attempt, 6)))
        raise RuntimeError(f"range request {start}-{end} failed after {retries} attempts: {last_error}")

    return reader, size


def file_range_reader(path: Path) -> tuple[RangeReader, int]:
    """``RangeReader`` over a local file (used by tests and for local archives)."""

    def reader(start: int, length: int) -> bytes:
        with path.open("rb") as handle:
            handle.seek(start)
            return handle.read(length)

    return reader, path.stat().st_size
