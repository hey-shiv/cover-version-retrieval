"""Download Da-TACOS metadata and HPCP features into data/external/da-tacos/.

Source: Zenodo record 10.5281/zenodo.4717628 (Da-TACOS v1.1.0). The Google Drive
links in the MTG/da-tacos README returned HTTP 404 when this project was built.
Metadata and features are CC BY-NC-SA 4.0 and git-ignored; never commit them.

Two modes:

1. Whole archives (resumable, MD5-verified, unpacked in place)::

       python scripts/download_datacos.py --what metadata
       python scripts/download_datacos.py --what coveranalysis_hpcp benchmark_hpcp

2. Selective member fetch via HTTP Range requests (for slow links). Reads the ZIP
   central directory remotely and fetches only the needed H5 files, CRC-checked,
   into exactly the layout the full archive would unpack to::

       # Cover Analysis: first WIDs of the seeded split order (all roles in the config)
       python scripts/download_datacos.py --fetch-members coveranalysis --config configs/base.yaml
       # Benchmark: every recording (all 15,000 files)
       python scripts/download_datacos.py --fetch-members benchmark --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cover_retrieval.data.datacos import (
    archive_member_name,
    feature_path,
    load_metadata,
    metadata_path,
    work_index,
)
from cover_retrieval.data.remote_zip import http_range_reader, list_members, read_member
from cover_retrieval.data.splits import role_sizes, seeded_wid_order
from cover_retrieval.utils.io import load_config

ZENODO_RECORD = "4717628"
_BASE = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files"
ARCHIVES: dict[str, tuple[str, int, str]] = {
    # key: (filename, size in bytes, md5) — from the Zenodo record API
    "metadata": ("da-tacos_metadata.zip", 3_513_878, "b8aed83c45687a6bac76de3da1799237"),
    "coveranalysis_hpcp": (
        "da-tacos_coveranalysis_subset_hpcp.zip",
        7_081_646_362,
        "961784fc2419214adf05504e9fc56cc2",
    ),
    "benchmark_hpcp": (
        "da-tacos_benchmark_subset_hpcp.zip",
        10_088_842_259,
        "f92cf3d00cc3195572381d6bbcc086de",
    ),
}
MIN_FREE_BYTES = 35 * 1024**3


def archive_url(filename: str) -> str:
    return f"{_BASE}/{filename}/content"


def md5sum(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------ whole archives
def _fetch(url: str, dest: Path, expected_size: int, retries: int = 50) -> None:
    """Stream ``url`` into ``dest``, resuming from a partial file when possible."""
    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if have == expected_size:
            return
        request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                mode = "ab" if have and response.status == 206 else "wb"
                done = have if mode == "ab" else 0
                last = time.monotonic()
                with dest.open(mode) as handle:
                    while block := response.read(1 << 20):
                        handle.write(block)
                        done += len(block)
                        if time.monotonic() - last > 60:
                            print(f"  {dest.name}: {done / 1e9:.2f}/{expected_size / 1e9:.2f} GB", flush=True)
                            last = time.monotonic()
        except Exception as error:  # noqa: BLE001 - any network failure: resume
            print(f"  attempt {attempt} interrupted ({error!r}); resuming", flush=True)
            time.sleep(min(60, 5 * attempt))
    if not dest.exists() or dest.stat().st_size != expected_size:
        raise RuntimeError(f"could not download {url} after {retries} attempts")


def download_archive(key: str, outdir: Path, keep_zip: bool) -> dict[str, object]:
    filename, size, md5 = ARCHIVES[key]
    marker = outdir / f".{key}.done"
    if marker.exists():
        print(f"[skip] {key}: already downloaded, verified and unpacked")
        return json.loads(marker.read_text())
    zip_path = outdir / filename
    print(f"[download] {key}: {size / 1e9:.2f} GB -> {zip_path}", flush=True)
    _fetch(archive_url(filename), zip_path, size)
    actual = md5sum(zip_path)
    if actual != md5:
        zip_path.unlink()
        raise RuntimeError(f"MD5 mismatch for {filename}: expected {md5}, got {actual}")
    print(f"[unpack] {filename}", flush=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(outdir)
        n_members = len(archive.namelist())
    if not keep_zip:
        zip_path.unlink()
    info: dict[str, object] = {
        "key": key,
        "mode": "archive",
        "source": f"https://doi.org/10.5281/zenodo.{ZENODO_RECORD}",
        "filename": filename,
        "zip_bytes": size,
        "md5": md5,
        "n_members": n_members,
    }
    marker.write_text(json.dumps(info, indent=2))
    return info


# ------------------------------------------------------------------ selective fetch
def wanted_pids(config: dict, subset: str, n_wids: int | None) -> list[tuple[str, str]]:
    """(WID, PID) pairs to fetch, in priority order."""
    works = work_index(load_metadata(metadata_path(config, subset)))  # type: ignore[arg-type]
    if subset == "benchmark":
        order = sorted(works)
    else:
        order = seeded_wid_order(list(works), int(config["splits"]["seed"]))
        if n_wids is None:
            sizes = role_sizes(config["splits"])
            if any(v is None for v in sizes.values()):
                n_wids = len(order)
            else:
                total = sum(v for v in sizes.values() if v is not None)
                n_wids = min(len(order), int(total * 1.03) + 10)  # margin for invalid WIDs
        order = order[:n_wids]
    return [(wid, pid) for wid in order for pid in works[wid]]


def fetch_members(
    config: dict, subset: str, outdir: Path, n_wids: int | None, threads: int
) -> dict[str, object]:
    filename = ARCHIVES[f"{subset}_hpcp"][0]
    features_dir = config["data"][f"{subset}_features_dir"]
    target_root = outdir / features_dir
    reader, size = http_range_reader(archive_url(filename))

    index_path = outdir / f".{subset}_hpcp_members.json"
    if index_path.exists():
        raw = json.loads(index_path.read_text())
    else:
        print(f"[index] reading central directory of {filename}", flush=True)
        raw = {name: vars(info) for name, info in list_members(reader, size).items()}
        index_path.write_text(json.dumps(raw))
    from cover_retrieval.data.remote_zip import MemberInfo

    members = {name: MemberInfo(**info) for name, info in raw.items()}
    pairs = wanted_pids(config, subset, n_wids)
    todo = []
    for wid, pid in pairs:
        dest = feature_path(target_root, wid, pid)
        member = members.get(archive_member_name(features_dir, wid, pid))
        if member is None:
            print(f"[missing] {wid}/{pid} not in archive", flush=True)
            continue
        if dest.exists() and dest.stat().st_size == member.file_size:
            continue
        todo.append((dest, member))
    print(f"[fetch] {subset}: {len(pairs)} wanted, {len(todo)} to download", flush=True)

    start = time.monotonic()
    done_bytes = 0

    def work(item: tuple[Path, MemberInfo]) -> tuple[int, str | None]:
        dest, member = item
        try:
            data = read_member(reader, member)
        except Exception as error:  # noqa: BLE001 - record and retry in a later pass
            return 0, f"{member.name}: {error}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
        return member.compress_size, None

    failures: list[str] = []
    for attempt in range(1, 4):
        failures = []
        retry = []
        with ThreadPoolExecutor(threads) as pool:
            for i, (item, (nbytes, error)) in enumerate(zip(todo, pool.map(work, todo), strict=True), 1):
                done_bytes += nbytes
                if error:
                    failures.append(error)
                    retry.append(item)
                if i % 100 == 0 or i == len(todo):
                    rate = done_bytes / max(time.monotonic() - start, 1e-9) / 1e6
                    print(
                        f"  {i}/{len(todo)} files, {done_bytes / 1e9:.2f} GB, {rate:.2f} MB/s, "
                        f"{len(failures)} failed",
                        flush=True,
                    )
        if not retry:
            break
        print(f"[retry] pass {attempt}: {len(retry)} failed files", flush=True)
        todo = retry
        time.sleep(60)
    if failures:
        raise RuntimeError(
            f"{len(failures)} members could not be fetched, e.g. {failures[:3]}; rerun to resume"
        )
    info = {
        "key": f"{subset}_hpcp",
        "mode": "members",
        "source": f"https://doi.org/10.5281/zenodo.{ZENODO_RECORD}",
        "filename": filename,
        "n_wanted": len(pairs),
        "n_wids": len({wid for wid, _ in pairs}),
    }
    (outdir / f".{subset}_hpcp_members.done").write_text(json.dumps(info, indent=2))
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--what", nargs="+", choices=sorted(ARCHIVES), help="whole archives to download")
    parser.add_argument("--fetch-members", choices=["coveranalysis", "benchmark"])
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--n-wids", type=int, default=None, help="coveranalysis: fetch the first N seeded WIDs"
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--keep-zip", action="store_true", help="keep archives after unpacking")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    outdir = args.outdir or Path(config["paths"]["datacos_root"])
    outdir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(outdir).free
    print(f"free disk space: {free / 1024**3:.1f} GiB")
    wants_features = args.fetch_members or any(k != "metadata" for k in args.what or [])
    if wants_features and free < MIN_FREE_BYTES:
        print("Refusing to download features: fewer than 35 GiB free.", file=sys.stderr)
        return 2

    records = []
    for key in args.what or []:
        records.append(download_archive(key, outdir, args.keep_zip))
    if args.fetch_members:
        if not metadata_path(config, args.fetch_members).exists():
            records.append(download_archive("metadata", outdir, args.keep_zip))
        records.append(fetch_members(config, args.fetch_members, outdir, args.n_wids, args.threads))
    if not records:
        parser.error("nothing to do: pass --what and/or --fetch-members")
    log = outdir / "download_log.jsonl"
    with log.open("a") as handle:
        for record in records:
            handle.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}) + "\n")
    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
