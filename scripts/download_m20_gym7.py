"""Opt-in download of ONE official Mendeley Gym7 archive, never Git-tracked.

Public metadata API: https://api.data.mendeley.com/datasets/kgbb3yn47p
Official version: 3. Dataset is CC BY 4.0, DOI 10.17632/kgbb3yn47p.3.
Download may be subject to provider access limits; manual fallback is documented.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "kgbb3yn47p"
VERSION = 3
DATASET_API = f"https://api.data.mendeley.com/datasets/{DATASET}"
ALLOWED_NAME = "Dataset Exercise-wise.zip"
CHUNK = 1024 * 1024


def _json_request(url: str):
    request = urllib.request.Request(url, headers={
        "Accept": "application/vnd.mendeley-public-dataset.1+json, application/json",
        "User-Agent": "GymPoseResearch/0.1 (owner opt-in dataset retrieval)",
    })
    with urllib.request.urlopen(request, timeout=40) as response:
        return json.load(response)


def select_official_file(metadata: dict) -> dict:
    if not isinstance(metadata, dict) or metadata.get("version") != VERSION:
        raise ValueError("Dataset version mismatch - expected the official v3")
    files = metadata.get("files", [])
    matches = [f for f in files if f.get("filename") == ALLOWED_NAME]
    if len(matches) != 1 or not matches[0].get("id"):
        raise ValueError("Exact Exercise-wise archive not found in official v3 metadata")
    return matches[0]


def download_archive(destination: Path, *, confirm_license: bool = False,
                     max_gb: float = 4.0) -> dict:
    if not confirm_license:
        raise ValueError("Explicit --accept-cc-by-4-0 is required")
    if max_gb < 3:
        raise ValueError("Download limit must allow a 2.9 GB archive")
    if destination.exists():
        raise FileExistsError("Archive already exists; will never replace user data")
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = _json_request(f"{DATASET_API}?version={VERSION}")
    record = select_official_file(metadata)
    expected = record.get("content_details") or {}
    digest = expected.get("sha256_hash") or record.get("sha256_hash")
    if digest and (not isinstance(digest, str) or len(digest) != 64):
        raise ValueError("Invalid SHA256 digest from publisher")
    size = expected.get("size") or record.get("size")
    if size is not None and int(size) > max_gb * 1024 ** 3:
        raise ValueError("Publisher archive exceeds the size limit")
    url = f"{DATASET_API}/files/{record['id']}/file_downloaded?version={VERSION}"
    part = destination.with_suffix(destination.suffix + ".part")
    if part.exists():
        raise FileExistsError("A prior partial archive exists; inspect it before retrying")
    h = hashlib.sha256()
    count = 0
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "GymPoseResearch/0.1"})
        with urllib.request.urlopen(request, timeout=90) as response:
            if urllib.parse.urlsplit(response.geturl()).scheme != "https":
                raise ValueError("Dataset redirect was not HTTPS")
            with part.open("xb") as output:
                while True:
                    chunk = response.read(CHUNK)
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > max_gb * 1024 ** 3:
                        raise ValueError("Dataset exceeded download safety cap")
                    h.update(chunk)
                    output.write(chunk)
        if size is not None and count != int(size):
            raise ValueError("Truncated archive - publisher length does not match")
        if digest and h.hexdigest().lower() != digest.lower():
            raise ValueError("Publisher SHA256 checksum mismatch")
        if count < 100000000:
            raise ValueError("Unexpectedly small dataset archive")
        # Verify ZIP header without unpacking untrusted archives into the working tree.
        with part.open("rb") as f:
            if f.read(4) != b"PK\\x03\\x04":
                raise ValueError("Publisher download is not a ZIP")
        os.replace(part, destination)
    except Exception:
        part.unlink(missing_ok=True)
        raise
    return {"dataset": DATASET, "version": VERSION, "license": "CC BY 4.0",
            "file": str(destination), "bytes": count, "sha256": h.hexdigest(),
            "publisher_checksum_verified": bool(digest),
            "note": "Do NOT unzip inside Git-tracked paths. Never combine the three duplicate archives."}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, default=Path(
        "data/external/mendeley_gym7/Dataset Exercise-wise.zip"))
    p.add_argument("--accept-cc-by-4-0", action="store_true")
    p.add_argument("--max-gb", type=float, default=4.0)
    args = p.parse_args(argv)
    try:
        result = download_archive(args.destination, confirm_license=args.accept_cc_by_4_0,
                                  max_gb=args.max_gb)
    except (OSError, ValueError, urllib.error.URLError, KeyError) as error:
        print("Download not completed. You may manually download ONE ZIP from:",
              "https://data.mendeley.com/datasets/kgbb3yn47p/3", file=sys.stderr)
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
