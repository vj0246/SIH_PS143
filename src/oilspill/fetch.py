"""Resumable, checksum verified downloader.

Decision D9. This runs wherever the operator has network access. It is written
defensively because these are multi-gigabyte transfers over links that fail:
every file resumes from a partial download with an HTTP Range request, every
completed file is hashed, and the hash is written back into a lock file so the
second run of the same command verifies rather than re-downloads.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests
import yaml

_DEFAULT_REGISTRY = Path(__file__).resolve().parents[2] / "configs" / "datasets.yaml"
_USER_AGENT = "oilspill-ard/0.1 (research; contact via repository)"


class FetchError(RuntimeError):
    pass


@dataclass
class RemoteFile:
    url: str
    relpath: str
    size: int | None = None
    sha256: str | None = None


def load_registry(path: str | Path | None = None) -> dict:
    path = Path(path or _DEFAULT_REGISTRY)
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})
    return session


def discover_zenodo(api_url: str, session: requests.Session | None = None) -> list[RemoteFile]:
    """Enumerate a Zenodo record's files through the record API.

    Zenodo publishes a per-file ``checksum`` of the form ``md5:<hex>``. It is
    recorded but not used for verification here, because the lock file stores
    sha256 computed locally; mixing digest algorithms across sources would make
    the lock file ambiguous.
    """
    session = session or _session()
    response = session.get(api_url, timeout=60)
    if response.status_code != 200:
        raise FetchError(f"Zenodo API returned {response.status_code} for {api_url}")
    payload = response.json()

    files = []
    for entry in payload.get("files", []):
        link = entry.get("links", {}).get("self") or entry.get("links", {}).get("download")
        if not link:
            continue
        files.append(
            RemoteFile(url=link, relpath=entry.get("key", link.rsplit("/", 1)[-1]), size=entry.get("size"))
        )
    if not files:
        raise FetchError(f"no files listed on Zenodo record {api_url}")
    return files


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(
    remote: RemoteFile,
    dest: Path,
    session: requests.Session | None = None,
    chunk_bytes: int = 8 << 20,
    retries: int = 5,
    backoff_seconds: float = 5.0,
    progress=None,
) -> Path:
    """Download one file, resuming a partial transfer if one is present.

    Args:
        remote: File description.
        dest: Final path. A sibling ``.part`` file holds the partial transfer.
        session: Reused HTTP session.
        chunk_bytes: Streaming chunk size.
        retries: Attempts before giving up.
        backoff_seconds: Base for exponential backoff between attempts.
        progress: Optional callable taking ``(bytes_done, total_or_None)``.

    Returns:
        The completed path.

    Raises:
        FetchError: If every attempt fails, or the server ignores a Range
            request in a way that would corrupt the resumed file.
    """
    session = session or _session()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    if dest.exists() and remote.size and dest.stat().st_size == remote.size:
        return dest

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            done = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={done}-"} if done else {}
            with session.get(remote.url, headers=headers, stream=True, timeout=120) as resp:
                if done and resp.status_code == 200:
                    # The server ignored the Range header and is sending the whole
                    # file. Appending would corrupt it, so restart cleanly.
                    part.unlink(missing_ok=True)
                    done = 0
                elif done and resp.status_code != 206:
                    raise FetchError(
                        f"resume of {remote.relpath} got HTTP {resp.status_code}, "
                        "expected 206 Partial Content"
                    )
                elif resp.status_code not in (200, 206):
                    raise FetchError(f"HTTP {resp.status_code} for {remote.url}")

                total = remote.size
                if total is None:
                    length = resp.headers.get("Content-Length")
                    total = (int(length) + done) if length else None

                mode = "ab" if done else "wb"
                with part.open(mode) as fh:
                    for block in resp.iter_content(chunk_size=chunk_bytes):
                        if not block:
                            continue
                        fh.write(block)
                        done += len(block)
                        if progress is not None:
                            progress(done, total)

            if remote.size and part.stat().st_size != remote.size:
                raise FetchError(
                    f"{remote.relpath}: got {part.stat().st_size} bytes, expected {remote.size}"
                )
            part.replace(dest)
            return dest

        except Exception as exc:  # noqa: BLE001 - retried and re-raised below
            last_error = exc
            if attempt < retries - 1:
                time.sleep(backoff_seconds * (2**attempt))

    raise FetchError(f"failed to download {remote.url} after {retries} attempts: {last_error}")


def _lock_path(root: Path) -> Path:
    return root / "fetch.lock.json"


def _read_lock(root: Path) -> dict:
    path = _lock_path(root)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _write_lock(root: Path, lock: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _lock_path(root).write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")


def fetch_source(
    name: str,
    registry: dict | None = None,
    root: str | Path | None = None,
    verify_existing: bool = True,
    progress=None,
) -> dict:
    """Download every file for one registry source.

    Returns:
        A report dict with ``downloaded``, ``verified``, ``skipped`` and any
        ``manual_instructions`` the registry supplies.

    Raises:
        FetchError: On an unresolvable download failure, or when a previously
            recorded sha256 no longer matches the file on disk.
    """
    registry = registry or load_registry()
    try:
        spec = registry["sources"][name]
    except KeyError as exc:
        raise FetchError(
            f"unknown source {name!r}; registry declares {sorted(registry['sources'])}"
        ) from exc

    root = Path(root or registry.get("defaults", {}).get("root", "data/raw")) / name
    discovery = spec.get("discovery", "manual")

    if discovery in ("manual", "unavailable"):
        return {
            "source": name,
            "status": discovery,
            "downloaded": [],
            "verified": [],
            "skipped": [],
            "manual_instructions": spec.get("manual_instructions") or spec.get("notes", ""),
            "landing": spec.get("landing"),
        }

    session = _session()
    if discovery == "zenodo_api":
        remotes = discover_zenodo(spec["api"], session)
    else:
        raise FetchError(
            f"discovery mode {discovery!r} is not implemented for source {name!r}"
        )

    lock = _read_lock(root)
    downloaded, verified, skipped = [], [], []
    defaults = registry.get("defaults", {})

    for remote in remotes:
        dest = root / remote.relpath
        recorded = lock.get(remote.relpath, {})

        if dest.exists() and recorded.get("sha256"):
            if verify_existing:
                actual = sha256_file(dest)
                if actual != recorded["sha256"]:
                    raise FetchError(
                        f"{dest} sha256 {actual} does not match the recorded "
                        f"{recorded['sha256']}. The file has been modified or the "
                        "download was corrupted. Delete it and re-fetch."
                    )
                verified.append(remote.relpath)
            else:
                skipped.append(remote.relpath)
            continue

        download_file(
            remote,
            dest,
            session=session,
            chunk_bytes=int(defaults.get("chunk_bytes", 8 << 20)),
            retries=int(defaults.get("retries", 5)),
            backoff_seconds=float(defaults.get("backoff_seconds", 5)),
            progress=progress,
        )
        lock[remote.relpath] = {
            "sha256": sha256_file(dest),
            "size": dest.stat().st_size,
            "url": remote.url,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_lock(root, lock)
        downloaded.append(remote.relpath)

    return {
        "source": name,
        "status": "ok",
        "root": str(root),
        "downloaded": downloaded,
        "verified": verified,
        "skipped": skipped,
        "licence": spec.get("licence"),
        "citation": spec.get("citation"),
    }


def estimate_disk(name: str, registry: dict | None = None) -> dict:
    """Total remote size for a source, so disk can be checked before starting."""
    registry = registry or load_registry()
    spec = registry["sources"][name]
    if spec.get("discovery") != "zenodo_api":
        return {"source": name, "known": False}
    remotes = discover_zenodo(spec["api"])
    total = sum(r.size or 0 for r in remotes)
    return {
        "source": name,
        "known": True,
        "n_files": len(remotes),
        "bytes": total,
        "gigabytes": round(total / 1e9, 2),
        "free_gigabytes": round(os.statvfs(".").f_bavail * os.statvfs(".").f_frsize / 1e9, 2),
    }
