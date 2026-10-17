"""Reproducible research utilities: file checksums and dependency provenance.

This is a traceable *record* of inputs and software, not proof that findings
are valid, reproducible across hardware, or protected against tampering.
Manifest content deliberately excludes personal absolute paths and user names.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform


DEFAULT_DISTRIBUTIONS = (
    "khaled-dsml", "python", "numpy", "pandas", "scipy",
    "scikit-learn", "statsmodels", "dask", "pyspark",
)


def sha256_file(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    """Return a SHA-256 hex digest by streaming bytes, not loading a whole file."""
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Not a regular input file: {source.name}")
    digest = sha256()
    with source.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def software_versions(packages=DEFAULT_DISTRIBUTIONS) -> dict[str, str | None]:
    """Capture installed distributions, with None for packages not installed."""
    result = {}
    for name in packages:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Package names must be nonempty strings")
        if name == "python":
            result[name] = platform.python_version()
        else:
            try:
                result[name] = version(name)
            except PackageNotFoundError:
                result[name] = None
    return result


def reproducibility_manifest(
    files, *, seed: int | None = None,
    include_timestamp: bool = False,
    packages=DEFAULT_DISTRIBUTIONS,
) -> dict:
    """Describe data inputs by basename, size and streaming hash.

    Files must be an iterable of input paths, not a directory. Basenames are
    included rather than machine-specific absolute paths. With timestamp=False
    (default), running twice on identical inputs and software is deterministic.
    """
    if isinstance(files, (str, Path)):
        paths = [Path(files)]
    else:
        paths = [Path(item) for item in files]
    if not paths:
        raise ValueError("Provide at least one input file")
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    entries = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path.name}")
        entries.append({
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    entries.sort(key=lambda item: (item["filename"], item["sha256"]))
    manifest = {
        "schema_version": 1,
        "inputs": entries,
        "software": software_versions(packages),
        "seed": seed,
        "interpretation_ar": (
            "يسجل الملف بصمات المدخلات وإصدارات البرمجيات لتسهيل إعادة التحليل. "
            "البصمة لا تثبت صحة النتائج ولا تمنع التلاعب دون توقيع موثوق."
        ),
        "interpretation_en": (
            "Input hashes and software versions help recreate an analysis. "
            "A hash alone neither validates results nor prevents tampering "
            "without a trusted signature."
        ),
    }
    if include_timestamp:
        manifest["created_utc"] = datetime.now(timezone.utc).isoformat()
    return manifest


def save_reproducibility_manifest(manifest: dict, path: str | Path) -> Path:
    """Write an auditable UTF-8 JSON record, preserving Arabic text."""
    if not isinstance(manifest, dict) or not isinstance(manifest.get("inputs"), list):
        raise ValueError("Pass a manifest returned by reproducibility_manifest")
    target = Path(path)
    target.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def verify_reproducibility_manifest(manifest: dict, files) -> dict:
    """Compare saved hashes with a supplied list of local input files.

    Compares filename, size, and digest, not the file system's location.
    Duplicate basenames are supported. Software equality is *not* checked.
    """
    if not isinstance(manifest, dict) or not isinstance(manifest.get("inputs"), list):
        raise ValueError("Pass a manifest returned by reproducibility_manifest")
    if isinstance(files, (str, Path)):
        paths = [Path(files)]
    else:
        paths = [Path(item) for item in files]
    observed = []
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"Input file not found: {p.name}")
        observed.append({
            "filename": p.name, "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })
    observed.sort(key=lambda item: (item["filename"], item["sha256"]))
    expected = sorted(
        manifest["inputs"], key=lambda item: (item["filename"], item["sha256"]),
    )
    matched = bool(observed == expected)
    return {
        "inputs_match": matched,
        "n_expected": len(expected),
        "n_observed": len(observed),
        "interpretation_ar": (
            "تطابقت الملفات المدخلة مع البصمات المسجلة؛ لم يُتحقق من إعادة الحساب."
            if matched else
            "لم تتطابق ملفات الإدخال مع البصمات المسجلة؛ راجع المصدر قبل التحليل."
        ),
        "interpretation_en": (
            "Input file hashes match the record; analysis rerun is not verified."
            if matched else
            "Input file hashes differ from the record; inspect provenance.",
        ),
    }
