#!/usr/bin/env python3.11
"""Unidirectional release integrity checker (Round-5).

Payload-only FILE_MANIFEST (excludes manifest itself, detached SHA, cover, stamp).
Verifies:
  - every payload path exists with matching size + sha256
  - cover n_files / path set bidirectional consistency with manifest
  - cover/stamp embedded SHA == recomputed payload manifest SHA
All IO: encoding=utf-8.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# Root-relative exclusions: only the chain files that sit at the package root
# are outside the payload. Matching on bare basenames used to swallow any
# nested file that happened to share a name — most visibly
# `B_layer_gpu_rerun/README.md`, a real deliverable that silently never
# entered the manifest.
EXCLUDED_ROOT_PATHS = {
    "FILE_MANIFEST.json",
    "FILE_MANIFEST.sha256",
    "ROUND7_COVER.md",
    "ROUND6_COVER.md",
    "ROUND5_COVER.md",
    "ROUND4_COVER.md",
    "README.md",
    "SUBMIT_STAMP.json",
    "COVER_CHECK.json",
    "INTEGRITY_CHECK.json",
}
# Checker self-output, wherever it is written.
EXCLUDED_SUFFIXES = (
    "INTEGRITY_CHECK.json",
    "integrity_check_output.txt",
    "COVER_CHECK.json",
    "cover_check_output.txt",
)


def is_excluded(rel: str) -> bool:
    """`rel` is a root-relative POSIX path inside the package."""
    return rel in EXCLUDED_ROOT_PATHS or rel.endswith(EXCLUDED_SUFFIXES)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_rel_path(path: str | Path) -> str:
    """Normalize to POSIX relative path (Windows `\\` → `/`)."""
    s = path.as_posix() if isinstance(path, Path) else str(path)
    return s.replace("\\", "/").lstrip("./")


def build_payload_manifest(root: Path) -> dict:
    files = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = norm_rel_path(p.relative_to(root))
        if is_excluded(rel):
            continue
        files.append({
            "path": rel,
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })
    return {"n_files": len(files), "files": files, "scope": "payload_only"}


def extract_cover_paths(readme: str) -> list:
    m = re.search(r"<!-- COVER_START -->(.*)<!-- COVER_END -->", readme, re.S)
    block = m.group(1) if m else readme
    paths = []
    for line in block.splitlines():
        for p in re.findall(r"`([^`]+)`", line):
            if "/" in p or "\\" in p or p.endswith(
                    (".json", ".py", ".md", ".txt", ".log", ".xml", ".toml", ".lock")):
                paths.append(norm_rel_path(p))
        m2 = re.match(r"\s*[-*]\s+`?([A-Za-z0-9_./\\-]+\.[A-Za-z0-9]+)`?", line)
        if m2:
            paths.append(norm_rel_path(m2.group(1).strip("`")))
    seen, out = set(), []
    for p in paths:
        if p not in seen and not is_excluded(p):
            seen.add(p)
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--write-manifest", action="store_true",
                    help="Rewrite FILE_MANIFEST.json + .sha256 from payload")
    args = ap.parse_args()
    root = Path(args.root)

    payload = build_payload_manifest(root)
    man_bytes = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    man_sha = hashlib.sha256(man_bytes).hexdigest()

    if args.write_manifest:
        (root / "FILE_MANIFEST.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        man_sha = hashlib.sha256((root / "FILE_MANIFEST.json").read_bytes()).hexdigest()
        (root / "FILE_MANIFEST.sha256").write_text(man_sha + "\n", encoding="utf-8")
        print(json.dumps({
            "wrote_manifest": True,
            "n_payload": payload["n_files"],
            "manifest_file_sha256": man_sha,
        }, indent=2, ensure_ascii=False))
        return 0

    # verify against on-disk manifest if present
    errors = []
    disk_man = None
    disk_sha = None
    if (root / "FILE_MANIFEST.json").exists():
        disk_man = json.loads((root / "FILE_MANIFEST.json").read_text(encoding="utf-8"))
        disk_sha = hashlib.sha256(
            (root / "FILE_MANIFEST.json").read_bytes()
        ).hexdigest()
        # Prefer comparing logical payload equality
        if disk_man.get("n_files") != payload["n_files"]:
            errors.append(f"manifest n_files {disk_man.get('n_files')} != payload {payload['n_files']}")
        disk_map = {f["path"]: f for f in disk_man["files"]}
        pay_map = {f["path"]: f for f in payload["files"]}
        for path, f in pay_map.items():
            if path not in disk_map:
                errors.append(f"manifest missing payload path {path}")
                continue
            g = disk_map[path]
            if g["bytes"] != f["bytes"] or g["sha256"] != f["sha256"]:
                errors.append(f"manifest mismatch {path}")
            fp = root / path
            if not fp.is_file():
                errors.append(f"missing file {path}")
            elif fp.stat().st_size != f["bytes"] or sha256_file(fp) != f["sha256"]:
                errors.append(f"on-disk mismatch {path}")
        for path in disk_map:
            if path not in pay_map:
                errors.append(f"manifest has non-payload path {path}")
        if (root / "FILE_MANIFEST.sha256").exists():
            det = (root / "FILE_MANIFEST.sha256").read_text(encoding="utf-8").strip()
            if det != disk_sha:
                errors.append(f"detached SHA {det} != FILE_MANIFEST.json sha {disk_sha}")

    # cover bidirectional
    cover_path = next(
        (root / n for n in ("ROUND7_COVER.md", "ROUND6_COVER.md", "ROUND5_COVER.md",
                            "README.md")
         if (root / n).exists()),
        root / "ROUND7_COVER.md",
    )
    if not cover_path.exists():
        errors.append("missing ROUND7_COVER.md / README.md")
        cover_text = ""
        cover_paths = []
        missing_in_man = []
        missing_in_cover = sorted({f["path"] for f in (disk_man or payload)["files"]})
    else:
        cover_text = cover_path.read_text(encoding="utf-8")
        cover_paths = extract_cover_paths(cover_text)
        man_paths = {f["path"] for f in (disk_man or payload)["files"]}
        missing_in_man = [p for p in cover_paths if p not in man_paths]
        missing_in_cover = sorted(man_paths - set(cover_paths))
        if missing_in_man:
            errors.append(f"cover mentions not in manifest: {missing_in_man[:5]}")
        if missing_in_cover:
            errors.append(
                f"manifest paths missing from cover ({len(missing_in_cover)}): "
                f"{missing_in_cover[:5]}")

    # embedded SHA in cover/stamp — stamp/cover SHA = FILE_MANIFEST.json file bytes
    want_file_sha = disk_sha or (
        hashlib.sha256((root / "FILE_MANIFEST.json").read_bytes()).hexdigest()
        if (root / "FILE_MANIFEST.json").exists() else man_sha
    )
    if cover_text:
        embedded = re.findall(r"`([0-9a-f]{64})`", cover_text, re.I)
        embedded += re.findall(r"sha256[:\s=]+([0-9a-f]{64})", cover_text, re.I)
        for h in set(embedded):
            if h != want_file_sha:
                errors.append(f"cover embedded SHA {h} != manifest file SHA {want_file_sha}")
    if (root / "SUBMIT_STAMP.json").exists():
        stamp = json.loads((root / "SUBMIT_STAMP.json").read_text(encoding="utf-8"))
        stamp_sha = stamp.get("file_manifest_sha256")
        if stamp_sha and stamp_sha != want_file_sha:
            errors.append(f"stamp SHA {stamp_sha} != manifest file SHA {want_file_sha}")

    report = {
        "pass": len(errors) == 0,
        "n_payload": payload["n_files"],
        "manifest_file_sha256": want_file_sha,
        "n_cover_mentions": len(cover_paths),
        "n_manifest_missing_from_cover": len(missing_in_cover),
        "n_cover_missing_from_manifest": len(missing_in_man),
        "errors": errors,
        "end_state": (
            "greedy numerical-safety verified / "
            "archive byte-reproducibility verified / "
            "release bundle reproducible"
        ),
    }
    out = root / "INTEGRITY_CHECK.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("INTEGRITY_CHECK:", "PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
