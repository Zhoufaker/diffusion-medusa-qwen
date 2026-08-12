#!/usr/bin/env python3.11
"""Assert every path mentioned in README cover section exists in FILE_MANIFEST.

Cover section markers in README:
  <!-- COVER_START -->
  ... lines containing `path/to/file` or `- path` ...
  <!-- COVER_END -->

Or: all fenced paths / backtick paths under a '## Packet contents' heading.

Exit 0 on pass; writes COVER_CHECK.json beside the README.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def extract_cover_paths(readme: str) -> list:
    m = re.search(r"<!-- COVER_START -->(.*)<!-- COVER_END -->", readme, re.S)
    block = m.group(1) if m else readme
    paths = []
    for line in block.splitlines():
        # backtick paths
        for p in re.findall(r"`([^`]+)`", line):
            if "/" in p or p.endswith(".json") or p.endswith(".py") or p.endswith(".md"):
                paths.append(p.strip("/"))
        # markdown list "- path"
        m2 = re.match(r"\s*[-*]\s+([A-Za-z0-9_./-]+\.(?:json|py|md|txt|log))", line)
        if m2:
            paths.append(m2.group(1))
    # unique preserve order
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="release packet root")
    args = ap.parse_args()
    root = Path(args.root)
    readme = (root / "README.md").read_text(encoding="utf-8")
    manifest = json.loads((root / "FILE_MANIFEST.json").read_text())
    man_paths = {f["path"] for f in manifest["files"]}
    cover = extract_cover_paths(readme)
    missing = [p for p in cover if p not in man_paths and p != "FILE_MANIFEST.json"]
    # also allow prefix match for directories mentioned
    still = []
    for p in missing:
        if any(mp == p or mp.startswith(p.rstrip("/") + "/") for mp in man_paths):
            continue
        still.append(p)
    report = {
        "n_cover_mentions": len(cover),
        "n_manifest": len(man_paths),
        "missing": still,
        "pass": len(still) == 0,
    }
    out = root / "COVER_CHECK.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("COVER_CHECK:", "PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
