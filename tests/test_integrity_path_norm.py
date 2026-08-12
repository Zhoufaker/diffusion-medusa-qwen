"""Integrity checker path normalization and root-relative exclusions."""
import json
from pathlib import Path

from scripts.check_release_integrity import (
    build_payload_manifest,
    extract_cover_paths,
    is_excluded,
    norm_rel_path,
)


def test_norm_rel_path_backslash_and_slash_equivalent():
    a = norm_rel_path(r"A_layer_no_torch\artifacts\foo.json")
    b = norm_rel_path("A_layer_no_torch/artifacts/foo.json")
    assert a == b == "A_layer_no_torch/artifacts/foo.json"
    assert norm_rel_path(Path("B_layer_gpu_rerun") / "pbs" / "x.pbs") == \
        "B_layer_gpu_rerun/pbs/x.pbs"


def test_extract_cover_paths_normalizes_separators():
    cover = """
<!-- COVER_START -->
- `A_layer_no_torch/artifacts/a.json`
- `B_layer_gpu_rerun\\scripts\\eval_acceptance_tree.py`
<!-- COVER_END -->
"""
    paths = extract_cover_paths(cover)
    assert "A_layer_no_torch/artifacts/a.json" in paths
    assert "B_layer_gpu_rerun/scripts/eval_acceptance_tree.py" in paths
    assert all("\\" not in p for p in paths)


def test_exclusions_are_root_relative_not_basename():
    """Only the chain files at the package root are outside the payload."""
    assert is_excluded("README.md")
    assert is_excluded("FILE_MANIFEST.json")
    assert not is_excluded("B_layer_gpu_rerun/README.md")
    assert not is_excluded("A_layer_no_torch/scripts/README.md")
    # checker self-output stays excluded wherever it lands
    assert is_excluded("A_layer_no_torch/artifacts/integrity_check_output.txt")
    assert is_excluded("A_layer_no_torch/artifacts/INTEGRITY_CHECK.json")


def test_nested_readme_enters_manifest(tmp_path: Path):
    """Regression: B_layer_gpu_rerun/README.md used to be dropped by basename."""
    (tmp_path / "B_layer_gpu_rerun").mkdir()
    (tmp_path / "A_layer_no_torch/artifacts").mkdir(parents=True)
    (tmp_path / "README.md").write_text("root cover copy", encoding="utf-8")
    (tmp_path / "B_layer_gpu_rerun/README.md").write_text("rerun guide", encoding="utf-8")
    (tmp_path / "A_layer_no_torch/artifacts/a.json").write_text("{}", encoding="utf-8")
    (tmp_path / "A_layer_no_torch/artifacts/integrity_check_output.txt").write_text(
        "self output", encoding="utf-8")
    (tmp_path / "FILE_MANIFEST.json").write_text(json.dumps({}), encoding="utf-8")

    man = build_payload_manifest(tmp_path)
    paths = {f["path"] for f in man["files"]}
    assert paths == {
        "A_layer_no_torch/artifacts/a.json",
        "B_layer_gpu_rerun/README.md",
    }
    assert man["n_files"] == 2
