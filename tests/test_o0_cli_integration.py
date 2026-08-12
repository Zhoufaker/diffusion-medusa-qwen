"""True CLI integration: subprocess eval_acceptance_tree O0 exit codes."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
MOD = ["-m", "scripts.eval_acceptance_tree"]


def _run(extra: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PY, *MOD, *extra],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_cli_o0_archive_requires_check_greedy_bytes():
    """--o0-archive without --check-greedy-bytes → parser/usage error (non-zero)."""
    r = _run(["--o0-archive", "/tmp/fake_arch.json", "--out", "/tmp/x.json"])
    assert r.returncode != 0
    blob = (r.stderr or "") + (r.stdout or "")
    assert "--o0-archive requires --check-greedy-bytes" in blob


def test_cli_o0_write_archive_requires_check_greedy_bytes():
    """--o0-write-archive without --check-greedy-bytes → usage error (non-zero)."""
    r = _run(["--o0-write-archive", "/tmp/fake_write.json", "--out", "/tmp/x.json"])
    assert r.returncode != 0
    blob = (r.stderr or "") + (r.stdout or "")
    assert "--o0-write-archive requires --check-greedy-bytes" in blob


def test_cli_o0_write_greedy_archive_requires_check_greedy_bytes():
    """--o0-write-greedy-archive without --check-greedy-bytes → usage error."""
    r = _run([
        "--o0-write-greedy-archive", "/tmp/fake_gwrite.json", "--out", "/tmp/x.json",
    ])
    assert r.returncode != 0
    blob = (r.stderr or "") + (r.stdout or "")
    assert "--o0-write-greedy-archive requires --check-greedy-bytes" in blob


def test_cli_archive_pass_exit_0():
    r = _run(["--check-greedy-bytes", "--o0-cli-selftest", "archive_pass"])
    assert r.returncode == 0
    assert "exit=0" in r.stdout


def test_cli_archive_fail_exit_2():
    r = _run(["--check-greedy-bytes", "--o0-cli-selftest", "archive_fail"])
    assert r.returncode == 2
    assert "exit=2" in r.stdout


def test_cli_archive_not_run_exit_5():
    r = _run(["--check-greedy-bytes", "--o0-cli-selftest", "archive_not_run"])
    assert r.returncode == 5
    assert "exit=5" in r.stdout


def test_cli_archive_incomplete_exit_6():
    r = _run(["--check-greedy-bytes", "--o0-cli-selftest", "archive_incomplete"])
    assert r.returncode == 6
    assert "exit=6" in r.stdout


def test_cli_safety_only_fail_exit_3():
    r = _run(["--check-greedy-bytes", "--o0-cli-selftest", "safety_fail"])
    assert r.returncode == 3
    assert "exit=3" in r.stdout
