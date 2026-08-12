"""Release PBS scripts must carry no site-specific log path.

A reviewer without the `/scratch/li96/mz9869` layout has to be able to submit
these unchanged: the log destination comes in via `qsub -o "$MEDUSA_LOG_DIR"`,
and `MEDUSA_LOG_DIR` is on each script's fail-fast checklist.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PBS_DIR = ROOT / "pbs"
RELEASE_PBS = [
    "tier_d_repro_fullcover.pbs",
    "tier_d_tier_c_rerun.pbs",
    "v2_rebaseline_in_domain.pbs",
    "v2_rebaseline_ood.pbs",
    "round6_candidate_reprobe.pbs",
]
REQUIRED_RE = re.compile(r'^:\s*"\$\{([A-Z0-9_]+):\?', re.M)
# Directive lines only — prose mentioning `#PBS -o` in a comment is fine.
OUT_DIRECTIVE_RE = re.compile(r"^#PBS\s+-[oe]\s", re.M)


def _prologue(text: str) -> str:
    """Everything up to the first runtime line (banner echo)."""
    cut = text.find('echo "===')
    assert cut > 0, "no banner line found"
    return text[:cut]


@pytest.mark.parametrize("name", RELEASE_PBS)
def test_no_fixed_output_directive(name: str):
    text = (PBS_DIR / name).read_text(encoding="utf-8")
    assert not OUT_DIRECTIVE_RE.search(text), f"{name} pins a log path in the script"
    assert "/scratch/li96/mz9869/logs" not in text, f"{name} mkdirs a site log dir"


@pytest.mark.parametrize("name", RELEASE_PBS)
def test_log_dir_is_fail_fast(name: str):
    text = (PBS_DIR / name).read_text(encoding="utf-8")
    assert "MEDUSA_LOG_DIR" in REQUIRED_RE.findall(text), \
        f"{name} does not fail-fast on MEDUSA_LOG_DIR"


@pytest.mark.parametrize("name", RELEASE_PBS + ["round7_evaluator_smoke.pbs"])
def test_no_hardcoded_claim_string(name: str):
    """A pasted claim outlives the next predicate change; import it instead."""
    text = (PBS_DIR / name).read_text(encoding="utf-8")
    assert "algorithmic greedy lossless" not in text, \
        f"{name} embeds a claim literal; use O0_CLAIM_OFFICIAL"


@pytest.mark.parametrize("name", RELEASE_PBS)
def test_prologue_passes_with_env_and_dies_without_log_dir(name: str, tmp_path: Path):
    """A foreign-site submission fails on the missing variable, not on a path."""
    text = (PBS_DIR / name).read_text(encoding="utf-8")
    prologue = _prologue(text)
    script = tmp_path / f"prologue_{name}.sh"
    script.write_text(prologue, encoding="utf-8")

    required = REQUIRED_RE.findall(text)
    env = {**os.environ}
    with tempfile.TemporaryDirectory() as elsewhere:
        for var in required:
            env[var] = elsewhere  # no /scratch/li96 layout anywhere

        ok = subprocess.run(["bash", str(script)], env=env,
                            capture_output=True, text=True)
        assert ok.returncode == 0, f"{name} prologue failed with full env: {ok.stderr}"

        env.pop("MEDUSA_LOG_DIR")
        bad = subprocess.run(["bash", str(script)], env=env,
                             capture_output=True, text=True)
        assert bad.returncode != 0, f"{name} accepted a missing MEDUSA_LOG_DIR"
        assert "MEDUSA_LOG_DIR" in bad.stderr
