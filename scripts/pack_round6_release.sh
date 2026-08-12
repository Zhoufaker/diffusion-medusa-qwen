#!/bin/bash
# Assemble gpt56_round6_release with unidirectional integrity chain (Round-5 process).
set -euo pipefail
SRC="${HOME}/medusa-qwen"
C1=/scratch/li96/mz9869/medusa_outputs/linked_medusa_c1_eagle
PKT=$C1/gpt56_round6_release
A=$PKT/A_layer_no_torch
B=$PKT/B_layer_gpu_rerun

rm -rf "$PKT"
mkdir -p "$A"/{artifacts/v2_rebaseline,artifacts/tier_c,artifacts/tier_d_repro,artifacts/ood_offline,artifacts/v1_lineage,artifacts/round6_reprobe,o0_reports,archives,pytest_logs,scripts}
mkdir -p "$B"/{decode,scripts,model,tests,pbs}

# --- A scripts ---
cp "$SRC/scripts/recompute_all.py" "$A/scripts/"
cp "$SRC/scripts/check_release_integrity.py" "$A/scripts/"
cp "$SRC/scripts/scrub_greedy_exact_aliases.py" "$A/scripts/"
cp "$SRC/scripts/fill_round6_cover.py" "$A/scripts/"
cp "$SRC/scripts/reprobe_near_tie_candidate.py" "$A/scripts/" 2>/dev/null || true

# --- v2 artifacts ---
for f in "$C1"/v2_rebaseline/*.json; do
  bn=$(basename "$f")
  case "$bn" in
    *.spec_archive.json|*.greedy_archive.json) cp "$f" "$A/archives/" ;;
    *.o0_report.json) cp "$f" "$A/o0_reports/v2_$bn" ;;
    *.o0_prov.json) cp "$f" "$A/archives/" ;;
    *) cp "$f" "$A/artifacts/v2_rebaseline/" ;;
  esac
done

cp "$C1/final_eval/c1_d3.json" "$A/artifacts/v1_lineage/gate100_c1_d3.json"
cp "$C1/fanout_sweep/c10_6432_32.json" "$A/artifacts/v1_lineage/"
cp "$C1/final_eval/b2_wide.json" "$A/artifacts/v1_lineage/"
cp "$C1/bridge_300/c1_d3.json" "$A/artifacts/v1_lineage/bridge300_c1_d3.json"
cp "$C1/bridge_300/c1_6432.json" "$A/artifacts/v1_lineage/bridge300_c1_6432.json"
cp "$C1/bridge_300/b2_d3.json" "$A/artifacts/v1_lineage/bridge300_b2_d3.json"
cp "$C1/bridge_300/b2_6432.json" "$A/artifacts/v1_lineage/bridge300_b2_6432.json"
cp "$C1/dynamic_sweep_300/dyn_k8_n24.json" "$A/artifacts/v1_lineage/"
cp "$C1/dynamic_sweep_300/dyn_k8_n32.json" "$A/artifacts/v1_lineage/"
cp "$C1/ood_mmvet_218/O1_c1_d3.json" "$C1/ood_mmvet_218/O2_c1_6432.json" \
   "$C1/ood_mmvet_218/O3_b2_d3.json" "$C1/ood_mmvet_218/O4_b2_6432.json" \
   "$A/artifacts/v1_lineage/"

if [[ -d $C1/tier_d_repro_fullcover ]]; then
  cp "$C1"/tier_d_repro_fullcover/*.o0_report.json "$A/o0_reports/" 2>/dev/null || true
  cp "$C1"/tier_d_repro_fullcover/*.json "$A/artifacts/tier_d_repro/" 2>/dev/null || true
fi
TC=$C1/tier_c_interleaved_speed_d
cp "$TC"/tier_c_summary.json "$TC"/tier_c_summary_paired.json "$TC"/tier_c_summary_raw.json \
  "$A/artifacts/tier_c/"
cp "$C1/ood_mmvet_218/ood_offline_analyses.json" "$A/artifacts/ood_offline/" 2>/dev/null || true

# Round-6 re-probe outputs (required for final pack)
RP=$C1/round6_candidate_reprobe
if [[ ! -f $RP/reprobe_summary.json ]]; then
  echo "ERROR: missing $RP/reprobe_summary.json — wait for GPU job 175813855"
  exit 1
fi
cp "$RP"/reprobe_summary.json "$RP"/reprobe_table.tsv "$RP"/reprobe_rows.json \
  "$A/artifacts/round6_reprobe/"

# --- B layer ---
cp "$SRC"/decode/*.py "$B/decode/"
cp "$SRC"/scripts/eval_acceptance_tree.py "$B/scripts/"
cp "$SRC"/scripts/tier_c_interleaved_speed.py "$B/scripts/"
cp "$SRC"/scripts/diag_prefill_mem.py "$B/scripts/"
cp "$SRC"/scripts/reprobe_near_tie_candidate.py "$B/scripts/"
cp "$SRC"/scripts/eval_acceptance.py "$B/scripts/" 2>/dev/null || true
cp "$SRC"/model/*.py "$B/model/"
cp "$SRC"/tests/test_o0_and_ordered.py "$SRC"/tests/test_o0_cli_integration.py \
   "$SRC"/tests/test_integrity_path_norm.py "$SRC"/tests/test_max_new_cap.py \
   "$SRC"/tests/test_max_new_cap_gpu.py "$SRC"/tests/test_dynamic_tree.py \
   "$SRC"/tests/test_tree.py "$B/tests/" 2>/dev/null || true
cp "$SRC"/pbs/tier_d_repro_fullcover.pbs "$SRC"/pbs/tier_d_tier_c_rerun.pbs \
   "$SRC"/pbs/v2_rebaseline_in_domain.pbs "$SRC"/pbs/v2_rebaseline_ood.pbs \
   "$SRC"/pbs/round6_candidate_reprobe.pbs "$B/pbs/"
cp "$SRC/pyproject.toml" "$SRC/RERUN.md" "$SRC/requirements.lock" "$B/"
# B pyproject readme is RERUN.md (already); also expose as README.md
cp "$B/RERUN.md" "$B/README.md"

# refresh pytest + fingerprint
cd "$SRC"
source ~/medusa-env/bin/activate
python -m pytest tests/test_o0_and_ordered.py tests/test_o0_cli_integration.py \
  tests/test_integrity_path_norm.py tests/test_max_new_cap.py \
  --junitxml="$A/pytest_logs/tier_d_negctrl.junit.xml" -q --tb=line \
  | tee "$A/pytest_logs/tier_d_negctrl.log"

# Full-suite skip census. Kept OUTSIDE the payload (payload file list is audited);
# the cover registers the breakdown and cites this path as a run record.
FS=$C1/round6_pytest_full_suite
mkdir -p "$FS"
python -m pytest tests/ -q -rs --tb=no | tee "$FS/full_suite.log"
python3.11 - <<PY
import json, re
from pathlib import Path
txt = Path("$FS/full_suite.log").read_text(encoding="utf-8", errors="replace")
m = re.search(r"(\d+) passed(?:, (\d+) skipped)?", txt)
groups = re.findall(r"^SKIPPED \[(\d+)\] (\S+?): (.*)", txt, re.M)
Path("$FS/full_suite_summary.json").write_text(json.dumps({
    "passed": int(m.group(1)) if m else None,
    "skipped": int(m.group(2)) if (m and m.group(2)) else 0,
    "skip_groups": [{"n": int(n), "loc": loc, "reason": r.strip()} for n, loc, r in groups],
}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("full-suite census ->", "$FS/full_suite_summary.json")
PY

python3.11 - <<PY
import json, platform, socket, torch, transformers
from pathlib import Path
from datetime import datetime
Path("$A/pytest_logs/env_fingerprint.json").write_text(json.dumps({
  "hostname": socket.gethostname(),
  "timestamp": datetime.now().astimezone().isoformat(),
  "platform": platform.platform(),
  "python": platform.python_version(),
  "torch": torch.__version__,
  "transformers": transformers.__version__,
}, indent=2), encoding="utf-8")
PY

cp "$SRC/scripts/scrub_greedy_exact_aliases.py" "$A/scripts/"
/usr/bin/python3.11 "$A/scripts/scrub_greedy_exact_aliases.py" --root "$PKT"

/usr/bin/python3.11 -c "import torch" 2>/dev/null && { echo "torch leaked into python3.11"; exit 1; } || true
/usr/bin/python3.11 "$A/scripts/recompute_all.py" --root "$A" --tol 1e-6 \
  | tee "$A/artifacts/recompute_all_output.txt"

# Integrity chain
/usr/bin/python3.11 "$A/scripts/check_release_integrity.py" --root "$PKT" --write-manifest
/usr/bin/python3.11 "$SRC/scripts/fill_round6_cover.py"
MAN_SHA=$(/usr/bin/python3.11 -c "import hashlib;from pathlib import Path;print(hashlib.sha256(Path('$PKT/FILE_MANIFEST.json').read_bytes()).hexdigest())")
echo "$MAN_SHA" > "$PKT/FILE_MANIFEST.sha256"
/usr/bin/python3.11 - <<PY
import json
from pathlib import Path
root = Path("$PKT")
man = json.loads((root/"FILE_MANIFEST.json").read_text(encoding="utf-8"))
stamp = {
  "packet": str(root),
  "file_manifest_sha256": "$MAN_SHA",
  "n_files": man["n_files"],
  "cover_file": "ROUND6_COVER.md",
  "end_state": "greedy numerical-safety verified / archive byte-reproducibility verified / release bundle reproducible",
  "submitted": True,
}
(root/"SUBMIT_STAMP.json").write_text(json.dumps(stamp, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
print(stamp)
PY

/usr/bin/python3.11 "$A/scripts/check_release_integrity.py" --root "$PKT" \
  | tee "$A/artifacts/integrity_check_output.txt"

echo "PACKED $PKT"
echo "MANIFEST_SHA $MAN_SHA"
