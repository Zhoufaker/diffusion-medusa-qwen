#!/bin/bash
# Assemble gpt56_round4_release A/B layers. Run AFTER Tier D jobs finish.
set -euo pipefail
SRC="${HOME}/medusa-qwen"
C1=/scratch/li96/mz9869/medusa_outputs/linked_medusa_c1_eagle
PKT=$C1/gpt56_round4_release
A=$PKT/A_layer_no_torch
B=$PKT/B_layer_gpu_rerun

rm -rf "$PKT"
mkdir -p "$A"/{artifacts/v2_rebaseline,artifacts/tier_c,artifacts/tier_d_repro,artifacts/ood_offline,o0_reports,archives,pytest_logs,scripts}
mkdir -p "$B"/{decode,scripts,model,tests,pbs}

# --- A: scripts ---
cp "$SRC/scripts/recompute_all.py" "$A/scripts/"
cp "$SRC/scripts/assert_cover_from_manifest.py" "$A/scripts/"

# --- A: v2 raw evaluator JSONs + O0 reports ---
for f in "$C1"/v2_rebaseline/*.json; do
  bn=$(basename "$f")
  case "$bn" in
    *.spec_archive.json|*.greedy_archive.json) cp "$f" "$A/archives/" ;;
    *.o0_report.json) cp "$f" "$A/o0_reports/v2_$bn" ;;
    *.o0_prov.json) cp "$f" "$A/archives/" ;;
    *) cp "$f" "$A/artifacts/v2_rebaseline/" ;;
  esac
done

# v1 lineage raws needed for Δ recompute (frozen)
mkdir -p "$A/artifacts/v1_lineage"
cp "$C1/dynamic_sweep_300/dyn_k8_n32.json" "$A/artifacts/v1_lineage/" 
cp "$C1/bridge_300/c1_6432.json" "$A/artifacts/v1_lineage/bridge300_c1_6432.json"
cp "$C1/fanout_sweep/c10_6432_32.json" "$A/artifacts/v1_lineage/"
cp "$C1/final_eval/c1_d3.json" "$C1/final_eval/b2_wide.json" "$A/artifacts/v1_lineage/" 2>/dev/null || true
cp "$C1/bridge_300/c1_d3.json" "$C1/bridge_300/b2_d3.json" "$C1/bridge_300/b2_6432.json" \
   "$A/artifacts/v1_lineage/" 2>/dev/null || true
cp "$C1/dynamic_sweep_300/dyn_k8_n24.json" "$A/artifacts/v1_lineage/" 2>/dev/null || true
cp "$C1/ood_mmvet_218/O1_c1_d3.json" "$C1/ood_mmvet_218/O2_c1_6432.json" \
   "$C1/ood_mmvet_218/O3_b2_d3.json" "$C1/ood_mmvet_218/O4_b2_6432.json" \
   "$A/artifacts/v1_lineage/" 2>/dev/null || true
# first Tier C (official absolute spd cites) + D hash rerun
mkdir -p "$A/artifacts/tier_c_official_175738321"
cp "$C1/tier_c_interleaved_speed/tier_c_summary.json" \
   "$A/artifacts/tier_c_official_175738321/" 2>/dev/null || true

# Tier D repro reports if present
if [[ -d $C1/tier_d_repro_fullcover ]]; then
  cp "$C1"/tier_d_repro_fullcover/*.o0_report.json "$A/o0_reports/" 2>/dev/null || true
  cp "$C1"/tier_d_repro_fullcover/repro_summary.json "$A/artifacts/tier_d_repro/" 2>/dev/null || true
  cp "$C1"/tier_d_repro_fullcover/*.json "$A/artifacts/tier_d_repro/" 2>/dev/null || true
fi

# Tier C (prefer D rerun)
TC=$C1/tier_c_interleaved_speed_d
[[ -d $TC ]] || TC=$C1/tier_c_interleaved_speed
cp "$TC"/tier_c_summary.json "$A/artifacts/tier_c/"
cp "$TC"/tier_c_summary_paired.json "$A/artifacts/tier_c/" 2>/dev/null || true
cp "$TC"/tier_c_summary_raw.json "$A/artifacts/tier_c/" 2>/dev/null || true

# OOD offline / capability inputs
cp "$C1/ood_mmvet_218/ood_offline_analyses.json" "$A/artifacts/ood_offline/" 2>/dev/null || true

# pytest log
cp /scratch/li96/mz9869/medusa_outputs/linked_medusa_c1_eagle/tier_d_meta/pytest_full.log "$A/pytest_logs/test_o0_and_max_new_cap.log" 2>/dev/null || true
# env fingerprint sidecar
python3.11 - <<PY
import json, os, platform, socket
from pathlib import Path
p = Path("$A/pytest_logs/env_fingerprint.json")
try:
    import torch, transformers
    tv, tr = torch.__version__, transformers.__version__
except Exception:
    tv = tr = None
p.write_text(json.dumps({
    "hostname": socket.gethostname(),
    "platform": platform.platform(),
    "python": platform.python_version(),
    "torch": tv,
    "transformers": tr,
    "pbs_jobs": {
        "v2_in": "175680071", "v2_ood": "175680072",
        "tier_c_v1": "175738321",
        "tier_d_repro": os.environ.get("TIER_D_REPRO_JOB"),
        "tier_d_tc": os.environ.get("TIER_D_TC_JOB"),
    },
}, indent=2))
PY

# --- B: executable tree ---
cp "$SRC"/decode/*.py "$B/decode/"
cp "$SRC"/scripts/eval_acceptance_tree.py "$B/scripts/"
cp "$SRC"/scripts/tier_c_interleaved_speed.py "$B/scripts/"
cp "$SRC"/scripts/eval_acceptance.py "$B/scripts/" 2>/dev/null || true
cp -r "$SRC"/model/*.py "$B/model/"
cp "$SRC"/tests/test_o0_and_ordered.py "$SRC"/tests/test_max_new_cap.py \
   "$SRC"/tests/test_max_new_cap_gpu.py "$SRC"/tests/test_dynamic_tree.py \
   "$SRC"/tests/test_tree.py "$B/tests/" 2>/dev/null || true
cp "$SRC"/pbs/tier_d_repro_fullcover.pbs "$SRC"/pbs/tier_d_tier_c_rerun.pbs \
   "$SRC"/pbs/v2_rebaseline_in_domain.pbs "$SRC"/pbs/v2_rebaseline_ood.pbs \
   "$B/pbs/"
/home/562/mz9869/medusa-env/bin/pip freeze > "$B/requirements.lock" 2>/dev/null \
  || pip freeze > "$B/requirements.lock"

# FILE_MANIFEST over whole packet
python3.11 - <<'PY'
import json, hashlib
from pathlib import Path
root = Path("/scratch/li96/mz9869/medusa_outputs/linked_medusa_c1_eagle/gpt56_round4_release")
files = []
for p in sorted(root.rglob("*")):
    if p.is_file() and p.name not in ("FILE_MANIFEST.json", "COVER_CHECK.json", "README.md"):
        files.append({
            "path": str(p.relative_to(root)),
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        })
(root/"FILE_MANIFEST.json").write_text(json.dumps({"n_files": len(files), "files": files}, indent=2))
print("manifest files", len(files))
PY

# Generate README cover from manifest
python3.11 - <<'PY'
import json
from pathlib import Path
root = Path("/scratch/li96/mz9869/medusa_outputs/linked_medusa_c1_eagle/gpt56_round4_release")
man = json.loads((root/"FILE_MANIFEST.json").read_text())
# group
a = [f["path"] for f in man["files"] if f["path"].startswith("A_layer_no_torch/")]
b = [f["path"] for f in man["files"] if f["path"].startswith("B_layer_gpu_rerun/")]
lines = []
lines.append("# GPT-5.6 Round-4 Release — Linked Medusa (v2 / Tier D)")
lines.append("")
lines.append("**Official cites (unchanged numerically):** σ champ **2.825**; R **0.433**;")
lines.append("speed **并列/未分** (Tier C). Numbers were not recomputed — verdict semantics +")
lines.append("reproducibility evidence + packet layout only.")
lines.append("")
lines.append("## Layer contract")
lines.append("")
lines.append("- **A_layer_no_torch**: zero-dependency independent recompute of all official numbers")
lines.append("  (`python3 A_layer_no_torch/scripts/recompute_all.py --root A_layer_no_torch`).")
lines.append("- **B_layer_gpu_rerun**: end-to-end path requiring V100 + torch (decode/scripts/model/tests/pbs).")
lines.append("")
lines.append("## Official claim (O0 vs independent greedy)")
lines.append("")
lines.append("Algorithmic greedy lossless; implementation-level no hard/material divergence;")
lines.append("near_tie calibrated. **Not** byte-equivalent vs independent greedy.")
lines.append("Byte-exact reserved for archive reproducibility (`archive_gate_status=PASS`).")
lines.append("")
lines.append("<!-- COVER_START -->")
lines.append("## Packet contents (auto from FILE_MANIFEST)")
lines.append("")
lines.append("### A_layer_no_torch")
for p in a:
    lines.append(f"- `{p}`")
lines.append("")
lines.append("### B_layer_gpu_rerun")
for p in b:
    lines.append(f"- `{p}`")
lines.append("<!-- COVER_END -->")
lines.append("")
lines.append("## Verify")
lines.append("")
lines.append("```bash")
lines.append("python3 A_layer_no_torch/scripts/recompute_all.py --root A_layer_no_torch")
lines.append("python3 A_layer_no_torch/scripts/assert_cover_from_manifest.py --root .")
lines.append("```")
lines.append("")
lines.append("Advisor third version: held until round-4 pass.")
(root/"README.md").write_text("\n".join(lines)+"\n")
print("README written", len(lines), "lines")
PY

# A-layer verify: system python3.11 without torch
/usr/bin/python3.11 -c "import torch" 2>/dev/null && { echo "ERROR: torch present in python3.11"; exit 1; } || true
/usr/bin/python3.11 "$A/scripts/recompute_all.py" --root "$A" --tol 1e-6
/usr/bin/python3.11 "$A/scripts/assert_cover_from_manifest.py" --root "$PKT"

# refresh manifest to include COVER_CHECK + README
python3.11 - <<'PY'
import json, hashlib
from pathlib import Path
root = Path("/scratch/li96/mz9869/medusa_outputs/linked_medusa_c1_eagle/gpt56_round4_release")
files = []
for p in sorted(root.rglob("*")):
    if p.is_file() and p.name != "FILE_MANIFEST.json":
        files.append({
            "path": str(p.relative_to(root)),
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        })
(root/"FILE_MANIFEST.json").write_text(json.dumps({"n_files": len(files), "files": files}, indent=2))
print("final manifest files", len(files))
PY

echo "PACKED $PKT"
