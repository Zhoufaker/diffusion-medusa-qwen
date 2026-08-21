"""CPU 单测:extract_ctx_longform 适配层(登录节点可跑,无 GPU/模型)。"""
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.append(str(_ROOT / "scripts"))  # 纪律:append

from gen_longform_rollouts import make_record  # noqa: E402
from extract_ctx_longform import (  # noqa: E402
    SHARD_SIZE,
    TARGET_LAYER_IDS,
    flush_shard,
    load_locked_rollouts,
    make_spans,
    merge_manifest,
    mismatch_head16,
    sha256_file,
)


def _write_locked_dir(tmp_path, n=3):
    """构造最小合法锁定资产:1 个 shard JSONL + manifest(complete=true)。"""
    fp = tmp_path / "rollouts_shard_00000.jsonl"
    with open(fp, "w") as fo:
        for i in range(n):
            rec = make_record(i, f"s{i}", [10 + i, 11 + i, 151645], True)
            fo.write(json.dumps(rec) + "\n")
    manifest = {"kind": "longform_rollouts_v2", "total": n, "n_cached": n,
                "complete": True, "config": {"code_commit": "deadbee"},
                "shard_files": {fp.name: {"n": n, "sha256": sha256_file(fp)}}}
    json.dump(manifest, open(tmp_path / "manifest.json", "w"))
    return fp


def test_layer_ids_match_w1():
    assert TARGET_LAYER_IDS == [1, 7, 13, 19, 25]
    assert SHARD_SIZE == 256


def test_load_locked_rollouts_ok(tmp_path):
    _write_locked_dir(tmp_path, n=3)
    by_idx, lineage = load_locked_rollouts(tmp_path, expect_total=3)
    assert sorted(by_idx) == [0, 1, 2]
    assert by_idx[1]["tokens"] == [11, 12, 151645]
    assert lineage["rollout_code_commit"] == "deadbee"
    assert len(lineage["rollout_manifest_sha256"]) == 64


def test_load_locked_rejects_tampered_row(tmp_path):
    fp = _write_locked_dir(tmp_path, n=3)
    lines = open(fp).read().splitlines()
    rec = json.loads(lines[1])
    rec["tokens"][0] += 1                      # 换 token,不改行内 sha → 行级先验必拒
    lines[1] = json.dumps(rec)
    fp.write_text("\n".join(lines) + "\n")
    m = json.load(open(tmp_path / "manifest.json"))
    m["shard_files"][fp.name]["sha256"] = sha256_file(fp)  # 文件 sha 同步,孤立行级校验
    json.dump(m, open(tmp_path / "manifest.json", "w"))
    with pytest.raises(ValueError, match="validate_record"):
        load_locked_rollouts(tmp_path, expect_total=3)


def test_load_locked_rejects_file_and_state(tmp_path):
    fp = _write_locked_dir(tmp_path, n=3)
    with open(fp, "a") as fo:                  # 追加一行,文件 sha 与 manifest 失配
        fo.write(json.dumps(make_record(9, "s9", [7, 151645], True)) + "\n")
    with pytest.raises(ValueError, match="sha256"):
        load_locked_rollouts(tmp_path)
    m = json.load(open(tmp_path / "manifest.json"))
    m["complete"] = False                      # 未收官资产拒读
    json.dump(m, open(tmp_path / "manifest.json", "w"))
    with pytest.raises(ValueError, match="complete"):
        load_locked_rollouts(tmp_path)


def test_make_spans_longform_structure():
    pad = 151655
    # prompt = [模板 2 tok, 图像 pad ×4, 问题 3 tok],P=9,L=5
    ids = torch.tensor([1, 2, pad, pad, pad, pad, 5, 6, 7, 90, 91, 92, 93, 94])
    spans = make_spans(ids, P=9, L=5, image_pad_id=pad)
    assert spans.tolist() == [2, 6, 0, 9, 9, 14]
    assert spans.dtype == torch.int64
    with pytest.raises(ValueError, match="vision"):   # longform 全池有图,缺图即停
        make_spans(torch.tensor([1, 2, 3]), P=3, L=0, image_pad_id=pad)


# ------------------------------------------------------------ 审查修订(2026-08-21)


def test_merge_manifest_two_ranges_complete():
    def mk(lo, hi, shard):
        return {"samples": {str(i): {"idx": i} for i in range(lo, hi)},
                "shards": {shard: f"shard_{shard:>05}.safetensors"},
                "range": [lo, hi], "aggregate": {"n_done": hi - lo,
                                                 "run_range": [lo, hi]}}
    a, b = mk(0, 2, 0), mk(2, 4, 1)
    m = merge_manifest(a, b)
    assert sorted(m["samples"]) == ["0", "1", "2", "3"]     # 索引完整无冲突
    assert m["range"] == [0, 4]
    assert sorted(m["shards"]) == ["0", "1"]
    assert [r["run_range"] for r in m["aggregate_runs"]] == [[0, 2], [2, 4]]
    assert merge_manifest(None, a) is a                     # 首跑直通
    m2 = merge_manifest(m, mk(2, 4, 1))                     # 同段重跑=幂等覆盖
    assert sorted(m2["samples"]) == ["0", "1", "2", "3"]


def test_mismatch_head16_schema_and_truncation():
    L, V = 40, 8
    exp = torch.zeros(L, dtype=torch.int64)
    pred = torch.zeros(L, dtype=torch.int64)
    pred[10:30] = 1                                         # 20 处失配 > cap 16
    logits = torch.zeros(L, V)
    logits[torch.arange(L), pred] = 1.5                     # gap = 1.5
    out = mismatch_head16(pred, exp, logits)
    assert len(out) == 16                                   # 截断到前 16
    assert [e["pos"] for e in out] == list(range(10, 26))   # W2 读取键 "pos"
    assert all(set(e) == {"pos", "gap"} and abs(e["gap"] - 1.5) < 1e-6
               for e in out)                                # 35K 缓存同 schema
    assert mismatch_head16(exp, exp, logits) == []          # 全对 → 空列表


def test_shard_round_trip(tmp_path):
    from safetensors.torch import load_file
    buf = {}
    for i in (256, 257):
        ctx = torch.randn(5, 4, 8).to(torch.bfloat16)
        ids = torch.arange(4, dtype=torch.int64)
        spans = torch.tensor([0, 2, 0, 3, 3, 4], dtype=torch.int64)
        buf[i] = (ctx, ids, spans)
    name = flush_shard(tmp_path, 1, buf, partial=True)
    assert name == "shard_00001.safetensors"
    back = load_file(tmp_path / name)
    assert sorted(back) == ["256.ctx", "256.ids", "256.spans",
                            "257.ctx", "257.ids", "257.spans"]
    assert torch.equal(back["257.ctx"], buf[257][0])
    assert torch.equal(back["256.spans"], buf[256][2])
    assert back["256.ctx"].dtype == torch.bfloat16
