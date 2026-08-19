"""gen_longform_rollouts 单测(CPU,不加载模型):左 pad 正确性(pad 不进
attention)、断点续跑幂等(哈希优先修复)、manifest/record schema、
EOS 口径、连续区间分片。"""
import json
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# scripts/ 只允许 append(scripts/train.py 与 train/ 包同名,见 2026-08-16 事故)
if str(_ROOT / "scripts") not in sys.path:
    sys.path.append(str(_ROOT / "scripts"))

from gen_longform_rollouts import (  # noqa: E402
    EFFECTIVE_VOCAB,
    assert_left_padded,
    build_manifest,
    code_commit,
    iter_batches,
    make_record,
    repair_and_load,
    scan_shard_file,
    shard_range,
    token_sha256,
    trim_new_tokens,
    validate_record,
    verify_pool,
)

PAD = 151643
EOS = 151645


# ---------------------------------------------------------------- left-pad


def test_left_pad_ok():
    ids = torch.tensor([[PAD, PAD, 5, 6], [1, 2, 3, 4]])
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    assert_left_padded(ids, mask, PAD)  # no raise


def test_right_pad_rejected():
    ids = torch.tensor([[5, 6, PAD, PAD]])
    mask = torch.tensor([[1, 1, 0, 0]])
    with pytest.raises(AssertionError):
        assert_left_padded(ids, mask, PAD)


def test_interior_pad_rejected():
    ids = torch.tensor([[5, PAD, 6, 7]])
    mask = torch.tensor([[1, 0, 1, 1]])
    with pytest.raises(AssertionError):
        assert_left_padded(ids, mask, PAD)


def test_pad_inside_attention_rejected():
    # mask==0 位置上不是 pad token:pad 语义被破坏,必须拒绝
    ids = torch.tensor([[7, PAD, 5, 6]])
    mask = torch.tensor([[0, 0, 1, 1]])
    with pytest.raises(AssertionError):
        assert_left_padded(ids, mask, PAD)


def test_fully_masked_row_rejected():
    ids = torch.tensor([[PAD, PAD]])
    mask = torch.tensor([[0, 0]])
    with pytest.raises(AssertionError):
        assert_left_padded(ids, mask, PAD)


# ---------------------------------------------------------------- EOS 口径


def test_trim_includes_first_eos():
    toks, hit = trim_new_tokens([11, 22, EOS, PAD, PAD], EOS)
    assert toks == [11, 22, EOS] and hit is True


def test_trim_no_eos_registers_false():
    toks, hit = trim_new_tokens([11, 22, 33], EOS)
    assert toks == [11, 22, 33] and hit is False


def test_trim_eos_first_position():
    toks, hit = trim_new_tokens([EOS, PAD], EOS)
    assert toks == [EOS] and hit is True


# ---------------------------------------------------------------- record/manifest


def test_record_roundtrip_valid():
    rec = make_record(3, "docci/train_x", [1, 2, EOS], True)
    assert validate_record(rec)
    assert rec["n_tokens"] == 3
    assert rec["sha256"] == token_sha256([1, 2, EOS])


def test_record_tamper_detected():
    rec = make_record(3, "docci/train_x", [1, 2, EOS], True)
    rec["tokens"][0] = 9
    assert not validate_record(rec)


def test_record_schema_violations():
    rec = make_record(0, "s", [1], False)
    bad_missing = {k: v for k, v in rec.items() if k != "sha256"}
    assert not validate_record(bad_missing)
    bad_extra = dict(rec, extra=1)
    assert not validate_record(bad_extra)
    bad_count = dict(rec, n_tokens=2)
    assert not validate_record(bad_count)
    assert not validate_record(dict(rec, idx=-1))
    assert not validate_record(dict(rec, tokens=[]))
    assert not validate_record("not a dict")


def test_manifest_schema_and_complete_flag():
    sf = {"rollouts_shard_00000.jsonl": {"n": 2, "sha256": "ab"},
          "rollouts_shard_00001.jsonl": {"n": 1, "sha256": "cd"}}
    m = build_manifest(sf, total=3, config={"max_pixels": 501760})
    assert m["complete"] is True and m["n_cached"] == 3
    assert m["config"]["max_pixels"] == 501760
    m2 = build_manifest(sf, total=4, config={})
    assert m2["complete"] is False


# ---------------------------------------------------------------- 续跑幂等


def _write_lines(path, recs, tail=b""):
    with open(path, "wb") as f:
        for r in recs:
            f.write(json.dumps(r).encode() + b"\n")
        f.write(tail)


def test_resume_clean_file_untouched(tmp_path):
    fp = tmp_path / "rollouts_shard_00000.jsonl"
    recs = [make_record(i, f"s{i}", [1, 2 + i, EOS], True) for i in range(3)]
    _write_lines(fp, recs)
    before = fp.read_bytes()
    done = repair_and_load(fp)
    assert done == {0, 1, 2}
    assert fp.read_bytes() == before  # 幂等:完好文件字节不变


def test_resume_truncated_tail_dropped_then_idempotent(tmp_path):
    fp = tmp_path / "rollouts_shard_00000.jsonl"
    recs = [make_record(i, f"s{i}", [1, EOS], True) for i in range(2)]
    half = json.dumps(make_record(2, "s2", [1, EOS], True)).encode()[:20]
    _write_lines(fp, recs, tail=half)  # 模拟写到一半被杀
    done = repair_and_load(fp)
    assert done == {0, 1}
    repaired = fp.read_bytes()
    valid, dirty = scan_shard_file(fp)
    assert not dirty and len(valid) == 2
    assert repair_and_load(fp) == {0, 1}
    assert fp.read_bytes() == repaired  # 第二次扫描不再改写


def test_resume_bad_sha_and_duplicate_dropped(tmp_path):
    fp = tmp_path / "rollouts_shard_00000.jsonl"
    good = make_record(0, "s0", [1, EOS], True)
    tampered = dict(make_record(1, "s1", [1, EOS], True), sha256="0" * 64)
    dup = make_record(0, "s0", [9, EOS], True)
    _write_lines(fp, [good, tampered, dup])
    done = repair_and_load(fp)
    assert done == {0}
    valid, dirty = scan_shard_file(fp)
    assert not dirty and [r["idx"] for r in valid] == [0]
    assert valid[0]["tokens"] == [1, EOS]  # 重复 idx 保首行


def test_resume_two_phase_completion(tmp_path):
    fp = tmp_path / "rollouts_shard_00000.jsonl"
    all_recs = [make_record(i, f"s{i}", [i + 1, EOS], True) for i in range(5)]
    _write_lines(fp, all_recs[:2])
    done = repair_and_load(fp)
    todo = [i for i in range(5) if i not in done]
    assert todo == [2, 3, 4]
    with open(fp, "a") as f:
        for i in todo:
            f.write(json.dumps(all_recs[i]) + "\n")
    assert repair_and_load(fp) == {0, 1, 2, 3, 4}
    valid, dirty = scan_shard_file(fp)
    assert not dirty and [r["idx"] for r in valid] == list(range(5))


# ---------------------------------------------------------------- 分片/分批


def test_shard_range_partition():
    total, k = 37079, 8
    ranges = [shard_range(total, k, i) for i in range(k)]
    assert ranges[0].start == 0 and ranges[-1].stop == total
    for a, b in zip(ranges, ranges[1:]):
        assert a.stop == b.start  # 连续无缝
    sizes = [len(r) for r in ranges]
    assert max(sizes) - min(sizes) <= 1
    with pytest.raises(ValueError):
        shard_range(total, k, k)


def test_iter_batches():
    assert list(iter_batches(range(5), 2)) == [[0, 1], [2, 3], [4]]
    assert list(iter_batches([], 2)) == []


def test_effective_vocab_matches_decode_common():
    from decode.common import EFFECTIVE_VOCAB as EV
    assert EFFECTIVE_VOCAB == EV


# ---------------------------------------------------------------- 审查修订(2026-08-20)


def test_verify_pool_assertions_trigger(tmp_path):
    import hashlib
    fp = tmp_path / "triplets.jsonl"
    fp.write_bytes(b'{"a":1}\n{"a":2}\n')
    rows = [{"a": 1}, {"a": 2}]
    good_sha = hashlib.sha256(fp.read_bytes()).hexdigest()
    verify_pool(fp, rows, expect_total=2, expect_sha256=good_sha)  # no raise
    verify_pool(fp, rows)  # 两断言均可选
    with pytest.raises(ValueError):
        verify_pool(fp, rows, expect_total=3)
    with pytest.raises(ValueError):
        verify_pool(fp, rows, expect_total=2, expect_sha256="0" * 64)


def test_code_commit_field_in_manifest_config():
    c = code_commit()
    assert c != "unknown" and all(ch in "0123456789abcdef" for ch in c)
    m = build_manifest({}, 0, {"code_commit": c})
    assert m["config"]["code_commit"] == c
