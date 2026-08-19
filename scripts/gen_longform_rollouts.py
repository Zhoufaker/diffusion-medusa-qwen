"""gen_longform_rollouts.py — D1.5 v2 批量 rollout 生成器(必审队列 ①)。

管线位置:triplets.jsonl + 图像 shard tar → 逐 shard rollout JSONL
(只存 token;特征由队列 ② 的 W1 模板适配层另跑)。

谱系原则(artifact 锁定制):
    锁定对象是【生成产物本身】——逐条 token 序列 + 逐条 sha256 入 manifest,
    而非"任意批组成下的逐位可复现"。批量 HF generate 与 bs=1 逐步前向
    (gen_cache_rollout.rollout_one)之间、以及不同 batch 组成之间的 fp16
    数值扰动属【已知类别】:如实登记,不视为缺陷。产物一经生成即以
    sha256 锁定,下游(特征抽取/训练)只认 manifest 内的锁定序列。

与 bs=1 谱系口径的三处显式对齐(缺一即分叉,审查重点):
    1. repetition_penalty=1.0 —— 模型 generation_config 自带 1.05,
       bs=1 路径是裸 argmax,必须显式压回;
    2. eos_token_id=tokenizer.eos_token_id(151645)—— generation_config
       默认还会停 151643,bs=1 只停 151645;首个 EOS【计入】保存序列
       (与 rollout_one 一致);无 EOS 达 max_new → eos_hit=false 登记;
    3. 幻影词表屏蔽 —— lm_head 152064 > 有效 151936,bs=1 走 argmax_masked;
       批量路径用等价 LogitsProcessor(ids >= EFFECTIVE_VOCAB → -inf)。

PREPROCESS_SPEC(2026-08-20 裁决):apply_max_pixels(processor, 501760),
与 longform 特征抽取/评估三处同 spec,MM-Vet OOD 锚定同源。

分片:按行号【连续区间】切分(tar 局部性;shard_{i//1000:05d}.tar 顺序读),
PBS array 每 shard 独占输出文件 rollouts_shard_{shard_id:05d}.jsonl。
与 gen_cache_rollout 的 idx%K 方案不同——登记差异:37K 散 .pt 会触
inode 纪律(项目 inode 警戒中),JSONL 每文件 ~千行。

断点续跑(哈希优先,fixation 事故 eeefe14 教训):启动先扫自己 shard 的
JSONL——截断尾行丢弃、每行 sha256 对 token 复验、坏行/重复行剔除,
仅在确有修复时重写文件;已完成 idx 跳过。幂等:完整且全验通过的
shard 文件,重复运行字节不变。

用法(PBS 草案 pbs/d15_gen_rollouts.pbs;预算见送审件):
    python scripts/gen_longform_rollouts.py \
        --triplets .../longform_fixed_v2/triplets.jsonl \
        --images-tar-dir .../longform_fixed_v2/images \
        --out-dir .../longform_fixed_v2/rollouts_px501760 \
        --shard-id $PBS_ARRAY_INDEX --num-shards K
    收尾: --manifest-only 单跑,全片修复+校验+写总 manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
from io import BytesIO
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

EFFECTIVE_VOCAB = 151936          # = decode.common.EFFECTIVE_VOCAB(单测校验一致)
ROWS_PER_TAR = 1000               # v2 fixation 图像分片约定 shard_{i//1000:05d}.tar
MAX_PIXELS_SPEC = 501760          # longform PREPROCESS_SPEC(2026-08-20 裁决)
RECORD_KEYS = ("idx", "sample_id", "n_tokens", "eos_hit", "tokens", "sha256")


# ----------------------------------------------------------------------------
# 纯函数层(CPU 单测覆盖:tests/test_gen_longform_rollouts.py)
# ----------------------------------------------------------------------------


def token_sha256(tokens) -> str:
    """sha256 over int64-LE bytes of the token sequence (manifest 锁定键)。"""
    h = hashlib.sha256()
    for t in tokens:
        h.update(int(t).to_bytes(8, "little", signed=True))
    return h.hexdigest()


def make_record(idx: int, sample_id: str, tokens, eos_hit: bool) -> dict:
    return {"idx": int(idx), "sample_id": sample_id, "n_tokens": len(tokens),
            "eos_hit": bool(eos_hit), "tokens": [int(t) for t in tokens],
            "sha256": token_sha256(tokens)}


def validate_record(rec) -> bool:
    """Schema + 哈希优先完整性:字段齐、类型对、n_tokens 一致、sha256 复验。"""
    if not isinstance(rec, dict) or tuple(sorted(rec)) != tuple(sorted(RECORD_KEYS)):
        return False
    if not (isinstance(rec["idx"], int) and rec["idx"] >= 0
            and isinstance(rec["sample_id"], str)
            and isinstance(rec["eos_hit"], bool)
            and isinstance(rec["tokens"], list) and rec["tokens"]
            and all(isinstance(t, int) for t in rec["tokens"])):
        return False
    if rec["n_tokens"] != len(rec["tokens"]):
        return False
    return rec["sha256"] == token_sha256(rec["tokens"])


def scan_shard_file(path: Path):
    """-> (valid_records, dirty)。截断尾行/坏行/sha 不符/重复 idx 均判 dirty。"""
    if not Path(path).exists():
        return [], False
    raw = Path(path).read_bytes()
    valid, seen, dirty = [], set(), False
    segs = raw.split(b"\n")
    if segs and segs[-1] == b"":
        segs = segs[:-1]          # 规范收尾:最后一行后的换行
    for seg in segs:
        try:
            rec = json.loads(seg)
        except ValueError:
            dirty = True          # 截断/损坏行:丢弃
            continue
        if not validate_record(rec) or rec["idx"] in seen:
            dirty = True
            continue
        seen.add(rec["idx"])
        valid.append(rec)
    return valid, dirty


def repair_and_load(path: Path) -> set:
    """哈希优先续跑入口:仅确有修复时重写(幂等——完好文件字节不变)。"""
    path = Path(path)
    valid, dirty = scan_shard_file(path)
    if dirty:
        tmp = path.with_name(path.name + ".repair")
        with open(tmp, "w") as f:
            for rec in valid:
                f.write(json.dumps(rec) + "\n")
        os.replace(tmp, path)
        print(f"[resume] {path.name}: repaired -> {len(valid)} valid records kept",
              flush=True)
    return {r["idx"] for r in valid}


def shard_range(total: int, num_shards: int, shard_id: int) -> range:
    """连续区间切分(tar 局部性),块大小差 ≤1。"""
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"shard_id {shard_id} out of range 0..{num_shards - 1}")
    base, rem = divmod(total, num_shards)
    lo = shard_id * base + min(shard_id, rem)
    return range(lo, lo + base + (1 if shard_id < rem else 0))


def iter_batches(indices, batch_size: int):
    buf = []
    for i in indices:
        buf.append(i)
        if len(buf) == batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


def assert_left_padded(input_ids, attention_mask, pad_id: int) -> None:
    """pad 不进 attention:各行 mask 单调 0…0 1…1(左 pad),mask==0 处必为 pad。"""
    m = attention_mask
    if bool((m[:, 1:] < m[:, :-1]).any()):
        raise AssertionError("padding not strictly on the left (mask not 0..1 monotone)")
    if bool((m.sum(dim=1) == 0).any()):
        raise AssertionError("fully-masked row")
    if bool((input_ids[m == 0] != pad_id).any()):
        raise AssertionError("non-pad token at masked position")


def trim_new_tokens(row, eos_id: int):
    """generate() 输出 prompt 之后的切片 → (tokens, eos_hit)。
    首个 EOS 计入(与 gen_cache_rollout.rollout_one 口径一致)。"""
    toks = [int(t) for t in row]
    if eos_id in toks:
        k = toks.index(eos_id)
        return toks[: k + 1], True
    return toks, False


def verify_pool(triplets_path, rows, expect_total=None, expect_sha256=None) -> None:
    """加载后断言(审查修订 2026-08-20,不符即停):
    - 行数 == --expect-total(PBS 传 37079);
    - triplets.jsonl 全文 sha256 == --triplets-sha256(等价内在键——
      jsonl 无 idx 列,行号 ↔ 图像 tar 成员 {i:06d} 的绑定由字节级锁定
      保障,文件重排/截断/换版一律拦下)。"""
    if expect_total is not None and len(rows) != expect_total:
        raise ValueError(f"pool size {len(rows)} != expected {expect_total}")
    if expect_sha256:
        h = hashlib.sha256(Path(triplets_path).read_bytes()).hexdigest()
        if h != expect_sha256:
            raise ValueError(
                f"triplets sha256 {h[:16]}... != expected {expect_sha256[:16]}...")


def code_commit() -> str:
    """manifest 谱系字段:生成时代码 commit(取本文件所在仓库;无 git 时 unknown)。"""
    try:
        import subprocess
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, capture_output=True,
            text=True, timeout=10, check=True).stdout.strip()
    except Exception:
        return "unknown"


def build_manifest(shard_files: dict, total: int, config: dict) -> dict:
    """shard_files: {name: {"n": int, "sha256": str}}。"""
    n = sum(v["n"] for v in shard_files.values())
    return {"kind": "longform_rollouts_v2", "total": total, "n_cached": n,
            "complete": n >= total, "config": config,
            "shard_files": shard_files,
            "token_hash": "sha256(int64-LE per token)"}


# ----------------------------------------------------------------------------
# 数据/模型层(GPU 路径,不进单测)
# ----------------------------------------------------------------------------


class TarImageReader:
    """顺序友好的 fixation 图像分片读取器(连续区间分片下逐 tar 推进)。"""

    def __init__(self, tar_dir):
        self.tar_dir = Path(tar_dir)
        self._sid = None
        self._tf = None
        self._members = None

    def get(self, idx: int):
        from PIL import Image
        sid = idx // ROWS_PER_TAR
        if sid != self._sid:
            if self._tf is not None:
                self._tf.close()
            self._tf = tarfile.open(self.tar_dir / f"shard_{sid:05d}.tar")
            self._members = {m.name.split(".")[0]: m for m in self._tf.getmembers()
                             if not m.name.endswith(".json")}
            self._sid = sid
        m = self._members[f"{idx:06d}"]
        return Image.open(BytesIO(self._tf.extractfile(m).read())).convert("RGB")


def generate_batch(base, processor, questions, images, max_new, eos_id, pad_id,
                   device, logits_processor):
    """左 pad 批量 greedy generate → [(tokens, eos_hit)],口径对齐见模块 docstring。"""
    import torch
    texts = []
    for q in questions:
        messages = [{"role": "user",
                     "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        texts.append(processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True))
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    assert_left_padded(inputs["input_ids"], inputs["attention_mask"], pad_id)
    inputs = inputs.to(device)
    with torch.no_grad():
        out = base.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                            num_beams=1, repetition_penalty=1.0,
                            eos_token_id=eos_id, pad_token_id=pad_id,
                            use_cache=True, logits_processor=logits_processor)
    p_len = inputs["input_ids"].shape[1]
    return [trim_new_tokens(out[b, p_len:], eos_id) for b in range(out.shape[0])]


def finalize_manifest(out_dir: Path, total: int, config: dict) -> bool:
    shard_files = {}
    for fp in sorted(out_dir.glob("rollouts_shard_*.jsonl")):
        done = repair_and_load(fp)
        shard_files[fp.name] = {"n": len(done),
                                "sha256": hashlib.sha256(fp.read_bytes()).hexdigest()}
    manifest = build_manifest(shard_files, total, config)
    json.dump(manifest, open(out_dir / "manifest.json", "w"), indent=1)
    print(f"[manifest] n_cached={manifest['n_cached']} total={total} "
          f"complete={manifest['complete']}")
    return manifest["complete"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplets",
                    default="/scratch/li96/mz9869/dflash_data/longform_fixed_v2/triplets.jsonl")
    ap.add_argument("--images-tar-dir",
                    default="/scratch/li96/mz9869/dflash_data/longform_fixed_v2/images")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS_SPEC)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="cap total rows (smoke)")
    ap.add_argument("--expect-total", type=int, default=None,
                    help="断言全池行数(PBS 传 37079),不符即停")
    ap.add_argument("--triplets-sha256", default=None,
                    help="断言 triplets.jsonl 全文 sha256(等价内在键),不符即停")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--manifest-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(args.triplets)]
    verify_pool(args.triplets, rows, args.expect_total, args.triplets_sha256)
    if args.limit is not None:
        rows = rows[: args.limit]
    total = len(rows)
    config = {"model_id": args.model_id, "max_pixels": args.max_pixels,
              "max_new": args.max_new, "batch_size": args.batch_size,
              "num_shards": args.num_shards, "sharding": "contiguous",
              "greedy": True, "repetition_penalty": 1.0,
              "eos": "tokenizer.eos_token_id only, first EOS included",
              "phantom_mask": f">= {EFFECTIVE_VOCAB} suppressed",
              "code_commit": code_commit()}

    if args.manifest_only:
        return 0 if finalize_manifest(out_dir, total, config) else 1

    import torch
    from transformers.generation.logits_process import (LogitsProcessor,
                                                        LogitsProcessorList)
    import decode.common as C

    class PhantomVocabMask(LogitsProcessor):
        """批量等价物 of decode.common.mask_phantom_(ids >= EFFECTIVE_VOCAB → -inf)。"""

        def __call__(self, input_ids, scores):
            if scores.size(-1) > EFFECTIVE_VOCAB:
                scores[..., EFFECTIVE_VOCAB:] = float("-inf")
            return scores

    assert EFFECTIVE_VOCAB == C.EFFECTIVE_VOCAB
    base, processor = C.load_base(args.model_id, args.device)
    C.apply_max_pixels(processor, args.max_pixels)
    processor.tokenizer.padding_side = "left"
    eos_id = processor.tokenizer.eos_token_id
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id
    lp = LogitsProcessorList([PhantomVocabMask()])

    my_idx = shard_range(total, args.num_shards, args.shard_id)
    shard_file = out_dir / f"rollouts_shard_{args.shard_id:05d}.jsonl"
    done = repair_and_load(shard_file)
    todo = [i for i in my_idx if i not in done]
    print(f"[gen] shard {args.shard_id}/{args.num_shards}: rows "
          f"[{my_idx.start},{my_idx.stop}) todo={len(todo)} resume-done={len(done)} "
          f"bs={args.batch_size} max_pixels={args.max_pixels} eos={eos_id} "
          f"pad={pad_id}", flush=True)

    reader = TarImageReader(args.images_tar_dir)
    n_done = n_fail = 0
    t0 = time.time()
    with open(shard_file, "a") as fo:
        for batch in iter_batches(todo, args.batch_size):
            try:
                qs = [rows[i]["question"] for i in batch]
                imgs = [reader.get(i) for i in batch]
                results = generate_batch(base, processor, qs, imgs, args.max_new,
                                         eos_id, pad_id, args.device, lp)
            except Exception as e:
                n_fail += len(batch)
                print(f"[gen] WARN batch {batch[0]}..{batch[-1]} failed: {e}",
                      flush=True)
                continue
            for i, (toks, eos_hit) in zip(batch, results):
                fo.write(json.dumps(make_record(i, rows[i]["sample_id"], toks,
                                                eos_hit)) + "\n")
            fo.flush()
            n_done += len(batch)
            if (n_done // args.batch_size) % 25 == 0:
                dt = time.time() - t0
                rate = n_done / max(1e-6, dt)
                eta = (len(todo) - n_done) / max(1e-6, rate)
                print(f"[gen] {n_done}/{len(todo)} fail={n_fail} "
                      f"{rate:.3f} row/s ETA {eta / 60:.0f}min", flush=True)

    print(f"[gen] shard done: generated={n_done} fail={n_fail} resume-skip="
          f"{len(done)} in {(time.time() - t0) / 60:.1f} min", flush=True)
    if args.num_shards == 1:
        finalize_manifest(out_dir, total, config)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
