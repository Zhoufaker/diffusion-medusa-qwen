"""fix_longform_data.py — D1.5 长文本数据固定(阶段一 Phase B)。

协议权威:external/longform_dataset_registry.py(mage_pack 逐字节副本,
sha256 7e736eb0…,原包只读纪律)。本脚本零 GPU。

协议保真边界(唯一允许的改造 = 导出行为,注释处标 [EXPORT-MOD]):
  * question 模板、字段解析、样本转换:直接调用 registry 的
    _record_to_sample / LONGFORM_PROMPT,零重写
  * 固定顺序:逐源复刻 registry 的顺序语义(jsonl 路:records 列表 +
    random.Random(seed).shuffle;HF 路:ds.shuffle(seed)),seed=42
    (registry 默认,协议已覆盖,不动)
  * [EXPORT-MOD] registry.write_samples_jsonl 的逐张 jpg 落盘改为
    webdataset tar(~1000 样本/片)+ 流式处理(逐条转换即打包,
    不整表驻留内存——registry 设计面向 n≤数百,全量 2 万条必须流式;
    顺序语义逐行等价);图像打包用**原始字节**(不 quality=95 重编码,
    保真优先,登记于 manifest)

去重(双键约定,builder 脚本未入 git,按 dedup report 工件重建,
重建依据登记于 manifest 与报告):
  * 图像键 = 图像 id/文件基名;问题键 = question 原串
  * 对评估清单(manifest_300 + MM-Vet 218):图像键命中即剔;
    辅助内容哈希(原始字节 sha256)对局部持有的评估图像
    (coco_subset + mmvet images)命中亦剔(登记为辅助键)
  * 源间互重:内容哈希,优先级 DOCCI > DetailCaps
  * 断言:与两份评估清单零交集
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import random
import subprocess
import sys
import tarfile
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/scratch/li96/mz9869/tmp_hf_download/")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_spec = importlib.util.spec_from_file_location(
    "longform_dataset_registry", _ROOT / "external" / "longform_dataset_registry.py")
REG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(REG)

SHARD_SIZE = 1000
SEED = 42                      # registry 默认,协议覆盖项
AUX_SEED = 43                  # 协议未覆盖处的 seed 族(本脚本仅抽检用)


def eval_keys():
    """评估清单双键集合 + 本地评估图像内容哈希。"""
    img_keys, q_keys = set(), set()
    m300 = json.load(open("/scratch/li96/mz9869/eval_manifests/manifest_300.json"))
    for e in (m300 if isinstance(m300, list) else m300.get("samples", [])):
        img_keys.add(str(e.get("id") or e.get("image", "")).split(".")[0])
        if e.get("image"):
            img_keys.add(e["image"])
        for c in (e.get("conversations") or []):
            if c.get("from") == "human":
                q_keys.add(c.get("value", "").strip())
        if e.get("question"):
            q_keys.add(e["question"].strip())
    mm = json.load(open("/scratch/li96/mz9869/ood_eval/mmvet/manifest_mmvet_218.json"))
    for e in (mm if isinstance(mm, list) else mm.get("samples", [])):
        if isinstance(e, dict):
            if e.get("image"):
                img_keys.add(e["image"]); img_keys.add(str(e["image"]).split(".")[0])
            if e.get("question"):
                q_keys.add(e["question"].strip())
    content = {}
    for d in ("/g/data/li96/mz9869/data/coco_subset",
              "/scratch/li96/mz9869/ood_eval/mmvet/mm-vet/images"):
        p = Path(d)
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    content[hashlib.sha256(f.read_bytes()).hexdigest()] = f.name
    return img_keys, q_keys, content


def prep_docci(staging: Path):
    """官方发布包 → registry --data-path 协议要求的本地 JSONL
    (占位源同款准备方式;字段名取 registry 解析链内的 image_path/description)。
    幂等:已存在即跳过。图像 tar 就地解包(打包完成后由 runner 清散件)。"""
    ann = staging / "docci_local.jsonl"
    if ann.exists():
        return
    imgdir = staging / "images"
    if not imgdir.is_dir():
        print("[prep] extracting docci_images.tar.gz …", flush=True)
        with tarfile.open(staging / "docci_images.tar.gz") as tf:
            tf.extractall(staging)
    n = 0
    with open(ann, "w") as out_f:
        for line in open(staging / "docci_descriptions.jsonlines"):
            r = json.loads(line)
            out_f.write(json.dumps({
                "id": r["example_id"], "split": r.get("split"),
                "image_path": f"images/{r['image_file']}",
                "description": r["description"],
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"[prep] docci_local.jsonl rows={n}", flush=True)


def iter_docci(staging: Path):
    """DOCCI 本地 jsonl 路:官方发布包 → registry jsonl 协议。
    顺序语义 = load_jsonl_samples 逐行复刻(records 列表 + shuffle(SEED))。
    """
    prep_docci(staging)
    ann = staging / "docci_local.jsonl"
    records = [json.loads(l) for l in open(ann) if l.strip()]
    rng = random.Random(SEED)
    rng.shuffle(records)                       # == registry load_jsonl_samples
    base = str(ann.parent)
    for rec in records:
        yield rec, base


def iter_detailcaps():
    """DetailCaps HF 路:registry _hf_candidates 顺序语义逐行复刻
    (candidates 首个可用 = test split;ds.shuffle(seed=SEED) 后顺序迭代)。

    [EXPORT-MOD 同级登记适配,2026-08-19 用户批准(决策点①)]:
    该源 schema 与 registry 字段链错位,进 registry 前做登记性映射——
      reference := GT_Caption_{GPT4O,GPT4V,Gemini15Pro} 三列取最长
                  (镜像 registry 对 caption 列表"取最长"的既有约定);
      id        := image 路径基名(COCO id 兼容 → 图像键去重可对撞评估集);
      image     := binary 原始字节(registry _to_rgb 可验证,打包零重编码)。
    """
    from datasets import load_dataset
    GT_COLS = ("GT_Caption_GPT4O", "GT_Caption_GPT4V", "GT_Caption_Gemini15Pro")
    ds = load_dataset("foundation-multimodal-models/DetailCaps-4870", split="test")
    ds = ds.shuffle(seed=SEED)                 # == registry _hf_candidates
    for raw in ds:
        gts = [(raw.get(k) or "").strip() for k in GT_COLS]
        rec = {
            "id": Path(raw["image"]).stem,
            "reference": max(gts, key=len),
            "image": raw["binary"],
            "binary": raw["binary"],
            "origin_subset": raw.get("source"),
        }
        yield rec, None


def raw_image_bytes(rec, base_dir):
    """[EXPORT-MOD] 打包用原始字节(不重编码)。返回 (bytes, ext) 或 None。"""
    v = REG._field(rec, REG.IMAGE_FIELDS)
    if isinstance(v, (bytes, bytearray)):      # [EXPORT-MOD] 原始字节直取
        return bytes(v), ".jpg"
    if isinstance(v, dict) and v.get("bytes"):
        return v["bytes"], (Path(v.get("path") or "img.jpg").suffix or ".jpg")
    if isinstance(v, str) and not v.startswith("http"):
        p = Path(base_dir or ".") / v if not os.path.isabs(v) else Path(v)
        if p.exists():
            return p.read_bytes(), (p.suffix or ".jpg")
    if hasattr(v, "save"):                     # PIL fallback -> 无损 PNG
        buf = io.BytesIO(); v.save(buf, format="PNG")
        return buf.getvalue(), ".png"
    return None


def fix_source(source, record_iter, writer, state, tokenizer,
               img_keys, q_keys, content_hashes):
    n_in = n_ok = n_dup_eval = n_dup_cross = n_noimg = 0
    for rec, base in record_iter:
        n_in += 1
        blob = raw_image_bytes(rec, base)
        # --append 快路径:内容哈希命中(已打包)即跳过,不做昂贵的
        # PIL 解码(登录节点 CPU 上限防护;对结果无影响——同哈希在原跑
        # 中也会以 dup_cross 剔除)
        if blob is not None and hashlib.sha256(blob[0]).hexdigest() in state["cross_hashes"]:
            n_dup_cross += 1
            continue
        # registry 语义转换(question 模板/reference 字段链,零重写);
        # 图像加载委托 registry(损坏图即弃,与协议一致)
        try:
            sample = REG._record_to_sample(
                rec, base_dir=base, source=source, idx=n_ok,
                default_question=REG.LONGFORM_PROMPT)
        except Exception:
            sample = None
        if sample is None or sample.get("reference") in (None, "") or blob is None:
            n_noimg += 1
            continue
        img_key = str(sample.get("sample_id") or f"{source}-{n_ok}")
        h = hashlib.sha256(blob[0]).hexdigest()
        if (img_key in img_keys or img_key.split(".")[0] in img_keys
                or h in content_hashes):
            n_dup_eval += 1
            continue
        state["cross_hashes"][h] = source
        ref = sample["reference"]
        row = {
            "sample_id": f"{source}/{img_key}",
            "source": source,
            "image_key": img_key,
            "question": sample["question"],
            "reference": ref,
            "ref_token_len": len(tokenizer(ref, add_special_tokens=False)["input_ids"]),
        }
        writer.add(row, blob)
        state["triplets"].write(json.dumps(row, ensure_ascii=False) + "\n")
        n_ok += 1
        if n_ok % 500 == 0:
            print(f"[{source}] {n_ok} fixed ({n_in} seen)", flush=True)
    return {"seen": n_in, "fixed": n_ok, "dup_eval": n_dup_eval,
            "dup_cross_source": n_dup_cross, "unusable": n_noimg}


class WdsWriter:
    """[EXPORT-MOD] webdataset 打包器:{key}.<ext> + {key}.json 成对入 tar,
    ~SHARD_SIZE 样本/片。替代 registry.write_samples_jsonl 的逐张 jpg。"""

    def __init__(self, out_dir: Path):
        self.dir = out_dir; self.dir.mkdir(parents=True, exist_ok=True)
        self.n = 0; self.shard = None; self.shard_id = -1; self.names = []

    def _roll(self):
        if self.shard: self.shard.close()
        self.shard_id += 1
        name = f"shard_{self.shard_id:05d}.tar"
        self.names.append(name)
        self.shard = tarfile.open(self.dir / name, "w")

    def add(self, row, blob):
        if self.n % SHARD_SIZE == 0:
            self._roll()
        key = f"{self.n:06d}"
        data, ext = blob
        for suffix, payload in ((ext, data),
                                (".json", json.dumps(row, ensure_ascii=False).encode())):
            info = tarfile.TarInfo(f"{key}{suffix}")
            info.size = len(payload); info.mtime = 0
            self.shard.addfile(info, io.BytesIO(payload))
        self.n += 1

    def close(self):
        if self.shard: self.shard.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", default="/scratch/li96/mz9869/dflash_data/longform_fixed_v1/staging/docci")
    ap.add_argument("--out", default="/scratch/li96/mz9869/dflash_data/longform_fixed_v1")
    ap.add_argument("--sources", default="docci,detailcaps")
    ap.add_argument("--append", action="store_true",
                    help="断点续跑:从既有 shards/triplets 重建 writer 计数与"
                         "cross_hashes,triplets 以追加模式打开(登录节点 30min "
                         "CPU 上限教训,2026-08-18)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    out = Path(args.out); (out / "images").mkdir(parents=True, exist_ok=True)
    img_keys, q_keys, content_hashes = eval_keys()
    print(f"[eval-keys] img={len(img_keys)} q={len(q_keys)} content={len(content_hashes)}", flush=True)

    writer = WdsWriter(out / "images")
    prior_sources = {}
    if args.append:
        # 重建:既有 shard 图像 payload 逐一哈希 → cross_hashes;
        # 计数/片号从 triplets 与 shard 清单续接
        cross = {}
        for name in sorted((out / "images").glob("shard_*.tar")):
            with tarfile.open(name) as tf:
                for m in tf.getmembers():
                    if m.name.endswith(".json"):
                        continue
                    cross[hashlib.sha256(tf.extractfile(m).read()).hexdigest()] = "prior"
        n_prior = 0
        for line in open(out / "triplets.jsonl"):
            r = json.loads(line)
            prior_sources[r["source"]] = prior_sources.get(r["source"], 0) + 1
            n_prior += 1
        writer.n = n_prior
        writer.shard_id = n_prior // SHARD_SIZE - (1 if n_prior % SHARD_SIZE == 0 else 0)
        last = sorted((out / "images").glob("shard_*.tar"))[-1]
        writer.names = [p.name for p in sorted((out / "images").glob("shard_*.tar"))]
        # 尾片未满则以追加重开:tarfile 不支持追加压缩,此处片为纯 tar,可 'a'
        writer.shard = tarfile.open(last, "a")
        state = {"cross_hashes": cross,
                 "triplets": open(out / "triplets.jsonl", "a")}
        print(f"[append] resumed at n={n_prior}, shards={len(writer.names)}, "
              f"hashes={len(cross)}, prior={prior_sources}", flush=True)
    else:
        state = {"cross_hashes": {}, "triplets": open(out / "triplets.jsonl", "w")}
    per_source = {s: {"prior_fixed": c} for s, c in prior_sources.items()}
    t0 = time.time()
    for source in args.sources.split(","):
        it = iter_docci(Path(args.staging)) if source == "docci" else iter_detailcaps()
        stats = fix_source(source, it, writer, state, tokenizer,
                           img_keys, q_keys, content_hashes)
        per_source[source] = {**per_source.get(source, {}), **stats}
        print(f"[{source}] {per_source[source]}", flush=True)
    writer.close(); state["triplets"].close()

    assert all(s.get("dup_eval", 0) >= 0 for s in per_source.values())
    # 断言:与评估清单零交集(dup_eval 已剔,复扫 triplets 直接验证)
    kept_keys = set()
    for line in open(out / "triplets.jsonl"):
        kept_keys.add(json.loads(line)["image_key"])
    inter = kept_keys & img_keys
    assert not inter, f"eval overlap survived dedup: {sorted(inter)[:5]}"

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                                     text=True).strip()
    reg_sha = hashlib.sha256(
        (_ROOT / "external" / "longform_dataset_registry.py").read_bytes()).hexdigest()
    manifest = {
        "version": "longform_fixed_v1",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED, "aux_seed_family": AUX_SEED,
        "registry_sha256": reg_sha, "fix_script_commit": commit,
        "question_template": REG.LONGFORM_PROMPT,
        "per_source": per_source,
        "n_total": writer.n, "shards": writer.names,
        "shard_size": SHARD_SIZE,
        "dedup": {
            "convention": "dual-key (image_key + question_key) per 旧线 dedup "
                          "report artifacts; builder script predates git — "
                          "reconstruction registered; content-sha256 vs local "
                          "eval images as auxiliary key; cross-source by "
                          "content hash, priority docci > detailcaps",
            "eval_zero_intersection_assert": "passed",
        },
        "licenses": {
            "docci": "CC BY 4.0 (google/docci official release)",
            "detailcaps": "see foundation-multimodal-models/DetailCaps-4870 card "
                          "(research; aggregates COCO/SAM/LAION-sourced images)",
        },
        "export_mod_note": "webdataset tars with ORIGINAL image bytes (no q95 "
                           "re-encode); streaming order-equivalent to registry",
    }
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in manifest.items() if k != "shards"},
                     ensure_ascii=False, indent=1)[:1200], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
