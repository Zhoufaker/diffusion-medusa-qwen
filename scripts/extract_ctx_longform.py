"""extract_ctx_longform.py — 队列②:longform rollout 5 层特征抽取(W1 适配层)。

以 W1 scripts/extract_ctx_features.py 为基,改动仅限四处(必审包登记):
  a. 输入源:rollouts_px501760 JSONL(manifest 锁定序列)+ fixation
     webdataset 图像 tar(复用 gen_longform_rollouts.TarImageReader)。
     锁定制下游第一读者纪律:load_locked_rollouts() 三级先验——
     ① manifest complete=true 且行数对账;② 每 shard 文件全文 sha256
     对 manifest 登记值;③ 逐行 validate_record(六键 schema + 逐 token
     sha256)——任一不符即停,绝不静默跳行。
  b. 像素 spec:apply_max_pixels(processor, 501760)(longform
     PREPROCESS_SPEC 正典,与生成端/MM-Vet OOD 同一实现);spec 与
     rollout manifest sha256 一并写入抽取 manifest(谱系闭环)。
  c. teacher-forcing 输入 = image + question chat template(与生成端
     generate_batch 逐参数镜像:同 message schema、apply_chat_template(
     add_generation_prompt=True)、processor(padding=True);单行无 pad,
     绕开左 pad/位置偏移问题)+ 锁定 rollout tokens。层 [1,7,13,19,25]、
     前向 fp16 sdpa、存储 bf16、(5,T,3584) 堆叠、spans 六元组
     [vision_s, vision_e, 0, P, P, P+L)——全部沿用 W1 约定零改动。
  d. 分片:SHARD_SIZE=256 → ceil(37079/256)=145 片 safetensors,写
     dflash_data/longform_ctx_cache/(独立于 ctx_cache_35k,互不触碰)。

TF argmax 诊断:登记不设门(生成为批量 generate 增量前向,TF 为整序列
前向,前向形态差异属已知扰动类别——见 drift_attribution_report 结论;
W1 的 99.5%/99.9% 门是旧缓存一致性门,不适用于本线,不移植)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from safetensors.torch import save_file

_SCRIPTS = Path(__file__).resolve().parent
_ROOT = _SCRIPTS.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))          # 仓库根(decode/ 等顶层包),W1 同款
if str(_SCRIPTS) not in sys.path:
    sys.path.append(str(_SCRIPTS))          # scripts/ 兄弟模块——纪律:append

from gen_longform_rollouts import (  # noqa: E402
    EFFECTIVE_VOCAB,
    ROWS_PER_TAR,
    TarImageReader,
    code_commit,
    validate_record,
    verify_pool,
)

sys.path.insert(0, str(_ROOT / "external" / "dflash"))
from dflash.model import build_target_layer_ids  # noqa: E402

SHARD_SIZE = 256
NUM_TARGET_LAYERS = 28
NUM_DRAFT_LAYERS = 5
TARGET_LAYER_IDS = build_target_layer_ids(NUM_TARGET_LAYERS, NUM_DRAFT_LAYERS)
HIDDEN_TUPLE_IDS = [l + 1 for l in TARGET_LAYER_IDS]  # tuple[0] = embedding 输出

MAX_PIXELS_SPEC = 501760
TRIPLETS_SHA256 = "06b6f55b3badb739f2c10b92141ddba8784f3a4a9026107a2b44caa31b51f81f"
POOL_TOTAL = 37079

# 镜像对象:scripts/gen_longform_rollouts.py(逐参数)。运行时才可知的字段
# 由 freeze_preprocess_spec() 回填并断言非 None。
PREPROCESS_SPEC = {
    "mirror_of": "scripts/gen_longform_rollouts.py",
    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
    "forward_dtype": "float16",          # load_base: fp16(与生成端同)
    "attn_implementation": "sdpa",       # load_base asserts sdpa
    "storage_dtype": "bfloat16",         # W1 spec 1c 零改动
    "chat_template": "processor.apply_chat_template(add_generation_prompt=True)",
    "message_schema": "[{role:user, content:[{type:image},{type:text,text:question}]}]",
    "processor_padding": True,           # 生成端同参;单行下无实际 pad
    "max_pixels_override": MAX_PIXELS_SPEC,  # longform PREPROCESS_SPEC 正典(对 W1 的登记差异)
    "runtime_max_pixels": None,          # freeze 时回填并断言 == 501760
    "runtime_min_pixels": None,
    "effective_vocab": EFFECTIVE_VOCAB,
    "rollout_max_new": 512,              # 生成端 --max-new
    "eos_token_id": None,
    "image_pad_token_id": None,
    "target_layer_ids": TARGET_LAYER_IDS,
    "hidden_tuple_indices": HIDDEN_TUPLE_IDS,
    "rollout_gen_hardware": "dgxa100 A100 fp16 sdpa bs=8 (jobs 176722377-85, 2026-08)",
    "extract_hardware": "dgxa100 A100 fp16 sdpa",
}


def sha256_file(fp: Path) -> str:
    return hashlib.sha256(Path(fp).read_bytes()).hexdigest()


def load_locked_rollouts(rollouts_dir, expect_total=None):
    """锁定 rollout 资产的下游第一读者:三级先验,任一不符 ValueError 即停。
    返回 ({idx: record}, source_lineage dict)。"""
    rollouts_dir = Path(rollouts_dir)
    mpath = rollouts_dir / "manifest.json"
    if not mpath.exists():
        raise ValueError(f"rollout manifest missing: {mpath}")
    manifest = json.load(open(mpath))
    if not manifest.get("complete", False):
        raise ValueError("rollout manifest complete != true — 资产未收官,拒读")
    if expect_total is not None and manifest.get("total") != expect_total:
        raise ValueError(f"manifest total {manifest.get('total')} != {expect_total}")
    by_idx: dict[int, dict] = {}
    for name, info in sorted(manifest["shard_files"].items()):
        fp = rollouts_dir / name
        h = sha256_file(fp)
        if h != info["sha256"]:
            raise ValueError(f"{name}: file sha256 {h[:16]}... != manifest 登记值")
        n_file = 0
        for ln, line in enumerate(open(fp)):
            rec = json.loads(line)
            if not validate_record(rec):
                raise ValueError(f"{name} line {ln}: validate_record 失败(schema/sha)")
            if rec["idx"] in by_idx:
                raise ValueError(f"{name} line {ln}: duplicate idx {rec['idx']}")
            by_idx[rec["idx"]] = rec
            n_file += 1
        if n_file != info["n"]:
            raise ValueError(f"{name}: rows {n_file} != manifest n {info['n']}")
    if expect_total is not None and len(by_idx) != expect_total:
        raise ValueError(f"rows loaded {len(by_idx)} != {expect_total}")
    lineage = {
        "rollouts_dir": str(rollouts_dir),
        "rollout_manifest_sha256": sha256_file(mpath),
        "rollout_code_commit": manifest.get("config", {}).get("code_commit"),
        "rollout_config": manifest.get("config", {}),
    }
    return by_idx, lineage


def find_vision_span(ids: torch.Tensor, image_pad_id: int) -> tuple[int, int]:
    """[start, end) of the contiguous <|image_pad|> run — W1 原样。"""
    pos = (ids == image_pad_id).nonzero(as_tuple=True)[0]
    if pos.numel() == 0:
        return 0, 0
    s, e = int(pos[0]), int(pos[-1]) + 1
    assert pos.numel() == e - s, f"non-contiguous vision span ({pos.numel()} vs {e - s})"
    return s, e


def make_spans(prompt_ids: torch.Tensor, P: int, L: int,
               image_pad_id: int) -> torch.Tensor:
    """六元组 [vision_s, vision_e, prompt_s, prompt_e, rollout_s, rollout_e)。
    W1 约定零改动:prompt 段 = 完整 prefill [0,P)(模板+图+问题,longform
    无多轮结构);rollout 段 = [P, P+L)(锁定序列)。longform 全池有图,
    vision span 必须非空。"""
    vs, ve = find_vision_span(prompt_ids[:P], image_pad_id)
    if ve <= vs:
        raise ValueError("longform 行缺 vision span — 输入构造异常,即停")
    return torch.tensor([vs, ve, 0, P, P, P + L], dtype=torch.int64)


def flush_shard(out_dir: Path, shard_id: int, buf: dict, partial: bool) -> str:
    """W1 原样:keys {i}.ctx (5,T,3584) bf16 / {i}.ids (T,) i64 / {i}.spans (6,) i64。"""
    name = f"shard_{shard_id:05d}.safetensors"
    tensors = {}
    for i, (ctx, ids, spans) in sorted(buf.items()):
        tensors[f"{i}.ctx"] = ctx
        tensors[f"{i}.ids"] = ids
        tensors[f"{i}.spans"] = spans
    save_file(tensors, str(out_dir / name), metadata={"partial": str(partial)})
    return name


def freeze_preprocess_spec(processor) -> dict:
    spec = dict(PREPROCESS_SPEC)
    ip = processor.image_processor
    spec["runtime_max_pixels"] = getattr(ip, "max_pixels", None)
    spec["runtime_min_pixels"] = getattr(ip, "min_pixels", None)
    spec["runtime_size"] = dict(getattr(ip, "size", None) or {})
    spec["runtime_image_processor_class"] = type(ip).__name__
    spec["eos_token_id"] = int(processor.tokenizer.eos_token_id)
    spec["image_pad_token_id"] = int(
        processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    runtime_cap = spec["runtime_max_pixels"] or spec["runtime_size"].get("longest_edge")
    assert runtime_cap == MAX_PIXELS_SPEC, \
        f"longform spec 要求 max_pixels=501760 生效,实际 {runtime_cap}"
    for k in ("eos_token_id", "image_pad_token_id"):
        assert spec[k] is not None, f"PREPROCESS_SPEC field {k} unresolved"
    return spec


@torch.no_grad()
def extract_one(base, inputs, rollout_tokens: torch.Tensor, device: str):
    """W1 原样(去门):整序列 teacher forcing,TF argmax 诊断登记不设门。"""
    from decode.common import mask_phantom_
    prompt_ids = inputs["input_ids"]                       # (1, P)
    P = prompt_ids.shape[1]
    L = rollout_tokens.shape[0]
    full_ids = torch.cat(
        [prompt_ids, rollout_tokens.view(1, -1).to(device)], dim=1)
    out = base(
        input_ids=full_ids,
        attention_mask=torch.ones_like(full_ids),
        pixel_values=inputs.get("pixel_values"),
        image_grid_thw=inputs.get("image_grid_thw"),
        output_hidden_states=True,
        use_cache=False,
    )
    ctx = torch.stack([out.hidden_states[t][0] for t in HIDDEN_TUPLE_IDS])
    logits = mask_phantom_(out.logits[0, P - 1 : P + L - 1, :].float())
    n_match = int((logits.argmax(-1) == rollout_tokens.to(device)).sum())
    ctx_cpu = ctx.to("cpu", torch.bfloat16).contiguous()
    ids_cpu = full_ids[0].to("cpu", torch.int64).contiguous()
    del out, ctx, logits
    return ctx_cpu, ids_cpu, P, L, n_match


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts-dir",
                    default="/scratch/li96/mz9869/dflash_data/longform_fixed_v2/rollouts_px501760")
    ap.add_argument("--triplets",
                    default="/scratch/li96/mz9869/dflash_data/longform_fixed_v2/triplets.jsonl")
    ap.add_argument("--images-tar-dir",
                    default="/scratch/li96/mz9869/dflash_data/longform_fixed_v2/images")
    ap.add_argument("--out-dir",
                    default="/scratch/li96/mz9869/dflash_data/longform_ctx_cache")
    ap.add_argument("--model-id", default=PREPROCESS_SPEC["model_id"])
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--end-idx", type=int, default=None, help="exclusive; default = all")
    ap.add_argument("--expect-total", type=int, default=POOL_TOTAL)
    ap.add_argument("--triplets-sha256", default=TRIPLETS_SHA256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--manifest-name", default="ctx_manifest.json")
    args = ap.parse_args()

    assert args.start_idx % SHARD_SIZE == 0, "start-idx must be shard-aligned"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.triplets)]
    verify_pool(args.triplets, rows, args.expect_total, args.triplets_sha256)
    rollouts, lineage = load_locked_rollouts(args.rollouts_dir, args.expect_total)
    total = len(rows)
    end = total if args.end_idx is None else min(args.end_idx, total)

    import decode.common as C
    assert EFFECTIVE_VOCAB == C.EFFECTIVE_VOCAB
    base, processor = C.load_base(args.model_id, args.device)
    C.apply_max_pixels(processor, MAX_PIXELS_SPEC)
    spec = freeze_preprocess_spec(processor)
    image_pad_id = spec["image_pad_token_id"]
    reader = TarImageReader(args.images_tar_dir)

    records: dict[str, dict] = {}
    shard_names: dict[int, str] = {}
    buf: dict[int, tuple] = {}
    n_pos_total = n_match_total = n_done = n_failed = 0
    tok_count = 0
    t0 = time.time()

    for i in range(args.start_idx, end):
        shard_id = i // SHARD_SIZE
        try:
            rec = rollouts[i]
            rollout_tokens = torch.tensor(rec["tokens"], dtype=torch.int64)
            q = rows[i]["question"]
            img = reader.get(i)
            messages = [{"role": "user",
                         "content": [{"type": "image"},
                                     {"type": "text", "text": q}]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img],
                               return_tensors="pt", padding=True).to(args.device)
            ctx, ids, P, L, n_match = extract_one(
                base, inputs, rollout_tokens, args.device)
            spans = make_spans(ids, P, L, image_pad_id)
        except Exception as e:  # noqa: BLE001 — 逐样本故障隔离,计数+末尾对账
            n_failed += 1
            print(f"[ext] WARN idx {i} failed: {e}", flush=True)
            records[str(i)] = {"idx": i, "failed": True, "error": str(e)[:200]}
            continue
        buf[i] = (ctx, ids, spans)
        records[str(i)] = {
            "idx": i, "shard": shard_id, "T": P + L, "P": P, "L": L,
            "sample_id": rows[i]["sample_id"], "source": rows[i]["source"],
            "rollout_sha256": rec["sha256"], "eos_hit": rec["eos_hit"],
            "n_pos": L, "n_match": n_match,
        }
        n_pos_total += L
        n_match_total += n_match
        n_done += 1
        tok_count += P + L
        if (i + 1) % SHARD_SIZE == 0 or (i + 1) == end:
            partial = (i + 1) % SHARD_SIZE != 0 or (i % SHARD_SIZE + 1) != len(buf)
            shard_names[shard_id] = flush_shard(out_dir, shard_id, buf, partial)
            buf = {}
        if n_done and n_done % 50 == 0:
            dt = time.time() - t0
            print(f"[ext] {i + 1 - args.start_idx}/{end - args.start_idx} "
                  f"done={n_done} fail={n_failed} "
                  f"{n_done / dt * 60:.1f} samples/min {tok_count / dt:.0f} tok/s "
                  f"tf_posmatch={n_match_total / max(1, n_pos_total):.6f}", flush=True)

    dt = time.time() - t0
    peak_gb = (torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0
    agg = {
        "n_done": n_done, "n_failed": n_failed,
        "tf_pos_match_rate": n_match_total / max(1, n_pos_total),
        "tf_audit_policy": "diagnostic only, NO gates (前向形态差异属已知类别,"
                           "见 drift_attribution_report;W1 旧缓存门不移植)",
        "throughput_samples_per_min": n_done / dt * 60,
        "throughput_tok_per_s": tok_count / dt,
        "peak_gpu_mem_gib": round(peak_gb, 2),
        "wall_s": round(dt, 1),
    }
    manifest = {
        "kind": "longform_ctx_cache_v1",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "code_commit": code_commit(),
        "hardware": os.environ.get(
            "EXTRACT_HARDWARE_TAG",
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "range": [args.start_idx, end], "total_rows": total,
        "shard_size": SHARD_SIZE, "shards": shard_names,
        "preprocess_spec": spec,
        "source_lineage": {**lineage, "triplets_sha256": args.triplets_sha256,
                           "images_tar_dir": args.images_tar_dir,
                           "rows_per_tar": ROWS_PER_TAR},
        "aggregate": agg, "samples": records,
    }
    mpath = out_dir / args.manifest_name
    json.dump(manifest, open(mpath, "w"), indent=1)
    print(f"[ext] DONE range=[{args.start_idx},{end}) done={n_done} fail={n_failed}\n"
          f"[ext] tf_pos_match={agg['tf_pos_match_rate']:.6f} (diagnostic, no gate)\n"
          f"[ext] {agg['throughput_samples_per_min']:.1f} samples/min "
          f"{agg['throughput_tok_per_s']:.0f} tok/s peak_mem={peak_gb:.1f}GiB\n"
          f"[ext] manifest -> {mpath}", flush=True)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
