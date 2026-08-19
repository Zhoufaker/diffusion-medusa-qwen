"""prep_longform_pilot.py — D1.5 rollout 长度 pilot 的登录节点预备(零 GPU)。

产出使 gen_cache_rollout.py 可【零改动】运行的输入(其 --prompts 格式
{id,image,question} 与 triplets 天然同构;--max-new 为既有参数):
  1. seed=43 分层抽样 50 条(DOCCI 40 / DetailCaps 10,按源占比,预注册)
  2. 从 webdataset 分片解包对应图像到 pilot 目录(~50 个文件,临时)
  3. 写 pilot_prompts.json:[{id, image, question}]
"""
from __future__ import annotations

import argparse
import json
import random
import tarfile
from io import BytesIO
from pathlib import Path

from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

SEED = 43
# PREPROCESS_SPEC 适配(2026-08-20 裁决,longform 线 max_pixels=501760):
# 仅对 cap binding 的图像按 smart_resize 目标尺寸预缩放(PIL BICUBIC,
# PNG 无损),使 gen_cache_rollout.py 【零改动】即得 capped vision token 数
# (与 decode.common.apply_max_pixels 的 processor 侧 cap 逐位同 token 数;
# 重采样核微差属登记类数值扰动)。未触线图像原字节透传。
FACTOR, MIN_PX = 28, 3136  # mirrors decode.common.apply_max_pixels
# v2 修订配额(用户 2026-08-20):占比调平 + 每源下限 10
N_PER_SOURCE = {"docci": 12, "detailcaps": 10, "sp": 16, "ln": 12}
# max_new 裁定规则(预注册,写入 pilot_manifest):
MAX_NEW_RULE = ("worst-source trunc@384 <=5% -> max_new=384; otherwise 512 "
                "(512 is the measurement ceiling; if exceeded, register "
                "truthfully and still take 512)")
BASE = Path("/scratch/li96/mz9869/dflash_data/longform_fixed_v2")
OUT = BASE / "pilot"


def save_image(raw: bytes, name: str, out_dir: Path, cap: int | None) -> tuple[str, bool]:
    """Write one pilot image; pre-resize iff the pixel cap binds. -> (fname, bound)"""
    if cap is not None:
        img = Image.open(BytesIO(raw))
        w, h = img.size
        if round(h / FACTOR) * FACTOR * round(w / FACTOR) * FACTOR > cap:
            h2, w2 = smart_resize(h, w, factor=FACTOR, min_pixels=MIN_PX,
                                  max_pixels=cap)
            out_name = name.rsplit(".", 1)[0] + ".png"
            img.convert("RGB").resize((w2, h2), Image.BICUBIC).save(out_dir / out_name)
            return out_name, True
    (out_dir / name).write_bytes(raw)
    return name, False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pixels", type=int, default=None,
                    help="longform PREPROCESS_SPEC cap (501760); None = v1 行为")
    args = ap.parse_args()
    sfx = f"_px{args.max_pixels}" if args.max_pixels else ""
    rows = [json.loads(l) for l in open(BASE / "triplets.jsonl")]
    by_src = {}
    for i, r in enumerate(rows):
        by_src.setdefault(r["source"], []).append(i)
    rng = random.Random(SEED)
    picks = []
    for src, k in N_PER_SOURCE.items():
        picks += sorted(rng.sample(by_src[src], k))
    img_dir = OUT / f"images{sfx}"
    img_dir.mkdir(parents=True, exist_ok=True)
    prompts = []
    n_bound = 0
    byshard = {}
    for i in picks:
        byshard.setdefault(i // 1000, []).append(i)
    for sid, idxs in sorted(byshard.items()):
        with tarfile.open(BASE / "images" / f"shard_{sid:05d}.tar") as tf:
            names = {m.name.split(".")[0]: m for m in tf.getmembers()
                     if not m.name.endswith(".json")}
            for i in idxs:
                m = names[f"{i:06d}"]
                fname, bound = save_image(tf.extractfile(m).read(), m.name,
                                          img_dir, args.max_pixels)
                n_bound += bound
                r = rows[i]
                prompts.append({"id": r["sample_id"].replace("/", "_"),
                                "image": fname, "question": r["question"]})
    json.dump(prompts, open(OUT / f"pilot_prompts{sfx}.json", "w"),
              ensure_ascii=False, indent=1)
    meta = {"seed": SEED, "n_per_source": N_PER_SOURCE,
            "max_new_decision_rule_prereg": MAX_NEW_RULE,
            "picked_row_indices": picks}
    if args.max_pixels:
        meta["preprocess_spec"] = {
            "max_pixels": args.max_pixels, "factor": FACTOR, "min_pixels": MIN_PX,
            "impl": "prep-side pre-resize of cap-bound images only "
                    "(smart_resize dims, PIL BICUBIC, lossless PNG); "
                    "token-count-exact vs processor-side apply_max_pixels",
            "n_prebound": n_bound}
    json.dump(meta, open(OUT / f"pilot_manifest{sfx}.json", "w"), indent=1)
    print(f"[pilot-prep] {len(prompts)} prompts, images -> {img_dir}"
          + (f", cap-bound resized: {n_bound}" if args.max_pixels else ""))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
