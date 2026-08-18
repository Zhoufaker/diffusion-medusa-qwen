"""prep_longform_pilot.py — D1.5 rollout 长度 pilot 的登录节点预备(零 GPU)。

产出使 gen_cache_rollout.py 可【零改动】运行的输入(其 --prompts 格式
{id,image,question} 与 triplets 天然同构;--max-new 为既有参数):
  1. seed=43 分层抽样 50 条(DOCCI 40 / DetailCaps 10,按源占比,预注册)
  2. 从 webdataset 分片解包对应图像到 pilot 目录(~50 个文件,临时)
  3. 写 pilot_prompts.json:[{id, image, question}]
"""
from __future__ import annotations

import json
import random
import tarfile
from pathlib import Path

SEED = 43
N_PER_SOURCE = {"docci": 40, "detailcaps": 10}
BASE = Path("/scratch/li96/mz9869/dflash_data/longform_fixed_v1")
OUT = BASE / "pilot"


def main() -> int:
    rows = [json.loads(l) for l in open(BASE / "triplets.jsonl")]
    by_src = {}
    for i, r in enumerate(rows):
        by_src.setdefault(r["source"], []).append(i)
    rng = random.Random(SEED)
    picks = []
    for src, k in N_PER_SOURCE.items():
        picks += sorted(rng.sample(by_src[src], k))
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    prompts = []
    byshard = {}
    for i in picks:
        byshard.setdefault(i // 1000, []).append(i)
    for sid, idxs in sorted(byshard.items()):
        with tarfile.open(BASE / "images" / f"shard_{sid:05d}.tar") as tf:
            names = {m.name.split(".")[0]: m for m in tf.getmembers()
                     if not m.name.endswith(".json")}
            for i in idxs:
                m = names[f"{i:06d}"]
                (OUT / "images" / m.name).write_bytes(tf.extractfile(m).read())
                r = rows[i]
                prompts.append({"id": r["sample_id"].replace("/", "_"),
                                "image": m.name, "question": r["question"]})
    json.dump(prompts, open(OUT / "pilot_prompts.json", "w"),
              ensure_ascii=False, indent=1)
    json.dump({"seed": SEED, "n_per_source": N_PER_SOURCE,
               "picked_row_indices": picks},
              open(OUT / "pilot_manifest.json", "w"), indent=1)
    print(f"[pilot-prep] {len(prompts)} prompts, images -> {OUT/'images'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
