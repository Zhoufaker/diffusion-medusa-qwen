"""prep_v2_sources.py — longform_fixed_v2 新源预备(copyq 编排的一环)。

SP(Stanford Paragraph):paragraphs_v1 + 官方 train_split(14,575)→
  按清单从 VG 两个 zip 选择性解取 → sp_local.jsonl(registry --data-path 契约,
  reference 字段名 paragraph 在 REFERENCE_FIELDS 链内原生命中)
LN(Localized Narratives, Open Images train):captions jsonl → 唯一 image_id
  seed=43 采 8,000(留去重/下载损耗余量)→ CVDF 逐张下载
  (失败者若存在→拉 OI 元数据 csv 以 OriginalURL/Flickr 兜底重试,再败剔除计数)
  → ln_local.jsonl(reference 字段名 caption 同上原生命中)
幂等:各步产物存在即跳过;下载 wget -c 续传。
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import zipfile
from pathlib import Path

S = Path("/scratch/li96/mz9869/dflash_data/longform_fixed_v2/staging")
LN_SAMPLE = 8000
SEED = 43          # 协议未覆盖处 seed 族(采样是本脚本新增行为,非 registry 顺序)
CVDF = "https://s3.amazonaws.com/open-images-dataset/train/{}.jpg"
OI_META = ("https://storage.googleapis.com/openimages/2018_04/train/"
           "train-images-boxable-with-rotation.csv")


def prep_sp():
    ann = S / "sp" / "sp_local.jsonl"
    if ann.exists():
        print("[sp] jsonl exists, skip"); return
    paras = json.load(zipfile.ZipFile(S / "sp" / "paragraphs_v1.json.zip")
                      .open("paragraphs_v1.json"))
    train_ids = set(json.load(open(S / "sp" / "train_split.json")))
    rows = [p for p in paras if p["image_id"] in train_ids]
    print(f"[sp] train rows {len(rows)} (expect 14575)")
    want = {}                      # basename -> (subdir, row)
    for p in rows:
        sub = "VG_100K_2" if "VG_100K_2" in p["url"] else "VG_100K"
        want[f"{sub}/{p['image_id']}.jpg"] = p
    (S / "sp" / "images").mkdir(exist_ok=True)
    got = set()
    for zname in ("images.zip", "images2.zip"):
        with zipfile.ZipFile(S / "vg" / zname) as z:
            names = set(z.namelist())
            for member in sorted(set(want) & names):
                out = S / "sp" / "images" / member.replace("/", "_")
                if not out.exists():
                    out.write_bytes(z.read(member))
                got.add(member)
    print(f"[sp] extracted {len(got)}/{len(want)}")
    with open(ann, "w") as f:
        for member in sorted(got):
            p = want[member]
            f.write(json.dumps({
                "id": f"vg_{p['image_id']}",
                "image_path": f"images/{member.replace('/', '_')}",
                "paragraph": p["paragraph"],
            }, ensure_ascii=False) + "\n")
    print(f"[sp] jsonl rows {len(got)}")


def prep_ln():
    ann = S / "ln" / "ln_local.jsonl"
    if ann.exists():
        print("[ln] jsonl exists, skip"); return
    caps = {}
    for line in open(S / "ln" / "open_images_train_v6_captions.jsonl"):
        r = json.loads(line)
        caps.setdefault(r["image_id"], r["caption"])   # 每图取首条 narrative
    ids = sorted(caps)
    rng = random.Random(SEED)
    picked = rng.sample(ids, LN_SAMPLE)
    print(f"[ln] unique images {len(ids)}, sampled {len(picked)}")
    imgdir = S / "ln" / "images"; imgdir.mkdir(exist_ok=True)
    failed = []
    for i, iid in enumerate(picked):
        out = imgdir / f"{iid}.jpg"
        if out.exists() and out.stat().st_size > 0:
            continue
        r = subprocess.run(["wget", "-q", "-c", "-O", str(out),
                            CVDF.format(iid)], timeout=120)
        if r.returncode != 0 or out.stat().st_size == 0:
            out.unlink(missing_ok=True); failed.append(iid)
        if (i + 1) % 500 == 0:
            print(f"[ln] {i + 1}/{LN_SAMPLE} downloaded", flush=True)
    if failed:
        print(f"[ln] CVDF failed {len(failed)} -> Flickr fallback via OI meta csv")
        meta = S / "ln" / "oi_meta.csv"
        if not meta.exists():
            subprocess.run(["wget", "-q", "-c", "-O", str(meta), OI_META],
                           check=True)
        url_of = {}
        fs = set(failed)
        for line in open(meta):
            p = line.split(",")
            if p and p[0] in fs and len(p) > 2:
                url_of[p[0]] = p[2]
        still = []
        for iid in failed:
            out = imgdir / f"{iid}.jpg"
            u = url_of.get(iid)
            ok = False
            if u:
                r = subprocess.run(["wget", "-q", "-O", str(out), u], timeout=120)
                ok = r.returncode == 0 and out.exists() and out.stat().st_size > 0
            if not ok:
                out.unlink(missing_ok=True); still.append(iid)
        failed = still
    kept = [i for i in picked if (imgdir / f"{i}.jpg").exists()]
    with open(ann, "w") as f:
        for iid in kept:
            f.write(json.dumps({"id": f"oi_{iid}",
                                "image_path": f"images/{iid}.jpg",
                                "caption": caps[iid]}, ensure_ascii=False) + "\n")
    json.dump({"sampled": LN_SAMPLE, "seed": SEED, "download_failed": len(failed),
               "kept": len(kept)}, open(S / "ln" / "ln_prep_stats.json", "w"))
    print(f"[ln] jsonl rows {len(kept)}, failed {len(failed)}")


if __name__ == "__main__":
    prep_sp()
    prep_ln()
    sys.exit(0)
