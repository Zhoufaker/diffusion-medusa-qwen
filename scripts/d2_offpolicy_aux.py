"""d2_offpolicy_aux.py — D2 off-policy 附属实验(协议预注册,~10 SU)。

val-500 上:target A100-TF 前向取每 rollout 位置 verify-argmax(图像从
onpolicy tar 现场解包——散件已收编);drafter 以每个合法位置为 anchor 的
slot-1 top-1 预测。按"缓存 token == verify-argmax"分层(≈96%/4%),
统计两层的 drafter top-1 命中 verify-argmax 率。
预注册方向性预期:4% 层命中率显著低于 96% 层。
产出:off-policy 对整体接受率损失的上界 ≈ 4% × 两层命中率差。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

_ROOT = Path(__file__).resolve().parent.parent
for p in (str(_ROOT), str(_ROOT / "external" / "dflash")):
    if p not in sys.path:
        sys.path.insert(0, p)

from data.ctx_dataset import CtxShardDataset, MASK_TOKEN_ID  # noqa: E402
from decode.common import load_base, make_image_inputs, mask_phantom_  # noqa: E402
from decode.dflash_vlm import load_drafter_for_inference  # noqa: E402


def stage_inputs(tar_path: str, val_indices: list[int], work: Path):
    """解包 rollout_prompts.json + val 集所需图像(不重散全部 30K 文件)。"""
    subprocess.run(["tar", "-xf", tar_path, "-C", str(work),
                    "onpolicy_data/rollout_prompts.json"], check=True)
    prompts = json.load(open(work / "onpolicy_data/rollout_prompts.json"))
    names = [f"onpolicy_data/images/{prompts[i]['image']}" for i in val_indices]
    lst = work / "img.list"
    lst.write_text("\n".join(names) + "\n")
    subprocess.run(["tar", "-xf", tar_path, "-C", str(work), "-T", str(lst)],
                   check=True)
    return prompts, work / "onpolicy_data/images"


@torch.inference_mode()
def drafter_slot1_preds(drafter, embed, lm_head, ctx, ids, P, T, device,
                        chunk_anchors: int = 128):
    """每个合法 anchor p∈[P,T-2] 的 slot-1 top-1(打包 + 4D mask,分块)。"""
    from data.ctx_dataset import pack_blocks, allowed_bool_mask
    B = drafter.block_size
    H = ctx.shape[-1]
    anchors_all = list(range(P, T - 1))
    ctx_flat = ctx.permute(1, 0, 2).reshape(1, T, -1).to(device, torch.bfloat16)
    preds = {}
    for s in range(0, len(anchors_all), chunk_anchors):
        anc = anchors_all[s:s + chunk_anchors]
        pk = pack_blocks(ids, anc, B, MASK_TOKEN_ID)
        allow = allowed_bool_mask(pk.anchor_pos, pk.block_of, T,
                                  len(anc) * B)
        amask = torch.where(allow[None, None].to(device), 0.0,
                            torch.finfo(torch.bfloat16).min).to(torch.bfloat16)
        pos = torch.cat([torch.arange(T), pk.noise_pos])[None].to(device)
        hid = drafter(position_ids=pos, attention_mask=amask,
                      noise_embedding=embed(pk.noise_ids[None].to(device)),
                      target_hidden=ctx_flat, is_causal=False)
        logits = lm_head(hid)[0]                       # (K*B, V)
        top1 = mask_phantom_(logits.float()).argmax(-1)
        for k, p_ in enumerate(anc):
            preds[p_] = int(top1[k * B + 1])           # slot-1 -> 预测 ids[p+1]
    return preds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/scratch/li96/mz9869/dflash_data/ctx_cache_35k")
    ap.add_argument("--val-indices", default="/scratch/li96/mz9869/dflash_data/drafter_ckpt_full/val_indices.json")
    ap.add_argument("--onpolicy-tar", default="/scratch/li96/mz9869/archives/onpolicy_data_legacy.tar")
    ap.add_argument("--drafter-ckpt", required=True)
    ap.add_argument("--hf-snapshot", default="/scratch/li96/mz9869/tmp_hf_download/hub/"
                    "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/"
                    "cc594898137f460bfe9f0759e9844b3ce807cfb5")
    ap.add_argument("--lm-head", default="/scratch/li96/mz9869/medusa_assets/base_lm_head.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work-dir", default=os.environ.get("PBS_JOBFS", tempfile.gettempdir()))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    val_idx = json.load(open(args.val_indices))
    work = Path(args.work_dir) / "offpolicy_stage"
    work.mkdir(parents=True, exist_ok=True)
    prompts, images_dir = stage_inputs(args.onpolicy_tar, val_idx, work)
    base, processor = load_base("Qwen/Qwen2.5-VL-7B-Instruct", args.device)
    drafter, embed, lm_head, _ = load_drafter_for_inference(
        args.drafter_ckpt, args.hf_snapshot, args.lm_head,
        torch.device(args.device), torch.bfloat16)
    ds = CtxShardDataset(args.cache_dir, indices=val_idx)

    S = {"on": {"n": 0, "hit": 0}, "off": {"n": 0, "hit": 0}}
    for j in range(len(ds)):
        it = ds[j]
        idx = it["meta"]["idx"]
        spans = it["spans"].tolist(); P, T = spans[4], spans[5]
        ids = it["ids"]
        p = prompts[idx]
        inputs = make_image_inputs(processor, p["question"],
                                   images_dir / p["image"], args.device)
        out = base(input_ids=ids[None].to(args.device),
                   attention_mask=torch.ones(1, T, dtype=torch.long,
                                             device=args.device),
                   pixel_values=inputs.get("pixel_values"),
                   image_grid_thw=inputs.get("image_grid_thw"),
                   use_cache=False)
        # 位置 p-1 的 logits 预测 ids[p]:verify-argmax over 位置 [P-1, T-2]
        vlog = mask_phantom_(out.logits[0, P - 1: T - 1, :].float())
        vargmax = vlog.argmax(-1)                       # 对应 ids[P..T-1]
        del out
        dpred = drafter_slot1_preds(drafter, embed, lm_head, it["ctx"], ids,
                                    P, T, args.device)
        # anchor p 的 slot-1 预测目标 = 位置 p+1;verify-argmax 同位对照
        for p_ in range(P, T - 1):
            va = int(vargmax[p_ - P + 1]) if p_ - P + 1 < vargmax.shape[0] else None
            if va is None:
                continue
            layer = "on" if int(ids[p_ + 1]) == va else "off"
            S[layer]["n"] += 1
            S[layer]["hit"] += int(dpred[p_] == va)
        if j % 50 == 0:
            print(f"[aux] {j + 1}/{len(ds)}", flush=True)

    on_r = S["on"]["hit"] / max(1, S["on"]["n"])
    off_r = S["off"]["hit"] / max(1, S["off"]["n"])
    off_share = S["off"]["n"] / max(1, S["on"]["n"] + S["off"]["n"])
    res = {"n_on": S["on"]["n"], "n_off": S["off"]["n"],
           "hit_rate_on_layer": on_r, "hit_rate_off_layer": off_r,
           "off_layer_share": off_share,
           "prereg_direction_holds": off_r < on_r,
           "acceptance_loss_upper_bound": off_share * (on_r - off_r)}
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
