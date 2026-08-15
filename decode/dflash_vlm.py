"""dflash_vlm.py — DFlash block-diffusion drafter inference on Qwen2.5-VL.

D2 臂③ 适配器(reports/d2_eval_protocol.md)。z-lab dflash_generate 的 VLM
移植,机制逐行对应(draft KV cache + crop、bonus 锚点、cumprod 接受):
  * prefill 带 pixel_values/image_grid_thw,预处理经 decode.common.
    make_image_inputs——与 W1 抽取 PREPROCESS_SPEC 同源(processor 默认
    像素参数,无 501760 cap)
  * target 侧 verify/prefill 不传 position_ids(M-RoPE 由模型内部
    rope_deltas 处理,与 vanilla_greedy 同一已验证路径)
  * drafter 侧一维扁平 position_ids(训练约定 w2_train_design §1d)
  * mask_token=151662,block 16;greedy 输出用 argmax_masked
    (EFFECTIVE_VOCAB=151936,与臂①② 同口径)
  * 模型组装复用 train.train_drafter 的加载函数(禁止第二套实现)
  * D3 埋点:逐 cycle 记每槽位 drafter top-1 概率 / accept / reject 处
    verify argmax
  * G2 自测钩子 run_twice_check:同输入双跑,输出逐 token 一致
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from transformers import DynamicCache

_ROOT = Path(__file__).resolve().parent.parent
for p in (str(_ROOT), str(_ROOT / "external" / "dflash")):
    if p not in sys.path:
        sys.path.insert(0, p)

from decode.common import EFFECTIVE_VOCAB, argmax_masked, mask_phantom_  # noqa: E402
from data.ctx_dataset import MASK_TOKEN_ID  # noqa: E402


def load_drafter_for_inference(ckpt_path: str, hf_snapshot: str,
                               lm_head_path: str, device,
                               dtype=torch.bfloat16):
    """训练档 → 推理组装,全部复用训练侧函数(无第二套实现)。"""
    from train.train_drafter import (TrainConfig, build_drafter,
                                     load_frozen_embed_lmhead)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    fields = {f.name for f in __import__("dataclasses").fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in ck["config"].items() if k in fields})
    model = build_drafter(cfg)
    model.load_state_dict(ck["state_dict"])
    model = model.to(device=device, dtype=dtype).eval()
    embed, lm_head = load_frozen_embed_lmhead(
        cfg, hf_snapshot, lm_head_path, device, dtype)
    return model, embed, lm_head, cfg


def extract_ctx(hidden_states, layer_ids) -> torch.Tensor:
    """5 层 cat(与 extract_ctx_features.py / z-lab 同式:tuple 索引 l+1)。"""
    return torch.cat([hidden_states[l + 1] for l in layer_ids], dim=-1)


@torch.inference_mode()
def dflash_vlm_generate(target, drafter, embed, lm_head, inputs,
                        max_new_tokens: int, eos_id: int,
                        block_size: int = 16,
                        mask_token_id: int = MASK_TOKEN_ID,
                        d3_log: list | None = None):
    """单 prompt 生成。返回 (tokens, stats)。

    stats: cycles、accept_lengths(含 bonus 的每轮推进量)、
    tau_accept_only / tau_with_bonus 两口径(协议:现场写死——
    tau_accept_only = mean(每轮纯接受 draft 数) = mean(acc_len),
    tau_with_bonus  = mean(每轮推进量) = mean(acc_len + 1)),
    prefill_s / decode_s。
    """
    device = target.device
    P = inputs["input_ids"].shape[1]
    layer_ids = drafter.target_layer_ids
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    out = target(**inputs, use_cache=True, output_hidden_states=True)
    past_t = out.past_key_values
    past_d = DynamicCache()
    first = argmax_masked(out.logits[0, -1, :])
    # B1 谱系:训练特征 = target fp16 前向 → bf16 落盘;推理 ctx 同源地
    # 经 bf16 再入 drafter,消除 fp16-ctx 与训练分布的 dtype 失配
    ctx = extract_ctx(out.hidden_states, layer_ids).to(torch.bfloat16)  # (1,P,5H)
    del out
    torch.cuda.synchronize() if device.type == "cuda" else None
    prefill_s = time.perf_counter() - t0

    max_len = P + max_new_tokens
    ids = torch.full((1, max_len + block_size), mask_token_id,
                     dtype=torch.long, device=device)
    ids[0, :P] = inputs["input_ids"][0]
    ids[0, P] = first
    pos = torch.arange(ids.shape[1], device=device).unsqueeze(0)

    t1 = time.perf_counter()
    accept_lengths = []
    start = P
    while start < max_len:
        blk = ids[:, start:start + block_size].clone()
        noise_emb = embed(blk)
        hid = drafter(
            target_hidden=ctx,
            noise_embedding=noise_emb,
            position_ids=pos[:, past_d.get_seq_length(): start + block_size],
            past_key_values=past_d,
            use_cache=True,
            is_causal=False,
        )[:, 1 - block_size:, :]
        draft_logits = lm_head(hid)                          # (1, B-1, V)
        past_d.crop(start)
        draft_pred = mask_phantom_(draft_logits.float()).argmax(-1)
        blk[:, 1:] = draft_pred

        vout = target(blk, past_key_values=past_t, use_cache=True,
                      output_hidden_states=True)
        vlogits = mask_phantom_(vout.logits.float())
        posterior = vlogits.argmax(-1)                       # (1, B)
        match = (blk[:, 1:] == posterior[:, :-1])
        acc = int(match.cumprod(dim=1).sum())
        ids[:, start:start + acc + 1] = blk[:, :acc + 1]
        ids[0, start + acc + 1] = posterior[0, acc]
        if d3_log is not None:
            probs = torch.softmax(draft_logits.float(), dim=-1)
            top1p = probs.max(-1).values[0]                  # (B-1,)
            d3_log.append({
                "cycle": len(accept_lengths), "anchor_pos": start,
                "slot_top1_prob": [round(float(x), 5) for x in top1p],
                "slot_accept": match[0].tolist(),
                "accept_len": acc,
                "reject_verify_argmax": (int(posterior[0, acc])
                                         if acc < block_size - 1 else None),
            })
        start += acc + 1
        past_t.crop(start)
        # B1 谱系:同上,verify 增量特征也走 bf16(与训练缓存 dtype 一致)
        ctx = extract_ctx(vout.hidden_states,
                          layer_ids)[:, :acc + 1, :].to(torch.bfloat16)
        accept_lengths.append(acc)
        del vout
        if eos_id in ids[0, P:start].tolist():
            break
    torch.cuda.synchronize() if device.type == "cuda" else None
    decode_s = time.perf_counter() - t1

    seq = ids[0, P:min(start, max_len)].tolist()
    if eos_id in seq:
        seq = seq[: seq.index(eos_id) + 1]
    n = max(1, len(accept_lengths))
    stats = {
        "cycles": len(accept_lengths),
        "accept_lengths": accept_lengths,
        "tau_accept_only": sum(accept_lengths) / n,
        "tau_with_bonus": sum(a + 1 for a in accept_lengths) / n,
        "prefill_s": prefill_s, "decode_s": decode_s,
        "n_tokens": len(seq),
    }
    return seq, stats


def run_twice_check(target, drafter, embed, lm_head, inputs,
                    max_new_tokens: int, eos_id: int, **kw) -> dict:
    """G2 自测钩子:同输入双跑,输出逐 token 一致。"""
    a, _ = dflash_vlm_generate(target, drafter, embed, lm_head, inputs,
                               max_new_tokens, eos_id, **kw)
    b, _ = dflash_vlm_generate(target, drafter, embed, lm_head, inputs,
                               max_new_tokens, eos_id, **kw)
    return {"identical": a == b, "len_a": len(a), "len_b": len(b),
            "first_div": next((i for i, (x, y) in enumerate(zip(a, b))
                               if x != y), None)}
