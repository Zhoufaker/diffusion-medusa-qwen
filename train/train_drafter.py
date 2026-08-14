"""train_drafter.py — W2 block diffusion drafter training loop.

Implements reports/w2_train_design.md:
  §1e TrainConfig (DFlash App.A.1 hyperparameters, Eq.4 loss decay gamma=7)
  §1f off-policy monitoring aggregation
  §2  model assembly: reuse external/dflash DFlashDraftModel VERBATIM
      (no reimplementation), frozen shared embed/lm_head from base weights.

Checkpoints store DRAFTER params only, cast bf16 (~2.5G — 旧线 21G 教训).
No GPU job is submitted by this module; smoke training needs separate
authorization (必审档).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from torch import nn

_ROOT = Path(__file__).resolve().parent.parent
for p in (str(_ROOT), str(_ROOT / "external" / "dflash")):
    if p not in sys.path:
        sys.path.insert(0, p)

from dflash.model import DFlashDraftModel, build_target_layer_ids  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config  # noqa: E402

from data.ctx_dataset import (  # noqa: E402
    MASK_TOKEN_ID, IGNORE_INDEX, CtxShardDataset, block_weights,
    collate_packed,
)


@dataclasses.dataclass
class TrainConfig:
    """Design §1e. Paper-sourced values: DFlash App.A.1 + Eq.4."""
    block_size: int = 16
    gamma: float = 7.0            # Eq.4; block 16->7 (10->5, 8->4)
    alpha: float = 1.5            # anchors = min(512, ceil(alpha*L)) — W1 revision
    max_anchors: int = 512
    lr: float = 6e-4
    schedule: str = "cosine"
    warmup_ratio: float = 0.04
    epochs: int = 6
    grad_clip: float = 1.0
    weight_decay: float = 0.01    # paper unstated; AdamW convention, flagged §3.4
    batch_seqs: int = 4
    grad_accum: int = 1
    seed: int = 42
    data_seed: int = 43           # F2/F3: val split + anchor rng derivation
    val_size: int = 500           # F2: fixed 500-index val, seed=43, on-disk
    mask_token_id: int = MASK_TOKEN_ID
    # drafter architecture (design §3.1/§3.2: Qwen3-structure drafter with
    # Qwen2.5-VL-7B dimensions; independent model, not a target-layer clone)
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    head_dim: int = 128
    num_hidden_layers: int = 5
    num_target_layers: int = 28
    vocab_size: int = 152064
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6


def build_drafter(cfg: TrainConfig) -> DFlashDraftModel:
    """§2: DFlashDraftModel reused verbatim; config carries Qwen2.5-VL dims."""
    qcfg = Qwen3Config(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        vocab_size=cfg.vocab_size,
        rope_theta=cfg.rope_theta,
        rms_norm_eps=cfg.rms_norm_eps,
        attention_bias=False,          # drafter's own choice (design §3.2)
        sliding_window=None,
        max_position_embeddings=4096,
    )
    qcfg.layer_types = ["full_attention"] * cfg.num_hidden_layers
    qcfg.num_target_layers = cfg.num_target_layers
    qcfg.block_size = cfg.block_size
    qcfg.dflash_config = {
        "mask_token_id": cfg.mask_token_id,
        "target_layer_ids": build_target_layer_ids(
            cfg.num_target_layers, cfg.num_hidden_layers),
    }
    qcfg._attn_implementation = "sdpa"
    return DFlashDraftModel(qcfg)


def load_frozen_embed_lmhead(cfg: TrainConfig, hf_snapshot: str,
                             lm_head_path: str, device, dtype):
    """Frozen shared embed_tokens (from HF snapshot shards) + lm_head (from
    the existing base_lm_head.safetensors asset). requires_grad=False."""
    from safetensors import safe_open as _so
    idx = json.load(open(Path(hf_snapshot) / "model.safetensors.index.json"))
    shard = idx["weight_map"]["model.embed_tokens.weight"]
    with _so(str(Path(hf_snapshot) / shard), framework="pt") as f:
        w = f.get_tensor("model.embed_tokens.weight")
    embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size, _weight=w.to(dtype))
    with _so(lm_head_path, framework="pt") as f:
        hw = f.get_tensor(sorted(f.keys())[0])
    lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    lm_head.weight.data = hw.to(dtype)
    for m in (embed, lm_head):
        m.requires_grad_(False)
    return embed.to(device), lm_head.to(device)


def weighted_ce(logits: torch.Tensor, labels: torch.Tensor,
                weights: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Sum_i w_i * CE_i / Sum_i w_i over non-ignored positions, chunked over
    the flattened token dim to bound the fp32 softmax workspace (§1g)."""
    V = logits.shape[-1]
    flat_logits = logits.reshape(-1, V)
    flat_labels = labels.reshape(-1)
    flat_w = weights.reshape(-1)
    keep = flat_labels != IGNORE_INDEX
    flat_logits, flat_labels, flat_w = (
        flat_logits[keep], flat_labels[keep], flat_w[keep])
    num = flat_labels.shape[0]
    loss_sum = flat_logits.new_zeros((), dtype=torch.float32)
    for s in range(0, num, chunk):
        ce = nn.functional.cross_entropy(
            flat_logits[s:s + chunk].float(), flat_labels[s:s + chunk],
            reduction="none")
        loss_sum = loss_sum + (ce * flat_w[s:s + chunk]).sum()
    return loss_sum / flat_w.sum().clamp(min=1e-8)


def slot_weights_like(labels: torch.Tensor, block_size: int,
                      gamma: float) -> torch.Tensor:
    """Per-position weights: block-cyclic Eq.4 pattern, zeroed on IGNORE."""
    Bs, N = labels.shape
    w = block_weights(block_size, gamma).to(labels.device)
    full = w.repeat(N // block_size)[None].expand(Bs, -1).clone()
    full[labels == IGNORE_INDEX] = 0.0
    return full


def make_val_split(all_indices: list[int], data_seed: int,
                   val_size: int) -> tuple[list[int], list[int]]:
    """F2: val = fixed `val_size` indices drawn seed-deterministically from
    the full (post-filter) index list; train = hard-excluded remainder."""
    rng = random.Random(data_seed)
    val = sorted(rng.sample(all_indices, min(val_size, len(all_indices))))
    vset = set(val)
    train = [i for i in all_indices if i not in vset]
    return train, val


def offpolicy_stats(metas: list[dict]) -> dict:
    """§1f aggregation: exact share from n_match/n_pos, positional detail
    from mismatch_head16 (approximate; truncation coverage reported)."""
    tot = match = trunc = 0
    decile = defaultdict(int)
    for m in metas:
        tot += m["n_pos"]; match += m["n_match"]
        n_mm = m["n_pos"] - m["n_match"]
        if n_mm > len(m["mismatch_pos"]):
            trunc += 1
        for p in m["mismatch_pos"]:
            decile[min(9, int(10 * p / max(1, m["L"])))] += 1
    return {"offpolicy_share_exact": 1.0 - match / max(1, tot),
            "mismatch_decile_head16": dict(sorted(decile.items())),
            "samples_truncated_head16": trunc, "n_samples": len(metas)}


def run_epoch(model, embed, lm_head, loader, cfg: TrainConfig, device,
              optimizer=None, scheduler=None, log_every: int = 50):
    training = optimizer is not None
    model.train(training)
    tot_loss = tot_steps = 0
    metas_seen: list[dict] = []
    t0 = time.time()
    for step, batch in enumerate(loader):
        ctx = batch["ctx"].to(device)                      # (Bs,5,T,H)
        Bs, five, T, H = ctx.shape
        ctx_flat = ctx.permute(0, 2, 1, 3).reshape(Bs, T, five * H)
        noise_emb = embed(batch["noise_ids"].to(device))
        allow = batch["allow"].to(device)                  # (Bs,N,T+N)
        amask = torch.where(
            allow.unsqueeze(1), 0.0, torch.finfo(noise_emb.dtype).min
        ).to(noise_emb.dtype)                              # (Bs,1,N,T+N)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            hidden = model(
                position_ids=batch["position_ids"].to(device),
                attention_mask=amask,
                noise_embedding=noise_emb,
                target_hidden=ctx_flat,
                is_causal=False,
            )
            logits = lm_head(hidden)
        labels = batch["labels"].to(device)
        w = slot_weights_like(labels, cfg.block_size, cfg.gamma)
        loss = weighted_ce(logits, labels, w)
        if training:
            (loss / cfg.grad_accum).backward()
            if (step + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        tot_loss += float(loss.detach()); tot_steps += 1
        metas_seen.extend(batch["metas"])
        if training and step % log_every == 0:
            print(f"[train] step {step} loss {loss:.4f} "
                  f"lr {optimizer.param_groups[0]['lr']:.2e} "
                  f"{(time.time()-t0):.0f}s", flush=True)
    return {"mean_loss": tot_loss / max(1, tot_steps),
            "offpolicy": offpolicy_stats(metas_seen)}


def save_drafter(model, cfg: TrainConfig, out: Path, tag: str, step: int):
    out.mkdir(parents=True, exist_ok=True)
    sd = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
    torch.save({"state_dict": sd, "config": dataclasses.asdict(cfg),
                "step": step}, out / f"drafter_{tag}.pt")


def self_check(model, embed, lm_head, val_loader, cfg: TrainConfig, device,
               ds: CtxShardDataset, val_idx: list[int]) -> dict:
    """Smoke 门 2+5 (design §1c dual path / §1f plumbing).

    门2: first val batch through the model twice in fp32 (sdpa vs eager),
    max |Δhidden| < 1e-4, plus shape/finiteness asserts.
    门5: offpolicy_stats over the full val set == direct recomputation from
    the manifest (share + decile histogram).
    """
    batch = next(iter(val_loader))
    noise_emb = embed(batch["noise_ids"].to(device)).float()
    ctx = batch["ctx"].to(device).float()
    Bs, five, T, H = ctx.shape
    ctx_flat = ctx.permute(0, 2, 1, 3).reshape(Bs, T, five * H)
    allow = batch["allow"].to(device)
    amask = torch.where(allow.unsqueeze(1), 0.0, torch.finfo(torch.float32).min)
    outs = {}
    orig_impl = model.config._attn_implementation
    orig_dtype = next(model.parameters()).dtype
    model_fp32 = model.float()
    with torch.no_grad():
        for impl in ("sdpa", "eager"):
            model_fp32.config._attn_implementation = impl
            outs[impl] = model_fp32(
                position_ids=batch["position_ids"].to(device),
                attention_mask=amask.float(),
                noise_embedding=noise_emb,
                target_hidden=ctx_flat,
                is_causal=False)
    model.config._attn_implementation = orig_impl
    model.to(orig_dtype)
    diff = float((outs["sdpa"] - outs["eager"]).abs().max())
    expect_shape = (Bs, batch["noise_ids"].shape[1], cfg.hidden_size)
    assert tuple(outs["sdpa"].shape) == expect_shape, \
        f"hidden shape {tuple(outs['sdpa'].shape)} != {expect_shape}"
    assert torch.isfinite(outs["sdpa"]).all(), "non-finite hidden (sdpa)"
    gate2 = diff < 1e-4
    # 门5: aggregation vs manifest direct recompute (metas straight off manifest)
    metas = [{"idx": i, "n_match": ds.samples[i]["n_match"],
              "n_pos": ds.samples[i]["n_pos"],
              "mismatch_pos": [e["pos"] for e in ds.samples[i].get("mismatch_head16", [])],
              "L": ds.samples[i]["L"]} for i in val_idx]
    agg = offpolicy_stats(metas)
    tot = sum(ds.samples[i]["n_pos"] for i in val_idx)
    match = sum(ds.samples[i]["n_match"] for i in val_idx)
    direct_share = 1.0 - match / max(1, tot)
    from collections import defaultdict as _dd
    direct_dec = _dd(int)
    for i in val_idx:
        L = ds.samples[i]["L"]
        for e in ds.samples[i].get("mismatch_head16", []):
            direct_dec[min(9, int(10 * e["pos"] / max(1, L)))] += 1
    gate5 = (abs(agg["offpolicy_share_exact"] - direct_share) < 1e-12
             and agg["mismatch_decile_head16"] == dict(sorted(direct_dec.items())))
    res = {"gate2_sdpa_eager_max_abs_diff": diff, "gate2_pass": gate2,
           "gate5_offpolicy_share": agg["offpolicy_share_exact"],
           "gate5_pass": gate5}
    print(f"[self-check] {res}", flush=True)
    assert gate2 and gate5, f"self-check failed: {res}"
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/scratch/li96/mz9869/dflash_data/ctx_cache_35k")
    ap.add_argument("--manifest-name", default="ctx_manifest.json")
    ap.add_argument("--hf-snapshot", default="/scratch/li96/mz9869/tmp_hf_download/hub/"
                    "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/"
                    "cc594898137f460bfe9f0759e9844b3ce807cfb5")
    ap.add_argument("--lm-head", default="/scratch/li96/mz9869/medusa_assets/base_lm_head.safetensors")
    ap.add_argument("--out-dir", default="/scratch/li96/mz9869/dflash_data/drafter_ckpt")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap TRAIN set to first N non-val indices (F2 smoke)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--init-from", default=None,
                    help="F5: load a drafter checkpoint before training/eval")
    ap.add_argument("--eval-only", action="store_true",
                    help="F5: run the fixed val epoch only, write JSON, exit")
    ap.add_argument("--self-check", action="store_true",
                    help="run 门2/门5 asserts before training")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = TrainConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    torch.manual_seed(cfg.seed)
    device = torch.device(args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = CtxShardDataset(args.cache_dir, args.manifest_name)
    train_idx, val_idx = make_val_split(ds.indices, cfg.data_seed, cfg.val_size)
    json.dump(val_idx, open(out / "val_indices.json", "w"))
    if args.limit:
        train_idx = train_idx[: args.limit]
    json.dump({"n_filtered_short_L_lt_2": len(ds.filtered_short),
               "filtered_short_indices": ds.filtered_short,
               "n_train": len(train_idx), "n_val": len(val_idx),
               "val_file": "val_indices.json", "limit": args.limit,
               "config": dataclasses.asdict(cfg)},
              open(out / "train_manifest.json", "w"), indent=1)

    mk = lambda sub: CtxShardDataset(args.cache_dir, args.manifest_name, indices=sub)  # noqa: E731
    def coll(epoch: int):
        return lambda items: collate_packed(
            items, cfg.block_size, cfg.alpha, cfg.max_anchors,
            cfg.data_seed, epoch)
    def train_loader_for(epoch: int):
        return torch.utils.data.DataLoader(
            mk(train_idx), batch_size=cfg.batch_seqs, shuffle=True,
            collate_fn=coll(epoch), num_workers=0)
    # F3: val pins epoch=0 -> fixed anchors across all evaluations
    val_loader = torch.utils.data.DataLoader(
        mk(val_idx), batch_size=cfg.batch_seqs, shuffle=False,
        collate_fn=coll(0), num_workers=0)

    model = build_drafter(cfg).to(device)
    model.gradient_checkpointing_enable()
    embed, lm_head = load_frozen_embed_lmhead(
        cfg, args.hf_snapshot, args.lm_head, device, torch.bfloat16)
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["state_dict"])
        print(f"[setup] loaded drafter from {args.init_from} "
              f"(step {ck.get('step')})", flush=True)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[setup] drafter trainable params: {n_trainable/1e9:.3f}B", flush=True)

    if args.self_check:
        self_check(model, embed, lm_head, val_loader, cfg, device, ds, val_idx)

    if args.eval_only:
        with torch.no_grad():
            va = run_epoch(model, embed, lm_head, val_loader, cfg, device)
        json.dump(va, open(out / "eval_only.json", "w"), indent=1)
        print(f"[eval-only] val {va['mean_loss']:.6f}", flush=True)
        return 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train_idx) / cfg.batch_seqs / cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = int(cfg.warmup_ratio * total_steps)
    def lr_lambda(s):
        if s < warmup:
            return s / max(1, warmup)
        prog = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best = float("inf")
    gstep = 0
    t_train0 = time.time()
    for ep in range(cfg.epochs):
        tr = run_epoch(model, embed, lm_head, train_loader_for(ep), cfg,
                       device, optimizer, scheduler)
        with torch.no_grad():
            va = run_epoch(model, embed, lm_head, val_loader, cfg, device)
        gstep += steps_per_epoch
        print(f"[epoch {ep}] train {tr['mean_loss']:.4f} val {va['mean_loss']:.4f} "
              f"offpolicy {va['offpolicy']['offpolicy_share_exact']:.4f}", flush=True)
        save_drafter(model, cfg, out, f"step{gstep}", gstep)
        if va["mean_loss"] < best:
            best = va["mean_loss"]
            save_drafter(model, cfg, out, "best", gstep)
        json.dump({"epoch": ep, "train": tr, "val": va},
                  open(out / f"epoch_{ep}_stats.json", "w"), indent=1)

    # 门4 load fidelity: reload best checkpoint, rerun fixed val, |ΔCE|<1e-3
    ck = torch.load(out / "drafter_best.pt", map_location="cpu",
                    weights_only=True)
    model.load_state_dict(ck["state_dict"])
    with torch.no_grad():
        va2 = run_epoch(model, embed, lm_head, val_loader, cfg, device)
    fidelity = abs(va2["mean_loss"] - best)
    peak_gb = (torch.cuda.max_memory_allocated() / 2**30
               if torch.cuda.is_available() else 0.0)
    summary = {"best_val_ce": best, "reloaded_val_ce": va2["mean_loss"],
               "gate4_load_fidelity_abs_diff": fidelity,
               "gate4_pass": fidelity < 1e-3,
               "peak_gpu_mem_gib": round(peak_gb, 2),
               "total_steps": gstep, "epochs": cfg.epochs,
               "train_wall_s": round(time.time() - t_train0, 1),
               "n_train": len(train_idx), "n_val": len(val_idx)}
    json.dump(summary, open(out / "train_summary.json", "w"), indent=1)
    print(f"[summary] {summary}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
