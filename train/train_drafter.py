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
    save_every_steps: int = 2000
    val_fraction: float = 0.02
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/scratch/li96/mz9869/dflash_data/ctx_cache_35k")
    ap.add_argument("--manifest-name", default="ctx_manifest.json")
    ap.add_argument("--hf-snapshot", default="/scratch/li96/mz9869/tmp_hf_download/hub/"
                    "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/"
                    "cc594898137f460bfe9f0759e9844b3ce807cfb5")
    ap.add_argument("--lm-head", default="/scratch/li96/mz9869/medusa_assets/base_lm_head.safetensors")
    ap.add_argument("--out-dir", default="/scratch/li96/mz9869/dflash_data/drafter_ckpt")
    ap.add_argument("--limit", type=int, default=None, help="smoke: cap samples")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = TrainConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    device = torch.device(args.device)

    ds = CtxShardDataset(args.cache_dir, args.manifest_name)
    idxs = ds.indices[: args.limit] if args.limit else ds.indices
    n_val = max(1, int(len(idxs) * cfg.val_fraction))
    train_idx, val_idx = idxs[:-n_val], idxs[-n_val:]
    mk = lambda sub: CtxShardDataset(args.cache_dir, args.manifest_name, indices=sub)  # noqa: E731
    coll = lambda items: collate_packed(  # noqa: E731
        items, cfg.block_size, cfg.alpha, cfg.max_anchors, rng)
    train_loader = torch.utils.data.DataLoader(
        mk(train_idx), batch_size=cfg.batch_seqs, shuffle=True,
        collate_fn=coll, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        mk(val_idx), batch_size=cfg.batch_seqs, shuffle=False,
        collate_fn=coll, num_workers=0)

    model = build_drafter(cfg).to(device)
    model.gradient_checkpointing_enable()
    embed, lm_head = load_frozen_embed_lmhead(
        cfg, args.hf_snapshot, args.lm_head, device, torch.bfloat16)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[setup] drafter trainable params: {n_trainable/1e9:.3f}B", flush=True)

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

    out = Path(args.out_dir)
    best = float("inf")
    gstep = 0
    for ep in range(cfg.epochs):
        tr = run_epoch(model, embed, lm_head, train_loader, cfg, device,
                       optimizer, scheduler)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
