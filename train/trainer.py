"""Trainer — full training loop for Linked Medusa heads.

Spec ref: linked_medusa_spec.md §7 + §3 + §9.4.

Design contract (read before changing anything):

  - Optimizer is bitsandbytes 8-bit AdamW. This is NOT a fallback. The only
    other choice we expose is the legacy fp32 AdamW, and it is reserved for
    smoke testing (no GPU available). On GPU runs we hard-require AdamW8bit
    because fp32 AdamW state would OOM the V100 (spec §9.4).
  - Mixed precision is fp16 + GradScaler. NOT bf16 (V100 doesn't support it).
  - Loss is multi-head weighted CE with `ignore_index=-100`, offsets per
    head_k = k+1 (spec §6.4). See train/loss.py for the actual math.
  - lm_heads are seeded from the base model's lm_head (spec §2 confirmed
    decision), giving near-zero head_0 loss on real cache at step 0.

The training loop is a flat for-loop over `total_steps`. We exhaust the
DataLoader as many times as needed (synthetic dataset has fixed size; cache
has 46k samples). At each step:

    forward -> compute_loss -> scaler.scale(loss).backward()
    -> scaler.unscale_(optimizer) -> clip_grad_norm_
    -> scaler.step(optimizer) -> scaler.update() -> scheduler.step()
    -> zero_grad

Every `log_every` we print loss + LR + (every now and then) grad norm. Every
`eval_every` we run val. Every `save_every` we write a checkpoint.

Checkpoint format: torch.save({model, optimizer, scheduler, scaler, step,
cfg}) into <output_dir>/ckpt_step<step>.pt. Latest checkpoint is also
symlinked as <output_dir>/latest.pt for trivial resume.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, random_split

# Ensure project root on sys.path so `from data import ...` works when this
# module is imported by scripts/train.py.
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data import CachedVLMDataset, SyntheticVLMDataset, collate_fn
from model import LinkedMedusaHeads, load_base_lm_head_weight
from train.loss import compute_loss, IGNORE_INDEX
from train.scheduler import cosine_warmup_schedule
from train.evaluate import evaluate


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _human_count(n: int) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.2f}{unit}" if isinstance(n, float) else f"{n}{unit}"
        n /= 1000.0
    return f"{n:.2f}T"


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _parse_eval_split(eval_split: Any) -> int | None:
    """Parse cfg.dataset.eval_split into a 'last N' count, or None for random.

    Accepted values:
      None / null / 0 / 'none'  -> None  (use val_fraction random_split)
      int N                     -> N
      'last_<N>'                -> N
    """
    if eval_split is None or eval_split == 0 or eval_split == "none":
        return None
    if isinstance(eval_split, int):
        return eval_split
    if isinstance(eval_split, str) and eval_split.startswith("last_"):
        try:
            return int(eval_split.split("_", 1)[1])
        except ValueError as e:
            raise ValueError(
                f"bad eval_split format {eval_split!r}: expected 'last_<int>'"
            ) from e
    raise ValueError(f"unrecognized dataset.eval_split: {eval_split!r}")


def build_dataset(cfg: Dict[str, Any]) -> tuple[Dataset, Dataset | None]:
    """Factory selecting the dataset implementation from cfg.dataset.kind.

    Returns:
        (train_set, val_set)  — val_set may be None if no split requested.

    Split selection rules:
      - dataset.eval_split = 'last_N' (or int N)
            train = first len(ds) - N samples, val = last N samples (by
            numeric stem for cached; by index for synthetic).
            This is the production split for cached training — reproducible
            across runs, no RNG seeding required.
      - dataset.eval_split = None and dataset.val_fraction > 0
            random_split with val_fraction (the v1 default for development).
      - dataset.val_fraction <= 0 too
            val = None (eval calls become no-ops in Trainer.fit()).
    """
    ds_cfg = cfg["dataset"]
    kind = ds_cfg["kind"]
    eval_n = _parse_eval_split(ds_cfg.get("eval_split"))

    if kind == "synthetic":
        model_vocab = int(cfg["model"]["vocab_size"])
        # Synthetic tokens use the EFFECTIVE tokenizer vocab (151936) when the
        # model is the real Qwen lm_head; for smoke tests with a tiny model
        # we clamp to the model's own vocab so CE has valid targets. Real
        # cache always uses effective_vocab=151936 (spec §2 + §6.2).
        effective = min(151936, model_vocab)
        full = SyntheticVLMDataset(
            num_samples=int(ds_cfg["num_samples"]),
            seq_len_range=tuple(ds_cfg["seq_len_range"]),
            hidden_dim=int(cfg["model"]["hidden_dim"]),
            vocab_size=effective,
            seed=int(cfg["seed"]),
        )
        if eval_n is not None and eval_n > 0:
            if eval_n >= len(full):
                raise ValueError(
                    f"dataset.eval_split={eval_n} >= synthetic num_samples="
                    f"{len(full)}; nothing left for train."
                )
            train_set = Subset(full, list(range(len(full) - eval_n)))
            val_set = Subset(full, list(range(len(full) - eval_n, len(full))))
            return train_set, val_set
        # fall through to caller's random_split path
        return full, None

    if kind == "cached":
        cache_dir = str(ds_cfg["cache_dir"])
        max_length = int(ds_cfg.get("max_length", 256))
        H = int(cfg["model"]["hidden_dim"])
        if eval_n is not None and eval_n > 0:
            train_set = CachedVLMDataset(
                cache_dir=cache_dir, max_length=max_length,
                hidden_dim_expected=H, eval_split_size=eval_n, split="train",
            )
            val_set = CachedVLMDataset(
                cache_dir=cache_dir, max_length=max_length,
                hidden_dim_expected=H, eval_split_size=eval_n, split="eval",
            )
        else:
            train_set = CachedVLMDataset(
                cache_dir=cache_dir, max_length=max_length, hidden_dim_expected=H,
            )
            val_set = None
        max_samples = ds_cfg.get("max_samples")
        if max_samples is not None:
            n = min(int(max_samples), len(train_set))
            train_set = Subset(train_set, list(range(n)))
        return train_set, val_set
    raise ValueError(f"unknown dataset.kind: {kind}")


def build_optimizer(model: nn.Module, train_cfg: Dict[str, Any], device: torch.device,
                    eagle_cond: bool = False) -> torch.optim.Optimizer:
    """Build AdamW8bit by default; allow torch AdamW only for CPU smoke tests."""
    name = train_cfg.get("optimizer", "adamw8bit").lower()
    lr = float(train_cfg["lr"])
    wd = float(train_cfg["weight_decay"])
    betas = tuple(train_cfg.get("betas", [0.9, 0.999]))

    if name == "adamw8bit":
        if device.type != "cuda":
            raise RuntimeError(
                "optimizer=adamw8bit requires CUDA. For CPU smoke testing, "
                "switch to optimizer=adamw via --override training.optimizer=adamw "
                "or use --smoke."
            )
        from bitsandbytes.optim import AdamW8bit
        if eagle_cond:
            # C1 locked param groups: heads @ 1e-4, bonus_proj @ 5e-4.
            head_params = [p for n, p in model.named_parameters() if n != "bonus_proj.weight"]
            return AdamW8bit([
                {"params": head_params, "lr": lr},
                {"params": [model.bonus_proj.weight], "lr": 5e-4},
            ], betas=betas, weight_decay=wd)
        return AdamW8bit(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
    if name == "adamw":
        if eagle_cond:
            head_params = [p for n, p in model.named_parameters() if n != "bonus_proj.weight"]
            return torch.optim.AdamW([
                {"params": head_params, "lr": lr},
                {"params": [model.bonus_proj.weight], "lr": 5e-4},
            ], betas=betas, weight_decay=wd)
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
    raise ValueError(f"unknown training.optimizer: {name}")


# ----------------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------------


@dataclass
class TrainerState:
    step: int = 0
    epoch: int = 0
    best_eval_loss: float = float("inf")


class Trainer:
    def __init__(self, cfg: Dict[str, Any], device: torch.device | str = "auto"):
        self.cfg = cfg
        self.device = self._pick_device(device)
        self.state = TrainerState()

        torch.manual_seed(int(cfg["seed"]))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(cfg["seed"]))

        # ---- data --------------------------------------------------------
        train_set, val_set = build_dataset(cfg)
        if val_set is None:
            # No explicit eval_split — fall back to random val split.
            val_frac = float(cfg["dataset"].get("val_fraction", 0.05))
            if val_frac > 0.0 and len(train_set) >= 2:
                n_val = max(1, int(len(train_set) * val_frac))
                n_train = len(train_set) - n_val
                train_set, val_set = random_split(
                    train_set,
                    [n_train, n_val],
                    generator=torch.Generator().manual_seed(42),
                )
        print(f"[init] train_set: {len(train_set)} samples" + (
            f", val_set: {len(val_set)} samples" if val_set is not None else ", val_set: None"
        ))

        train_cfg = cfg["training"]
        # num_workers=0 on CPU avoids forking the synthetic dataset's torch
        # generators (which complicates determinism on some platforms).
        nw = int(train_cfg.get("num_workers", 4)) if self.device.type == "cuda" else 0
        pin = bool(train_cfg.get("pin_memory", True)) and self.device.type == "cuda"

        self.train_loader = DataLoader(
            train_set,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=nw,
            pin_memory=pin,
            drop_last=True,
        )
        self.val_loader = (
            DataLoader(
                val_set,
                batch_size=int(train_cfg["batch_size"]),
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=nw,
                pin_memory=pin,
            )
            if val_set is not None
            else None
        )

        # ---- model -------------------------------------------------------
        m_cfg = cfg["model"]
        self.model = LinkedMedusaHeads(
            hidden_dim=int(m_cfg["hidden_dim"]),
            vocab_size=int(m_cfg["vocab_size"]),
            num_heads=int(m_cfg["num_heads"]),
            num_blocks=int(m_cfg["num_blocks"]),
            expansion=int(m_cfg["expansion"]),
            detach_chain=bool(m_cfg.get("detach_chain", False)),
        ).to(self.device)
        if self.model.detach_chain:
            print("[init] detach_chain=True — inter-head hidden is stop-grad "
                  "(fallback mode; linked gradient flow DISABLED)")

        if bool(m_cfg.get("init_lm_head_from_base", True)):
            path = m_cfg["base_lm_head_path"]
            print(f"[init] loading base lm_head from {path}")
            base_W = load_base_lm_head_weight(path)
            self.model.init_lm_heads_from_base(base_W.to(self.device))
            del base_W

        self.eagle_cond = bool(m_cfg.get("eagle_cond", False))
        self.embed_layer = None
        if self.eagle_cond:
            model_id = str(m_cfg.get(
                "base_model_id",
                "/scratch/li96/mz9869/tmp_hf_download/hub/"
                "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/"
                "cc594898137f460bfe9f0759e9844b3ce807cfb5",
            ))
            if not os.path.isdir(model_id):
                raise FileNotFoundError(
                    f"eagle_cond base_model_id must be a local snapshot directory "
                    f"(compute nodes have no HF hub access); not found: {model_id}"
                )
            # Load ONLY embed_tokens on CPU (fp16, frozen) — never a GPU copy.
            # Per-step lookup stays on CPU; only the small (B,L,H) result moves.
            self.embed_layer = self._load_embed_tokens_only(model_id)  # stays on CPU
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Partial warm-start (B1: heads 0..2 from Phase A; C1: full B2 + fresh
        # bonus_proj). MUST run AFTER init_lm_heads_from_base when enabled.
        ws = m_cfg.get("warm_start_from")
        if ws:
            print(f"[init] warm-starting matching heads from {ws}")
            sd_full = torch.load(str(ws), map_location="cpu", weights_only=False)
            head_sd = sd_full.get("model", sd_full.get("state_dict", sd_full))
            res = self.model.load_state_dict(head_sd, strict=False)
            if res.unexpected_keys:
                raise RuntimeError(
                    f"warm_start_from has keys not in this model (arch mismatch): "
                    f"{res.unexpected_keys[:8]}"
                )
            if self.eagle_cond:
                if set(res.missing_keys) != {"bonus_proj.weight"}:
                    raise RuntimeError(
                        f"C1 warm-start from B2: expected missing exactly "
                        f"bonus_proj.weight, got {res.missing_keys}"
                    )
                print("[init] C1 warm-start: B2 heads loaded; bonus_proj stays zero-init")
            else:
                loaded = len(head_sd)
                warm_heads = sorted({k.split(".")[1] for k in head_sd if k.startswith("heads.")})
                fresh = sorted({k.split(".")[1] for k in res.missing_keys if k.startswith("heads.")})
                print(f"[init] warm-start: loaded {loaded} tensors into heads {warm_heads}; "
                      f"heads {fresh} stay fresh-init ({len(res.missing_keys)} missing keys, expected)")
            del sd_full, head_sd

        # ---- optimizer / scheduler / scaler ------------------------------
        self.optimizer = build_optimizer(self.model, train_cfg, self.device,
                                           eagle_cond=self.eagle_cond)
        print(f"[init] optimizer = {type(self.optimizer).__module__}.{type(self.optimizer).__name__}")
        self.scheduler = cosine_warmup_schedule(
            self.optimizer,
            warmup_steps=int(train_cfg["warmup_steps"]),
            total_steps=int(train_cfg["total_steps"]),
            final_lr_multiplier=float(train_cfg["final_lr_multiplier"]),
        )

        self.use_amp = bool(train_cfg.get("use_amp", True)) and self.device.type == "cuda"
        # init_scale=2**10 (1024) instead of the PyTorch default 2**16 (65536).
        #
        # Rationale: in Linked Medusa, head_k's lm_head receives ~(k+1)*h_t as
        # input at init (because the body ResBlocks are zero-init residuals,
        # exactly identity at step 0; so h_{k-1}' = k * h_t). For k>=1 this
        # amplifies the input to head_k's lm_head by (k+1)x compared to a
        # vanilla single-head model. The lm_head weights are copied from the
        # base, so head_k's logits also amplify k+1 x, sharpening softmax on
        # the wrong token (target_k = tokens[t+k]) — hence the legitimately
        # large head_1/head_2 initial losses (~27 / ~49 on Qwen2.5-VL cache).
        # That doesn't itself overflow CE, but it inflates the grad w.r.t.
        # head_k's lm_head weight by (k+1)x via the outer product with the
        # amplified h_input. With |h_t| ~ 5 and (k+1)=3, grad_W entries can
        # reach ~15 in fp32. Scaled by the default 2**16 they become ~1e6,
        # far above fp16 max 65504 → every backward overflows and the scaler
        # skips every optimizer.step() for the first ~14 halvings.
        # 2**10 puts initial scaled grads comfortably in fp16 range. The
        # scaler can still grow from this seed (growth_factor=2, every 2000
        # good steps), so it ends up where the network needs it.
        scaler_init_scale = float(train_cfg.get("grad_scaler_init_scale", 2.0 ** 10))
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp, init_scale=scaler_init_scale,
        )
        if self.use_amp:
            print(f"[init] GradScaler init_scale = {scaler_init_scale:.0f}")

        self.grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.loss_weights = list(train_cfg["loss_weights"])
        self.total_steps = int(train_cfg["total_steps"])

        # ---- logging / IO -----------------------------------------------
        log_cfg = cfg["logging"]
        self.output_dir = Path(log_cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = int(log_cfg["log_every"])
        self.eval_every = int(log_cfg["eval_every"])
        self.save_every = int(log_cfg["save_every"])
        self.keep_recent_n = int(log_cfg.get("keep_recent_n", 3))
        # Tracks paths of "regular" (untagged) ckpts, oldest-first, so we can
        # prune all but the most recent `keep_recent_n` after each save.
        # Tagged checkpoints (best / final) are NEVER tracked here and never
        # auto-pruned. On resume this list starts empty — the cleanup policy
        # only governs ckpts written in the current run.
        self._recent_ckpts: List[Path] = []
        # Persist a snapshot of the merged config for reproducibility.
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2, default=str)
            f.write("\n")

        n_params = count_parameters(self.model)
        print(f"[init] model has {_human_count(n_params)} trainable params")
        if self.eagle_cond:
            fnorm = float(self.model.bonus_proj.weight.detach().float().norm().item())
            print(f"[init] bonus_proj.weight Frobenius norm (pre-train) = {fnorm:.6f}")
        print(f"[init] device={self.device}, use_amp={self.use_amp}, "
              f"grad_accum={self.grad_accum_steps}, total_steps={self.total_steps}")
        if self.device.type == "cuda":
            alloc_gb = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            print(f"[init] cuda memory_allocated = {alloc_gb:.2f} GiB "
                  f"(expect ~14.0 GiB B1-resident; embed table is CPU-only)")

        # ---- optional resume --------------------------------------------
        resume = train_cfg.get("resume_from")
        if resume:
            self.load_checkpoint(resume)

    @staticmethod
    def _load_embed_tokens_only(snapshot_dir: str) -> nn.Embedding:
        """Load frozen nn.Embedding from snapshot safetensors — no full base."""
        from safetensors import safe_open

        snap = Path(snapshot_dir)
        index_path = snap / "model.safetensors.index.json"
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        key = "model.embed_tokens.weight"
        if key not in weight_map:
            raise KeyError(f"{key} not in {index_path}")
        shard = snap / weight_map[key]
        print(f"[init] eagle_cond=True — loading embed_tokens ONLY from {shard.name}")
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            W = f.get_tensor(key)  # (V, H)
        V, H = W.shape
        emb = nn.Embedding(V, H)
        with torch.no_grad():
            emb.weight.copy_(W.float())  # copy then cast storage to fp16 below
        emb.weight.data = emb.weight.data.half().contiguous()
        emb.weight.requires_grad_(False)
        emb.eval()
        # MUST remain on CPU — training peak on V100 has no room for the 1.09GB table.
        print(f"[init] embed_tokens ready on CPU: V={V} H={H} dtype={emb.weight.dtype} "
              f"size={emb.weight.numel() * emb.weight.element_size() / 1e9:.2f}GB (frozen)")
        del W
        return emb

    # ------- device / dtype --------------------------------------------------

    @staticmethod
    def _pick_device(device: torch.device | str) -> torch.device:
        if isinstance(device, torch.device):
            return device
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    # ------- checkpoint ------------------------------------------------------

    def save_checkpoint(self, tag: str | None = None) -> Path:
        """Save a checkpoint.

        If tag is None: writes ckpt_step<N>.pt and prunes regular checkpoints
        beyond `keep_recent_n`. The newly-written ckpt is always preserved
        (it's the most recent).
        If tag is given (e.g. 'best', 'final'): writes ckpt_<tag>.pt and is
        NEVER auto-pruned. Subsequent calls with the same tag overwrite.
        Symlink `latest.pt` is updated regardless.
        """
        step = self.state.step
        name = f"ckpt_step{step}.pt" if tag is None else f"ckpt_{tag}.pt"
        path = self.output_dir / name
        torch.save(
            {
                "step": step,
                "epoch": self.state.epoch,
                "best_eval_loss": self.state.best_eval_loss,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "cfg": self.cfg,
            },
            str(path),
        )
        print(f"[ckpt] saved -> {path}")

        # Track + prune regular checkpoints. Never prune tagged ones.
        if tag is None:
            self._recent_ckpts.append(path)
            while len(self._recent_ckpts) > self.keep_recent_n:
                old = self._recent_ckpts.pop(0)
                try:
                    if old.is_file():
                        old.unlink()
                        print(f"[ckpt] pruned -> {old}")
                except OSError as e:
                    print(f"[ckpt] WARN: failed to prune {old}: {e}")

        latest = self.output_dir / "latest.pt"
        try:
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            latest.symlink_to(path.name)
        except OSError:
            import shutil

            shutil.copy(str(path), str(latest))
        return path

    def load_checkpoint(self, path: str | os.PathLike) -> None:
        print(f"[ckpt] loading {path}")
        sd = torch.load(str(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(sd["model"])
        self.optimizer.load_state_dict(sd["optimizer"])
        self.scheduler.load_state_dict(sd["scheduler"])
        try:
            self.scaler.load_state_dict(sd["scaler"])
        except Exception:
            pass
        self.state.step = int(sd.get("step", 0))
        self.state.epoch = int(sd.get("epoch", 0))
        self.state.best_eval_loss = float(sd.get("best_eval_loss", float("inf")))
        print(f"[ckpt] resumed at step {self.state.step}")

    # ------- core loop helpers ----------------------------------------------

    def _move_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        h_t = batch["hidden"].to(self.device, non_blocking=True)
        if self.use_amp:
            h_t = h_t.half()
        else:
            h_t = h_t.float()
        tokens = batch["tokens"].to(self.device, non_blocking=True)
        return {"hidden": h_t, "tokens": tokens}

    def _cond_embed(self, tokens: Tensor) -> Tensor | None:
        """C1: embed(bonus) at anchor position t = tokens[t].

        §2.2 lock: cond index MUST match head_0 target gather (tokens[t]).
        NOT tokens[t-1] (already-consumed input, redundant) and NOT tokens[t+1]
        (answer leak). Reuse the head_0 target id tensor verbatim.
        """
        if not self.eagle_cond or self.embed_layer is None:
            return None
        head0_target_ids = tokens
        cond_ids = head0_target_ids
        assert (cond_ids == head0_target_ids).all(), (
            "C1 cond index regression: cond_ids must equal head_0 target ids "
            "(tokens[t] at each anchor t)"
        )
        with torch.no_grad():
            # CPU-resident embed table: lookup on CPU, move only (B,L,H) to GPU.
            cond_ids_cpu = cond_ids.detach().cpu()
            embed_ids = cond_ids_cpu.clamp(min=0)
            emb = self.embed_layer(embed_ids)
            if (cond_ids_cpu == IGNORE_INDEX).any():
                emb = emb.masked_fill(
                    (cond_ids_cpu == IGNORE_INDEX).unsqueeze(-1), 0.0
                )
            emb = emb.to(self.device, non_blocking=True)
        return emb

    def _forward_heads(self, h_t: Tensor, tokens: Tensor) -> list:
        return self.model(h_t, cond_embed=self._cond_embed(tokens))

    def _autocast(self):
        if self.use_amp:
            return torch.amp.autocast("cuda", dtype=torch.float16)
        return torch.amp.autocast(self.device.type, enabled=False)

    def _global_grad_norm(self) -> float:
        total_sq = 0.0
        for p in self.model.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            total_sq += float(g.norm().item()) ** 2
        return total_sq ** 0.5

    # ------- §7.1 startup sanity prints -------------------------------------

    def startup_sanity(self) -> None:
        """Print spec §7.1 + §11.2 sanity numbers on one mini-batch BEFORE step 0.

        Replays the REAL training numeric path (fp16 autocast + GradScaler) on a
        tiny batch (bs=1, seq<=64): checks graph connectivity, dtype, no NaN.
        Does NOT use a separate fp32 full-size backward (that path is not used
        in training and OOMs a 32GB V100 under 5-head + embed).

        GradScaler hygiene: scale().backward() only — no unscale_/step/update —
        so the scaler stays ready for fit()'s first step. Then zero_grad +
        empty_cache so nothing leaks into the training loop.
        """
        print("\n[sanity §7.1] mini-batch (bs=1,seq<=64) via train path "
              "(fp16 autocast + GradScaler backward)")
        it = iter(self.train_loader)
        batch = next(it)
        h_t = batch["hidden"][:1, :64].to(self.device, non_blocking=True)
        tokens = batch["tokens"][:1, :64].to(self.device, non_blocking=True)
        if self.use_amp:
            h_t = h_t.half()
        else:
            h_t = h_t.float()

        # ---- pass 1: per-head loss with default weights (no_grad) ---------
        with torch.no_grad():
            with self._autocast():
                all_logits = self._forward_heads(h_t, tokens)
                losses = compute_loss(all_logits, tokens, weights=self.loss_weights)
        log_V = float(torch.log(torch.tensor(self.model.vocab_size, dtype=torch.float32)).item())
        print(
            f"[sanity §7.1] head_0_loss = {losses['head_0_loss']:.4f}   "
            f"(expect < 1.0 on REAL cache under TARGET conv.; on synthetic ~log(V)≈{log_V:.2f})"
        )
        for k in range(1, self.model.num_heads):
            print(f"[sanity §7.1] head_{k}_loss = {losses[f'head_{k}_loss']:.4f}")

        # ---- pass 2: chain connectivity via train numeric path ------------
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            all_logits = self._forward_heads(h_t, tokens)
            mask_weights = [0.0] * (self.model.num_heads - 1) + [1.0]
            losses2 = compute_loss(all_logits, tokens, weights=mask_weights)
        self.scaler.scale(losses2["total_loss"]).backward()

        # Aggregate L2 over head_0's resblock params (lm_head is expected to
        # have zero grad when only loss_K-1 is active). Grads are scaler-
        # scaled but sign/nonzero is what we need for the connectivity gate.
        sq = 0.0
        for p in list(self.model.heads[0].input_resblock.parameters()) + list(
            self.model.heads[0].body.parameters()
        ):
            if p.grad is not None:
                sq += float(p.grad.detach().float().norm().item()) ** 2
        head0_grad_norm = sq ** 0.5
        print(
            f"[sanity §7.1] ||head_0 (resblock only) grads||_2 = {head0_grad_norm:.3e}  "
            "(should be > 0; proves chain h_t->h_0'->h_1'->loss_2 is connected)"
        )
        self.optimizer.zero_grad(set_to_none=True)
        del all_logits, losses, losses2, h_t, tokens, batch
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            alloc_gb = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            print(f"[sanity §7.1] post-cleanup cuda memory_allocated = {alloc_gb:.2f} GiB")

    # ------- main loop ------------------------------------------------------

    def fit(self) -> None:
        self.startup_sanity()
        print(f"\n[train] starting at step {self.state.step} / {self.total_steps}    {_now()}")

        running_loss = 0.0
        running_n = 0
        step_start = time.time()

        data_iter = iter(self.train_loader)
        while self.state.step < self.total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                self.state.epoch += 1
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            batch = self._move_batch(batch)
            h_t, tokens = batch["hidden"], batch["tokens"]

            self.model.train()
            with self._autocast():
                all_logits = self._forward_heads(h_t, tokens)
                losses = compute_loss(all_logits, tokens, weights=self.loss_weights)
            loss_for_bwd = losses["total_loss"] / self.grad_accum_steps

            self.scaler.scale(loss_for_bwd).backward()

            # gradient accumulation
            do_step = ((self.state.step + 1) % self.grad_accum_steps) == 0
            if do_step:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip
                ).item()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
            else:
                grad_norm = float("nan")

            running_loss += float(losses["total_loss"].item())
            running_n += 1
            self.state.step += 1

            if self.state.step % self.log_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                mean_loss = running_loss / max(1, running_n)
                dt = time.time() - step_start
                ips = running_n / max(1e-6, dt)
                running_loss, running_n, step_start = 0.0, 0, time.time()
                per_head = " ".join(
                    f"h{k}={losses[f'head_{k}_loss']:.3f}" for k in range(self.model.num_heads)
                )
                print(
                    f"[step {self.state.step:>6d}/{self.total_steps}] "
                    f"loss={mean_loss:.4f}  {per_head}  "
                    f"lr={lr:.2e}  gnorm={grad_norm:.2e}  ips={ips:.1f}"
                )

            if self.eval_every > 0 and self.state.step % self.eval_every == 0 and self.val_loader is not None:
                metrics = evaluate(
                    self.model,
                    self.val_loader,
                    device=self.device,
                    loss_weights=self.loss_weights,
                    use_amp=self.use_amp,
                    max_batches=None,
                    embed_layer=self.embed_layer,
                )
                line = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"[eval step={self.state.step}]  {line}")
                if metrics["eval/total_loss"] < self.state.best_eval_loss:
                    self.state.best_eval_loss = metrics["eval/total_loss"]
                    self.save_checkpoint(tag="best")

            if self.save_every > 0 and self.state.step % self.save_every == 0:
                self.save_checkpoint()

        # final save
        self.save_checkpoint(tag="final")
        if self.eagle_cond:
            fnorm = float(self.model.bonus_proj.weight.detach().float().norm().item())
            print(f"[train] bonus_proj.weight Frobenius norm = {fnorm:.6f}")
        print(f"[train] done at step {self.state.step}    {_now()}")
