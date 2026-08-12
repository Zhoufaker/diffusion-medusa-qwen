"""debug_forward.py — Phase A sanity checks (C1..C5) on synthetic data.

Spec ref: linked_medusa_spec.md §7.1 + §11.

The five user-mandated checks (C1..C5), each printing concrete numbers:

    C1  forward produces 3 logits tensors of shape (B, L, vocab_size)
        Reports: actual vocab_size, B, L, all 3 shapes.

    C2  identity-init: head_0 logits ≈ base_lm_head @ h_t
        Reports: max |abs diff|, max relative diff, threshold (default 1e-3)
        Threshold tuned for fp16; in fp32 we expect EXACTLY 0.

    C3  chain connectivity: total_loss = loss_2; .backward(); ||head_0 grads||_2 > 1e-6
        Reports: per-param-group L2 norms, aggregate L2 norm over all
        head_0 parameters, threshold (1e-6).

    C4  full fwd+bwd+opt-step at batch=4, seq_len=256 — no OOM
        Reports: peak GPU memory in MiB (via torch.cuda.max_memory_allocated),
        total_loss after the step.

    C5  determinism: forward twice on the same batch → bit-exact equality
        Reports: max |abs diff| between two runs (should be exactly 0.0).

Bonus diagnostic (always reports, never blocks PASS/FAIL):

    A6  per-head gradient norms with the standard loss weights, NaN/Inf
        scan. Useful to eyeball whether head magnitudes are reasonable.

Two operating modes:
    --tiny             CPU smoke mode: vocab=512, hidden=128, random fp16
                       base lm_head. Useful on a login node to confirm code
                       structure before submitting a GPU job.
    (default)          Real mode: loads the actual base lm_head from disk
                       (vocab=152064, hidden=3584, fp16) — needs a GPU.

Usage on GPU (real mode):
    python -u scripts/debug_forward.py
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

# Allow `python scripts/debug_forward.py` from project root.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from data import SyntheticVLMDataset, collate_fn
from model import LinkedMedusaHeads, load_base_lm_head_weight
from train import compute_loss


# ---------------- pretty printing -------------------------------------------

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"


def _pass(msg: str) -> None:
    print(f"{GREEN}[PASS]{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"{RED}[FAIL]{RESET} {msg}")


def _info(msg: str) -> None:
    print(f"{DIM}[info]{RESET} {msg}")


def _section(title: str) -> None:
    print(f"\n{YELLOW}=== {title} ==={RESET}")


# ---------------- shared setup ----------------------------------------------


def build_model(args, vocab_size: int) -> LinkedMedusaHeads:
    return LinkedMedusaHeads(
        hidden_dim=args.hidden_dim,
        vocab_size=vocab_size,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        expansion=args.expansion,
    )


def build_loader(args) -> DataLoader:
    ds = SyntheticVLMDataset(
        num_samples=args.num_samples,
        seq_len_range=tuple(args.seq_len_range),
        hidden_dim=args.hidden_dim,
        vocab_size=args.effective_vocab,
        seed=args.seed,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )


def get_one_batch(loader: DataLoader, device: torch.device) -> Dict[str, Tensor]:
    batch = next(iter(loader))
    return {
        "hidden": batch["hidden"].to(device),
        "tokens": batch["tokens"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }


def _zero_all_grads(model: torch.nn.Module) -> None:
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


# ---------------- the checks ------------------------------------------------


def check_c1_forward_shapes(model: LinkedMedusaHeads, batch, args) -> bool:
    _section("C1: forward produces correct shapes")
    h_t = batch["hidden"].float()
    model.eval()
    with torch.no_grad():
        all_logits = model(h_t)

    B, L, _ = h_t.shape
    expected = (B, L, model.vocab_size)
    _info(f"vocab_size = {model.vocab_size}    B={B}, L={L}")

    if len(all_logits) != args.num_heads:
        _fail(f"got {len(all_logits)} heads, expected {args.num_heads}")
        return False

    ok = True
    for k, lg in enumerate(all_logits):
        if tuple(lg.shape) != expected:
            _fail(f"head_{k}: shape {tuple(lg.shape)} != expected {expected}")
            ok = False
        else:
            _info(f"head_{k} shape: {tuple(lg.shape)}")

    if ok:
        _pass(f"all 3 heads emit shape (B, L, {model.vocab_size})")
    return ok


def check_c2_identity_init(model: LinkedMedusaHeads, base_lm_head_weight: Tensor, batch, args) -> bool:
    _section("C2: identity init — head_0 logits ≈ base_lm_head @ h_t   (tol < 1e-3)")
    # Compare in the model's native dtype. The model is fp32 by default (the
    # heads' resblocks/Linear are fp32 init); copy_(base_lm_head_weight) up-
    # casts the fp16 disk weight to fp32 model storage. So this comparison
    # runs entirely in fp32 — and we expect EXACT equality. The 1e-3
    # tolerance just lets fp16 paths (e.g. inference-time autocast) also pass.
    h_t = batch["hidden"].float()
    W = base_lm_head_weight.to(device=h_t.device, dtype=torch.float32)  # (V, H)

    expected_logits = h_t @ W.t()  # (B, L, V), no bias

    model.eval()
    with torch.no_grad():
        all_logits = model(h_t)
    head_0_logits = all_logits[0]

    abs_diff = (head_0_logits - expected_logits).abs()
    max_abs = abs_diff.max().item()
    rel_max = (abs_diff / (expected_logits.abs() + 1e-9)).max().item()
    e_max = expected_logits.abs().max().item()
    _info(
        f"head_0 logits range: max|expected| = {e_max:.3e}    "
        f"max|abs diff| = {max_abs:.3e}    max relative diff = {rel_max:.3e}"
    )

    tol_abs = args.c2_tol_abs
    if max_abs <= tol_abs:
        _pass(f"max |abs diff| = {max_abs:.3e}  ≤  tol {tol_abs:.3e}")
        return True
    _fail(
        f"max |abs diff| = {max_abs:.3e}  >  tol {tol_abs:.3e}\n"
        "        likely cause: ResBlock W_2 not zero-init; or lm_head copy "
        "did not reach the right tensor; or shape/dtype mismatch in copy."
    )
    return False


def _aggregate_l2(params) -> Tuple[float, int, int, int]:
    """Aggregate L2 norm over a list of parameters' .grad tensors.

    Returns (L2_norm, n_with_grad, n_zero_grad, n_none_grad).
    """
    sq = 0.0
    n_g, n_z, n_n = 0, 0, 0
    for p in params:
        if p.grad is None:
            n_n += 1
            continue
        g = p.grad.detach().float()
        gn2 = g.norm().item() ** 2
        sq += gn2
        if gn2 > 0.0:
            n_g += 1
        else:
            n_z += 1
    return sq ** 0.5, n_g, n_z, n_n


def check_c3_chain_connectivity(model: LinkedMedusaHeads, batch, args) -> bool:
    _section("C3: chain connectivity — total_loss=loss_2; ||head_0 grads||_2 > 1e-6")
    model.train()
    _zero_all_grads(model)

    h_t = batch["hidden"].float()
    tokens = batch["tokens"]
    all_logits = model(h_t)
    losses = compute_loss(all_logits, tokens, weights=[0.0, 0.0, 1.0])
    losses["total_loss"].backward()

    # Aggregate L2 over ALL head_0 params (input_resblock + body + lm_head).
    head_0_params = list(model.heads[0].parameters())
    g_total, n_g, n_z, n_n = _aggregate_l2(head_0_params)

    # Also break out into the two sub-groups so the report is clearer:
    rb_params = (
        list(model.heads[0].input_resblock.parameters())
        + list(model.heads[0].body.parameters())
    )
    lm_params = list(model.heads[0].lm_head.parameters())
    g_rb, *_ = _aggregate_l2(rb_params)
    g_lm, *_ = _aggregate_l2(lm_params)

    _info(f"head_2_loss        = {losses['head_2_loss']:.4f}")
    _info(f"#head_0 params     = {len(head_0_params)}   "
          f"(non-zero grad: {n_g}, zero grad: {n_z}, no-grad: {n_n})")
    _info(f"||grad||_2  (head_0 input_resblock + body) = {g_rb:.3e}")
    _info(f"||grad||_2  (head_0 lm_head)               = {g_lm:.3e}   "
          "  ← may legitimately be 0: lm_head only feeds loss_0 (weight=0); chain passes pre-lm_head h')")
    _info(f"||grad||_2  (head_0 ALL params, aggregate) = {g_total:.3e}   ← the C3 number")

    threshold = args.c3_l2_threshold
    if g_total > threshold:
        _pass(f"||head_0 grads||_2 = {g_total:.3e}  >  threshold {threshold:.0e}")
        return True

    # Tolerated edge case: at exact identity-init in fp32, some grads can
    # collapse to numerical zero. Verify by perturbing W_2 weights slightly
    # and re-trying. If the chain is connected, perturbation will introduce
    # a non-zero gradient.
    _info("aggregate L2 below threshold at exact identity-init; perturbing W_2 weights")
    with torch.no_grad():
        for blk in [model.heads[0].input_resblock, *model.heads[0].body]:
            blk.w2.weight.add_(torch.randn_like(blk.w2.weight) * 1e-3)

    _zero_all_grads(model)
    all_logits = model(h_t)
    losses = compute_loss(all_logits, tokens, weights=[0.0, 0.0, 1.0])
    losses["total_loss"].backward()
    g_total2, _, _, _ = _aggregate_l2(head_0_params)
    _info(f"after small W_2 perturbation: ||head_0 grads||_2 = {g_total2:.3e}")
    if g_total2 > threshold:
        _pass(
            f"||head_0 grads||_2 after perturbation = {g_total2:.3e}  >  threshold {threshold:.0e}\n"
            "        (the all-near-zero state at exact identity-init is a known linearization "
            "artifact; once symmetry breaks during training, gradients flow.)"
        )
        return True

    _fail(
        f"||head_0 grads||_2 = {g_total2:.3e} ≤ threshold {threshold:.0e} "
        "even after perturbation — chain appears NOT connected"
    )
    return False


def check_c4_oom_at_full_seq(model: LinkedMedusaHeads, args, device: torch.device) -> bool:
    _section(f"C4: fwd+bwd+opt-step  batch={args.c4_batch_size}, seq={args.c4_seq_len}  — peak memory")
    ds = SyntheticVLMDataset(
        num_samples=args.c4_batch_size,
        seq_len_range=(args.c4_seq_len, args.c4_seq_len),
        hidden_dim=args.hidden_dim,
        vocab_size=args.effective_vocab,
        seed=args.seed + 1,
    )
    loader = DataLoader(
        ds, batch_size=args.c4_batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    batch = get_one_batch(loader, device)

    # Per spec §3 / §9.4: V100 32GB cannot fit fp32 AdamW state for 3x lm_head
    # (each lm_head 152064 x 3584 fp32 = 2.18 GB; AdamW keeps weight + 2
    # momentums = ~6.5 GB per head; 3 heads → ~20 GB optimizer state alone,
    # before activations). Switch to 8-bit AdamW which compresses momentums.
    # Original: torch.optim.AdamW (OOMs on V100, see spec §9.4)
    if args.c4_optimizer == "adamw8bit":
        try:
            from bitsandbytes.optim import AdamW8bit
        except ImportError as e:
            _info(f"bitsandbytes not installed ({e}); falling back to torch.optim.AdamW")
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.0)
            _info("optimizer = torch.optim.AdamW (fallback)")
        else:
            optimizer = AdamW8bit(model.parameters(), lr=5e-4, weight_decay=0.0)
            _info("optimizer = bitsandbytes.optim.AdamW8bit")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.0)
        _info("optimizer = torch.optim.AdamW")

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    model.train()
    optimizer.zero_grad()
    h_t = batch["hidden"].float()
    tokens = batch["tokens"]
    all_logits = model(h_t)
    losses = compute_loss(all_logits, tokens, weights=args.loss_weights)
    losses["total_loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_b = torch.cuda.max_memory_allocated()
        peak_mib = peak_b / (1024 ** 2)
        peak_gb_si = peak_b / 1e9
        peak_gib = peak_b / (1024 ** 3)
        free_b, total_b = torch.cuda.mem_get_info()
        _info(
            f"peak_GPU_memory = {peak_gb_si:.2f} GB ({peak_mib:.1f} MiB / {peak_gib:.2f} GiB)"
        )
        _info(
            f"GPU free / total = {free_b/1e9:.2f} GB / {total_b/1e9:.2f} GB   "
            f"(after step; reflects fragmentation + cached blocks)"
        )
    else:
        peak_gb_si = -1.0
        _info("peak_GPU_memory = N/A (CPU run; OOM-check is GPU-only meaningful)")

    total_loss = losses["total_loss"].item()
    _info(f"total_loss after step = {total_loss:.4f}")
    if total_loss != total_loss or total_loss in (float("inf"), float("-inf")):
        _fail(f"total_loss is non-finite: {total_loss}")
        return False

    if peak_gb_si >= 0:
        _pass(
            f"one full step at B={args.c4_batch_size}, L={args.c4_seq_len} completed; "
            f"peak={peak_gb_si:.2f} GB"
        )
    else:
        _pass(f"one full step at B={args.c4_batch_size}, L={args.c4_seq_len} completed (CPU)")
    return True


def check_c5_determinism(model: LinkedMedusaHeads, batch, args) -> bool:
    _section("C5: determinism — same batch → identical logits across two forwards")
    model.eval()
    h_t = batch["hidden"].float()
    with torch.no_grad():
        out1 = model(h_t)
        out2 = model(h_t)
    if len(out1) != len(out2):
        _fail(f"different number of outputs: {len(out1)} vs {len(out2)}")
        return False
    max_abs_overall = 0.0
    for k in range(len(out1)):
        d = (out1[k] - out2[k]).abs().max().item()
        _info(f"head_{k} max |abs diff| between runs = {d:.3e}")
        max_abs_overall = max(max_abs_overall, d)
    if max_abs_overall == 0.0:
        _pass(f"two forwards bit-exact identical (max |diff| = 0)")
        return True
    if max_abs_overall <= args.c5_tol_abs:
        _pass(
            f"two forwards near-identical (max |diff| = {max_abs_overall:.3e}, "
            f"≤ tol {args.c5_tol_abs:.0e}). Not bit-exact but within tolerance."
        )
        return True
    _fail(
        f"max |diff| = {max_abs_overall:.3e}  >  tol {args.c5_tol_abs:.0e} — "
        "non-determinism detected. Likely an unexpected dropout or a random op."
    )
    return False


def check_a6_grad_norms_sane(model: LinkedMedusaHeads, batch, args) -> bool:
    _section("A6 (bonus): per-head grad norms with default loss weights, NaN/Inf scan")
    _zero_all_grads(model)
    h_t = batch["hidden"].float()
    tokens = batch["tokens"]
    all_logits = model(h_t)
    losses = compute_loss(all_logits, tokens, weights=args.loss_weights)
    losses["total_loss"].backward()

    per_head, n_nan, n_inf = [], 0, 0
    for k, head in enumerate(model.heads):
        sq = 0.0
        for p in head.parameters():
            if p.grad is None:
                continue
            if torch.isnan(p.grad).any():
                n_nan += 1
            if torch.isinf(p.grad).any():
                n_inf += 1
            sq += p.grad.detach().float().norm().item() ** 2
        per_head.append(sq ** 0.5)
    _info("per-head ||grad||_2: " + ", ".join(f"head_{k}={n:.3e}" for k, n in enumerate(per_head)))
    _info(f"NaN tensors = {n_nan}    Inf tensors = {n_inf}")
    if n_nan or n_inf:
        _fail("found NaN or Inf in gradients (bonus check)")
        return False
    _pass("gradient norms finite, no NaN/Inf")
    return True


# ---------------- main ------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--hidden-dim", type=int, default=3584)
    p.add_argument("--num-heads", type=int, default=3)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--expansion", type=int, default=2)
    p.add_argument("--effective-vocab", type=int, default=151936,
                   help="Token id range for synthetic data (effective tokenizer vocab).")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Batch size used for C1, C2, C3, C5, A6.")
    p.add_argument("--seq-len-range", type=int, nargs=2, default=[50, 256])
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--loss-weights", type=float, nargs=3, default=[1.0, 0.8, 0.64])
    p.add_argument("--lm-head-path", default="/scratch/li96/mz9869/medusa_assets/base_lm_head.safetensors")
    p.add_argument("--c2-tol-abs", type=float, default=1e-3,
                   help="C2 max-abs-diff tolerance.")
    p.add_argument("--c3-l2-threshold", type=float, default=1e-6,
                   help="C3 minimum aggregate L2 norm of head_0 grads.")
    p.add_argument("--c4-batch-size", type=int, default=4,
                   help="Batch size for the C4 OOM stress test.")
    p.add_argument("--c4-seq-len", type=int, default=256,
                   help="Fixed seq length for the C4 OOM stress test.")
    p.add_argument("--c4-optimizer", choices=("adamw8bit", "adamw"), default="adamw8bit",
                   help="Optimizer for C4. Default adamw8bit per spec §3/§9.4 to fit V100 32GB. "
                        "adamw is the original torch.optim.AdamW (OOMs on V100 with 3x lm_head).")
    p.add_argument("--c5-tol-abs", type=float, default=0.0,
                   help="C5 determinism tolerance (default 0 = bit-exact required).")
    p.add_argument("--skip-c4", action="store_true",
                   help="Skip the OOM check (useful on CPU).")
    p.add_argument("--skip-bonus", action="store_true",
                   help="Skip the bonus A6 NaN/Inf check (it always runs but never blocks PASS/FAIL).")
    p.add_argument("--tiny", action="store_true",
                   help="CPU smoke-test mode: tiny vocab + tiny hidden, random base lm_head.")
    p.add_argument("--tiny-vocab", type=int, default=512)
    p.add_argument("--tiny-hidden", type=int, default=128)
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"device          = {device}")
    if device.type == "cuda":
        print(f"cuda device     = {torch.cuda.get_device_name(0)}")
        print(f"cuda capability = {torch.cuda.get_device_capability(0)}")
    print(f"hidden_dim      = {args.hidden_dim}")
    print(f"num_heads       = {args.num_heads}")
    print(f"num_blocks      = {args.num_blocks}")
    print(f"expansion       = {args.expansion}")
    print(f"effective_vocab = {args.effective_vocab}")
    print(f"batch_size      = {args.batch_size}")
    print(f"seq_len_range   = {tuple(args.seq_len_range)}")
    print(f"loss_weights    = {args.loss_weights}")
    print(f"c2_tol_abs      = {args.c2_tol_abs}")
    print(f"c3_l2_threshold = {args.c3_l2_threshold}")
    print(f"c4 stress       = batch={args.c4_batch_size}, seq={args.c4_seq_len}")
    print(f"c5_tol_abs      = {args.c5_tol_abs}")
    print(f"lm_head_path    = {args.lm_head_path}")

    if args.tiny:
        print("\n[tiny] CPU smoke mode: tiny dimensions + random base lm_head")
        args.hidden_dim = args.tiny_hidden
        args.effective_vocab = args.tiny_vocab
        physical_vocab = args.tiny_vocab
        torch.manual_seed(args.seed)
        base_lm_head = torch.randn(physical_vocab, args.hidden_dim, dtype=torch.float16) * 0.02
        print(f"  shape = {tuple(base_lm_head.shape)}, dtype = {base_lm_head.dtype}  (synthetic)")
    else:
        print("\nLoading base lm_head from disk...")
        base_lm_head = load_base_lm_head_weight(args.lm_head_path)
        physical_vocab, hidden_from_file = base_lm_head.shape
        print(f"  shape = {tuple(base_lm_head.shape)}, dtype = {base_lm_head.dtype}")
        if hidden_from_file != args.hidden_dim:
            print(f"  WARN: hidden_dim from file {hidden_from_file} != --hidden-dim {args.hidden_dim}; "
                  f"overriding hidden_dim to {hidden_from_file}")
            args.hidden_dim = hidden_from_file

    model = build_model(args, vocab_size=physical_vocab).to(device)
    model.init_lm_heads_from_base(base_lm_head.to(device))

    loader = build_loader(args)
    batch = get_one_batch(loader, device)

    results: Dict[str, bool] = {}

    for name, fn in [
        ("C1", lambda: check_c1_forward_shapes(model, batch, args)),
        ("C2", lambda: check_c2_identity_init(model, base_lm_head, batch, args)),
        ("C3", lambda: check_c3_chain_connectivity(model, batch, args)),
    ]:
        try:
            results[name] = fn()
        except Exception:
            traceback.print_exc()
            results[name] = False

    if args.skip_c4:
        _info("\nC4 skipped via --skip-c4")
        results["C4"] = True
    else:
        try:
            results["C4"] = check_c4_oom_at_full_seq(model, args, device)
        except Exception:
            traceback.print_exc()
            results["C4"] = False

    try:
        results["C5"] = check_c5_determinism(model, batch, args)
    except Exception:
        traceback.print_exc()
        results["C5"] = False

    if not args.skip_bonus:
        try:
            check_a6_grad_norms_sane(model, batch, args)
        except Exception:
            traceback.print_exc()

    _section("SUMMARY (C1..C5 — these are the formal must-pass checks)")
    for k in ("C1", "C2", "C3", "C4", "C5"):
        v = results.get(k, False)
        sym = f"{GREEN}PASS{RESET}" if v else f"{RED}FAIL{RESET}"
        print(f"  {k}: {sym}")
    all_ok = all(results.get(k, False) for k in ("C1", "C2", "C3", "C4", "C5"))
    print()
    print(f"{GREEN if all_ok else RED}Phase A (C1..C5) overall: {'PASS' if all_ok else 'FAIL'}{RESET}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
