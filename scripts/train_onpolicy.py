"""train_onpolicy.py — B2: on-policy fine-tune of the linked heads.

Loss is IDENTICAL to B1 (CE; head_k predicts the base's argmax at offset k),
the ONLY change is the input distribution: heads draft from the live folded-
rollout anchor (the last-accepted hidden produced by the tree-masked verify
forward), not from clean cache hiddens. Warm-start from the B1 ckpt.

Per round (reusing decode/tree.py primitives, gradients enabled on heads only):
    prefill (no_grad) -> loop:
        all_logits = heads(h_anchor)            # WITH grad, fp16 autocast
        tree       = build_tree_folded(detach)  # wide sweet spot [.,6,4,2,1]x24
        verify     = base(tree)                 # no_grad, gives v_logits/v_hidden
        round_loss = sum_k w_k * CE(head_k logits, GT_k);  BACKWARD PER ROUND
        accept greedy prefix + bonus; carry anchor; gather-reorg KV

GT convention (ALL proposed tree positions, not only accepted ones — accepted-
only would be a biased sample):
    head_0            : the known root token (base's argmax at the anchor) —
                        same task as offline head_0.
    node i at depth d>=2 (proposed by head_{d-1}): base's argmax AFTER the
                        parent's path = argmax(v_logits[parent(i)]). One head
                        distribution scores against every parent-path target at
                        its depth (the tree samples the anchor-conditional
                        path distribution).

MEMORY DESIGN (learned the hard way — first smoke OOM'd an 80GB A100):
    Rounds are independent graphs (the anchor comes from the no_grad verify),
    so we call backward() EVERY ROUND and let grads accumulate. Summing round
    losses across a prompt and backwarding once retains each round's autocast
    fp16 copies of the head weights (~7GB/round for 3.5B fp32 params) in the
    graph -> OOM within ~4 rounds. Per-round backward is mathematically
    identical (grad_W accumulates once per weight-use either way) and bounds
    graph memory to a single round. Gradient accumulation is therefore at
    ROUND granularity (--round-accum), and the optimizer may step mid-prompt —
    which is fine (even more on-policy) since each round drafts with the
    current weights. Budget is defined directly in optimizer steps
    (--total-steps); prompts are consumed from the shuffled pool as needed.

Base is frozen (no_grad) throughout; backprop touches heads only.
fp32 master weights + fp16 autocast + GradScaler + AdamW8bit.

Val: last --val-reserve prompts of the rollout pool are excluded from training;
folded-rollout sigma on the first --val-prompts of that slice is the plateau
monitor (still disjoint from the canonical eval-100, which lives in
llava_subset_2k.json). Best-by-val-sigma checkpoint is kept.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from decode.common import (argmax_masked, continuation_base, load_base,
                           make_image_inputs, mask_phantom_)
from decode.tree import (accept, build_mask_and_positions, build_tree_folded,
                         reorg_kv_gather, tree_tokens)
from model import LinkedMedusaHeads
from train.scheduler import cosine_warmup_schedule


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init-ckpt",
                   default="/scratch/li96/mz9869/medusa_outputs/linked_medusa_5head_b1/ckpt_best.pt")
    p.add_argument("--prompts", default="/scratch/li96/mz9869/onpolicy_data/rollout_prompts.json")
    p.add_argument("--images-dir", default="/scratch/li96/mz9869/onpolicy_data/images")
    p.add_argument("--out-dir", default="/scratch/li96/mz9869/medusa_outputs/linked_medusa_5head_b2")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--fanout", type=int, nargs="+", default=[1, 6, 4, 2, 1],
                   help="training rollout tree (wide samples the anchor distribution)")
    p.add_argument("--max-nodes", type=int, default=24)
    p.add_argument("--max-new", type=int, default=150)
    p.add_argument("--total-steps", type=int, default=1500, help="optimizer steps (budget)")
    p.add_argument("--round-accum", type=int, default=256,
                   help="rounds per optimizer step (grad accumulation unit)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--final-lr-mult", type=float, default=0.33)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--loss-weights", type=float, nargs="+",
                   default=[1.0, 0.8, 0.64, 0.512, 0.41])
    p.add_argument("--scaler-init-scale", type=float, default=1024.0)
    p.add_argument("--log-every", type=int, default=10, help="steps")
    p.add_argument("--val-every", type=int, default=200, help="steps; 0 disables")
    p.add_argument("--ckpt-every", type=int, default=200, help="steps")
    p.add_argument("--val-reserve", type=int, default=500,
                   help="last N pool prompts excluded from training (val slice)")
    p.add_argument("--val-prompts", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def load_trainable_heads(ckpt_path: str, device: str) -> LinkedMedusaHeads:
    print(f"[init] warm-start heads from {ckpt_path}")
    sd_full = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head_sd = sd_full.get("model", sd_full.get("state_dict", sd_full))
    m = (sd_full.get("cfg") or {}).get("model", {})
    head = LinkedMedusaHeads(
        hidden_dim=int(m.get("hidden_dim", 3584)),
        vocab_size=int(m.get("vocab_size", 152064)),
        num_heads=int(m.get("num_heads", 5)),
        num_blocks=int(m.get("num_blocks", 2)),
        expansion=int(m.get("expansion", 2)),
    )
    head.load_state_dict(head_sd, strict=True)
    del sd_full, head_sd
    print(f"[init] heads: K={head.num_heads} params="
          f"{sum(p.numel() for p in head.parameters())/1e9:.2f}B (fp32, trainable)")
    return head.to(device).float().train()


def rollout(base, heads, processor, prompt, images_dir, args, on_round_loss=None):
    """One folded rollout. If on_round_loss is given (training), it is called
    with the per-round weighted loss tensor (graph alive) and must consume it
    (backward) immediately — nothing round-graph-related is retained here."""
    device = args.device
    K = heads.num_heads
    W = args.loss_weights
    train = on_round_loss is not None
    with torch.no_grad():
        inputs = make_image_inputs(processor, prompt["question"],
                                   images_dir / prompt["image"], device)
        out = base(**inputs, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        P = past_kv.get_seq_length()
        h_anchor = out.hidden_states[-1][0, -1, :].clone()
        known_next = argmax_masked(out.logits[0, -1, :])
        del out

    emitted = 0
    rounds = 0
    accept_log = []
    ce_sum = [0.0] * K
    ce_cnt = [0] * K

    while emitted < args.max_new:
        rounds += 1
        cont_base = continuation_base(base, P)
        with torch.autocast("cuda", torch.float16):
            with torch.set_grad_enabled(train):
                all_logits = heads(h_anchor.view(1, 1, -1))
        det = [l.detach() for l in all_logits]
        nodes = build_tree_folded(det, known_next, args.fanout, args.max_nodes)
        mask, pos = build_mask_and_positions(nodes, P, cont_base, base.dtype, device)
        toks = tree_tokens(nodes, device)
        with torch.no_grad():
            v_out = base(input_ids=toks, attention_mask=mask, past_key_values=past_kv,
                         position_ids=pos, use_cache=True, output_hidden_states=True)
            v_logits = v_out.logits[0]
            v_hidden = v_out.hidden_states[-1][0]

        if train:
            with torch.no_grad():
                gts = mask_phantom_(v_logits.float()).argmax(-1)  # (N,) GT after each node
            per_head_t = [[] for _ in range(K)]
            per_head_t[0].append(known_next)
            for nd in nodes:
                if nd.depth >= 2:
                    per_head_t[nd.depth - 1].append(int(gts[nd.parent].item()))
            round_loss = None
            for k in range(K):
                t = per_head_t[k]
                if not t:
                    continue
                lg = all_logits[k].reshape(1, -1).float()
                tgt = torch.tensor(t, device=device, dtype=torch.long)
                ce = F.cross_entropy(lg.expand(len(t), -1), tgt)
                round_loss = W[k] * ce if round_loss is None else round_loss + W[k] * ce
                ce_sum[k] += float(ce.item()) * len(t)
                ce_cnt[k] += len(t)
            if round_loss is not None and torch.isfinite(round_loss):
                on_round_loss(round_loss)
            del all_logits, round_loss

        accepted, accept_len, next_known, acc_depths = accept(nodes, v_logits, known_next)
        accept_log.append(accept_len)
        acc_tokens = [nodes[i].token for i in accepted]
        emitted += len(acc_tokens)
        if args.eos_id in acc_tokens:
            break
        last = accepted[-1]
        h_anchor = v_hidden[last].clone()
        known_next = next_known
        reorg_kv_gather(v_out.past_key_values, P, accepted, device)
        past_kv = v_out.past_key_values
        del v_out
        P = past_kv.get_seq_length()

    return {
        "sigma": emitted / max(1, rounds),
        "rounds": rounds,
        "accept_log": accept_log,
        "ce_sum": ce_sum,
        "ce_cnt": ce_cnt,
    }


@torch.no_grad()
def run_val(base, heads, processor, val_prompts, images_dir, args):
    heads.eval()
    sig, acc_gt = [], [0] * heads.num_heads
    total_rounds = 0
    for p in val_prompts:
        st = rollout(base, heads, processor, p, images_dir, args)
        sig.append(st["sigma"])
        total_rounds += st["rounds"]
        for a in st["accept_log"]:
            for k in range(heads.num_heads):
                if a > k:
                    acc_gt[k] += 1
    heads.train()
    ppa = [x / max(1, total_rounds) for x in acc_gt]
    return statistics.mean(sig), ppa


def save_ckpt(heads, args, step, tag, val_sigma=None):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / (f"ckpt_{tag}.pt" if tag else f"ckpt_step{step}.pt")
    torch.save({
        "model": heads.state_dict(),
        "step": step,
        "val_sigma": val_sigma,
        "cfg": {"model": {"hidden_dim": heads.hidden_dim, "vocab_size": heads.vocab_size,
                          "num_heads": heads.num_heads, "num_blocks": heads.num_blocks,
                          "expansion": heads.expansion},
                "onpolicy_args": {k: v for k, v in vars(args).items()}},
    }, str(path))
    print(f"[ckpt] saved -> {path}", flush=True)
    return path


def main() -> int:
    args = parse_args()
    for k, v in vars(args).items():
        print(f"[args] {k} = {v}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    base, processor = load_base(args.model_id, args.device)
    for p in base.parameters():
        p.requires_grad_(False)
    heads = load_trainable_heads(args.init_ckpt, args.device)
    assert len(args.loss_weights) == heads.num_heads
    assert len(args.fanout) == heads.num_heads
    args.eos_id = processor.tokenizer.eos_token_id
    images_dir = Path(args.images_dir)

    pool = json.load(open(args.prompts))
    val_prompts = [p for p in pool[-args.val_reserve:]
                   if (images_dir / p["image"]).exists()][: args.val_prompts]
    train_pool = pool[: -args.val_reserve]
    order = list(range(len(train_pool)))
    random.Random(args.seed).shuffle(order)
    print(f"[data] train pool {len(train_pool)}, val slice {len(val_prompts)} "
          f"(last {args.val_reserve} reserved; eval-100 untouched & disjoint)")

    from bitsandbytes.optim import AdamW8bit
    opt = AdamW8bit(heads.parameters(), lr=args.lr, betas=(0.9, 0.999),
                    weight_decay=args.weight_decay)
    sched = cosine_warmup_schedule(opt, warmup_steps=args.warmup_steps,
                                   total_steps=args.total_steps,
                                   final_lr_multiplier=args.final_lr_mult)
    scaler = torch.amp.GradScaler("cuda", init_scale=args.scaler_init_scale)
    print(f"[init] AdamW8bit lr={args.lr} total_steps={args.total_steps} "
          f"round_accum={args.round_accum} (~{args.total_steps*args.round_accum} rounds)")

    state = {"step": 0, "rounds_in_step": 0, "gnorm": float("nan"), "best_val": -1.0}
    win = {"sig": [], "ce_sum": [0.0] * heads.num_heads, "ce_cnt": [0] * heads.num_heads,
           "prompts": 0, "t0": time.time()}

    def on_round_loss(round_loss):
        scaler.scale(round_loss / args.round_accum).backward()
        state["rounds_in_step"] += 1
        if state["rounds_in_step"] >= args.round_accum:
            scaler.unscale_(opt)
            state["gnorm"] = torch.nn.utils.clip_grad_norm_(
                heads.parameters(), args.grad_clip).item()
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            state["rounds_in_step"] = 0
            state["step"] += 1
            step = state["step"]

            if step % args.log_every == 0:
                per_head = " ".join(
                    f"h{k}={win['ce_sum'][k]/max(1,win['ce_cnt'][k]):.3f}"
                    for k in range(heads.num_heads))
                dt = time.time() - win["t0"]
                print(f"[step {step:>5d}/{args.total_steps}] {per_head}  "
                      f"rollout_sigma={statistics.mean(win['sig']) if win['sig'] else 0:.3f}  "
                      f"lr={sched.get_last_lr()[0]:.2e} gnorm={state['gnorm']:.2e} "
                      f"scale={scaler.get_scale():.0f} "
                      f"prompt/s={win['prompts']/max(1e-6,dt):.2f}", flush=True)
                win["sig"] = []
                win["ce_sum"] = [0.0] * heads.num_heads
                win["ce_cnt"] = [0] * heads.num_heads
                win["prompts"] = 0
                win["t0"] = time.time()

            if args.val_every > 0 and step % args.val_every == 0:
                vs, vppa = run_val(base, heads, processor, val_prompts, images_dir, args)
                print(f"[val step={step}] folded_sigma={vs:.3f} "
                      f"accept={[f'{x:.3f}' for x in vppa]}", flush=True)
                if vs > state["best_val"]:
                    state["best_val"] = vs
                    save_ckpt(heads, args, step, "best", val_sigma=vs)

            if args.ckpt_every > 0 and step % args.ckpt_every == 0:
                save_ckpt(heads, args, step, None)

    i_order = 0
    while state["step"] < args.total_steps:
        p = train_pool[order[i_order % len(order)]]
        i_order += 1
        if not (images_dir / p["image"]).exists():
            continue
        st = rollout(base, heads, processor, p, images_dir, args,
                     on_round_loss=on_round_loss)
        win["sig"].append(st["sigma"])
        win["prompts"] += 1
        for k in range(heads.num_heads):
            win["ce_sum"][k] += st["ce_sum"][k]
            win["ce_cnt"][k] += st["ce_cnt"][k]

    vs, vppa = run_val(base, heads, processor, val_prompts, images_dir, args)
    print(f"[val FINAL] folded_sigma={vs:.3f} accept={[f'{x:.3f}' for x in vppa]}")
    if vs > state["best_val"]:
        save_ckpt(heads, args, state["step"], "best", val_sigma=vs)
    save_ckpt(heads, args, state["step"], "final", val_sigma=vs)
    print(f"[train] done: {state['step']} steps, prompts consumed ~{i_order}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
