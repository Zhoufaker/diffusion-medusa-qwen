"""eval_acceptance.py — speculative-decoding σ (acceptance length) on LLaVA prompts.

Measures the average number of tokens emitted per base-model verification
round when Linked Medusa draft heads propose tokens. Decoding is GREEDY:
no temperature, no tree fanout, just a linear chain of K head candidates.

Per-round semantics (TARGET convention, see linked_medusa_spec.md §11.1):

    Position L = next response position to emit
    Past KV   covers prompt + previously accepted tokens (0..L-1)
    h_t       = base last-layer hidden at position L-1
                (under TARGET conv, this IS the cache "hidden[L]" — its
                 lm_head argmax = base's emission at pos L)

    Step A: heads draft candidates
        c_0 = head_0(h_t)  predicts pos L
        c_1 = head_1(h_t)  predicts pos L+1
        c_2 = head_2(h_t)  predicts pos L+2

    Step B: verify forward
        input = [c_0, c_1, c_2]
        v_logits[i] = base's logits at pos L+i (after seeing c_0..c_i)
                    = base's natural prediction for pos L+i+1

    Step C: base predictions at positions L, L+1, L+2:
        base_pred[0] = base_pred_t  (from previous round / prefill)
        base_pred[1] = argmax(v_logits[0])
        base_pred[2] = argmax(v_logits[1])
        bonus_logits = v_logits[K-1] (for the "free" next-pos prediction)

    Step D: accept_len = longest matching prefix where c_i == base_pred[i]

    Step E: emit
        accepted_tokens = c_0..c_{accept_len-1}  (accept_len tokens)
        bonus_token     = base_pred at pos L+accept_len
                          (= argmax(v_logits[accept_len-1]) if accept_len>=1
                           else base_pred_t from prior)
        total emitted this round = accept_len + 1

    Step F: cache + state update
        truncate past_kv to L+accept_len positions (drop the rejected K/V)
        ONE extra single-token base forward with bonus_token:
            -> extends past_kv to L+accept_len+1
            -> gives h_new at pos L+accept_len for next round
            -> gives base_pred_new at pos L+accept_len+1

σ (per prompt) = total tokens emitted / number of verify rounds
              = (sum_round (accept_len_round + 1)) / num_rounds

Static cache eval (per-head MARGINAL top1 accuracy, step 11000):
    h0_top1 = 0.9224 , h1_top1 = 0.7765 , h2_top1 = 0.6444
Using these as a chain of CONDITIONAL accept rates gives
    σ_est = 1 + 0.9224 + 0.9224*0.7765 + ... ≈ 3.10
but that is a known OVERESTIMATE: marginal top1 on the static cache is not the
conditional accept rate seen during speculative decoding.

Measured live baseline (v1, 100 LLaVA prompts, fp16, A100):
    σ_mean = 1.95 , per-position accept = [0.647, 0.218, 0.063]
The gap from the static-cache estimate is attributed to the PROCESS-LEVEL OOD
of speculative decoding: the heads are trained on the base model's clean cached
hidden states, but at inference they consume hidden states produced inside the
draft/verify chain. (This is NOT a dataset cross-domain effect — the cache and
this eval share their prompt source.)

NOTE on cache provenance: the exact construction of the training cache is
PENDING final confirmation. The supervisor describes it as LLaVA-Bench-derived;
decoded cache samples skew toward chart-describing content, so a 'long' subset
length filter may bias it. Do not assert a specific dataset here until confirmed.
"""
from __future__ import annotations

import os

# MUST be set BEFORE importing transformers (HF_HOME caches model files;
# OFFLINE flags prevent the load from trying to hit HuggingFace).
os.environ["HF_HOME"] = "/scratch/li96/mz9869/tmp_hf_download"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image

# Project root on path so we can import model.
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model import LinkedMedusaHeads  # noqa: E402

# Effective tokenizer vocab. lm_head's physical dim is 152064 (padded for
# hardware alignment); IDs in [EFFECTIVE_VOCAB .. 152063] are padding rows
# whose logits are unbounded. We mask those to -inf before argmax so a
# trained head can't accidentally pick a "phantom" token.
EFFECTIVE_VOCAB = 151936


# ----------------------------------------------------------------------------
# Argparse
# ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        default="/scratch/li96/mz9869/medusa_outputs/linked_medusa_v1_full/ckpt_best.pt",
        help="Linked Medusa head checkpoint (best by eval/total_loss).",
    )
    p.add_argument(
        "--manifest",
        default="/g/data/li96/mz9869/data/llava_subset_2k.json",
        help="LLaVA-style JSON list with id/image/conversations fields.",
    )
    p.add_argument(
        "--images-dir",
        default="/g/data/li96/mz9869/data/coco_subset",
    )
    p.add_argument(
        "--out",
        default="/scratch/li96/mz9869/medusa_outputs/linked_medusa_v1_full/eval_acceptance_v1_fp16.json",
    )
    p.add_argument("--n-prompts", type=int, default=100)
    p.add_argument(
        "--min-ref-words",
        type=int,
        default=80,
        help="Filter: keep only prompts whose reference GPT answer has >= this many "
             "whitespace-tokens. Long-form responses are what we want to measure σ on.",
    )
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Override: n_prompts=5, max_new_tokens=30. For pipeline validation.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HF repo id; cached under HF_HOME.",
    )
    args = p.parse_args()

    if args.dry_run:
        args.n_prompts = 5
        args.max_new_tokens = 30
        # Redirect to a separate JSON so the full-run output path stays
        # untouched until we explicitly bless the dry-run results.
        if args.out.endswith("eval_acceptance_v1_fp16.json"):
            args.out = args.out.replace(
                "eval_acceptance_v1_fp16.json",
                "eval_acceptance_v1_fp16_dryrun.json",
            )
    return args


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------


def _cfg_attr(cfg, name: str, default=None):
    """Read attribute from a possibly-nested HF config.

    transformers 5.x Qwen2_5_VLConfig keeps llm hyperparams under
    `cfg.text_config.*` rather than at the top level. Earlier versions
    had them at top level. This helper falls back across both layouts.
    """
    if hasattr(cfg, name):
        return getattr(cfg, name)
    for sub in ("text_config", "llm_config", "language_config"):
        sub_cfg = getattr(cfg, sub, None)
        if sub_cfg is not None and hasattr(sub_cfg, name):
            return getattr(sub_cfg, name)
    return default


def load_base(model_id: str):
    print(f"[load] base model: {model_id} (fp16, cuda:0)")
    t0 = time.time()
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    hidden = _cfg_attr(base.config, "hidden_size")
    vocab = _cfg_attr(base.config, "vocab_size")
    print(
        f"[load] base ready in {time.time() - t0:.1f}s  "
        f"hidden={hidden}  vocab={vocab}"
    )
    return base, processor


def load_head(ckpt_path: str, hidden_dim: int, vocab_size: int) -> LinkedMedusaHeads:
    print(f"[load] linked heads from {ckpt_path}")
    t0 = time.time()
    sd_full = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head_sd = sd_full.get("model", sd_full.get("state_dict", sd_full))

    m_cfg = (sd_full.get("cfg") or {}).get("model", {})
    H = int(m_cfg.get("hidden_dim", hidden_dim))
    V = int(m_cfg.get("vocab_size", vocab_size))
    num_heads = int(m_cfg.get("num_heads", 3))
    num_blocks = int(m_cfg.get("num_blocks", 2))
    expansion = int(m_cfg.get("expansion", 2))

    head = LinkedMedusaHeads(
        hidden_dim=H,
        vocab_size=V,
        num_heads=num_heads,
        num_blocks=num_blocks,
        expansion=expansion,
    )
    missing, unexpected = head.load_state_dict(head_sd, strict=True)
    head = head.cuda().half().eval()
    print(
        f"[load] head ready in {time.time() - t0:.1f}s  "
        f"K={num_heads} num_blocks={num_blocks} expansion={expansion}  "
        f"params={sum(p.numel() for p in head.parameters()) / 1e9:.2f}B"
    )
    return head


def filter_prompts(manifest_path: str, min_ref_words: int, seed: int) -> List[Dict]:
    with open(manifest_path) as f:
        data = json.load(f)
    out = []
    for item in data:
        convs = item.get("conversations") or []
        if len(convs) < 2:
            continue
        q = convs[0].get("value", "")
        a = convs[1].get("value", "")
        if len(a.split()) < min_ref_words:
            continue
        out.append(
            {
                "id": item["id"],
                "image": item["image"],
                "question": q.replace("<image>", "").strip(),
                "answer": a,
                "answer_word_count": len(a.split()),
            }
        )
    rng = random.Random(seed)
    rng.shuffle(out)
    return out


# ----------------------------------------------------------------------------
# Verify loop
# ----------------------------------------------------------------------------


def _argmax_masked(logits_1d: torch.Tensor, max_id: int = EFFECTIVE_VOCAB) -> int:
    """argmax restricted to [0, max_id), -inf'ing the padded rows."""
    if logits_1d.size(-1) > max_id:
        logits_1d = logits_1d.clone()
        logits_1d[..., max_id:] = float("-inf")
    return int(logits_1d.argmax(-1).item())


def make_inputs(processor, question: str, image_path: Path) -> Dict[str, torch.Tensor]:
    img = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True)
    return inputs.to("cuda:0")


@torch.no_grad()
def vanilla_greedy(
    base,
    processor,
    question: str,
    image_path: Path,
    max_new: int,
    eos_id: int,
) -> List[int]:
    """Plain autoregressive greedy decoding for sanity-comparing to our speculative output."""
    inputs = make_inputs(processor, question, image_path)
    out = base(**inputs, use_cache=True)
    past_kv = out.past_key_values
    next_tok = _argmax_masked(out.logits[0, -1, :])
    emitted = [next_tok]
    for _ in range(max_new - 1):
        if next_tok == eos_id:
            break
        b_in = torch.tensor([[next_tok]], device="cuda:0", dtype=torch.long)
        out = base(input_ids=b_in, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        next_tok = _argmax_masked(out.logits[0, -1, :])
        emitted.append(next_tok)
    return emitted


@torch.no_grad()
def run_one_prompt(
    base,
    head: LinkedMedusaHeads,
    processor,
    prompt: Dict,
    images_dir: Path,
    max_new: int,
    eos_id: int,
) -> Dict:
    """Run greedy speculative decoding for one prompt; return per-prompt stats."""
    K = head.num_heads
    inputs = make_inputs(processor, prompt["question"], images_dir / prompt["image"])

    # ---- prefill -------------------------------------------------------------
    out = base(**inputs, use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    prefill_len = past_kv.get_seq_length()
    h_t = out.hidden_states[-1][0, -1, :].clone()
    base_pred_t = _argmax_masked(out.logits[0, -1, :])

    # SANITY: verify h_t IS the post-norm hidden that lm_head expects.
    # If hidden_states[-1] is pre-final-norm, base.lm_head(h_t) argmax will
    # disagree with the logits' argmax, signalling a wrong-hidden-state bug.
    with torch.no_grad():
        lm_argmax_from_h = _argmax_masked(base.lm_head(h_t.unsqueeze(0))[0])
    sanity_ok = (lm_argmax_from_h == base_pred_t)
    del out
    torch.cuda.empty_cache()

    accept_log: List[int] = []
    emitted_tokens: List[int] = []
    rounds = 0
    hit_eos = False

    # ---- decode loop ---------------------------------------------------------
    while len(emitted_tokens) < max_new:
        rounds += 1

        # heads → K candidates
        all_logits = head(h_t.view(1, 1, -1).half())  # list of (1,1,V) tensors
        candidates: List[int] = [_argmax_masked(L[0, 0]) for L in all_logits]
        del all_logits

        # verify forward
        v_in = torch.tensor([candidates], device="cuda:0", dtype=torch.long)
        v_out = base(
            input_ids=v_in,
            past_key_values=past_kv,
            use_cache=True,
            output_hidden_states=False,  # we don't need hidden from verify (bonus fwd will produce it)
        )
        v_logits = v_out.logits[0]  # (K, V)

        # base predictions at pos t+i, i in 0..K-1:
        #   i==0  -> base_pred_t (from prior step / prefill)
        #   i>=1  -> argmax(v_logits[i-1])
        base_preds = [base_pred_t] + [
            _argmax_masked(v_logits[i]) for i in range(K - 1)
        ]

        # accept_len = longest matching prefix where candidates[i] == base_preds[i]
        accept_len = 0
        for i in range(K):
            if candidates[i] == base_preds[i]:
                accept_len = i + 1
            else:
                break

        # bonus token at pos t+accept_len = base's natural pick there
        if accept_len == 0:
            bonus = base_pred_t
        else:
            bonus = _argmax_masked(v_logits[accept_len - 1])

        accept_log.append(accept_len)
        accepted_only = candidates[:accept_len]
        emitted_this_round = accepted_only + [bonus]
        prev_emitted = len(emitted_tokens)
        emitted_tokens.extend(emitted_this_round)

        if eos_id in emitted_this_round:
            hit_eos = True
            break

        # ---- update KV + state ----
        # v_out.past_key_values has prefill_len + K positions; truncate to
        # prefill_len + (prev_emitted + accept_len): keep prefill + previously-
        # accepted tokens + this round's accepted candidates. The bonus's KV
        # will be added by the dedicated bonus forward below.
        v_kv = v_out.past_key_values
        v_kv.crop(prefill_len + prev_emitted + accept_len)
        del v_out

        # Bonus forward: append bonus token to extend past_kv by 1 AND produce
        # the hidden state at the bonus's position (= h_t for the NEXT round)
        # plus base's natural argmax at the position AFTER the bonus
        # (= base_pred_t for the NEXT round).
        b_in = torch.tensor([[bonus]], device="cuda:0", dtype=torch.long)
        b_out = base(
            input_ids=b_in,
            past_key_values=v_kv,
            use_cache=True,
            output_hidden_states=True,
        )
        past_kv = b_out.past_key_values
        h_t = b_out.hidden_states[-1][0, -1, :].clone()
        base_pred_t = _argmax_masked(b_out.logits[0, -1, :])
        del b_out

    total_emitted = len(emitted_tokens)
    sigma = total_emitted / max(1, rounds)

    return {
        "id": prompt["id"],
        "image": prompt["image"],
        "answer_word_count": prompt["answer_word_count"],
        "sigma": sigma,
        "rounds": rounds,
        "total_emitted": total_emitted,
        "hit_eos": hit_eos,
        "sanity_lm_head_h_t_matches_logits_argmax": sanity_ok,
        "sanity_lm_argmax_from_h": lm_argmax_from_h,
        "sanity_base_pred_t": base_pred_t,
        "accept_log": accept_log,
        "emitted_tokens": emitted_tokens,
        "accept_distribution": dict(Counter(accept_log)),
        "per_position_accept": [
            sum(1 for a in accept_log if a > k) / max(1, len(accept_log)) for k in range(head.num_heads)
        ],
    }


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    print(f"[args] ckpt           = {args.ckpt}")
    print(f"[args] manifest       = {args.manifest}")
    print(f"[args] images_dir     = {args.images_dir}")
    print(f"[args] out            = {args.out}")
    print(f"[args] n_prompts      = {args.n_prompts}")
    print(f"[args] min_ref_words  = {args.min_ref_words}")
    print(f"[args] max_new_tokens = {args.max_new_tokens}")
    print(f"[args] dry_run        = {args.dry_run}")
    print(f"[args] seed           = {args.seed}")

    base, processor = load_base(args.model_id)
    base_hidden = _cfg_attr(base.config, "hidden_size")
    base_vocab = _cfg_attr(base.config, "vocab_size")
    if base_hidden is None or base_vocab is None:
        raise RuntimeError(
            f"Could not find hidden_size/vocab_size on base.config or nested configs. "
            f"Got hidden={base_hidden} vocab={base_vocab}. Config type: {type(base.config).__name__}"
        )
    head = load_head(args.ckpt, base_hidden, base_vocab)
    eos_id = processor.tokenizer.eos_token_id
    print(f"[init] eos_id = {eos_id}")

    prompts = filter_prompts(args.manifest, args.min_ref_words, args.seed)
    print(
        f"[data] {len(prompts)} prompts pass filter (>= {args.min_ref_words} ref words)"
    )
    prompts = prompts[: args.n_prompts]

    print("[data] === preview of first 5 selected prompts ===")
    images_dir = Path(args.images_dir)
    for i, p in enumerate(prompts[:5]):
        img_path = images_dir / p["image"]
        ok = "✓" if img_path.is_file() else "✗"
        q_preview = p["question"][:80].replace("\n", " ")
        print(
            f"  [{i}] {ok} {p['image']} | ref_words={p['answer_word_count']} | Q: {q_preview}..."
        )

    # ---- run -----
    print(f"\n[run] starting σ measurement over {len(prompts)} prompts...")
    results = []
    t0 = time.time()
    for i, p in enumerate(prompts):
        rt = time.time()
        try:
            r = run_one_prompt(
                base=base,
                head=head,
                processor=processor,
                prompt=p,
                images_dir=images_dir,
                max_new=args.max_new_tokens,
                eos_id=eos_id,
            )
        except Exception as e:
            print(f"[run {i + 1}/{len(prompts)}] FAILED on {p['image']}: {e!r}")
            raise
        dt = time.time() - rt
        print(
            f"[run {i + 1:>3}/{len(prompts)}] σ={r['sigma']:.3f}  emit={r['total_emitted']}  "
            f"rounds={r['rounds']}  dist={r['accept_distribution']}  "
            f"eos={r['hit_eos']}  sanity={r['sanity_lm_head_h_t_matches_logits_argmax']}  ({dt:.1f}s)"
        )

        # Dry-run only: produce vanilla greedy baseline + decoded text + first-divergence point
        if args.dry_run:
            vg = vanilla_greedy(
                base=base,
                processor=processor,
                question=p["question"],
                image_path=images_dir / p["image"],
                max_new=args.max_new_tokens,
                eos_id=eos_id,
            )
            # find longest common prefix
            common = 0
            while common < min(len(vg), len(r["emitted_tokens"])) and vg[common] == r["emitted_tokens"][common]:
                common += 1
            text_spec = processor.tokenizer.decode(r["emitted_tokens"], skip_special_tokens=False)
            text_van = processor.tokenizer.decode(vg, skip_special_tokens=False)
            print(f"        vanilla_greedy  : len={len(vg)}  text: {text_van[:120]!r}")
            print(f"        speculative_dec : len={len(r['emitted_tokens'])}  text: {text_spec[:120]!r}")
            print(f"        common_prefix_len = {common}/{min(len(vg), len(r['emitted_tokens']))}")
            if common < min(len(vg), len(r["emitted_tokens"])):
                print(f"        FIRST DIVERGE @ idx {common}: "
                      f"vanilla={vg[common]} ({processor.tokenizer.decode([vg[common]])!r})  "
                      f"vs spec={r['emitted_tokens'][common]} ({processor.tokenizer.decode([r['emitted_tokens'][common]])!r})")
            r["vanilla_greedy_tokens"] = vg
            r["common_prefix_len"] = common
            r["text_spec"] = text_spec
            r["text_vanilla"] = text_van

        results.append(r)
    total_time = time.time() - t0
    print(f"\n[run] complete in {total_time:.1f}s ({total_time / max(1, len(results)):.2f}s/prompt)")

    # ---- aggregate ----
    sigmas = [r["sigma"] for r in results]
    K = head.num_heads
    total_rounds_all = sum(r["rounds"] for r in results)
    per_pos_accept = [
        sum(int(a > k) for r in results for a in r["accept_log"]) / max(1, total_rounds_all)
        for k in range(K)
    ]
    agg_dist: Counter = Counter()
    for r in results:
        agg_dist.update(r["accept_log"])

    agg: Dict[str, Any] = {
        "n_prompts": len(results),
        "sigma_mean": statistics.mean(sigmas) if sigmas else 0.0,
        "sigma_median": statistics.median(sigmas) if sigmas else 0.0,
        "sigma_p25": statistics.quantiles(sigmas, n=4)[0] if len(sigmas) >= 4 else None,
        "sigma_p75": statistics.quantiles(sigmas, n=4)[2] if len(sigmas) >= 4 else None,
        "sigma_min": min(sigmas) if sigmas else 0.0,
        "sigma_max": max(sigmas) if sigmas else 0.0,
        "per_position_accept_rate": per_pos_accept,
        "accept_distribution_aggregate": {str(k): v for k, v in sorted(agg_dist.items())},
        "total_rounds": total_rounds_all,
        "total_tokens_emitted": sum(r["total_emitted"] for r in results),
        "wall_time_s": total_time,
        "args": vars(args),
        "per_prompt_results": results,
    }

    print("\n=== AGGREGATE ===")
    print(f"  σ_mean   = {agg['sigma_mean']:.3f}")
    print(f"  σ_median = {agg['sigma_median']:.3f}")
    print(f"  σ_range  = [{agg['sigma_min']:.3f}, {agg['sigma_max']:.3f}]")
    if agg["sigma_p25"] is not None:
        print(f"  σ_p25/p75 = {agg['sigma_p25']:.3f} / {agg['sigma_p75']:.3f}")
    print(f"  per-position accept (head_k accepts): "
          f"{[f'{p:.3f}' for p in per_pos_accept]}")
    print(f"  accept length distribution: {dict(sorted(agg_dist.items()))}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n[save] -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
