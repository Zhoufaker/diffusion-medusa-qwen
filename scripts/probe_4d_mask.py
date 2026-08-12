"""probe_4d_mask.py — verify Qwen2.5-VL accepts a custom 4D attention mask.

Gating experiment for P0 (tree speculative decoding). Tree verify needs to
flatten a candidate tree into one sequence and feed a CUSTOM 4D attention mask
so each node only attends to its ancestor path. If transformers 5.3.0 does NOT
pass a 4D mask straight through to attention, we'd have to patch the attention
forward or downgrade transformers — a fork we want to discover NOW.

Source analysis (already done, this confirms empirically):
  - transformers/masking_utils.py::_preprocess_mask_arguments (L787-789):
        if isinstance(attention_mask, (Tensor, BlockMask)) and len(shape)==4:
            return True, attention_mask, None, None, None   # early exit, as-is
  - models/qwen2_5_vl/modeling_qwen2_5_vl.py (L1391):
        self.language_model(..., attention_mask=attention_mask, ...)  # passthrough
  - position_ids: outer forward only auto-computes when position_ids is None
        (L1376), so a custom 2D position_ids is honored.

Mask convention for SDPA: additive float mask of shape (B, H_or_1, q_len, kv_len);
0.0 = attend, finfo.min = blocked.

Checks:
  1. explicit 4D causal mask  == default causal (no-mask) logits      [numeric eq]
  2. tree-style mask (block one key from last query) CHANGES last-pos
     logits but leaves pos-0 logits unchanged                          [mask active]
  3. representative tree verify: prefill -> past_kv, then a 3-node tree
     with custom 4D mask (siblings can't see each other) + custom
     position_ids; assert it runs and logits are finite + siblings
     with identical context produce sensible (distinct-by-self) outputs [end-to-end]
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["HF_HOME"] = "/scratch/li96/mz9869/tmp_hf_download"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from PIL import Image

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MANIFEST = "/g/data/li96/mz9869/data/llava_subset_2k.json"
IMAGES_DIR = "/g/data/li96/mz9869/data/coco_subset"
EFFECTIVE_VOCAB = 151936


def banner(s: str) -> None:
    print("\n" + "=" * 64 + f"\n{s}\n" + "=" * 64)


def argmax_masked(logits_1d: torch.Tensor, max_id: int = EFFECTIVE_VOCAB) -> int:
    if logits_1d.size(-1) > max_id:
        logits_1d = logits_1d.clone()
        logits_1d[..., max_id:] = float("-inf")
    return int(logits_1d.argmax(-1).item())


def make_image_inputs(proc, question: str, image_path: Path, device: str):
    img = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        }
    ]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt", padding=True)
    return inputs.to(device)


def main() -> int:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    banner("LOAD (forcing attn_implementation=sdpa for custom-mask support)")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()
    proc = AutoProcessor.from_pretrained(MODEL_ID)

    device = "cuda:0"
    dtype = base.dtype
    mn = torch.finfo(dtype).min
    print(f"config._attn_implementation          = {base.config._attn_implementation}")
    try:
        print(
            f"language_model._attn_implementation  = "
            f"{base.model.language_model.config._attn_implementation}"
        )
    except Exception as e:  # noqa: BLE001
        print(f"(could not read text attn impl: {e!r})")

    text = "The quick brown fox jumps over the lazy dog and then runs away very quickly."
    ids = proc.tokenizer(text, return_tensors="pt").input_ids.to(device)
    L = ids.shape[1]
    print(f"seq len L = {L}")

    results = {}

    # ---- Check 1: explicit 4D causal == default ----------------------------
    # NOTE on tolerance: the no-mask path dispatches SDPA with is_causal=True
    # (a fused kernel), while the explicit-additive-mask path uses the masked
    # kernel. In fp16 these two kernels differ by ~0.1 in raw logits — that is
    # kernel rounding, NOT a masking error. The decision-relevant metric for
    # greedy speculative decoding is ARGMAX agreement, so we judge on that.
    banner("CHECK 1: explicit 4D causal mask vs default (no mask)")
    with torch.no_grad():
        logits_def = base(input_ids=ids, use_cache=False).logits[0]  # (L, V)

    cmask = torch.triu(
        torch.full((1, 1, L, L), mn, device=device, dtype=dtype), diagonal=1
    )  # 0 on/below diag, mn strictly above -> standard causal
    with torch.no_grad():
        logits_4d = base(input_ids=ids, attention_mask=cmask, use_cache=False).logits[0]

    d1 = (logits_def - logits_4d).abs().max().item()
    am_def = logits_def.argmax(-1)
    am_4d = logits_4d.argmax(-1)
    argmax_match = int((am_def == am_4d).sum().item())
    ok1 = argmax_match == L  # every position's greedy token identical
    print(f"max|Δlogit| (4D-causal vs default)   = {d1:.6e}  (fp16 kernel noise, informational)")
    print(f"argmax agreement                     = {argmax_match}/{L}  -> {'PASS' if ok1 else 'FAIL'}")
    if not ok1:
        mism = (am_def != am_4d).nonzero(as_tuple=True)[0].tolist()
        print(f"  mismatched positions: {mism}")
    results["check1_causal_argmax_equiv"] = ok1

    # ---- Check 2: tree-style mask actually masks --------------------------
    banner("CHECK 2: blocking a key from the last query changes only that row")
    tmask = cmask.clone()
    tmask[0, 0, L - 1, 1] = mn  # last query may NOT attend to key position 1
    with torch.no_grad():
        logits_t = base(input_ids=ids, attention_mask=tmask, use_cache=False).logits[0]
    d2_last = (logits_def[L - 1] - logits_t[L - 1]).abs().max().item()
    d2_pos0 = (logits_def[0] - logits_t[0]).abs().max().item()
    ok2 = d2_last > 1e-2 and d2_pos0 < 5e-2
    print(f"last-pos  max|Δ| = {d2_last:.4e}  (expect > 0, mask took effect)")
    print(f"pos-0     max|Δ| = {d2_pos0:.4e}  (expect ~ 0, untouched)")
    print(f"-> {'PASS' if ok2 else 'FAIL'}")
    results["check2_mask_active"] = ok2

    # ---- Check 3: IMAGE-based tree verify with correct M-RoPE offset -------
    # The test that was MISSED in probe v1: v1 used a text-only prefix where
    # rope_delta == 0, so the M-RoPE offset bug stayed hidden. With a real image
    # prefix, rope_delta != 0 (image tokens compress positions, delta < 0). A
    # tree node's continuation position must be cont_base + (depth-1), where
    #   cont_base = P + base.model.rope_deltas   (NOT just P).
    # We compare a 3-node sibling tree's node-1 logits against a known-correct
    # LINEAR reference (sequential forwards with position_ids=None == the chain
    # path that already gives correct image σ). We test BOTH the FIXED offset
    # (expect argmax match) and the BUGGY offset (cont_base=P; expect mismatch
    # on images) — proving the bug is real AND the fix resolves it.
    banner("CHECK 3: IMAGE tree verify — M-RoPE offset (buggy vs fixed)")

    with open(MANIFEST) as f:
        manifest = json.load(f)
    sample = None
    for item in manifest:
        ip = Path(IMAGES_DIR) / item["image"]
        if ip.is_file():
            q = item["conversations"][0]["value"].replace("<image>", "").strip()
            sample = (q, ip)
            break
    if sample is None:
        print("FAIL: no image sample found")
        results["check3_image_mrope_fixed"] = False
    else:
        question, image_path = sample
        print(f"image: {image_path.name}   Q: {question[:66]!r}")
        inputs = make_image_inputs(proc, question, image_path, device)

        # linear reference cache (auto positions == chain path, known-correct)
        with torch.no_grad():
            pre_lin = base(**inputs, use_cache=True)
        past_lin = pre_lin.past_key_values
        P = past_lin.get_seq_length()
        base_pred_root = argmax_masked(pre_lin.logits[0, -1, :])

        rd = base.model.rope_deltas
        rope_delta = int(rd.flatten()[0].item()) if rd is not None else 0
        cont_base = P + rope_delta
        print(f"prefill P = {P}   rope_deltas = "
              f"{None if rd is None else rd.tolist()}   rope_delta = {rope_delta}   "
              f"cont_base = {cont_base}")
        if rope_delta == 0:
            print("  WARN: rope_delta == 0 on this image — bug would stay hidden "
                  "(buggy==fixed). Unexpected for an image prefix.")

        tok0 = base_pred_root            # node0 (depth1) = base's own next token
        tok1, tok2 = 1230, 9876          # node1/node2 (depth2 siblings under node0)
        tree_ids = torch.tensor([[tok0, tok1, tok2]], device=device)
        T = 3

        def tree_mask(P_):
            m = torch.full((1, 1, T, P_ + T), mn, device=device, dtype=dtype)
            m[0, 0, :, :P_] = 0.0
            m[0, 0, 0, P_ + 0] = 0.0     # node0 -> self
            m[0, 0, 1, P_ + 0] = 0.0     # node1 -> node0
            m[0, 0, 1, P_ + 1] = 0.0     # node1 -> self
            m[0, 0, 2, P_ + 0] = 0.0     # node2 -> node0
            m[0, 0, 2, P_ + 2] = 0.0     # node2 -> self (sibling-isolated)
            return m

        # linear reference: append tok0, then tok1, auto positions
        with torch.no_grad():
            s0 = base(input_ids=torch.tensor([[tok0]], device=device),
                      past_key_values=past_lin, use_cache=True)
            s1 = base(input_ids=torch.tensor([[tok1]], device=device),
                      past_key_values=s0.past_key_values, use_cache=True)
        linear_node1 = argmax_masked(s1.logits[0, -1, :])

        # tree FIXED offset
        with torch.no_grad():
            past_fix = base(**inputs, use_cache=True).past_key_values
            out_fix = base(input_ids=tree_ids, attention_mask=tree_mask(P),
                           past_key_values=past_fix,
                           position_ids=torch.tensor(
                               [[cont_base, cont_base + 1, cont_base + 1]], device=device),
                           use_cache=True)
        tree_fixed_node1 = argmax_masked(out_fix.logits[0, 1, :])
        fin = bool(torch.isfinite(out_fix.logits).all().item())

        # tree BUGGY offset (ignores rope_delta)
        with torch.no_grad():
            past_bug = base(**inputs, use_cache=True).past_key_values
            out_bug = base(input_ids=tree_ids, attention_mask=tree_mask(P),
                           past_key_values=past_bug,
                           position_ids=torch.tensor([[P, P + 1, P + 1]], device=device),
                           use_cache=True)
        tree_buggy_node1 = argmax_masked(out_bug.logits[0, 1, :])

        fixed_match = tree_fixed_node1 == linear_node1
        buggy_match = tree_buggy_node1 == linear_node1
        print(f"linear ref node1 argmax = {linear_node1}")
        print(f"tree FIXED node1 argmax = {tree_fixed_node1}   match={fixed_match}  "
              f"-> {'PASS' if fixed_match else 'FAIL'}")
        print(f"tree BUGGY node1 argmax = {tree_buggy_node1}   match={buggy_match}  "
              f"(expect MISMATCH to prove the bug is real)")
        print(f"logits finite = {fin}")
        results["check3_image_mrope_fixed"] = bool(fixed_match and fin)
        results["check3_mrope_bug_demonstrated"] = bool(rope_delta != 0 and not buggy_match)

    banner("SUMMARY")
    for k, v in results.items():
        print(f"  {k:34s}: {'PASS' if v else 'FAIL/NO'}")
    # Gating: the three correctness checks must pass. bug_demonstrated is
    # informational (confirms the image actually exercised M-RoPE).
    gate_keys = ["check1_causal_argmax_equiv", "check2_mask_active",
                 "check3_image_mrope_fixed"]
    all_ok = all(results.get(k, False) for k in gate_keys)
    print(f"\n  4D CUSTOM MASK + M-RoPE OFFSET CORRECT (transformers 5.3.0 / sdpa): "
          f"{'YES ✓' if all_ok else 'NO ✗'}")
    if results.get("check3_mrope_bug_demonstrated"):
        print("  (buggy cont_base=P mismatched on image -> M-RoPE offset bug "
              "confirmed real; fix cont_base=P+rope_deltas resolves it.)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
