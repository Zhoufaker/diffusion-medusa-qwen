"""diag_kernel_argmax.py — quantify SDPA kernel argmax flips on image continuations.

Explains the tree(width-1) vs chain regression divergence. chain verify passes
attention_mask=None (SDPA dispatches the is_causal FUSED kernel); tree verify
passes an explicit additive 4D causal mask (SDPA dispatches the MASKED kernel).
The two kernels differ by ~0.1 fp16 logit (probe CHECK 1). At a near-tie this
flips the greedy argmax -> a single mid-stream divergence with identical length.

For each prompt: greedy-decode N tokens, then re-evaluate that exact token
sequence two ways in a single multi-token forward on top of the prefill cache:
  A: no attention_mask  (auto-causal == chain verify kernel)
  B: explicit additive causal 4D mask + explicit cont_base positions (== tree)
Compare per-position argmax; for every flip print path-A top-2 margin.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", "/scratch/li96/mz9869/tmp_hf_download")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from decode.common import (  # noqa: E402
    EFFECTIVE_VOCAB,
    continuation_base,
    filter_prompts,
    load_base,
    make_image_inputs,
    mask_phantom_,
)

IMAGES = ["000000055049.jpg"]   # the diverging prompt; [] => first 5 by filter
N_TOK = 60


@torch.no_grad()
def diag_prompt(base, processor, question, image_path, device="cuda:0"):
    dtype = base.dtype
    mn = torch.finfo(dtype).min
    inputs = make_image_inputs(processor, question, image_path, device)

    # greedy decode N_TOK tokens (auto path)
    out = base(**inputs, use_cache=True)
    past = out.past_key_values
    P = past.get_seq_length()
    cont_base = continuation_base(base, P)
    nxt = int(mask_phantom_(out.logits[0, -1, :]).argmax().item())
    toks = [nxt]
    for _ in range(N_TOK - 1):
        out = base(input_ids=torch.tensor([[nxt]], device=device),
                   past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = int(mask_phantom_(out.logits[0, -1, :]).argmax().item())
        toks.append(nxt)
    T = len(toks)
    seq = torch.tensor([toks], device=device)

    # path A: no mask (auto-causal == chain verify kernel)
    pastA = base(**inputs, use_cache=True).past_key_values
    logitsA = base(input_ids=seq, past_key_values=pastA, use_cache=False).logits[0]

    # path B: explicit additive causal 4D mask + explicit positions (== tree verify)
    pastB = base(**inputs, use_cache=True).past_key_values
    causal = torch.triu(torch.full((T, T), mn, device=device, dtype=dtype), diagonal=1)
    maskB = torch.zeros((1, 1, T, P + T), device=device, dtype=dtype)
    maskB[0, 0, :, P:] = causal
    posB = torch.arange(cont_base, cont_base + T, device=device).unsqueeze(0)
    logitsB = base(input_ids=seq, attention_mask=maskB, past_key_values=pastB,
                   position_ids=posB, use_cache=False).logits[0]

    amA = mask_phantom_(logitsA).argmax(-1)
    amB = mask_phantom_(logitsB).argmax(-1)
    flips = (amA != amB).nonzero(as_tuple=True)[0].tolist()
    max_dlogit = (logitsA[:, :EFFECTIVE_VOCAB] - logitsB[:, :EFFECTIVE_VOCAB]).abs().max().item()

    print(f"\nimage {image_path.name}  P={P} rope_delta={cont_base - P} cont_base={cont_base}")
    print(f"  positions compared        = {T}")
    print(f"  argmax flips (A vs B)      = {len(flips)} / {T}   at {flips}")
    print(f"  max|Δlogit| over seq       = {max_dlogit:.4e}  (fp16 kernel noise scale)")
    for j in flips:
        a = mask_phantom_(logitsA[j]).float()
        top2 = a.topk(2).values
        margin = (top2[0] - top2[1]).item()
        print(f"    flip@{j}: pathA top2 margin = {margin:.4e}  "
              f"(A={int(amA[j])} B={int(amB[j])})  -> near-tie => kernel noise")
    return len(flips), T, flips


def main() -> int:
    base, processor = load_base("Qwen/Qwen2.5-VL-7B-Instruct")
    images_dir = Path("/g/data/li96/mz9869/data/coco_subset")
    prompts = filter_prompts("/g/data/li96/mz9869/data/llava_subset_2k.json", 80, 42)
    if IMAGES:
        sel = [p for p in prompts if p["image"] in IMAGES]
    else:
        sel = prompts[:5]
    total_flips = total_pos = 0
    for p in sel:
        f, t, _ = diag_prompt(base, processor, p["question"], images_dir / p["image"])
        total_flips += f; total_pos += t
    print(f"\n=== SUMMARY: {total_flips} argmax flips / {total_pos} positions "
          f"({100 * total_flips / max(1, total_pos):.2f}%) ===")
    print("If all flips have tiny pathA margins, the tree(width-1) vs chain "
          "divergence is SDPA kernel fp16 noise (is_causal vs masked), not a "
          "mask/position/accept/reorg logic bug.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
