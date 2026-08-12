"""decode.common — shared loaders + vocab masking + M-RoPE helper.

SINGLE SOURCE OF TRUTH for:
  * EFFECTIVE_VOCAB (phantom-token mask boundary)
  * argmax_masked / topk_masked (mask padded vocab rows BEFORE argmax/top-k)
  * base + head loaders (base ALWAYS loaded with attn_implementation="sdpa",
    required for custom 4D tree masks; verified by scripts/probe_4d_mask.py)
  * make_image_inputs (chat-template + image -> processor tensors)
  * filter_prompts (long-form LLaVA prompt selection)
  * vanilla_greedy (plain AR decode, for the tok/s baseline)
  * continuation_base (M-RoPE: cont_base = past_len + base.model.rope_deltas)

The chain evaluator (scripts/eval_acceptance.py) keeps its own copies for now
(it produced the validated σ=1.95 and must not regress); new code (tree, P1)
imports from here so 151936 / the SDPA requirement / the rope offset are never
re-hardcoded inconsistently.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image

# lm_head physical dim is 152064 (padded for hardware alignment); ids in
# [EFFECTIVE_VOCAB .. 152063] are padding rows with unbounded logits. Mask them
# to -inf before any argmax / top-k so a trained head can't pick a phantom token.
EFFECTIVE_VOCAB = 151936


# ----------------------------------------------------------------------------
# Vocab masking
# ----------------------------------------------------------------------------


def mask_phantom_(logits: torch.Tensor, max_id: int = EFFECTIVE_VOCAB) -> torch.Tensor:
    """Set logits[..., max_id:] = -inf, in a fresh tensor. Returns the masked copy."""
    if logits.size(-1) > max_id:
        logits = logits.clone()
        logits[..., max_id:] = float("-inf")
    return logits


def argmax_masked(logits_1d: torch.Tensor, max_id: int = EFFECTIVE_VOCAB) -> int:
    """argmax over the effective vocab only."""
    return int(mask_phantom_(logits_1d, max_id).argmax(-1).item())


def topk_masked(
    logits_1d: torch.Tensor, k: int, max_id: int = EFFECTIVE_VOCAB
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (logprobs, token_ids) of the top-k over the effective vocab.

    logprobs are log-softmax values (so path joint logprob = sum of these).
    """
    masked = mask_phantom_(logits_1d, max_id)
    logprobs = torch.log_softmax(masked.float(), dim=-1)
    p, idx = logprobs.topk(k)
    return p, idx


def truncate_emit_path(
    tokens: Sequence[int],
    remaining: int,
    eos_id: int,
) -> Tuple[List[int], bool]:
    """Hard-cap an emit path to ``remaining`` tokens, stopping at EOS inclusive.

    Runner soft-cap bug (round-2 review P1): rounds used to append a full
    accepted(+bonus) path after only checking ``len(emitted) < max_new`` at
    loop entry, so outputs could exceed ``max_new`` by up to one tree width.

    Returns ``(to_emit, hit_eos)`` with ``len(to_emit) <= max(0, remaining)``.
    """
    if remaining <= 0:
        return [], False
    out: List[int] = []
    hit_eos = False
    for t in tokens:
        out.append(int(t))
        if int(t) == eos_id:
            hit_eos = True
            break
        if len(out) >= remaining:
            break
    return out, hit_eos


# ----------------------------------------------------------------------------
# Config helper
# ----------------------------------------------------------------------------


def cfg_attr(cfg, name: str, default=None):
    """Read attr from a possibly-nested HF config (transformers 5.x nests under text_config)."""
    if hasattr(cfg, name):
        return getattr(cfg, name)
    for sub in ("text_config", "llm_config", "language_config"):
        sub_cfg = getattr(cfg, sub, None)
        if sub_cfg is not None and hasattr(sub_cfg, name):
            return getattr(sub_cfg, name)
    return default


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------


def load_base(model_id: str, device: str = "cuda:0"):
    """Load Qwen2.5-VL base, fp16, FORCING sdpa attention.

    sdpa is REQUIRED for the tree's custom 4D additive mask (FA2 rejects
    arbitrary masks). Verified by scripts/probe_4d_mask.py check 1-3.
    """
    print(f"[load] base model: {model_id} (fp16, {device}, attn=sdpa)")
    t0 = time.time()
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="sdpa",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    hidden = cfg_attr(base.config, "hidden_size")
    vocab = cfg_attr(base.config, "vocab_size")
    assert base.config._attn_implementation == "sdpa", (
        f"expected sdpa, got {base.config._attn_implementation}"
    )
    print(f"[load] base ready in {time.time() - t0:.1f}s  hidden={hidden}  vocab={vocab}")
    return base, processor


def load_head(ckpt_path: str, hidden_dim: int, vocab_size: int, device: str = "cuda:0"):
    """Load LinkedMedusaHeads from a training checkpoint; cuda+half+eval."""
    # Local import so this module is importable on a CPU login node for syntax checks.
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from model import LinkedMedusaHeads

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
        hidden_dim=H, vocab_size=V, num_heads=num_heads,
        num_blocks=num_blocks, expansion=expansion,
    )
    has_bonus = any(k.startswith("bonus_proj") for k in head_sd)
    if has_bonus:
        res = head.load_state_dict(head_sd, strict=True)
    else:
        res = head.load_state_dict(head_sd, strict=False)
        if res.unexpected_keys:
            raise RuntimeError(
                f"unexpected keys loading {ckpt_path}: {res.unexpected_keys}"
            )
        if set(res.missing_keys) != {"bonus_proj.weight"}:
            raise RuntimeError(
                f"pre-C1 ckpt must miss exactly bonus_proj.weight; "
                f"got missing={res.missing_keys}"
            )
    head = head.to(device).half().eval()
    print(
        f"[load] head ready in {time.time() - t0:.1f}s  "
        f"K={num_heads} num_blocks={num_blocks} expansion={expansion}  "
        f"params={sum(p.numel() for p in head.parameters()) / 1e9:.2f}B"
    )
    return head


# ----------------------------------------------------------------------------
# M-RoPE continuation base (BLOCKING fix, verified on image prefix)
# ----------------------------------------------------------------------------


def rope_delta(base) -> int:
    """Scalar M-RoPE delta cached on the model after a (vision) prefill.

    For an image prefix this is NEGATIVE (image tokens compress positions);
    for text-only it is 0. See scripts/probe_4d_mask.py CHECK 3.
    """
    rd = base.model.rope_deltas
    return 0 if rd is None else int(rd.flatten()[0].item())


def continuation_base(base, past_len: int) -> int:
    """The RoPE position the base model itself assigns to the next token.

    cont_base = past_len + rope_delta. Any forward that passes EXPLICIT
    position_ids (the tree does) MUST offset by rope_delta, otherwise on image
    prompts the positions are wrong by ~hundreds and attention degenerates
    (verified: buggy cont_base=P emitted EOS instead of the correct token).
    """
    return past_len + rope_delta(base)


# ----------------------------------------------------------------------------
# Inputs / prompts
# ----------------------------------------------------------------------------


def make_image_inputs(processor, question: str, image_path: Path, device: str = "cuda:0"):
    img = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "user",
         "content": [{"type": "image"}, {"type": "text", "text": question}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True)
    return inputs.to(device)


def filter_prompts(
    manifest_path: str,
    min_ref_words: int,
    seed: int,
    ordered: bool = False,
) -> List[Dict]:
    """Load LLaVA-style prompts; optionally preserve file order (no shuffle).

    ordered=True: keep survivors in manifest order (nested-300 / MM-Vet fixed
    manifests). ordered=False (default): Random(seed).shuffle — legacy gate path.
    """
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
        out.append({
            "id": item["id"],
            "image": item["image"],
            "question": q.replace("<image>", "").strip(),
            "answer": a,
            "answer_word_count": len(a.split()),
        })
    if not ordered:
        random.Random(seed).shuffle(out)
    return out


# O0 kernel band: candidate-specific near_tie requires the speculative
# candidate's own full-precision logit to sit within `band` of the greedy
# argmax. See docs/ood_mmvet_protocol.md.
O0_KERNEL_BAND = 0.15

# Round-5 narrowed wording, kept for provenance of superseded artifacts.
O0_CLAIM_NARROWED = (
    "algorithmic greedy lossless; no first divergence with greedy "
    "top1-top2 gap > 0.15 under the current O0 heuristic"
)
# Round-6 wording. Superseded: its `spec_tok == greedy_top2` disjunct could
# admit an arbitrarily large gap. Kept for provenance of Round-6 artifacts.
O0_CLAIM_CANDIDATE_SPECIFIC = (
    "algorithmic greedy lossless; every first mid-sequence divergence is a "
    "candidate-specific near_tie "
    "(spec_tok == greedy_top2 or logit[top1]-logit[spec] <= 0.15)"
)
# Official claim from Round-7 on: gap-only, no rank shortcut.
O0_CLAIM_GAP_ONLY = (
    "algorithmic greedy lossless; every first mid-sequence divergence is a "
    "candidate-specific near_tie "
    "(0 <= logit[greedy_top1]-logit[spec_tok] <= 0.15)"
)
O0_CLAIM_OFFICIAL = O0_CLAIM_GAP_ONLY


def compare_token_sequences(
    speculative: List[int],
    greedy: List[int],
) -> Optional[Dict]:
    """Byte-level sequence compare.

    Returns None if sequences are identical; otherwise a dict with:
      pos, spec_tok, greedy_tok, spec_len, greedy_len,
      spec_context (±5 toks), greedy_context (±5 toks).
    Does not classify near-tie vs bug — caller / human decides.
    """
    if speculative == greedy:
        return None
    n = min(len(speculative), len(greedy))
    pos = next((i for i in range(n) if speculative[i] != greedy[i]), n)
    # length mismatch with common prefix counts as diverge at first missing index
    def _ctx(seq: List[int], i: int) -> Dict[str, List[int]]:
        return {
            "before": seq[max(0, i - 5): i],
            "at_and_after": seq[i: i + 6],
        }
    return {
        "pos": pos,
        "spec_tok": speculative[pos] if pos < len(speculative) else None,
        "greedy_tok": greedy[pos] if pos < len(greedy) else None,
        "spec_len": len(speculative),
        "greedy_len": len(greedy),
        "spec_context": _ctx(speculative, pos),
        "greedy_context": _ctx(greedy, pos),
    }


def is_candidate_near_tie(
    spec_tok: Optional[int],
    greedy_top2: Optional[int],
    gap_spec: Optional[float],
    band: float = O0_KERNEL_BAND,
) -> bool:
    """Candidate-specific near_tie predicate (Round-7: gap-only).

    True iff a speculative candidate exists and its own full-precision logit
    sits within the band of the greedy argmax: ``0 ≤ gap_spec ≤ band``.

    ``greedy_top2`` is accepted for call-site symmetry but takes **no** part in
    the decision, and neither does ``spec_rank``: both are diagnostics. The
    Round-6 disjunct ``spec_tok == greedy_top2`` was a specification error —
    being ranked second bounds nothing about the size of the gap, so it could
    admit an arbitrarily separated candidate. A missing or negative
    ``gap_spec`` is never a pass; the caller classifies it ``hard``.
    """
    if spec_tok is None or gap_spec is None:
        return False
    gap = float(gap_spec)
    return 0.0 <= gap <= float(band)


def classify_o0_vs_ref(
    speculative: List[int],
    reference: List[int],
    top2_gap: Optional[float] = None,
    band: float = O0_KERNEL_BAND,
    *,
    greedy_top1: Optional[int] = None,
    greedy_top2: Optional[int] = None,
    gap_spec: Optional[float] = None,
    spec_rank: Optional[int] = None,
) -> Dict:
    """Classify speculative vs a reference sequence (O0口径).

    Returns dict with:
      kind: 'match' | 'len_boundary' | 'near_tie' | 'hard'
      div, top2_logit_gap, greedy_top1/top2, gap_spec, spec_rank
    len_boundary = public prefix agrees on [:min(len)]; only unilateral tail.
    near_tie (candidate-specific, gap-only) = mid diverge and 0 ≤ gap_spec ≤ band.
    hard = mid diverge that fails the candidate rule (incl. missing probe).
    ``top2_gap`` (greedy top1−top2), ``greedy_top2`` and ``spec_rank`` are
    recorded as diagnostics and grant nothing.
    """
    div = compare_token_sequences(speculative, reference)
    meta = {
        "top2_logit_gap": top2_gap,
        "greedy_top1": greedy_top1,
        "greedy_top2": greedy_top2,
        "gap_spec": gap_spec,
        "spec_rank": spec_rank,
    }
    if div is None:
        return {"kind": "match", "div": None, **meta}
    m = min(div["spec_len"], div["greedy_len"])
    if div["pos"] >= m:
        return {"kind": "len_boundary", "div": div, **meta}
    if is_candidate_near_tie(div["spec_tok"], greedy_top2, gap_spec, band):
        return {"kind": "near_tie", "div": div, **meta}
    return {"kind": "hard", "div": div, **meta}


def o0_fingerprint_triggers_fail(
    speculative: List[int],
    greedy: List[int],
    top2_gap: Optional[float] = None,
    band: float = O0_KERNEL_BAND,
    *,
    greedy_top1: Optional[int] = None,
    greedy_top2: Optional[int] = None,
    gap_spec: Optional[float] = None,
    spec_rank: Optional[int] = None,
) -> bool:
    """True when mid diverge is classified hard under candidate-specific rule."""
    return classify_o0_vs_ref(
        speculative, greedy, top2_gap, band,
        greedy_top1=greedy_top1, greedy_top2=greedy_top2,
        gap_spec=gap_spec, spec_rank=spec_rank,
    )["kind"] == "hard"


def greedy_numerical_safety_triggers_fail(
    speculative: List[int],
    greedy: List[int],
    top2_gap: Optional[float] = None,
    band: float = O0_KERNEL_BAND,
    *,
    greedy_top1: Optional[int] = None,
    greedy_top2: Optional[int] = None,
    gap_spec: Optional[float] = None,
    spec_rank: Optional[int] = None,
) -> bool:
    """True → greedy_numerical_safety gate must fail.

    Official claim: see ``O0_CLAIM_OFFICIAL`` (candidate-specific).
    ``match`` → byte-exact (n_exact). ``near_tie`` → not byte-exact, does
    **not** fail safety. ``len_boundary`` / ``hard`` → FAIL safety.
    Machine field name ``exact`` means byte-exact only (not this gate).
    """
    kind = classify_o0_vs_ref(
        speculative, greedy, top2_gap, band,
        greedy_top1=greedy_top1, greedy_top2=greedy_top2,
        gap_spec=gap_spec, spec_rank=spec_rank,
    )["kind"]
    return kind in ("hard", "len_boundary")


def greedy_byte_exact_pass(n_exact: int, n_prompts: int) -> bool:
    """True iff every prompt is token-byte-identical to independent greedy.

    Expected **FALSE** on real fp16 runs whenever any near_tie exists.
    """
    return int(n_exact) == int(n_prompts) and int(n_prompts) > 0


def greedy_numerical_safety_pass(n_len_boundary: int, n_hard: int) -> bool:
    """True iff no len_boundary and no hard vs independent greedy."""
    return int(n_len_boundary) == 0 and int(n_hard) == 0


# Archive reproducibility state machine (byte-exact only vs archive).
ARCHIVE_GATE_NOT_RUN = "NOT_RUN"
ARCHIVE_GATE_INCOMPLETE = "INCOMPLETE"
ARCHIVE_GATE_PASS = "PASS"
ARCHIVE_GATE_FAIL = "FAIL"

# Exit codes when --o0-archive is explicitly provided:
#   2 = FAIL, 5 = NOT_RUN, 6 = INCOMPLETE
ARCHIVE_EXIT_FAIL = 2
ARCHIVE_EXIT_NOT_RUN = 5
ARCHIVE_EXIT_INCOMPLETE = 6


def archive_gate_status(
    archive_provided: bool,
    n_covered: int,
    n_prompts: int,
    n_spec_fails: int,
) -> str:
    """Four-state archive reproducibility gate.

    - no archive **or** zero covered → NOT_RUN (never silently PASS)
    - partial coverage → INCOMPLETE (may attach covered-subset verdict; no promote)
    - full coverage + zero fails → PASS
    - full coverage + any fail → FAIL
    """
    if (not archive_provided) or int(n_covered) == 0:
        return ARCHIVE_GATE_NOT_RUN
    if int(n_covered) < int(n_prompts):
        return ARCHIVE_GATE_INCOMPLETE
    return ARCHIVE_GATE_FAIL if int(n_spec_fails) > 0 else ARCHIVE_GATE_PASS


def archive_gate_exit_code(status: str, archive_provided: bool) -> Optional[int]:
    """Non-zero exit when archive was explicitly passed and gate is not PASS.

    No ``--o0-archive`` → None (NOT_RUN is informational only).
    """
    if not archive_provided:
        return None
    if status == ARCHIVE_GATE_PASS:
        return None
    if status == ARCHIVE_GATE_FAIL:
        return ARCHIVE_EXIT_FAIL
    if status == ARCHIVE_GATE_NOT_RUN:
        return ARCHIVE_EXIT_NOT_RUN
    if status == ARCHIVE_GATE_INCOMPLETE:
        return ARCHIVE_EXIT_INCOMPLETE
    return ARCHIVE_EXIT_FAIL


def apply_max_pixels(processor, max_pixels: int) -> None:
    """Cap Qwen2.5-VL image processor longest_edge / max_pixels (V100-safe)."""
    ip = processor.image_processor
    ip.max_pixels = int(max_pixels)
    ip.min_pixels = int(getattr(ip, "min_pixels", None) or 3136)
    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size = {
            "shortest_edge": int(ip.size.get("shortest_edge", ip.min_pixels)),
            "longest_edge": int(max_pixels),
        }
    processor.max_pixels = int(max_pixels)


def o0_archive_triggers_fail(
    speculative: List[int],
    archive_tokens: List[int],
) -> bool:
    """True → archive byte-reproducibility mismatch (any non-identical)."""
    return classify_o0_vs_ref(speculative, archive_tokens)["kind"] != "match"


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_o0_archive_not_self(
    archive_path: str,
    out_path: str,
    run_job_id: Optional[str] = None,
) -> Dict:
    """Runtime anti-self-compare for archive hard gate.

    Refuses if:
      - resolved archive path == resolved out path
      - archive JSON source_out / o0_provenance.source_out == current out
      - archive o0_provenance.job_id == current PBS job id (when both set)
    Always returns sha256 of the archive file for provenance logging.
    """
    ap = Path(archive_path).resolve()
    op = Path(out_path).resolve()
    if ap == op:
        raise ValueError(
            f"O0 anti-self: archive path equals out path ({ap})"
        )
    if not ap.is_file():
        raise FileNotFoundError(f"O0 archive not found: {ap}")
    digest = sha256_file(ap)
    with open(ap) as f:
        data = json.load(f)
    prov = data.get("o0_provenance") or {}
    src_out = prov.get("source_out") or data.get("source_out")
    if src_out is not None and Path(str(src_out)).resolve() == op:
        raise ValueError(
            f"O0 anti-self: archive source_out equals current out ({op})"
        )
    src_job = prov.get("job_id") or data.get("job_id")
    if run_job_id and src_job and str(src_job) == str(run_job_id):
        raise ValueError(
            f"O0 anti-self: archive job_id={src_job} equals current run job"
        )
    # Sidecar provenance (optional): <archive>.o0_prov.json
    side = Path(str(ap) + ".o0_prov.json")
    side_meta = {}
    if side.is_file():
        side_meta = json.load(open(side))
        if side_meta.get("sha256") and side_meta["sha256"] != digest:
            raise ValueError(
                f"O0 archive sha mismatch vs sidecar {side}: "
                f"file={digest} sidecar={side_meta['sha256']}"
            )
        sj = side_meta.get("job_id")
        if run_job_id and sj and str(sj) == str(run_job_id):
            raise ValueError(
                f"O0 anti-self: sidecar job_id={sj} equals current run job"
            )
    return {
        "archive_path": str(ap),
        "sha256": digest,
        "job_id": src_job or side_meta.get("job_id"),
        "source_out": src_out or side_meta.get("source_out"),
        "sidecar": str(side) if side.is_file() else None,
    }


def load_o0_archive(path: str) -> Dict[str, List[int]]:
    """Load prompt_id -> emitted_tokens from an eval JSON (per_prompt_results)."""
    with open(path) as f:
        data = json.load(f)
    rows = data.get("per_prompt_results") or data.get("archive") or []
    out: Dict[str, List[int]] = {}
    for r in rows:
        pid = r.get("id") or r.get("prompt_id")
        toks = r.get("emitted_tokens")
        if pid is None or toks is None:
            raise ValueError(f"archive row missing id/emitted_tokens: keys={list(r)[:12]}")
        out[str(pid)] = list(toks)
    return out


@torch.no_grad()
def greedy_logits_after_prefix(
    base,
    processor,
    question: str,
    image_path: Path,
    prefix_tokens: List[int],
    device: str = "cuda:0",
):
    """Replay greedy prefix; return masked next-token logits or None on mismatch."""
    inputs = make_image_inputs(processor, question, image_path, device)
    out = base(**inputs, use_cache=True)
    past = out.past_key_values
    logits = mask_phantom_(out.logits[0, -1, :])
    for tok in prefix_tokens:
        if argmax_masked(logits) != tok:
            return None
        out = base(input_ids=torch.tensor([[tok]], device=device),
                   past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = mask_phantom_(out.logits[0, -1, :])
    return logits


def candidate_probe_from_logits(
    logits,
    spec_tok: Optional[int],
) -> Dict:
    """From masked logits: top1/top2, top2_gap, gap_spec, spec_rank."""
    top2 = torch.topk(logits, k=2)
    top1_id = int(top2.indices[0].item())
    top2_id = int(top2.indices[1].item())
    top1_logit = float(top2.values[0].item())
    top2_logit = float(top2.values[1].item())
    top2_gap = top1_logit - top2_logit
    gap_spec = None
    spec_rank = None
    if spec_tok is not None:
        st = int(spec_tok)
        spec_logit = float(logits[st].item())
        gap_spec = top1_logit - spec_logit
        # rank: 1 + number of tokens with strictly greater logit
        spec_rank = int((logits > logits[st]).sum().item()) + 1
    return {
        "greedy_top1": top1_id,
        "greedy_top2": top2_id,
        "top2_logit_gap": top2_gap,
        "gap_spec": gap_spec,
        "spec_rank": spec_rank,
        "spec_tok": None if spec_tok is None else int(spec_tok),
    }


@torch.no_grad()
def greedy_candidate_probe_at(
    base,
    processor,
    question: str,
    image_path: Path,
    prefix_tokens: List[int],
    spec_tok: Optional[int],
    device: str = "cuda:0",
) -> Optional[Dict]:
    """Candidate-specific probe after greedy[:pos].

    Returns dict with greedy_top1/top2, top2_logit_gap, gap_spec, spec_rank;
    None if replayed argmax diverges from the recorded prefix.
    """
    logits = greedy_logits_after_prefix(
        base, processor, question, image_path, prefix_tokens, device=device,
    )
    if logits is None:
        return None
    return candidate_probe_from_logits(logits, spec_tok)


@torch.no_grad()
def greedy_top2_gap_at(
    base,
    processor,
    question: str,
    image_path: Path,
    prefix_tokens: List[int],
    device: str = "cuda:0",
) -> Optional[float]:
    """Top-1 − top-2 logit gap for the next token after `prefix_tokens`.

    prefix_tokens = greedy[:pos] (tokens before the diverge index).
    Returns None if replayed argmax diverges from the recorded prefix.
    """
    logits = greedy_logits_after_prefix(
        base, processor, question, image_path, prefix_tokens, device=device,
    )
    if logits is None:
        return None
    top2 = torch.topk(logits, k=2).values
    return float(top2[0] - top2[1])


@torch.no_grad()
def greedy_dump_h_at(
    base,
    processor,
    question: str,
    image_path: Path,
    max_new: int,
    eos_id: int,
    target_pos: int,
    device: str = "cuda:0",
):
    """Greedy to ``target_pos``; capture last-layer hidden predicting that token.

    pos=0 → prefill last hidden; pos=k>0 → hidden after consuming emitted[k-1].
    Returns (h_cpu_fp16, emitted_prefix, mem_meta).
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    inputs = make_image_inputs(processor, question, image_path, device)
    thw = inputs.get("image_grid_thw")
    thw_list = thw.tolist() if thw is not None else None
    seq_len = int(inputs["input_ids"].shape[-1])
    n_vision = int(thw.prod().item() // 4) if thw is not None else None

    out = base(**inputs, use_cache=True, output_hidden_states=True)
    past = out.past_key_values
    h_last = out.hidden_states[-1][0, -1, :].detach().half().cpu()
    logits = mask_phantom_(out.logits[0, -1, :])
    del out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if target_pos == 0:
        peak = (torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available() else 0.0)
        del past, inputs
        return h_last, [], {
            "seq_len": seq_len, "image_grid_thw": thw_list,
            "n_vision_tokens_est": n_vision, "peak_alloc_gib": peak,
        }

    nxt = argmax_masked(logits)
    emitted = [nxt]
    while len(emitted) < max_new:
        need_h = (len(emitted) == target_pos)
        out = base(
            input_ids=torch.tensor([[nxt]], device=device),
            past_key_values=past, use_cache=True,
            output_hidden_states=need_h,
        )
        past = out.past_key_values
        if need_h:
            h_last = out.hidden_states[-1][0, -1, :].detach().half().cpu()
            del out
            break
        logits = mask_phantom_(out.logits[0, -1, :])
        del out
        nxt = argmax_masked(logits)
        if nxt == eos_id:
            emitted.append(nxt)
            break
        emitted.append(nxt)
        if len(emitted) % 32 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    peak = (torch.cuda.max_memory_allocated() / 1024**3
            if torch.cuda.is_available() else 0.0)
    del past, inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return h_last, emitted, {
        "seq_len": seq_len, "image_grid_thw": thw_list,
        "n_vision_tokens_est": n_vision, "peak_alloc_gib": peak,
    }


# ----------------------------------------------------------------------------
# Vanilla greedy (tok/s baseline; reused by tree eval §7 comparison)
# ----------------------------------------------------------------------------


@torch.no_grad()
def vanilla_greedy(
    base, processor, question: str, image_path: Path, max_new: int, eos_id: int,
    device: str = "cuda:0",
) -> List[int]:
    """Plain autoregressive greedy decode (no position_ids passed -> model
    auto-applies M-RoPE; this is the known-correct reference path)."""
    inputs = make_image_inputs(processor, question, image_path, device)
    out = base(**inputs, use_cache=True)
    past = out.past_key_values
    nxt = argmax_masked(out.logits[0, -1, :])
    emitted = [nxt]
    for _ in range(max_new - 1):
        if nxt == eos_id:
            break
        out = base(input_ids=torch.tensor([[nxt]], device=device),
                   past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = argmax_masked(out.logits[0, -1, :])
        emitted.append(nxt)
    return emitted
