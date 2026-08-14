"""ctx_dataset.py — W2 drafter training dataset over ctx_cache_35k shards.

Implements reports/w2_train_design.md:
  §1a mask token (=151662, runtime zero-occurrence guard)
  §1b anchor sampling (rollout span only) + multi-block packing layout
  §1c attention mask, dual path: FlexAttention mask_mod factory + 4D additive
  §1d ctx alignment (TARGET convention): block at anchor p sees ctx [0, p)
  §1f off-policy monitoring metadata pass-through

Packing layout per sequence (design §1b):
  keys    = [ctx_0..ctx_{T-1} | blk_1 .. blk_K | noise pad]
  queries = [                   blk_1 .. blk_K | noise pad]
  blk_k   = [ids[p_k], mask x (B-1)], positions p_k .. p_k+B-1 (flat absolute)
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open

MASK_TOKEN_ID = 151662  # <|fim_pad|> — design §1a
IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Anchor sampling — design §1b
# ---------------------------------------------------------------------------

def sample_anchors(P: int, T: int, alpha: float, max_anchors: int,
                   rng: random.Random) -> list[int]:
    """Anchor positions p in [P, T-2] (need >=1 supervisable slot), uniform
    without replacement, K = min(max_anchors, ceil(alpha*L), domain size)."""
    L = T - P
    lo, hi = P, T - 2
    domain = hi - lo + 1
    if domain <= 0:
        return []
    k = min(max_anchors, math.ceil(alpha * L), domain)
    return sorted(rng.sample(range(lo, hi + 1), k))


# ---------------------------------------------------------------------------
# Block packing — design §1b/§1d
# ---------------------------------------------------------------------------

@dataclass
class PackedSeq:
    ids: torch.Tensor            # (T,) full sequence ids (ctx side reference)
    noise_ids: torch.Tensor      # (K*B,) anchor + mask tokens
    noise_pos: torch.Tensor      # (K*B,) absolute positions p_k..p_k+B-1
    labels: torch.Tensor         # (K*B,) ids[p+m] or IGNORE_INDEX
    block_of: torch.Tensor       # (K*B,) block index of each noise slot
    anchor_pos: torch.Tensor     # (K,) anchor absolute positions
    T: int                       # ctx length
    meta: dict = field(default_factory=dict)  # off-policy monitoring (§1f)


def pack_blocks(ids: torch.Tensor, anchors: list[int], block_size: int,
                mask_token_id: int = MASK_TOKEN_ID,
                meta: dict | None = None) -> PackedSeq:
    """§1d: slot m of block at anchor p sits at seq position p+m; slot 0 is the
    anchor (no label); slot m>=1 label = ids[p+m] if p+m < T else IGNORE."""
    T = ids.shape[0]
    K, B = len(anchors), block_size
    noise_ids = torch.full((K * B,), mask_token_id, dtype=torch.long)
    noise_pos = torch.empty(K * B, dtype=torch.long)
    labels = torch.full((K * B,), IGNORE_INDEX, dtype=torch.long)
    block_of = torch.empty(K * B, dtype=torch.long)
    for k, p in enumerate(anchors):
        s = k * B
        noise_ids[s] = ids[p]
        noise_pos[s: s + B] = torch.arange(p, p + B)
        block_of[s: s + B] = k
        m_hi = min(B - 1, T - 1 - p)          # last in-range slot
        if m_hi >= 1:
            labels[s + 1: s + 1 + m_hi] = ids[p + 1: p + 1 + m_hi]
    return PackedSeq(ids=ids, noise_ids=noise_ids, noise_pos=noise_pos,
                     labels=labels, block_of=block_of,
                     anchor_pos=torch.tensor(anchors, dtype=torch.long),
                     T=T, meta=meta or {})


# ---------------------------------------------------------------------------
# Attention mask, dual path — design §1c
# ---------------------------------------------------------------------------

def allowed_bool_mask(anchor_pos: torch.Tensor, block_of: torch.Tensor,
                      ctx_len_padded: int, n_noise: int,
                      valid_noise: torch.Tensor | None = None) -> torch.Tensor:
    """(Q, KV) bool, Q = n_noise, KV = ctx_len_padded + n_noise.
    Rule 1 in-block bidirectional; rule 2 cross-block isolated;
    rule 3 ctx visible iff ctx position c < anchor of the query's block
    (strict: the anchor's own ctx feature is NOT visible, §1d).
    Padded noise queries (valid_noise False) attend key 0 only (softmax guard).
    """
    Q = n_noise
    anchor_of_q = anchor_pos[block_of]                       # (Q,)
    cols = torch.arange(ctx_len_padded + n_noise)
    ctx_part = cols[None, :ctx_len_padded] < anchor_of_q[:, None]  # (Q, C)
    same_block = block_of[None, :] == block_of[:, None]            # (Q, Q)
    allow = torch.cat([ctx_part, same_block], dim=1)
    if valid_noise is not None:
        pad_q = ~valid_noise
        allow[pad_q] = False
        allow[pad_q, 0] = True
    return allow


def additive_4d(allow: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """(Q,KV) bool -> (1,1,Q,KV) additive mask for sdpa/eager."""
    m = torch.zeros(allow.shape, dtype=dtype)
    return m.masked_fill(~allow, torch.finfo(dtype).min)[None, None]


def flex_mask_mod_factory(anchor_of_q: torch.Tensor, block_of: torch.Tensor,
                          ctx_len: int):
    """FlexAttention mask_mod closure — W3 speed-optimization CANDIDATE only
    (design §1c revised: D1 training main path is the 4D additive sdpa mask).
    Same boolean semantics as allowed_bool_mask; unit-tested for elementwise
    equivalence. Kept for the W3 ablation; not wired into the D1 loop."""
    def mask_mod(b, h, q_idx, kv_idx):
        is_ctx = kv_idx < ctx_len
        ctx_ok = kv_idx < anchor_of_q[q_idx]
        blk_ok = block_of[kv_idx - ctx_len] == block_of[q_idx]
        return torch.where(is_ctx, ctx_ok, blk_ok)
    return mask_mod


# ---------------------------------------------------------------------------
# Loss weights — design §1e (DFlash Eq. 4)
# ---------------------------------------------------------------------------

def block_weights(block_size: int, gamma: float) -> torch.Tensor:
    """w[0]=0 (anchor slot, no loss); w[m]=exp(-(m-1)/gamma) for m>=1.
    gamma pairs from the paper: block 16 -> 7, 10 -> 5, 8 -> 4."""
    w = torch.zeros(block_size)
    m = torch.arange(1, block_size, dtype=torch.float32)
    w[1:] = torch.exp(-(m - 1) / gamma)
    return w


# ---------------------------------------------------------------------------
# Shard-backed dataset — design §2
# ---------------------------------------------------------------------------

class CtxShardDataset(torch.utils.data.Dataset):
    """Lazy loader over ctx_cache safetensors shards.

    Keeps at most `cache_shards` open shard handles (LRU). Returns raw per-
    sample tensors; anchor sampling/packing happens in the collate path so
    each epoch resamples anchors (paper §4.2 random anchors).
    """

    def __init__(self, cache_dir: str, manifest_name: str = "ctx_manifest.json",
                 indices: list[int] | None = None, cache_shards: int = 2,
                 mask_token_id: int = MASK_TOKEN_ID):
        self.dir = Path(cache_dir)
        man = json.load(open(self.dir / manifest_name))
        self.samples = {int(k): v for k, v in man["samples"].items()
                        if not v.get("failed")}
        self.shard_size = man["shard_size"]
        cand = sorted(indices if indices is not None else self.samples.keys())
        # F1: L<2 has no legal anchor (need >=1 supervisable slot) — filter,
        # count goes into the training manifest.
        self.filtered_short = [i for i in cand if self.samples[i]["L"] < 2]
        self.indices = [i for i in cand if self.samples[i]["L"] >= 2]
        self.mask_token_id = mask_token_id
        self._handles: dict[str, object] = {}
        self._lru: list[str] = []
        self._cache_shards = cache_shards

    def __len__(self) -> int:
        return len(self.indices)

    def _shard(self, idx: int):
        name = f"shard_{idx // self.shard_size:05d}.safetensors"
        if name not in self._handles:
            if len(self._lru) >= self._cache_shards:
                old = self._lru.pop(0)
                del self._handles[old]
            self._handles[name] = safe_open(self.dir / name, framework="pt")
            self._lru.append(name)
        return self._handles[name]

    def __getitem__(self, i: int):
        idx = self.indices[i]
        f = self._shard(idx)
        ids = f.get_tensor(f"{idx}.ids")
        ctx = f.get_tensor(f"{idx}.ctx")
        spans = f.get_tensor(f"{idx}.spans")
        assert not (ids == self.mask_token_id).any(), \
            f"mask token {self.mask_token_id} occurs in sample {idx} (§1a guard)"
        r = self.samples[idx]
        mm_pos = [e["pos"] for e in r.get("mismatch_head16", [])]
        assert all(p < r["L"] for p in mm_pos), \
            f"sample {idx}: mismatch_pos out of rollout range (manifest 谱系损坏?)"
        meta = {"idx": idx, "n_match": r["n_match"], "n_pos": r["n_pos"],
                "mismatch_pos": mm_pos, "L": r["L"]}
        return {"ids": ids, "ctx": ctx, "spans": spans, "meta": meta}


def collate_packed(items: list[dict], block_size: int, alpha: float,
                   max_anchors: int, seed: int, epoch: int,
                   mask_token_id: int = MASK_TOKEN_ID) -> dict:
    """Sample anchors + pack + pad a batch (design §1b padding rules).

    F3: anchor rng derives from (seed, epoch, sample_idx) — same triple is
    bit-reproducible, epochs resample (paper random anchors), val pins
    epoch=0 for a fixed evaluation set.

    Returns ctx (Bs,5,Tmax,H) bf16, noise_ids (Bs,Nmax), noise_pos, labels,
    allow (Bs,Nmax,Tmax+Nmax) bool, valid_noise, position_ids (Bs,Tmax+Nmax),
    plus flex factory args and monitoring meta.
    """
    packed, ctxs = [], []
    for it in items:
        spans = it["spans"].tolist()        # [vs, ve, 0, P, P, T]
        P, T = spans[4], spans[5]           # rollout span [P, T)
        rng = random.Random(f"{seed}:{epoch}:{it['meta']['idx']}")
        anchors = sample_anchors(P, T, alpha, max_anchors, rng)
        assert anchors, (f"sample {it['meta']['idx']}: K=0 anchors "
                         f"(L<2 should have been filtered, F1)")
        packed.append(pack_blocks(it["ids"], anchors, block_size,
                                  mask_token_id, it["meta"]))
        ctxs.append(it["ctx"])
    Bs = len(packed)
    Tmax = max(p.T for p in packed)
    Nmax = max(p.noise_ids.shape[0] for p in packed)
    H = ctxs[0].shape[-1]
    ctx_b = torch.zeros(Bs, ctxs[0].shape[0], Tmax, H, dtype=ctxs[0].dtype)
    noise_ids = torch.full((Bs, Nmax), mask_token_id, dtype=torch.long)
    noise_pos = torch.zeros(Bs, Nmax, dtype=torch.long)
    labels = torch.full((Bs, Nmax), IGNORE_INDEX, dtype=torch.long)
    valid = torch.zeros(Bs, Nmax, dtype=torch.bool)
    allow = torch.zeros(Bs, Nmax, Tmax + Nmax, dtype=torch.bool)
    pos_ids = torch.zeros(Bs, Tmax + Nmax, dtype=torch.long)
    for b, p in enumerate(packed):
        n = p.noise_ids.shape[0]
        ctx_b[b, :, :p.T] = ctxs[b]
        noise_ids[b, :n] = p.noise_ids
        noise_pos[b, :n] = p.noise_pos
        labels[b, :n] = p.labels
        valid[b, :n] = True
        # block_of padded with distinct sentinel so pad never joins a block
        blk = torch.full((Nmax,), -1, dtype=torch.long)
        blk[:n] = p.block_of
        a = allowed_bool_mask(p.anchor_pos, blk.clamp(min=0),
                              Tmax, Nmax, valid_noise=valid[b])
        # clamp(min=0) maps pad rows to block 0; overwrite their rows:
        pad = ~valid[b]
        a[pad] = False
        a[pad, 0] = True
        # also forbid real queries from seeing pad noise keys
        a[:, Tmax:][:, pad] = False
        a[pad, 0] = True
        allow[b] = a
        pos_ids[b, :Tmax] = torch.arange(Tmax)
        pos_ids[b, Tmax:Tmax + n] = p.noise_pos
    metas = [p.meta for p in packed]
    return {"ctx": ctx_b, "noise_ids": noise_ids, "noise_pos": noise_pos,
            "labels": labels, "valid_noise": valid, "allow": allow,
            "position_ids": pos_ids, "ctx_len": Tmax, "metas": metas}
