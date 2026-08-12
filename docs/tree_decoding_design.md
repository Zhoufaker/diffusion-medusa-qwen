# Tree Speculative Decoding — Design Doc (P0)

> Status: **APPROVED — implementing** (review resolved 2026-06). Builds on the
> working linear-chain σ evaluator `scripts/eval_acceptance.py` (baseline
> σ_v1 = 1.95, 100 LLaVA prompts, fp16). Goal: tree-structured draft/verify to
> raise accepted tokens per round, then confirm the *net* wall-time speedup is
> positive.
>
> **Review resolutions (locked):**
> - Q1 prune: global top-N by joint `cum_logprob` **+ depth-1 floor** (force-keep
>   all `fanout[0]` depth-1 nodes; they're parentless so prefix-closure holds).
>   Per-round logging of (a) per-depth kept-width and (b) per-depth accept counts.
> - Q2 reorg: **INVERTED by evidence (see §0.2).** The regression proved
>   `reorg_kv_gather` is **byte-identical to the validated chain** (5/5, σ exact),
>   while `reorg_kv_safe` drifts at fp16 near-ties (its recompute uses the SDPA
>   `is_causal` kernel vs the verify's masked kernel). **Recommend gather as the
>   default reorg** (faster + bit-exact to chain); safe kept as fallback. (The
>   original "validate gather==safe bit-identical" gate cannot pass: safe is the
>   one that drifts, not gather.)
> - Q3 vocab: single source of truth `decode.common.EFFECTIVE_VOCAB=151936` +
>   `mask_phantom_/argmax_masked/topk_masked`, masked **before** top-k/argmax.
> - Q4/Q5 layout: new `decode/` package. `common.py` = loaders/vocab/greedy/
>   M-RoPE helper (shared by chain, tree, P1). `tree.py` = construction+mask+reorg
>   (reusable) + `accept` (inference-only greedy; P1 won't inherit it).
>   `scripts/eval_acceptance.py` is **untouched** (no σ=1.95 regression risk).
> - **BLOCKING M-RoPE bug (FIXED + verified):** any forward passing explicit
>   `position_ids` must offset by `rope_delta`. See §0.1.
>
> **Wall-time result (§0.3, 100 prompts) — tree ≈ break-even:**
> greedy 31.64 > tree 30.72 (0.97×) > chain 27.16 (0.86×) tok/s. The tree raises
> σ 1.95→2.41 (+23%) and is **+13% over the chain** in wall-time, nearly closing
> the gap to greedy (3% short).
>
> **NET POSITIVE achieved (§0.4, fold-bonus):** removing the per-round bonus
> forward (≈0.95 base-unit/round, 3a) makes the **folded** decoder **1.122×
> greedy** — speculative decoding is wall-time net-faster than greedy for the
> first time. Cost is σ (2.30→1.49, structural: head_0 spent on the known bonus).
> Two operating points kept: baseline tree (σ 2.405, 0.971×) and folded (σ 1.485,
> 1.122×). Folded correctness ≡ baseline tree vs greedy (same fp16 near-tie
> divergences, no folded-unique error).

---

## 0. Gating result (step (a), DONE)

`scripts/probe_4d_mask.py` on V100, transformers 5.3.0:

| Check | Result |
|-------|--------|
| explicit 4D causal mask vs default — **argmax** | 16/16 identical ✓ |
| custom mask actually blocks (last-pos Δ=17.5, pos-0 Δ=0.0) | ✓ |
| tree verify (past_kv + custom position_ids + sibling isolation) vs linear ref | argmax match ✓ |

**Decisions locked in:**
1. transformers 5.3.0 passes a 4D `attention_mask` straight through
   (`masking_utils._preprocess_mask_arguments` L787 early-exit;
   `Qwen2_5_VLModel.forward` L1391 passthrough). **No attention patch, no
   downgrade.**
2. Base **must** be loaded with `attn_implementation="sdpa"`. FA2 rejects
   arbitrary masks; eager works but is slower.
3. Mask convention: `(B, 1, q_len, kv_len)` additive **float16**, `0.0` = attend,
   `torch.finfo(fp16).min` = block.
4. Custom 2D `position_ids` are honored (outer forward only auto-computes RoPE
   positions when `position_ids is None`). Tree positions = depth-based.
5. `DynamicCache` layout in 5.3.0: `cache.layers[i].keys` / `.values`, shape
   `(B, n_kv_heads, seq, head_dim)`, seq on **dim=2**. `.crop(len)` truncates;
   we can `index_select(2, idx)` for gather-based reorg.

---

## 0.1 BLOCKING M-RoPE position fix (DONE + verified on image)

Root cause: the chain verify/bonus forwards never pass `position_ids`, so
Qwen2.5-VL auto-computes M-RoPE positions as `arange(P, P+n) + rope_delta`
(`modeling_qwen2_5_vl.py` L1309–1312 / L1667–1668). The tree is the project's
first code to pass **explicit** `position_ids`; using raw `P + (depth-1)` drops
`rope_delta`, which is **nonzero (negative) on image prompts** (image tokens
compress positions). The original probe used a text-only prefix where
`rope_delta == 0`, so the bug was masked (false green).

**Fix:** `cont_base = past_len + base.model.rope_deltas` (scalar, `(1,1)` tensor,
constant per prompt). Tree node position = `cont_base + (depth - 1)`. Centralized
in `decode.common.continuation_base`.

**Verified** (`scripts/probe_4d_mask.py` CHECK 3, image `000000294160.jpg`):

| quantity | value |
|----------|-------|
| prefill `P` | 425 |
| `base.model.rope_deltas` | `[[-368]]` |
| `cont_base = P + rope_delta` | **57** |
| linear-ref node1 argmax | 9217 |
| tree FIXED (`cont_base`) node1 | 9217 — **match ✓** |
| tree BUGGY (`P`) node1 | 151645 (EOS) — **mismatch** (bug is real) ✓ |

The buggy variant degenerating to EOS confirms the offset bug would silently
lower image σ; the fix restores byte-exact agreement with the chain path.

---

## 0.2 Degenerate-tree regression + reorg finding (DONE)

Gate: `tree(fanout=[1,1,1])` must reproduce the chain's emitted tokens
byte-for-byte (5 image prompts). Result:

| reorg | byte-identical | mean σ_tree vs σ_chain |
|-------|----------------|------------------------|
| `gather` | **5/5 ✓** | 1.865 == 1.865 (exact) |
| `safe`   | 4/5 (1 flips @ tok 34) | 1.855 vs 1.865 |

The `safe` 1/5 divergence was root-caused with a round-by-round lockstep
(`scripts/diag_lockstep.py`): caches stay aligned (same P, same candidates, same
verify argmax) for 14 rounds, then at round 15 a **carried near-tie argmax**
(`base_pred_t`: 7952 vs 304, top-2 margin ≈ 0.28) flips. Cause: `reorg_kv_safe`
recomputes the accepted path **without** an attention_mask → SDPA `is_causal`
fused kernel, whereas the verify (and chain) use the **masked** kernel; the two
differ by ~0.1 fp16 logit (quantified in §0 CHECK 1) and accumulate over rounds.

**Conclusion:** `gather` (reuse verify-computed KV via `index_select`) is the
*more faithful* reorg — it keeps the exact masked-kernel KV the chain keeps, so
it is byte-identical. This confirms the tree construction / mask / position_ids
(with M-RoPE offset) / accept logic are exactly chain-equivalent. **v1 ships
`reorg=gather` as default.**

---

## 0.3 Wall-time speedup (DONE, 100 prompts) — NET ≈ break-even

**100 image prompts, 150 new tokens, V100, fp16, same GPU**, `fanout=[4,3,2]`,
`max_nodes=16`, `reorg=gather`:

| decoder | σ | tok/s | speedup vs greedy |
|---------|----|-------|-------------------|
| vanilla greedy | — | 31.64 | 1.000 (ref) |
| linear chain | 1.953 | 27.16 | 0.859 |
| **tree** | **2.405** | 30.72 | **0.971** |

`tree/chain = 1.131` (tree is **13% faster** than the chain in wall-time). The
chain's measured σ=1.953 reproduces the v1 σ=1.95 (validates the shared-loader
reproduction). Tree accept rates rose to `[0.871, 0.384, 0.117]` (vs chain v1
`[0.647, 0.218, 0.063]`) — the depth-1 floor + fanout-4 directly widened the
layer where 35% of rounds were dying. σ quartiles: p25=2.19, p50=2.32, p75=2.56.
N̄=17.24, per-depth kept widths `[4.0, 5.59, 7.65]`.

**Tree nearly closes the gap to greedy (0.97×, 3% short of break-even); chain is
0.86×.** Per-round fixed cost = `head_fwd(2.1B)` + `verify_fwd(N tokens, 7B)` +
`bonus_fwd(1 token, 7B)`. The 2.1B head (≈30% of the 7B base) run every round +
the separate bonus forward are the overhead; the tree's higher σ amortizes the
head cost over more accepted tokens, so it recovers most of what the chain loses.
A smaller `max_nodes` (cheaper verify) or a lighter head should push net > 1.0.

**Implications / levers (need supervisor direction):**
1. The draft head is the dominant cost. A much lighter head (fewer blocks /
   shared vocab projection) is likely required for net positive.
2. The per-round **bonus forward** is a structural overhead (~1 base-unit/round);
   folding it into the next round's verify **was done (§0.4) and works** — it
   yields the first net-positive (1.122× greedy). The head-needs-current-hidden
   concern is resolved by drafting from the last-accepted node's hidden and
   making the known bonus the next tree's root (cost: head_0 layer).
3. Larger σ (better heads via P1 on-policy) raises the amortization ceiling but
   P1 does not reduce per-round head/bonus cost — it won't fix wall-time alone.

---

## 0.4 Fold-bonus spike (eliminate the per-round bonus forward)

Decision (supervisor): do **fold bonus** (pure inference-side eng, no retrain, no
arch change). Do **not** shrink `max_nodes`. Light head is parked for v2.

### 3a — Per-round cost breakdown (measured, 10 prompts, V100, gather, fanout=[4,3,2], N̄=17.25)

CUDA-synced timers, `--profile`. `base-unit` = single-token greedy forward = 31.93 ms.

| component | ms/round | % | base-units |
|-----------|----------|-----|-----------|
| head_fwd (2.1B) | 8.04 | 11.2% | 0.25 |
| **verify_fwd (N=17, 7B)** | 32.97 | 45.8% | **1.03** |
| **bonus_fwd (1 tok, 7B)** | 30.27 | 42.0% | **0.95** |
| reorg (gather) | 0.60 | 0.8% | — |
| other | 0.13 | 0.2% | — |

Two confirmations:
- **`bonus_fwd ≈ 0.95 base-units` and is the single largest removable item** (42%
  of the round). Removing it cuts per-round wall-time from ~72 ms to ~42 ms (~1.7×
  fewer ms/round) before any σ change.
- **`verify_fwd ≈ 1.03 base-units even at N=17`** — verify is memory-bound (one
  load of the 7B weights), so the width is nearly free. This is exactly why
  shrinking `max_nodes` is the wrong lever: it would cut σ for ~zero wall-time gain.

### 3b — Tradeoff of folding the bonus forward

The bonus forward's only outputs are, for the next round:
- (a) `h_t` — the hidden at the bonus position, the draft head's input;
- (b) the bonus token's **KV** in the cache;
- (c) `base_pred_root` — base's argmax after the bonus (the root the accept walk
  confirms against).

**Standard elimination (implemented):** draft the next round from the
**last-accepted node's hidden** (run verify with `output_hidden_states=True`); the
known bonus becomes the **forced depth-1 root** of the next tree. Then:
- (a) replaced by the last-accepted node's hidden from verify;
- (b) the bonus's KV is produced by the next verify forward itself (root token);
- (c) `base_pred_root` for the root = the bonus we already know (base's argmax at
  the anchor from the previous verify) → root is always accepted "for free".

**Cost — one fewer speculative draft layer.** The head drafts from the anchor's
hidden, so `head_0` is spent re-predicting the *known* bonus; only `head_1..head_{K-1}`
add new speculation. Effective draft depth K→K-1. Folded round can emit at most K
(root + K-1 spec) vs baseline K+1 (K spec + bonus) → σ drops slightly. Since the
deepest layer accepts only 11.7%, the σ loss is expected small while we remove a
full ~0.95-base-unit forward every round.

Linked-head caveat: `head_1`'s distribution is internally conditioned on `head_0`'s
predicted token, not on the (base-chosen) bonus. When `head_0` mispredicts the
bonus (~13% at depth-1) the deeper drafts are mildly off-policy, but base verify is
still the arbiter → **correctness is unaffected** (folded greedy ≡ vanilla greedy
modulo fp16 near-ties); only draft *quality* on those rounds degrades.

### 3c — Prototype + decision gate

New code (does **not** touch the proven `run_one_prompt_tree` loop):
- `decode/tree.py::build_tree_folded(all_logits, root_token, fanout, max_nodes)` —
  forced known root at depth-1, speculative layers from `head_1..` (unit-tested).
- `scripts/eval_acceptance_tree.py::run_one_prompt_tree_folded` — no bonus forward;
  carries `(h_anchor, known_next)`; verify with hidden states; reorg=gather.
- `--mode foldbonus_ab` — same prompts/GPU/process A/B: baseline gather vs folded
  vs vanilla greedy.

**Adoption gate:** `tok/s(folded) > tok/s(baseline)` on the same batch/card; report
σ change (expected slightly lower); correctness gate = folded greedy matches
vanilla-greedy argmax (common prefix; rare fp16 near-tie divergence allowed).
Headline numbers fixed to one card (V100); A100 σ work re-measured on A100.

*(P1 note: `continuation_base` uses `rope_deltas[0]` assuming batch=1; P1 batch
decode must go per-sample.)*

### 3c — Results (20 prompts, V100, fanout=[4,3,2], max_nodes=16, 150 tok)

| decoder | σ | tok/s | speedup vs greedy |
|---------|----|-------|-------------------|
| vanilla greedy | — | 31.66 | 1.000 (ref) |
| baseline tree (gather) | 2.296 | 29.18 | 0.921 |
| **folded** | **1.485** | **35.52** | **1.122** |

`fold/baseline = 1.217`. **Folding the per-round bonus forward is the first
configuration where speculative decoding is NET POSITIVE on wall-time (1.122×
greedy)**, vs baseline tree at 0.921×. The win comes entirely from removing the
~0.95-base-unit bonus forward (3a): per-round wall-time ~72 ms → ~42 ms.

**Cost is σ (2.30 → 1.49), as predicted — and it's structural, not a bug:**
folding spends the *strongest* head (head_0, depth-1 accept 0.871) on re-predicting
the already-known bonus, so speculation falls to the weaker head_1/head_2. The σ
arithmetic checks out: `E[accept_len] = 1 (root) + head_1 + head_2 ≈ 1 + 0.38 + 0.12
≈ 1.5`.

**Correctness — disambiguated by the baseline-tree control (NOT an absolute margin
threshold).** The kernel fp16 diff is only ~0.1 and cannot flip a margin-0.3..0.5
decision, so `margin<0.5` would mask a real bug. The real discriminator: folded's
divergence-vs-greedy profile must ≈ baseline-tree's. On 10 prompts × 150 tok:

| decoder | match | diverge | divergence margins |
|---------|-------|---------|--------------------|
| baseline tree | 7 | 3 | 0.016, 0.031 (near-tie) |
| folded | 7 | 3 | 0.016, 0.031 (near-tie) |

Folded diverges at **exactly the same points/margins** as baseline (e.g. both
@113 margin 0.031, both @42 margin 0.016) — **zero folded-unique divergences**.
This is the masked-kernel (tree verify) vs is_causal-kernel (greedy) fp16 flip,
shared by baseline; folded adds no error of its own. Hardened comparison-based
gate (`--mode greedy_agreement`, `KERNEL_BAND=0.15`) + a plumbing health check
(`head_0(h_anchor) top1 == bonus` rate should ≈0.871; folded head_1/head_2 accept
vs baseline `[_,0.384,0.117]`) re-running on 20 prompts to finalize adoption.

**Decision:** adopt folded as the **speed-optimized** path (net > 1.0) AND keep
baseline tree as the **high-σ** tradeoff point. Two operating points:
- baseline tree: σ=2.405, **0.971×** (max accepted-tokens/round).
- folded: σ=1.485, **1.122×** (max wall-time throughput).

> **Conclusion for supervisor:** removing the per-round independent bonus forward
> makes speculative decoding **net-faster than greedy in wall-time for the first
> time (1.122×)**, at the cost of σ. σ is the means, wall-time is the engineering
> objective — and the objective is now met. head_0's structural loss is not
> recoverable by widening; deeper speculation needs 4–5 heads (retrain, to discuss
> alongside the supervisor's 5-head direction). P1 on-policy lifts head_1/head_2
> live accept (folded's workhorses) → indirectly helps folded σ, but cannot refill
> the head_0 layer.

*(Queued only AFTER adoption is confirmed — do not run yet: σ-recovery sweep of
head_1/head_2 fanout (e.g. `[_,5,4]`, `[_,6,5]`) — verify is memory-bound so
widening is ~free; check how much σ returns at ~0 wall-time cost.)*

---

## 1. Why a tree helps (and the key subtlety)

`LinkedMedusaHeads.forward(h_t)` returns a list of `K=3` logit vectors
**from a single h_t**. head_k's input is `h_t + h'_{k-1}` (the previous head's
pre-lm_head hidden), which is **deterministic given h_t** — it does NOT depend
on *which token* was sampled at the previous level. So each round produces three
**fixed** per-level distributions:

```
head_0(h_t) -> logits_0  -> top-f0 candidate tokens  (default f0 = 4)
head_1(h_t) -> logits_1  -> top-f1 candidate tokens  (default f1 = 3)
head_2(h_t) -> logits_2  -> top-f2 candidate tokens  (default f2 = 2)
```

Linear chain (v1) drafted only top-1 of each level → 1 path of length 3.
A round was accepted only while the base's greedy argmax happened to equal our
single candidate at each level. Measured per-position accept = [0.65, 0.22, 0.06].

The tree drafts **top-f_k** candidates per level and verifies the cartesian
product of paths in ONE base forward. base still has exactly one argmax per
context (greedy), but now it only needs to match *one of f_k* candidates at each
level → strictly higher per-level accept probability, hence longer accepted
paths and higher σ.

Full expansion node count (depth d nodes = ∏_{i<d} f_i):

```
depth 1: f0            = 4
depth 2: f0*f1         = 12
depth 3: f0*f1*f2      = 24
                  total = 40 nodes
```

These are config-driven (`fanout: [4, 3, 2]`).

---

## 2. Module A — tree build + pruning

### 2.1 Node representation

Flatten the tree into an ordered list of `N` nodes (after pruning). Each node:

```python
@dataclass
class TreeNode:
    token: int           # candidate token id
    depth: int           # 1, 2, or 3
    parent: int          # flatten index of parent node, or -1 if depth==1 (root=prefix)
    logprob: float       # log p_k(token) from that level's head softmax
    cum_logprob: float   # sum of logprobs along path from root to this node (joint path logprob)
    flat_idx: int        # its own index in the flattened list
```

### 2.2 Build → prune

```python
def build_tree(all_logits, fanout, max_nodes):
    # all_logits: list of K tensors (1,1,V) from heads(h_t)
    # 1. per-level top-f_k tokens + log-softmax probs
    level_tok, level_lp = [], []
    for k, logits in enumerate(all_logits):
        lp = log_softmax(mask_phantom(logits[0,0]), dim=-1)   # mask_phantom: -inf on padded vocab rows
        p, idx = lp.topk(fanout[k])
        level_tok.append(idx.tolist()); level_lp.append(p.tolist())

    # 2. full expansion as paths (cartesian product of per-level choices)
    #    a depth-d path = (i0, i1, ..., i_{d-1}) with i_j in range(fanout[j])
    nodes = []
    # depth 1
    for a in range(fanout[0]):
        nodes.append(node(token=level_tok[0][a], depth=1, parent=-1,
                          logprob=level_lp[0][a], cum=level_lp[0][a]))
    # depth 2 (child of each depth-1 node)
    for a in range(fanout[0]):
        for b in range(fanout[1]):
            nodes.append(node(token=level_tok[1][b], depth=2, parent=idx_of(a),
                              logprob=level_lp[1][b], cum=nodes[idx_of(a)].cum + level_lp[1][b]))
    # depth 3 (child of each depth-2 node) ... analogous
    ...
    # 3. prune: keep top-N by cum_logprob (joint path probability)
    #    KEY INVARIANT: a parent's cum_logprob >= its child's (logprob <= 0),
    #    so the top-N-by-cum set is automatically PREFIX-CLOSED (connected):
    #    if a child is kept, its higher-prob ancestor is also kept.
    keep = sorted(nodes, key=lambda n: n.cum_logprob, reverse=True)[:max_nodes]
    # 4. re-flatten in a stable order (BFS by depth) and remap parent pointers
    return reindex(keep)
```

`max_nodes` default = 16 (config `tree.max_nodes`). Prune-by-joint-prob with the
prefix-closed property is the simple dynamic pruning from Medusa/EAGLE-2 that the
brief asks for; no learned tree.

**Flatten order convention (used everywhere below): sort kept nodes by
`(depth, parent_flat_idx, -cum_logprob)`, assign `flat_idx = 0..N-1`.** Depth-1
nodes come first, then depth-2, etc. This makes the attention mask lower-block
structured and is the order the base forward will see.

---

## 3. Module B — tree attention mask + position_ids

Let `P = past_kv.get_seq_length()` (prefix = original prompt + all previously
accepted tokens). The verify forward feeds `N` tree tokens; the attention mask
shape is `(1, 1, N, P + N)`.

```python
def build_mask_and_positions(nodes, P, dtype, device):
    N = len(nodes)
    mn = torch.finfo(dtype).min
    mask = torch.full((1, 1, N, P + N), mn, dtype=dtype, device=device)
    mask[0, 0, :, :P] = 0.0                      # every node sees the whole prefix
    pos = torch.empty(N, dtype=torch.long, device=device)
    for n in nodes:
        i = n.flat_idx
        mask[0, 0, i, P + i] = 0.0                # self
        # walk ancestors, unblock each
        a = n.parent
        while a != -1:
            mask[0, 0, i, P + a] = 0.0
            a = nodes[a].parent
        pos[i] = P + (n.depth - 1)                # depth-1 -> P, depth-2 -> P+1, ...
    return mask, pos.unsqueeze(0)                 # pos: (1, N)
```

- Each node attends to: full prefix + its ancestor chain + itself. Siblings and
  unrelated branches are blocked (probe check 2/3 confirmed this works).
- `position_ids`: all nodes at the same tree depth share the same RoPE position.
  This is what lets the base treat each path as if it were a contiguous
  continuation. Probe check 3 validated tree-node logits == linear-forward logits
  under this exact construction.
- Verify forward:
  ```python
  v_out = base(input_ids=tree_tokens[None,:], attention_mask=mask,
               past_key_values=past_kv, position_ids=pos,
               use_cache=True, output_hidden_states=False)
  # v_out.logits: (1, N, V); v_out.logits[0, i] = base's next-token dist AFTER node i's path
  ```

---

## 4. Acceptance walk (greedy)

Same greedy semantics as the chain verifier, generalized to a tree. We carry
`base_pred_root` from the previous round's bonus forward (base's argmax for the
first draft position) exactly like the chain code's `base_pred_t`.

```python
def accept(nodes, v_logits, base_pred_root):
    accepted = []                      # list of flat_idx along accepted path
    cur_children = [n for n in nodes if n.parent == -1]   # depth-1 nodes
    expected = base_pred_root          # base's argmax for the next position
    while True:
        match = next((c for c in cur_children if c.token == expected), None)
        if match is None:
            break                      # base's argmax not among our candidates at this level
        accepted.append(match.flat_idx)
        expected = argmax_masked(v_logits[match.flat_idx])   # base's argmax AFTER this node
        cur_children = [n for n in nodes if n.parent == match.flat_idx]
        if not cur_children:
            break                      # reached a leaf of the (pruned) tree
    accept_len = len(accepted)
    bonus = expected                   # base's free next token at the last accepted (or root)
    return accepted, accept_len, bonus
```

- Because verify is greedy/argmax, at most one child matches per level → the walk
  is a single path, no backtracking.
- Emitted this round = `[nodes[i].token for i in accepted] + [bonus]`
  → `accept_len + 1` tokens, identical accounting to the chain (σ comparable).
- EOS check on emitted tokens, same as chain.

---

## 5. Module C — KV reorganization (most bug-prone; detailed)

### 5.1 The problem

The verify forward appended **all N** tree nodes' K/V to the cache at positions
`[P, P+N)`, in **flatten order**. But:
- only the `accept_len` accepted-path nodes are real continuation tokens;
- their flatten indices are **non-contiguous** (e.g. accepted = flat idx `[2, 7]`);
- the rest (rejected branches) must be discarded.

We need the cache to end up as a clean sequential `[0, P + accept_len)`:
prefix + accepted path, ready for the bonus forward to append at `P+accept_len`.

### 5.2 Why no RoPE re-rotation is needed (critical correctness point)

Keys are stored **post-RoPE**, rotated by each node's `position_id`. We set the
accepted depth-d node's `position_id = P + (d-1)`. Along an accepted path the
depths are exactly `1, 2, ..., accept_len`, so their position_ids are
`P, P+1, ..., P+accept_len-1` — **exactly the sequential positions they will
occupy after reorg.** Therefore the cached (already-rotated) keys are valid
in-place; reorg is a pure **gather**, not a recompute.

### 5.3 Gather implementation (recommended, after validation)

```python
def reorg_kv(past_kv, P, accepted_flat_idx):
    # accepted_flat_idx: LongTensor of accepted nodes' flatten indices, IN PATH
    #   (depth) ORDER, e.g. tensor([2, 7]) for a depth-2 accepted path.
    keep = torch.cat([
        torch.arange(P, device=dev),          # prefix, unchanged
        P + accepted_flat_idx,                 # accepted nodes, in depth order
    ])                                         # -> length P + accept_len
    for layer in past_kv.layers:
        layer.keys   = layer.keys.index_select(2, keep).contiguous()
        layer.values = layer.values.index_select(2, keep).contiguous()
    # DynamicLayer.get_seq_length() reads keys.shape[-2], now P+accept_len. OK.
```

Worked example: `P=16`, pruned tree, accepted path = root → node@flat2(depth1)
→ node@flat7(depth2). `accept_len=2`. `keep = [0..15, 16+2, 16+7] = [0..15,18,23]`.
After `index_select`, cache length = 18 = `P+accept_len`. Positions 16,17 now
hold the (already correctly-rotated) K/V of the two accepted tokens.

### 5.4 Safe fallback (recommended for the FIRST implementation)

Because §5.3 directly mutates cache internals, validate it against a
re-forward reference (just like the probe's linear-vs-tree check):

```python
def reorg_kv_safe(base, past_kv, P, accepted_tokens, positions):
    past_kv.crop(P)                            # drop ALL tree-node K/V
    if accepted_tokens:                        # re-run accepted path linearly
        ids = torch.tensor([accepted_tokens], device=dev)
        pos = torch.arange(P, P+len(accepted_tokens), device=dev)[None]
        base(input_ids=ids, past_key_values=past_kv, position_ids=pos, use_cache=True)
    # cache now P+accept_len, identical to gather but via recompute
```

Cost: one extra forward of `accept_len (<=3)` tokens per round — negligible.
**Plan:** ship `reorg_kv_safe` in v1 for correctness; add `reorg_kv` (gather)
behind a flag and assert the two produce bit-identical next-round h_t / logits on
a 5-prompt run before switching. This de-risks the most bug-prone part.

### 5.5 Bonus forward + next-round state (unchanged from chain)

```python
b_out = base(input_ids=[[bonus]], past_key_values=past_kv,
             position_ids=[[P+accept_len]], use_cache=True, output_hidden_states=True)
past_kv = b_out.past_key_values                # now P+accept_len+1
h_t          = b_out.hidden_states[-1][0, -1, :]   # draft source for NEXT round
base_pred_root = argmax_masked(b_out.logits[0, -1, :])
```

Note: we draft from the **bonus** position's hidden, same invariant as the chain
verifier. The tree's per-node hidden states are NOT used for drafting.

---

## 6. Per-round structure (summary)

```
state in: past_kv (len P), h_t, base_pred_root
  1. all_logits = heads(h_t)                         # cheap, K small heads
  2. nodes = build_tree(all_logits, fanout, max_nodes)
  3. mask, pos = build_mask_and_positions(nodes, P, ...)
  4. v_out = base(tree_tokens, mask, past_kv, pos)    # 1 forward, N nodes
  5. accepted, accept_len, bonus = accept(nodes, v_out.logits[0], base_pred_root)
  6. reorg_kv(past_kv, P, accepted)                   # safe (recompute) in v1
  7. b_out = base([bonus], past_kv, pos=P+accept_len, hidden_states=True)
  8. h_t, base_pred_root <- b_out ; emit accepted+bonus ; EOS check
state out: past_kv (len P+accept_len+1), h_t, base_pred_root
```

Per round cost vs chain:
- chain: verify K=3 tokens + bonus 1 token  (+ heads)
- tree:  verify N≤16 tokens + bonus 1 token (+ heads) [+ safe-reorg recompute ≤3]

---

## 7. Wall-time speedup metric (must report, not just σ)

σ alone is misleading because tree rounds verify N tokens, not 3. We report:

1. **σ_tree** = total emitted / rounds (directly comparable to σ_chain = 1.95).
2. **mean verify width** `N̄` = average tree size per round (overhead proxy; brief
   asks to log per-round verify sequence length).
3. **Equivalent-forward count per round**: define a round's verify cost in units
   of single-token forwards. We measure it empirically rather than assume
   linearity:
   - `t_round` = wall time per decode round (tree verify + bonus + reorg).
   - throughput `tok/s = total_emitted / total_decode_wall_time`.
4. **Net speedup** = `tok/s(tree) / tok/s(plain_greedy)`. This is the headline
   number. It can be < 1 even if σ_tree > σ_chain, if the wider verify forward
   costs more wall time than the extra accepted tokens save. We will also report
   `tok/s(chain_v1)` for the 3-way comparison.
5. Plain-greedy baseline `tok/s` measured on the same 100 prompts / same GPU
   (reuse `vanilla_greedy` already in `eval_acceptance.py`).

Output JSON additions: `sigma_tree`, `mean_tree_width`, `tok_per_s_tree`,
`tok_per_s_greedy`, `net_speedup`, plus per-round `verify_width` list.

---

## 8. Config additions (`config/`), proposed

```yaml
tree:
  enabled: true
  fanout: [4, 3, 2]      # per-head top-k; len must == num_heads
  max_nodes: 16          # prune target (joint-prob top-N)
  reorg: safe            # 'safe' (recompute) | 'gather' (index_select); validate gather vs safe
```

---

## 9. Open questions for review

1. **Pruning policy**: joint-prob top-N is the simplest. Alternative: per-depth
   budget (e.g. keep ≤4 at depth1, ≤8 at depth2). Stick with global top-N for v1?
2. **reorg strategy for v1**: ship `safe` (recompute, ~0 risk) and add `gather`
   as a validated optimization — agree?
3. **Phantom-vocab masking**: reuse `EFFECTIVE_VOCAB=151936` mask from
   `eval_acceptance.py` before softmax/top-k. (Yes unless told otherwise.)
4. **Code organization**: new `scripts/eval_acceptance_tree.py` (share loaders +
   vanilla_greedy with the chain script via a small `scripts/_spec_common.py`),
   vs a `--tree` flag inside the existing script. Proposing a separate file +
   shared util module to keep the validated chain path untouched.
5. **P1 reuse**: the build_tree / mask / accept / reorg functions are written so
   the on-policy trainer can call the same verify primitives. Confirm we want
   them factored into an importable module (e.g. `decode/tree.py`) rather than
   script-local.
```
