# Dynamic Tree — Design Lock

> Status: **Round-5 packaging** (2026-08-08). σ **2.825**, R **0.433**,
> speed official **1.683/1.703 并列/未分** (primary n=204, job `175785218`).
> End-state naming: greedy numerical-safety verified / archive
> byte-reproducibility verified / release bundle reproducible.
> Equivalence: **tie-deterministic 键下逐位等价**. Segmented = **已废弃口径**.

---

## B1. Status quo (static path — cite-as-is)

Canonical builder: `decode/tree.py`. Folded eval (all C1/B2 champions) calls
`build_tree_folded`; non-folded baseline calls `build_tree`. Both share the
same prune tail.

### B1.1 Static expansion (`build_tree_folded`, lines 123–166)

```text
nodes = [forced root = known bonus]          # depth 1, cum=0
for k in 1..K-1:                             # heads 1..K-1
    (lp, idx) = topk_masked(logits[k], fanout[k])   # ONE shared top-f per level
    for each surviving parent at depth k:
        append fanout[k] children with cum = parent.cum + lp[b]
```

Key property of the **linked-head** product tree: every parent at depth `k`
gets the **same** child token set (head `k` emits one distribution from `h_t`,
not per-parent distributions). Width at depth `d` is therefore
`∏_{j=1}^{d-1} fanout[j]` (folded: depth-1 width = 1).

`build_tree` (non-folded, lines 48–96) is the same product construction with
depth-1 = `topk(logits[0], fanout[0])` instead of a forced root.

### B1.2 Prune tail (folded lines 168–178; non-folded 98–109)

Exact order, both builders:

1. **Global top-N by `cum_logprob`:**  
   `order = sorted(..., key=cum, reverse=True); keep = order[:max_nodes]`
2. **Root / depth-1 floor:**  
   - folded: always `keep.add(0)` (bonus root); if `depth1_floor`: keep **all
     depth-2** nodes (first speculative layer).  
   - non-folded: if `depth1_floor`: keep **all depth-1** nodes.
3. **Prefix-close (defensive):** walk each kept node’s `parent` chain upward
   and add missing ancestors.
4. **Re-flatten** depth-major: `(depth, parent, -cum)` with parent remap.

Docstring claim (non-folded L65–66): prefix-closed “by construction” because
`parent.cum >= child.cum` for log-probs ≤ 0, plus the floor only adds
parentless roots; the upward walk is defensive.

### B1.3 Call sites / reuse vs fork

| Piece | Reuse for dynamic? | Notes |
|-------|--------------------|-------|
| `TreeNode` dataclass | **reuse** | unchanged |
| `topk_masked` / `argmax_masked` | **reuse** | selection still uses masked vocab |
| `build_mask_and_positions` | **reuse** | mask/pos depend only on final node list |
| `accept` / KV reorg | **reuse** | operate on final nodes |
| `build_tree` / `build_tree_folded` | **do not modify** | static path stays bit-stable |
| Global top-N by cum (+ floor) | **semantic target** | achieved via best-first heap, not post-hoc full-tree prune |
| Shared per-level top-k | **reuse** | precompute once per depth (not inside parent loop) |

**Fork point:** replace “materialize full product tree → prune” with
**best-first heap expansion** that yields the same global top-N-by-cum set
without ever allocating the full product. Everything after a finalized
`List[TreeNode]` (mask, verify, accept, reorg) stays on the static path.

---

## B2. Proposed algorithm — best-first heap expand

### B2.1 Hard constraint

**Forbidden:** build the full product tree then prune. With large `cand_k` the
full tree is `Θ(cand_k^{D})` and explodes (sweep row #10 fullN=247 already at
modest fanouts; `cand_k=8` depth-5 is worse).  
**Required:** only expand nodes that enter the selected set; never materialize
pruned subtrees. Expansion order is **best-first by `cum_logprob`**, not
layer-synchronous greedy.

### B2.2 Parameters (independent)

| Param | Role |
|-------|------|
| `cand_k` | **per-depth** list of per-parent top-k widths from each level’s **shared** head logits (v1 required; a scalar is only syntactic sugar for a constant list) |
| `max_nodes` | global selected-set budget (same units as today’s `max_nodes`) |

Static `fanout[d]` is **not** used when the dynamic builder is selected, except
in the equivalence gates where we set `cand_k ≡ fanout` per depth (B3).

### B2.3 Selection criterion (best-first ≡ global top-N)

**Rejected (do not implement): layer-wise greedy.**  
A previous draft kept “top proposals at depth `d` among live parents, then
advance.” That contradicts “global top-N by cum” once the budget binds:

1. Weak shallow nodes are admitted whenever depth-local budget remains, and
   can **never** be displaced later by a stronger deep descendant of another
   branch.
2. Large `cand_k` lets shallow layers **consume the entire `max_nodes`**,
   leaving deeper layers empty — the opposite of “budget flows to the most
   valuable paths.”

Both pathologies diverge from static prune semantics and from the project
motive for a dynamic tree. **Recorded so this is not reintroduced.**

**Also rejected: floor via pre-admission into the budget.**  
Pre-admitting all depth-2 candidates *before* / *as part of* filling
`max_nodes` silently changes accounting. Concrete counter-example (folded,
`cand_k=(6,4,2,1)`, `max_nodes=24`, `depth1_floor` on): pre-admit yields
exactly **24** nodes and **never overshoots**, while static
`build_tree_folded` on the same config has **N̄ = 26.0 > 24**. The two
missing slots are deep nodes that static keeps after the floor union; the
dynamic tree would be short by 2 — a silent semantic drift that gate#1
(`max_nodes ≥ fullN`) cannot catch. Pre-admission is recorded alongside
layer-wise greedy as a **rejected** design.

**Adopted: best-first heap (EAGLE-2-style expand) + post-hoc floor union.**  
Maintain a max-heap of frontier candidates keyed by `cum_logprob` (tie-break
in B3). Because linked heads share one distribution per depth (B1.1), each
depth’s `(lp, idx) = topk_masked(logits[depth], cand_k[depth])` is
**precomputed once** — never inside the parent loop. Expanding a selected
parent only attaches those shared child tokens with
`child.cum = parent.cum + lp[b]`.

Pure best-first pops until `|selected| == max_nodes` (root always included).
Floor is applied **afterwards** (B2.4), not by consuming budget up front.

### B2.4 Root / floors / prefix-close

- **Bonus root forced** (folded): depth-1 known bonus, `cum=0`, always starts
  in the selected set (mirrors `keep.add(0)`).
- **`depth1_floor`: KEEP — as post-hoc union (not pre-admission).**  
  After pure best-first has filled `max_nodes`:  
  `selected ← selected ∪ { depth-2 candidates in level_cand[2] not yet selected }`.  
  These floor nodes **do not consume the `max_nodes` budget** — they are the
  sole source of overshoot, matching static `depth1_floor` (force-keep all
  depth-2 after top-N, allowing N̄ > `max_nodes`). No eviction of already
  selected deep nodes.
- **Prefix-close:** see B2.5 — best-first under monotone-nonincreasing cum is
  prefix-closed; floor only adds depth-2 children of the forced root, so the
  set remains prefix-closed. Keep a defensive assert in tests.

### B2.5 Pseudocode (folded) + equivalence argument

```text
# cand_k has length K-1 (speculative widths for depths 2..K).
# Index: cand_k[d-2] for tree depth d. CLI --fanout 1 w2..wK → pass fanout[1:].
for d in 2..K:   # head index = d-1
    level_cand[d] = topk_masked(logits[d-1], cand_k[d-2])  # slots b=0..width-1

root = Node(token=bonus, depth=1, parent=-1, logprob=0, cum=0,
            constr_ord=0)
selected = [root]
heap = MaxHeap()   # key: (-cum, depth, parent.rank, slot_b) — see B3
# node.rank = parent.rank * cand_k[depth-2] + slot_b  (root.rank = 0)

# Seed frontier from root's depth-2 candidates only (no floor pre-admit).
for b, (lp, tok) in enumerate(level_cand[2]):
    heap.push(candidate(parent=root, tok, lp, cum=0+lp, slot_b=b,
                        rank=root.rank*cand_k[0]+b, ...))

while |selected| < max_nodes and heap non-empty:
    u = heap.pop_max()
    selected.append(u)
    if |selected| < max_nodes and u.depth + 1 <= K:
        for b, (lp, tok) in enumerate(level_cand[u.depth+1]):
            heap.push(candidate(parent=u, tok, lp, cum=u.cum+lp, slot_b=b,
                                rank=u.rank*cand_k[u.depth-1]+b, ...))

# Post-hoc floor union (overshoot; does NOT count against max_nodes).
# Rejected alternative — pre-admit depth-2 into selected before/during the
# loop — deleted; it double-pushed root children (selected-seed + expand)
# and silently under-filled deep nodes (see B2.3).
if depth1_floor:
    for b, (lp, tok) in enumerate(level_cand[2]):
        if not already_in_selected(root → tok at depth-2):
            selected.append(child(root, tok, lp, cum=0+lp, slot_b=b, ...))

return reflatten(selected)   # presentation order (depth, rank) — see B3
```

**Why best-first selects exactly the global top-N by cum (and is prefix-closed):**

1. **Monotonicity:** every child log-prob ≤ 0 (log-softmax), so  
   `child.cum = parent.cum + lp ≤ parent.cum`. Cum is **monotone nonincreasing**
   along any root→leaf path.
2. **Best-first invariant:** the next node popped is the maximum-cum node not
   yet selected among all nodes that are children of already-selected nodes
   (the frontier). By monotonicity, any not-yet-generated descendant of a
   non-selected node has cum ≤ that non-selected node’s cum ≤ every
   already-popped node’s cum. Therefore no unseen node can outrank a popped
   node.
3. **Conclusion (pure top-N):** the first N nodes admitted by best-first
   (before floor) are exactly the N highest-cum nodes in the (implicit) full
   product tree — i.e. identical to static `sorted_by_cum[:max_nodes]` under
   the B3 selection tie-break.
4. **Prefix-closure:** a child is only generated from a selected parent, so
   every selected node’s ancestors are selected. Equivalently: if a node is in
   the global top-N by cum, monotonicity implies each ancestor has cum ≥ that
   node and hence is also in the top-N (or was force-kept as root). Static’s
   defensive upward walk is therefore a no-op on this set.
5. **Floor (post-hoc union):** `selected ∪ depth-2` matches static
   `keep = top_N ∪ {all depth-2}`. Descendants of a depth-2 node that lost the
   pure top-N cut have cum ≤ that depth-2’s cum; by monotonicity they cannot
   sit in the pure top-N either, so static does not keep them — and dynamic
   never generates them (their parent was only unioned after the heap phase).
   Overshoot cardinality matches static.

**Strengthened claim — equivalence at any `max_nodes`:**  
When `cand_k[d] = fanout[d]` per depth (same shared `topk_masked` lists), the
construction above is **node-identical** to static “full product tree + prune”
for **every** `max_nodes ≥ 1`, not only when `max_nodes ≥ fullN`:

- best-first’s first N = global top-N by cum (points 1–3);
- post-hoc floor union = static floor union (point 5);
- dropped depth-2 nodes’ descendants are excluded from top-N on both sides
  by cum monotonicity.

`max_nodes ≥ fullN` is then only a convenient special case (floor is a no-op
because every depth-2 node already sits in the top-N / full set).

---

## B3. Equivalence gate (first acceptance test — locked now)

**Setting:** folded; `cand_k` is a **per-depth** list equal to the static
fanout on speculative depths; `depth1_floor` on unless noted. Old-100
regression scale. Both rows below must pass; **either fail ⇒ gate fail**.
“σ within ε” must **not** replace bit-exact checks on either row.

| Gate | `cand_k` (speculative) | `max_nodes` | Display (rounded) | Bit-exact archive |
|------|------------------------|------------:|-------------------|-------------------|
| **#1** | `(3, 2, 1)` | 16 | σ **2.370**, N̄ **16.00** | `final_eval/c1_d3.json` — sha256 `b2cb6923…bcdd18` (job 173623490) |
| **#2** | `(6, 4, 2, 1)` | 24 | σ **2.619**, N̄ **26.0** | `final_eval/c1_wide.json` — sha256 `0313c9f7…4b0dee` (same lineage as static wide×24) |

**Bit-exact object (LOCKED 2026-07-20):** compare the live dynamic run’s
**per-prompt** `sigma` / `mean_width` (and thus aggregate σ / N̄) to the
**archived static artifact** above — not to the rounded display numbers.
Display values in tables/docs are **rounded for readability only**.

**Case (PBS false fail, 2026-07-20):** gate script required
`abs(N̄ − 26.0) < 0.0005`. Live dynamic after reflatten fix matched static
archive **bit-for-bit** at N̄ = **25.9961294159…**, but failed the rounded
threshold (Δ ≈ −0.00387). Lesson: gate numerics = archive equality; never
hard-code rounded display floats as tolerances.

**Flat-layout requirement (REVISED 2026-08-06, GPT-5.6 P1):** under the
**tie-deterministic** reflatten keys below, dynamic must emit a flat layout
**element-wise identical** to `build_tree_folded`: same ordered
`(flat_idx, token, depth, parent_flat_idx)` (hence same `tree_tokens` /
`position_ids` / attention `mask`), not merely the same node **multiset**.

**Claim scope:** **tie-deterministic 键下逐位等价** — **not** “universally
bit-exact” independent of sort inputs. Prior wording over-claimed.

**Case (own prior error, locked):** static prune converted the keep-set to a
`set`, then stable-sorted by `(depth, parent, -cum)`. Exact-`cum` ties then
followed **set iteration order**, which is not guaranteed across Python
versions and is not constructive rank. Dynamic already used constructive
`rank`. A targeted counterexample (spec widths `(2,2,2,4)`, `max_nodes=7`,
floor off) selected the same multiset but permuted tied depth-5 siblings
(`1,0` vs `0,1`). **Fix (static path, pre-approved):** append constructive
`rank` as the final static sort key
`(depth, parent, -cum, rank)`; dynamic reflatten stays `(depth, rank)`.
Synthetic coverage: `tests/test_dynamic_tree.py` (property ≥10k + review P1
case). Live gates must still pass archive bit-match after this change.

**Lesson (live gate#2 pre-fix):** multiset-equal but depth-5 flat permutation
≠ static → `mask` SHA differed → fp16 near-tie accept flip → σ **+0.003** /
N̄ **−0.01** vs static. Root cause then: reflatten keyed on
`(depth, selected-parent-index, -cum)`. Dynamic fix: reflatten `(depth, node.rank)`.
Static path later gained the rank final key (this revision). Post-fix live
gates matched `c1_d3` / `c1_wide` archives **per-prompt identical** (job 174154684);
re-verify after the static rank key (Tier-1 1b).

### Why this must be equivalent

When `cand_k = fanout` per depth (B2.5 strengthened claim):

1. Shared `topk_masked` lists equal the static product tree’s per-level sets.
2. Best-first admits exactly the global top-N by cum under the locked
   selection tie-break (below) — for **any** `max_nodes`.
3. Post-hoc floor union equals static `keep ∪ depth-2` (overshoot included).
4. Reflatten must reproduce static’s **presentation order** via constructive
   `(depth, rank)` so verify mask/pos (and thus accept) match byte-for-byte —
   multiset identity alone is **not** sufficient.

### Tie-breaking — selection consistency (locked)

Static top-N uses Python’s **stable** `sorted(..., key=cum, reverse=True)`, so
among equal `cum` the **construction order** decides who enters the top-N cut
— not token id values. Dynamic selection must match that cut, not merely sort
the same survivors differently after the fact.

**Constructive rank (full-product in-level lexicographic order — no full tree):**

```text
root.rank = 0
node.rank = parent.rank × cand_k[node.depth] + slot_b
```

where `slot_b ∈ {0 .. cand_k[node.depth]-1}` is the index into that depth’s
precomputed `topk_masked` list. This is the layer-wise dictionary order of the
implicit full product tree; it is computable incrementally when a candidate is
created. The heap’s `parent_constr_ord` field **is** `parent.rank` under this
definition.

**Forbidden:** using the dynamic `selected` list index (best-first **pop
order**) as `parent_constr_ord` / rank. Counter-example: parents P1, P2 with
`P2.cum > P1.cum` ⇒ P2 pops first and gets a smaller selected-index, but
static construction order still has P1 before P2. If children of P1 and P2
later tie on cum at the last budget slot, pop-order ranks reverse the static
stable cut → **multisets diverge**. Rank must be the constructive formula
above, never pop order.

**Heap comparison key (max-heap / min-tuple form), locked:**

```text
(-cum, depth, parent.rank, slot_b)
```

| Component | Role |
|-----------|------|
| `-cum` | primary: higher cum first |
| `depth` | then shallower first (matches how static construction tends to emit) |
| `parent.rank` | parent’s constructive rank (formula above) |
| `slot_b` | index into that depth’s `topk_masked` list (`b = 0 .. cand_k-1`) |

**Why `slot_b`, not `token id`:** static’s stable-sort tie order among equal-cum
siblings is the order those nodes were **appended during product expansion**,
which follows the shared top-k slot loop `for b in range(fanout[k])`. Token
numeric ids are unrelated to that order and must not appear in the selection
key. After the selected multiset is fixed, **reflatten** presentation order is
`(depth, node.rank)` on the dynamic path; static prune uses
`(depth, parent, -cum, rank)` with the **same** constructive `rank` as the
final key so exact-`cum` ties are deterministic. Do **not** rely on `set`
iteration order. Equivalence claim = **tie-deterministic 键下逐位等价**, not
universal bit-exactness independent of sort inputs.

### Edge cases

| Edge | Handling |
|------|----------|
| Top-k ties (equal logprob) | Same `topk_masked` as static; slot order is `b` |
| `max_nodes > fullN` | Keep all; floor union is a no-op |
| `max_nodes < fullN` + floor | Equivalence still required (gate#2); overshoot expected |
| Trailing zero fanouts / head truncation | Same active-head truncation as eval |
| Cum ties across branches | Heap key above — must match static stable top-N cut |

If any edge prevents bit-identity, **fail the gate** — do not weaken to
“σ within ε” (applies to gate#1 and gate#2 equally).

---

## B4. Code placement

- **`build_tree` / `build_tree_folded`:** Tier-1 added constructive `rank` as the
  final reflatten sort key (tie-determinism). No other semantic change.
- **New:** `build_tree_folded_dynamic(...)` (`cand_k` = K−1 speculative widths).
- **Switch:** `--tree-builder {static,dynamic}` (default `static`);
  `--e2e-wall` / `--mode greedy_e2e` for official speed protocol.
- **Acceptance triad:** gate#1 + gate#2 + `tests/test_dynamic_tree.py`
  (property ≥10k + review P1 case); Tier-1 re-verified job `175589163`.

---

## B5. Post-implementation experiment (CLOSED — job `174156696`)

1. **Equivalence gates** (old 100): dynamic ≡ archive `c1_d3` / `c1_wide`
   per-prompt (job `174154684`). Synthetic flat-layout tests **8/8**.
2. **Sweep** (300-prompt; C1 ckpt; `cand_k ∈ {4,6,8}` equal-width ×
   `max_nodes ∈ {16,24,32}`; depth1_floor on; speed denom = pooled greedy
   **29.918**):

### Dual champions (300-scale, dynamic builder + C1) — **v2 cites**

| Role | Config | Metric | vs static anchor (paired / band) |
|------|--------|--------|----------------------------------|
| **σ (LOCKED v2)** | `dyn_k8_n32` | **σ = 2.825**, N̄ = 36.49 | vs c1_6432: Δσ = **+0.101 ± 0.007** OOB; N̄ 36.49 vs 34.08 |
| **speed** | static_d3 / dyn_n24 | Tier C **1.683 / 1.703** (primary 204) | Δspd **+0.020±0.006**; 95% CI ∩ 1% band → **并列/未分** |

**v1 soft-cap lineage (do not new-cite):** σ 2.841 / e2e 1.732× under job `175598529`.
Legacy segmented = **已废弃口径**. Pre-hash Tier C 1.729/1.753 (`175738321`) =
historical descriptive only.

**Official operating points (v2):**
- **σ = 2.825** (`dyn_k8_n32`, 300, `v2_rebaseline/`)
- **speed = 并列/未分** — static_d3 **1.683×** / dyn_n24 **1.703×**
  (primary n=204, `tier_c_interleaved_speed_d/`, job `175785218`).

**Tier-3a difficulty-bias cross-check:** dyn n32 σ100/σ300 bias aligns with
static 6432 (v1 cross-check 2.774/2.841 vs 2.677/2.739) — bias reproduced on
dynamic tree; v2 levels shift ~−0.5% uniformly.

### Attribution (REVISED 2026-08-07 — GPT-5.6 P1)

Gain = **wider candidate envelope + global top-N selection**, made feasible by
**lazy exact top-N enumeration**. Static prune was already context-dependent
global top-N on the product tree; the heap does **not** introduce a new
allocation mechanism. At `cand_k == fanout` (any budget) the builders are
equivalent (tie-deterministic keys) — **Tier-3a confirmed:** static
materialize `[1,8,8,8,8]×32` ≡ `dyn_k8_n32` on old100, **100/100** bit-match
(σ=2.774, N̄=36.45; job `175598529`).

**Delete / avoid:** “path reallocation alone”, “budget flows to optimal paths”
as causal claims for the k8 vs `[.,6,4,3,2]` gap. That gap changes the
candidate envelope vs the static anchor.

**Architecture label:** **EAGLE-2-inspired lazy enumeration** specialized to
linked shared per-depth distributions — **not** literal EAGLE-2 (no per-node
draft distributions).

### Matched-width / matched-latency contrast (was “same-budget”)

CLI `max_nodes` is a **soft pre-floor** target; report post-floor N̄ always.

| Nominal | Dynamic | Static compare | N̄ note |
|---------|---------|----------------|--------|
| n16 | k8 → σ **2.714** (v1) | c1_d3 (N̄=16) | matched N̄=16 |
| n24 | k8 → σ **2.780** (v2; N̄=28.86) | c1_6432 (N̄=34.08) σ **2.724** (v2) | **not** matched-N̄ |
| n32 | k8 → σ **2.825** (v2; N̄=36.49) | c1_6432 (N̄=34.08) σ **2.724** (v2) | **not** matched-N̄; +0.101 paired |

### Expectation management — CORRECTED

Prior B5 claim: linked architecture ⇒ dynamic σ gains “mild”. **Measured
(174156696):** wider envelope (k8 vs static 6432 widths) + lazy top-N yielded
paired **+0.102** σ — **outside** the 0.5% band. Record as **expectation error
direction: underestimate**. Ceiling vs true per-node-draft EAGLE-2 remains a
separate architecture comparison.

### Scan boundary (CLOSED — optional k10/k12 done)

`cand_k` 4→6→8 is **monotonic** in σ at fixed `max_nodes` (no plateau).
Optional extension at n32 (job `174181383`):

| Config | σ300 | N̄ | Δσ vs prior k | Note |
|--------|------|-----|---------------|------|
| `dyn_k8_n32` | **2.825 (v2)** / 2.841 (v1) | **36.49** | — | σ champ (cite **v2**) |
| `dyn_k10_n32` | **2.865** (v1 scan) | **38.42** | +0.024 | N̄ also ↑ — **N̄ 混杂未排除** |
| `dyn_k12_n32` | **2.875** (v1 scan) | **40.40** | +0.010 | N̄ also ↑ — **N̄ 混杂未排除** |

**Marginal decay in σ; k12 = scan upper bound; k16 not scheduled.** Do not
claim pure-k gains without matched-N̄ / matched-latency controls.

### Scope notes (lossless / KV)

- Acceptance / “lossless” claims are **greedy-only** (no temperature>0
  rejection-sampling correction).
- `reorg_kv_gather` assumes single-device mutable cache layers (seq dim 2).
  **32B/72B / sharded multi-GPU** requires per-layer device index — open
  prerequisite before scaling.

3. **Anchors (v2):**  
   - **σ300 static:** 2.403 / 2.724 / 2.119 / 2.384 (`v2_rebaseline/bridge300_*`).  
   - **σ champ:** 2.825 (`dyn_k8_n32`).  
   - **speed:** Tier C hash-rerun **并列/未分** (1.683 / 1.703, primary 204).
     Segmented / pre-hash 1.729/1.753 = lineage only.  
   A-vs-B: **paired per-prompt diff ±SE only**.
4. Historical 100-scale lineage never mixed into 300 tables without a scale tag.

---

## B6. Effort / risk estimate

| Item | Estimate |
|------|----------|
| New builder function(s) | ~100–140 LOC in `decode/tree.py` (heap + floor post-hoc union) |
| Eval CLI + wiring | ~15–30 LOC in `scripts/eval_acceptance_tree.py` |
| Tests (synthetic equiv + smoke) | ~80–150 LOC |
| PBS / sweep scripts | ~1 short PBS, mirror fanout sweep |
| **Risk: budget accounting ≠ static** | **Downgraded (was High).** Best-first + monotone cum makes the selected set = global top-N by construction; no per-layer bookkeeping drift. Residual risk is **tie-break / floor overshoot wiring**, covered by B3. |
| **Risk: top-k tie order** | Medium — share `topk_masked` only |
| **Risk: truncation / skip_head0 interaction** | Low if builder sits behind same logits assembly |
| Non-goal this round | Learned / adaptive `cand_k`; per-parent distinct distributions (architecture still linked) |

---

## Open points for review (not blocking the design lock)

1. ~~Exact live-set accounting when `depth1_floor` overshoots~~ — **closed:**
   post-hoc union mirroring static overshoot (B2.4); pre-admission rejected.
2. ~~Scalar vs per-depth `cand_k` in v1~~ — **closed:** per-depth list is **v1
   required** (both equivalence gates use asymmetric fanouts). A scalar is
   only syntactic sugar for a constant per-depth list.
