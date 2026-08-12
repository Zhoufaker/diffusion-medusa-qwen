# OOD MM-Vet protocol (Part A)

Status: active. Gate anchors remain on **V100** (no A100) to preserve numeric lineage.

## Dataset

| Field | Value |
|-------|-------|
| Dataset | MM-Vet v1 **full 218** (no subsample) |
| Manifest | `/scratch/li96/mz9869/ood_eval/mmvet/manifest_mmvet_218.json` — fixed numeric order `v1_0`…`v1_217` |
| Images | `/scratch/li96/mz9869/ood_eval/mmvet/mm-vet/images/` |
| Prompt / decode | Same chat template, `max_new_tokens=150`, greedy / folded tree as LLaVA-Bench gates |

## O0 gate — dual verdict (LOCKED 2026-08 Tier D, runner v2_hardcap)

Implemented in `decode.common` / `scripts.eval_acceptance_tree` (`--check-greedy-bytes`).

**Official claim vs independent greedy (Round-7, gap-only):**
algorithmic greedy lossless; **every first mid-sequence divergence is a
candidate-specific near-tie** — in full precision
`0 ≤ logit[greedy_top1] − logit[spec_tok] ≤ 0.15`.
Do **not** claim “exact / byte-equivalent” vs independent greedy.
**Byte-exact** wording is reserved for **archive reproducibility** only.

**near_tie classifier (Round-7):** candidate-specific, **gap-only** —
`near_tie` iff mid-sequence diverge and `0 ≤ gap_spec ≤ 0.15`, where
`gap_spec = logit[greedy_top1] − logit[spec_tok]`. A missing gap is `hard`.
`spec_rank` and `spec_tok == greedy_top2` are recorded but **grant nothing**.

Two rules were tried and rejected before this one. The legacy top1−top2-only
heuristic never inspected the candidate (Round-5 P1). Round-6 then admitted
`spec_tok == greedy_top2` as a second, sufficient condition; that disjunct is a
specification error — rank 2 bounds nothing about separation, so it could pass
an arbitrarily distant candidate. Round-7 keeps the gap alone.

**Evidence (job `175813855`, re-scored offline in `round7_gap_only_reclass`):**
all **512/512** recorded divergences were re-probed at full precision
(`lm_head.float() @ h`); re-scoring the stored rows under the gap-only rule
leaves all 512 `near_tie` and changes **0** verdicts.
Tail context gated per row (replayed greedy prefix must match the recorded
`greedy_context.before`, a ≤5-token window): 512/512 verified, 0 `PREFIX_MISMATCH`.
`spec_rank` ∈ {1: 215, 2: 292, 3: 5}; `gap_spec` max **0.0325**, well inside the
0.15 band; the five rank-3 rows pass on their own gap, which is what shows the
gap-only rule is not rank ≤ 2 in disguise.
The 215 rank-1 rows are fp16→fp32 argmax flips where the recorded greedy token
is top-2. Historical item3 10/10 tree∈top-2 sampling remains as corroboration.
Packaged `*.o0_report.json` were labelled by the legacy rule and are
retroactively re-verified; the live classifier applies the gap-only rule.

| Verdict | Rule | Notes |
|---------|------|-------|
| **archive_gate_status** | `NOT_RUN` / `INCOMPLETE` / `PASS` / `FAIL` | no archive or 0 covered → **NOT_RUN** (never silent PASS); partial → **INCOMPLETE** (+ covered-subset flag, no promote); full coverage → PASS/FAIL |
| **greedy_byte_exact_pass** | `n_exact == n_prompts` | expected **FALSE** whenever near_tie>0 |
| **greedy_numerical_safety_pass** | `n_len_boundary==0 and n_hard==0` | near_tie does **not** fail |

| Exit | Condition |
|------|-----------|
| **2** | archive explicitly passed and gate = FAIL |
| **5** | archive explicitly passed and gate = NOT_RUN (incl. zero coverage) |
| **6** | archive explicitly passed and gate = INCOMPLETE |
| **3** | numerical_safety FAIL |
| **4** | archive FAIL + safety FAIL |
| **0** | byte_exact FALSE alone is OK; no `--o0-archive` → NOT_RUN is non-fatal |

Summary fields: `n_archive_covered`, `n_fingerprint`, `n_exact`,
`n_len_boundary`, `n_near_tie`, `n_hard`, plus the three verdicts above.
Raw: `*.o0_report.json`.

Archives (sidecar always `runner=v2_hardcap`):

| Flag | Contents |
|------|----------|
| `--o0-write-archive` | speculative archive (all prompts) when gate PASS or NOT_RUN |
| `--o0-write-greedy-archive` | byte-exact (`kind=match`) prompts only; require numerical_safety PASS |

**Anti-self (runtime):** unchanged (`validate_o0_archive_not_self`).

v1 soft-cap archives are **frozen lineage** (`runner=v1_softcap`); gates and new
archives anchor **v2 only**. Full-coverage repro: `tier_d_repro_fullcover/`.

## Image pixel cap (`max_pixels`) — V100 32GB

**Chosen value: `max_pixels = 501_760` (= 640 × 28 × 28).** Locked 2026-07-20 after
item5 mem forensics on V100 32GB:

| Probe | Result |
|-------|--------|
| v1_82 @ **501760** | peak_alloc **26.35** GiB, reserved 27.15, seq=650, `grid_thw=[[1,50,50]]`, n_vis=625 — fits with head |
| v1_82 @ **1003520** | **CUDA OOM** (~31.6 GiB allocated) — **rejected** |
| Formal smoke top-3 n_vis @ 501760 (c1_6432, job `174156697`) | peak_alloc **29.252** GiB, reserved **29.633** GiB (v1_73/40/16) |

**Smoke gate (calibrated):** `peak_alloc ≤ 30.0` GiB. Rationale: the three
smoke images **are** the full-218 n_vis top-3 (worst case); 29.252 / reserved
29.633 still leaves headroom to 32 GiB. Under this gate, `174156697` smoke =
**PASS** → full A authorized.

Prior target `1_003_520` remains documented as the OOM evidence, not the live
cap. Do **not** schedule A100 unless 501760 still cannot fit with head resident.

### Scope

Cap applies to **all A-batch rows** (G1–G3 and O1–O4), including greedy paths.

---

## A-batch results — **v2 anchors** (`v2_rebaseline/`, jobs `175680071`/`072`)

v1 job `174181382` remains frozen lineage. New cites / gates → v2 only.

### Gate rows (G1–G3 / old100, v2)

| Row | Config | σ (v2) | note |
|-----|--------|--------|------|
| G1 | c1_d3 | **2.359** | v1→v2 Δ −0.44% |
| G2 | c1_6432 | **2.660** | v1=2.677 (`fanout_sweep/c10_6432_32`); Δ **−0.64%** |
| G3 | b2_wide | **2.300** | v1→v2 Δ −0.40% |

### OOD rows (O1–O4, MM-Vet 218, v2)

| Row | Config | σ (v2) |
|-----|--------|--------|
| O1 | c1_d3 | **1.816** |
| O2 | c1_6432 | **1.992** |
| O3 | b2_d3 | **1.694** |
| O4 | b2_6432 | **1.858** |

(v1 spd columns from `174181382` remain lineage; do not mix into v2 σ tables.)

### Retention R (pre-registered bands) — **v2**

Paired O2−O4: \(\bar{d}=+0.134\pm0.012\) (n=218).

| Denominator | Source | R ± SE | Band |
|-------------|--------|--------|------|
| **0.310** | 100-scale historical C1−B2 (locked primary) | **0.433 ± 0.039** | mid [0.3, 0.6]; no straddle |
| **0.340** | σ300 v2 `c1_6432 − b2_6432` = 2.724−2.384 | **≈0.395** | still mid |

v1 primary R was **0.435**; move ≪ SE, same band.

### Discount echo (discussion material)

| Transition | Retention | Note |
|------------|-----------|------|
| held-out → live (in-domain) | ~**0.44** | prior bridge / live cash-out |
| in-domain → OOD (MM-Vet) | **0.433** (v2 primary R) | paired wide-tree |

**Repeatable pattern (draft):** “cross-distribution ~halving” of the C1−B2 gap — held-out→live and in-domain→OOD land at nearly the same retention. Use as discussion motif; not a causal claim.

### OOD speed flip (standalone; v1 e2e lineage)

In-domain, wider trees trade some speed for σ. On MM-Vet (v1 speeds):

| Row | Tree | σ (v2) | spd (v1) | vs O1 (narrow C1) |
|-----|------|--------|----------|-------------------|
| O1 | c1_d3 | 1.816 | **1.164×** | baseline |
| O2 | c1_6432 | 1.992 | **1.044×** | σ↑ but speed↓ |
| O4 | b2_6432 | 1.858 | **0.996×** | ≈ break-even vs greedy |

**Read:** after σ drops on OOD, wide-tree verify cost **dominates** — speed flips against the σ champion. **σ champion has no OOD deploy case** on this set. Speed–σ tradeoff is **domain-sensitive** (do not export in-domain speed ranking to OOD).

### Offline supplements (no GPU; `ood_offline_analyses.json`)

**(a) Capability buckets** (O2−O4 paired Δσ)

**Multi-label:** MM-Vet `mmvet_capability` is multi-label (169/218 prompts have ≥2 tags). Each prompt’s Δσ is assigned to **every** tagged bucket. Bucket *n*s overlap (Σ*n* = 511 ≠ 218); **do not sum across buckets**.

**Significance lock:** ±1SE interval excludes 0 →「增益显著」; includes 0 →「方向不定(小样本)」— latter **must not** be cited as “XX 能力偏弱”.

| Tag | n (of 218; overlapping) | mean ± SE | ±1SE interval | Verdict |
|-----|-------------------------|-----------|---------------|---------|
| rec | 150 | +0.165 ± 0.015 | [+0.150, +0.179] | 增益显著 |
| ocr | 96 | +0.082 ± 0.015 | [+0.067, +0.097] | 增益显著 |
| know | 84 | +0.144 ± 0.012 | [+0.132, +0.156] | 增益显著 |
| gen | 80 | +0.144 ± 0.011 | [+0.134, +0.155] | 增益显著 |
| spat | 75 | +0.068 ± 0.019 | [+0.050, +0.087] | 增益显著 |
| math | 26 | +0.051 ± 0.024 | [+0.027, +0.075] | 增益显著 |

**Buckets whose ±1SE contains 0:** **none.** Effect-size rank (mean only): rec/know/gen &gt; ocr/spat/math — all six still「增益显著」under the lock; math is smallest mean but **not** “方向不定”.

**(b) Per-prompt σ percentiles** (dynamic-tree OOD hypothesis prep — distribution only):

| Row | mean | p25 | p50 | p75 | frac&lt;1.5 | frac&lt;2.0 |
|-----|------|-----|-----|-----|----------|----------|
| O1 | 1.817 | 1.562 | 1.705 | 1.973 | 13.3% | 76.2% |
| O2 | 1.994 | 1.666 | 1.820 | 2.221 | 7.3% | 63.3% |
| O3 | 1.696 | 1.464 | 1.597 | 1.843 | 30.7% | 82.1% |
| O4 | 1.859 | 1.566 | 1.724 | 2.000 | 12.8% | 72.0% |

OOD mass sits well below in-domain σ300 (~2.4–2.7); most prompts &lt; 2.0. Supports *inspecting* whether dynamic trees would auto-shrink — **no OOD dynamic job scheduled from this table alone**.

**(c)** Dual-denom R: see table above. Artifact: `ood_mmvet_218/ood_offline_analyses.json`.
