# Speculative Decoding for Vision-Language Models

Research code for accelerating autoregressive decoding in vision-language models via
speculative drafting, targeting **Qwen2.5-VL-7B-Instruct**.

The project has gone through two architectural lines. The current line is a
**DFlash-style block diffusion drafter** with acceptance-rate calibration as the
central contribution. An earlier line — **Linked Medusa heads with dynamic tree
drafting** — is complete and archived; its results motivated the pivot.

> **Status:** active research, unpublished. Results below are internal measurements,
> not peer-reviewed claims.

---

## Motivation

Speculative decoding trades a cheap drafter's parallel guesses against one verification
pass of the target model. Throughput gain is governed by two quantities:

- **σ (sigma)** — mean accepted tokens per verification step
- **wall-clock speedup** — what σ actually buys after drafter overhead

The gap between the two is where most of the engineering lives. A drafter that raises σ
but costs more to run can be net-negative, and in the VLM setting the drafter must also
handle long multimodal contexts without the visual prefix dominating its cost.

---

## Current line — block diffusion drafter

A block diffusion drafter proposes a block of tokens in parallel rather than
autoregressively, then relies on the target model to verify. The research question is
whether **calibrating the drafter's acceptance behaviour** — rather than simply
maximising its raw accuracy — yields a better speed/quality operating point.

### Results

| Configuration | σ / speedup | Notes |
|---|---|---|
| AR drafter baseline | 1.852× | reference |
| Block diffusion drafter | **1.867×** | bootstrap 95% CI on paired Δ: [+0.0003, +0.029] |

The confidence interval excludes zero, but only just. This is a real but small effect at
the current training scale — reported as-is rather than overstated.

**Drafter training (W2):** best checkpoint at epoch 3, validation CE **3.2943**.

### Long-form data pipeline

Drafter quality on long generations is data-bound. Current training mix:

| Source | Samples |
|---|---|
| DOCCI (train) | 9,647 |
| DetailCaps | 4,868 |
| **Current total** | **14,515** |
| Target | ~35–37K |

Priority additions: Stanford Paragraph, Localized Narratives (Open Images subset).
The **DOCCI test split (5,000 samples) is withheld** as a future out-of-distribution
evaluation set and is never touched during training or model selection.

---

## Prior line — Linked Medusa + dynamic tree drafting

Archived at tag `linked-medusa-final`. Summarised here because the findings shaped the
current direction.

### Progression

| Phase | Change | Outcome |
|---|---|---|
| v1 | Baseline linked heads | top-1 92.2 / 77.7 / 64.4 (heads 0/1/2); σ = 1.950 |
| A | Self-distillation on general-domain data | resolved train/eval domain mismatch |
| B1 | Extension to 5 heads | exposed inference inefficiencies worth **+14.5pp** speed |
| B2 | On-policy fine-tuning | — |
| C1 | EAGLE-conditioning (zero-init `bonus_proj`) | head_1 accept **0.814**, σ **2.619**, speed **1.689×** |
| Dynamic tree | Best-first heap + post-hoc floor union | σ **2.841** (paired Δ +0.102 ± 0.007) |

### What the numbers actually showed

- **Chain-gain is real.** Conditioning head_1 also improved downstream heads that were
  left unmodified — the heads are not independent.
- **σ saturates.** A fanout re-sweep put the ceiling near **2.68** for static trees,
  which is what motivated dynamic tree construction.
- **σ gains stopped converting to wall-clock gains.** Dynamic tree reached σ = 2.841 but
  measured 1.703× vs. 1.683× on speed — the preregistered verdict was
  **tied / undetermined**. This decoupling is the main reason for the pivot.
- **OOD acceptance is much weaker.** On MM-Vet (218 prompts), R = **0.435 ± 0.039**,
  landing in the middle preregistered band.
- **On-policy vs. off-policy** showed a 38pp gap at position layers, but only ~1.6%
  impact on total acceptance rate.

---

## Repository layout

```
decode/        Speculative decoding runtime — tree construction, verification loop
train/         Drafter training (trainer, entry points)
scripts/       Evaluation, data generation, and diagnostic tooling
tests/         Unit tests for drafter and decode-path correctness
docs/          Architecture specs and design documents
reports/       Experiment reports and data inventories
```

Key entry points:

| File | Purpose |
|---|---|
| `train/train_drafter.py` | Drafter training entry point |
| `scripts/eval_acceptance.py` | Acceptance-rate evaluation |
| `scripts/eval_acceptance_tree.py` | Tree-drafting acceptance evaluation |
| `scripts/tier_c_interleaved_speed.py` | Wall-clock speed measurement |
| `scripts/gen_longform_rollouts.py` | Long-form rollout generation |
| `decode/tree.py` | Draft tree construction |

---

## Environment

Experiments run on **NCI Gadi** (A100, `dgxa100` queue, project `li96`).

```bash
module load python3/3.11.0 cuda/12.3.2
source ~/medusa-env/bin/activate
```

Job submission requires the storage directive:

```bash
qsub -l storage=gdata/li96+scratch/li96 ...
```

Compute nodes have no external network — set `HF_HUB_OFFLINE=1` and pre-stage all model
weights to scratch before submitting.

---

## Methodological notes

A few practices this project holds to, since they caught real errors:

- **Preregistered verdicts.** Speed comparisons declare their decision rule before the
  run. This is why the dynamic tree result is reported as "tied" rather than as a win.
- **Byte-exact regression gates.** The original 100-prompt evaluation set is retained
  unchanged as a regression check when the evaluation set expands.
- **Paired bootstrap CIs** on all speedup deltas, not point estimates alone.
- **Mandatory code review before compute.** An off-by-one in the C1 conditioning index
  was caught this way before any GPU time was spent.
- **Timing hygiene.** An early contamination incident — diagnostic probes left inside the
  greedy timing window, inflating speedup to ~2.4× — is why timing windows are now
  explicitly bounded.

---

## Reproducibility

Checkpoints, datasets, and raw logs are **not** in this repository (see `.gitignore`).
They live on Gadi scratch/gdata under project `li96`.

---

## Author

**Zhou Mo (Chris)** — Master of Computer Science (Data Science & AI),
University of Sydney. Supervised by Prof. Chang Xu.
