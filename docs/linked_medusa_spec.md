# Linked Medusa Head — Implementation Spec

> **Audience**: Coding agent (Cursor).
> **Goal**: Reimplement supervisor's Linked Medusa Head baseline for Qwen2.5-VL-7B-Instruct, training on a precomputed hidden-state cache. This document is a complete implementation spec — all design decisions below are either confirmed or have justified defaults.

---

## 1. Project context (read first)

This is a Medusa-style speculative decoding draft head for a Vision-Language Model (VLM). The novelty vs. vanilla Medusa is **sequentially-dependent heads** that pass continuous hidden states between each other. This is **NOT** Hydra — Hydra passes token embeddings; this design passes hidden states (more information, end-to-end differentiable).

**Training setup**: cache-only. The base model (Qwen2.5-VL-7B-Instruct) has already been run on training data, and its last-layer hidden states + ground-truth tokens are stored on disk. We never run the base model during head training. Each step is just `head.forward + head.backward` on cached tensors → ~50-100x faster than traditional Medusa training.

**Hardware**: NCI Gadi HPC, V100 32GB GPU, fp16 (V100 does NOT support bf16).

### 1.1 Current phase: cache not yet on server

The 77 GB cache is still being transferred to `/scratch/li96/mz9869/cached_data/qwen25vl_long/` and is **NOT available yet**. The current goal is to get the entire codebase structurally correct using synthetic data, so that when the cache arrives we can immediately start training.

**Implication for implementation**:

- Implement a `SyntheticVLMDataset` class alongside `CachedVLMDataset`, with the same interface and tensor schema.
- The training entrypoint (`scripts/train.py`) MUST support both via a config flag (e.g. `dataset.kind: 'synthetic' | 'cached'`).
- All structural validation (§7.1 + §11 except item 3) can and should be performed on synthetic data **before** the real cache arrives.
- Item 3 in §7.1 ("head_0 CE loss < 1.0") is the only check that requires real data — defer it.

```python
class SyntheticVLMDataset(Dataset):
    """
    Mock dataset matching CachedVLMDataset's interface and tensor schema.
    Use to validate code structure before real cache arrives.
    
    Hidden states are random Gaussian. Tokens are uniform random over vocab.
    Sequence lengths are sampled uniformly in seq_len_range to exercise padding.
    """
    def __init__(self, num_samples=200, seq_len_range=(50, 256),
                 hidden_dim=3584, vocab_size=152064, seed=42):
        ...
    
    def __getitem__(self, idx) -> dict:
        # Returns same dict schema as CachedVLMDataset:
        # {'hidden': Tensor(L, 3584) float16, 'tokens': Tensor(L,) int64}
        ...
```

**Workflow**:
1. Implement everything with `dataset.kind: 'synthetic'`.
2. Run §11 validation checklist on synthetic data (all items except #3 should pass).
3. Once cache arrives, switch to `dataset.kind: 'cached'`, re-run §11 item #3 on real data.
4. Begin actual training.

---

## 2. Confirmed design decisions (locked — do not change)

| Decision | Value | Rationale |
|----------|-------|-----------|
| Inter-head signal | Continuous hidden state `h_prev'` (NOT token embedding) | Confirmed by supervisor; ablation showed this beats token-embedding |
| Combination operation | Elementwise add: `head_k input = h_t + h_prev'` | Per supervisor handoff doc |
| Number of heads | 3 | Match vanilla Medusa baseline |
| Hidden dim | 3584 | Qwen2.5-VL-7B-Instruct |
| Vocab size | **152064** (lm_head physical output dim, padded for hardware alignment; effective tokenizer vocab is 151936) | Qwen2.5 series pads lm_head to 64/128-divisible dimension; the extra 128 rows are never emitted by tokenizer but must match for `lm_head` weight copy |
| Per-head lm_head | Independent (each head has its own `Linear(3584, 152064)`) | Match Medusa convention |
| lm_head init | Copy weights from base model's lm_head | Established Medusa trick — gives near-zero head_0 loss at init |
| Cache structure | `{'hidden': (L, 3584), 'tokens': (L,)}` per `.pt` file | Supervisor's cache format |
| Max sequence length | 256 (truncate longer) | Storage constraint |
| Base model frozen | Yes (we never load it during training) | Cache-only design |

---

## 3. Default hyperparameters (ablate later, don't touch in v1)

| Hyperparameter | Default | Notes |
|----------------|---------|-------|
| ResBlock layers per head | 2 | Supervisor said "你自己试" → ablate {1, 2, 4} later |
| ResBlock structure | `x + W_2(SiLU(W_1(LN(x))))` | MLP-style with LayerNorm, expansion=2; per supervisor handoff doc |
| ResBlock expansion | 2 | i.e. inner dim = 3584 × 2 = 7168 |
| Last linear init in ResBlock | Zero (weight + bias) | Makes block ≡ identity at init → stable training |
| Internal norm type | LayerNorm | Per supervisor handoff doc (NOT RMSNorm despite Qwen using RMSNorm internally) |
| Loss type | Cross-entropy with `ignore_index=-100` | CE first; teacher-loss is an ablation later |
| Loss weights across heads | `[1.0, 0.8, 0.64]` (exponential decay 0.8^k) | Vanilla Medusa default |
| Optimizer | **bitsandbytes 8-bit AdamW** (`bnb.optim.AdamW8bit`) | Standard AdamW state (fp32 m+v) takes ~16.7 GB for ~2.1B params, OOMs on V100 32GB. 8-bit Adam reduces this to ~2 GB with negligible accuracy loss for fine-tuning. Confirmed by C4 OOM debug. |
| Learning rate | 5e-4 | Hydra++ paper value; conservative for V100 |
| Weight decay | 0.0 | Heads are small; WD usually hurts |
| Warmup steps | 100 | |
| LR schedule | Cosine decay to `final_lr_multiplier=0.33` of peak | |
| Gradient clipping | 1.0 | Standard |
| Mixed precision | fp16 with loss scaling (NOT bf16, V100 doesn't support it) | |
| Batch size | 4-8 (whatever fits) + gradient accumulation | V100 32GB constraint |

---

## 4. Architecture diagram

```
                Cache provides h_t ∈ ℝ^3584 (per-position)
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               │               │
     ┌────────────────┐       │               │
     │ Head 0         │       │               │
     │  in_resblock   │       │               │
     │  resnet (2x)   │       │               │
     │  ─── h_0' ────┼───────┼───→ next head ─┼──┐
     │  lm_head       │       │               │  │
     │  ↓             │       │               │  │
     │  logits_0      │       │               │  │
     └────────────────┘       │               │  │
       predicts t+1           │               │  │
                              │               │  │
                              ▼               │  │
                     h_t + h_0' (add)         │  │
                              │               │  │
                              ▼               │  │
                     ┌────────────────┐       │  │
                     │ Head 1         │       │  │
                     │  in_resblock   │       │  │
                     │  resnet (2x)   │       │  │
                     │  ─── h_1' ────┼───→ next head ─┐
                     │  lm_head       │              │
                     │  ↓             │              │
                     │  logits_1      │              │
                     └────────────────┘              │
                       predicts t+2                  │
                                                     │
                                       ┌─── h_t + h_1'
                                       ▼
                              ┌────────────────┐
                              │ Head 2         │
                              │  in_resblock   │
                              │  resnet (2x)   │
                              │  lm_head       │
                              │  ↓             │
                              │  logits_2      │
                              └────────────────┘
                                predicts t+3
```

**Key invariant**: `h_k'` is the hidden state at the input of `head_k.lm_head` (after all ResBlocks, before vocab projection). Each head returns `(logits, h_prime)`. The chain passes `h_prime`; `logits` are used only for loss and final token output. **Both flow simultaneously, neither is detached.**

---

## 5. Module specs

### 5.1 `MLPResBlock`

```python
class MLPResBlock(nn.Module):
    """
    Pre-LN residual MLP block. Identity-initialized via zero-init of last linear.
    
    forward: x → x + W_2(SiLU(W_1(LayerNorm(x))))
    
    Args:
        hidden_dim:  3584
        expansion:   2  (inner dim = hidden_dim * expansion = 7168)
    
    Init:
        - W_1, b_1, LayerNorm: PyTorch defaults (Kaiming for Linear, ones/zeros for LN)
        - W_2.weight, W_2.bias: ZERO  (so block ≡ identity at init)
    """
    def __init__(self, hidden_dim: int, expansion: int = 2):
        ...
    
    def forward(self, x: Tensor) -> Tensor:  # (B, L, H) → (B, L, H)
        ...
```

### 5.2 `LinkedMedusaHead` (single head)

```python
class LinkedMedusaHead(nn.Module):
    """
    Single linked head. Consists of:
      - 1 input ResBlock      (consumes raw input)
      - N body ResBlocks      (the "ResNet" stack; default N=2)
      - 1 lm_head             (Linear projection to vocab)
    
    Returns BOTH logits and the pre-lm_head hidden state.
    The hidden state is what gets passed to the next head.
    
    Args:
        hidden_dim:   3584
        vocab_size:   152064
        num_blocks:   2  (number of body ResBlocks; configurable)
        expansion:    2
    """
    def __init__(self, hidden_dim, vocab_size, num_blocks=2, expansion=2):
        self.input_resblock = MLPResBlock(hidden_dim, expansion)
        self.body = nn.Sequential(*[
            MLPResBlock(hidden_dim, expansion) for _ in range(num_blocks)
        ])
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x:        (B, L, H)
        returns:
          logits: (B, L, V)        # V = 152064 for Qwen2.5-VL-7B (padded; effective 151936)
          h_prime: (B, L, H)   ← pre-lm_head hidden, for chaining
        """
        h = self.input_resblock(x)
        h = self.body(h)
        logits = self.lm_head(h)
        return logits, h
```

### 5.3 `LinkedMedusaHeads` (the full module — 3 heads chained)

```python
class LinkedMedusaHeads(nn.Module):
    """
    The full linked draft head module. Three heads chained via hidden-state passing.
    
    Args:
        hidden_dim:    3584
        vocab_size:    152064
        num_heads:     3
        num_blocks:    2
        expansion:     2
    """
    def __init__(self, hidden_dim, vocab_size, num_heads=3, num_blocks=2, expansion=2):
        self.heads = nn.ModuleList([
            LinkedMedusaHead(hidden_dim, vocab_size, num_blocks, expansion)
            for _ in range(num_heads)
        ])
    
    def forward(self, h_t: Tensor) -> List[Tensor]:
        """
        h_t:     (B, L, H)  — base model hidden states from cache
        returns: list of (B, L, V) — logits from each head
        
        head_0 input: h_t
        head_k input (k≥1): h_t + h_{k-1}_prime
        """
        all_logits = []
        h_prev_prime = None
        for k, head in enumerate(self.heads):
            head_input = h_t if h_prev_prime is None else h_t + h_prev_prime
            logits, h_prime = head(head_input)
            all_logits.append(logits)
            h_prev_prime = h_prime  # NO .detach() — gradients must flow
        return all_logits
    
    def init_lm_heads_from_base(self, base_lm_head_weight: Tensor):
        """
        Copy base model's lm_head weights into all 3 head lm_heads.
        Call this AFTER instantiation, before training.
        
        base_lm_head_weight: (V, H) = (152064, 3584) for Qwen2.5-VL-7B-Instruct
        Note: V is the padded physical dim, not the effective tokenizer vocab (151936).
        Both head lm_head and base lm_head must use the same padded dim for shape match.
        """
        for head in self.heads:
            head.lm_head.weight.data.copy_(base_lm_head_weight)
```

### 5.4 Critical implementation notes for the heads

- **DO NOT** call `.detach()` anywhere in the forward pass. Gradients must flow from `loss_2` all the way back to `head_0`. This is the core mechanism of the design.
- **DO NOT** add a final LayerNorm before `lm_head`. The body ResBlock's last operation is already a residual add; a final norm would break the lm_head copy trick (base lm_head expects an unnormalized input from the residual stream).
- **DO NOT** share `lm_head` weights across heads. Each head has its own.
- The combination is `h_t + h_prev_prime`, not concatenation.

---

## 6. Data spec

### 6.1 Cache layout (real data — when available)

```
/scratch/li96/mz9869/cached_data/qwen25vl_long/
├── manifest.json    # skip this when listing
├── 0.pt
├── 1.pt
├── ...
└── 46613.pt
```

Each `.pt` file is a dict:
```python
{
    'hidden': Tensor(L, 3584),  # float16
    'tokens': Tensor(L,),       # int64
}
```

`L` varies per sample (observed range: ~65 to >256). Cache is **response-only** — no image/prompt tokens, no special tokens. Truncate to max_length=256.

### 6.2 Dataset implementations

Both datasets MUST return the same dict schema and be selectable via config.

```python
class CachedVLMDataset(Dataset):
    def __init__(self, cache_dir: str, max_length: int = 256):
        # Enumerate all *.pt files (skip manifest.json)
        # Sort numerically by stem (0.pt, 1.pt, ..., 46613.pt — NOT lexicographic)
        ...
    
    def __getitem__(self, idx) -> dict:
        """
        Returns:
          {
            'hidden': Tensor(L, 3584) float16,  # truncated to <= max_length
            'tokens': Tensor(L,) int64,
          }
        """
        ...


class SyntheticVLMDataset(Dataset):
    """
    Mock dataset for testing code structure before the real cache is uploaded.
    Same interface and schema as CachedVLMDataset.
    """
    def __init__(self, num_samples: int = 200,
                 seq_len_range: Tuple[int, int] = (50, 256),
                 hidden_dim: int = 3584,
                 vocab_size: int = 152064,
                 seed: int = 42):
        # Generate fixed synthetic samples (deterministic via seed) so that
        # repeat training runs see the same data
        ...
    
    def __getitem__(self, idx) -> dict:
        # Same return schema as CachedVLMDataset
        ...


def build_dataset(cfg) -> Dataset:
    """Factory selecting dataset based on config."""
    if cfg.dataset.kind == 'synthetic':
        return SyntheticVLMDataset(
            num_samples=cfg.dataset.num_samples,
            seq_len_range=tuple(cfg.dataset.seq_len_range),
        )
    elif cfg.dataset.kind == 'cached':
        return CachedVLMDataset(
            cache_dir=cfg.dataset.cache_dir,
            max_length=cfg.dataset.max_length,
        )
    raise ValueError(f"Unknown dataset.kind: {cfg.dataset.kind}")
```

### 6.3 Collate function

```python
def collate_fn(batch: List[dict]) -> dict:
    """
    Pads variable-length sequences to the max length in the batch.
    
    Returns:
      {
        'hidden': Tensor(B, L_max, 3584) float16,
        'tokens': Tensor(B, L_max) int64,        # padded with -100
        'attention_mask': Tensor(B, L_max) bool, # True for real tokens
      }
    
    - Hidden states pad with 0.0
    - Tokens pad with -100 (CE ignore_index)
    - attention_mask used for any masked operations
    """
    ...
```

### 6.4 Loss computation (CRITICAL — get the offsets right)

**Convention**: cache uses TARGET convention. `tokens[t]` is what the base model emitted at step t; `hidden[t]` is the hidden state just before that emission. See §11.1 for verification details.

For head_k, the target is the token at position `t + k`, predicted from input at position `t`:

```python
def compute_loss(all_logits: List[Tensor], tokens: Tensor, weights: List[float]) -> dict:
    """
    all_logits: list of (B, L, V) — logits from each head
    tokens:     (B, L) — ground truth tokens, padded with -100
    weights:    [1.0, 0.8, 0.64]
    
    Cache uses TARGET convention: head_k predicts tokens[t+k] from input at position t.
    So for head_0 (k=0): pred is full-length, target is full-length.
    For head_k (k>=1):  pred drops last k positions, target drops first k positions.
    """
    losses = {}
    total_loss = 0.0
    
    for k, (logits_k, w_k) in enumerate(zip(all_logits, weights)):
        if k == 0:
            pred_k = logits_k                          # (B, L, V)
            target_k = tokens                          # (B, L)
        else:
            pred_k = logits_k[:, :-k, :].contiguous()  # (B, L-k, V)
            target_k = tokens[:, k:].contiguous()      # (B, L-k)
        
        loss_k = F.cross_entropy(
            pred_k.view(-1, pred_k.size(-1)),
            target_k.view(-1),
            ignore_index=-100,
        )
        losses[f'head_{k}_loss'] = loss_k.item()
        total_loss = total_loss + w_k * loss_k
    
    losses['total_loss'] = total_loss
    return losses
```

**Sequence length guard**: with K heads, require `L >= K` (head_K-1 needs `tokens[:, K-1:]` non-empty).

### 6.5 Eval metrics (compute every N steps)

For each head k, on a held-out subset:

- **top1_acc**: fraction of positions where `argmax(logits_k) == target_k` (excluding ignore_index)
- **top5_acc**: fraction where target is in top-5
- **mean_loss**: per-head CE loss

---

## 7. Training loop spec

```python
def train(cfg):
    # 1. Load dataset
    dataset = CachedVLMDataset(cfg.cache_dir, max_length=cfg.max_length)
    train_set, val_set = random_split(dataset, [...], generator=Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    # 2. Build model
    model = LinkedMedusaHeads(
        hidden_dim=3584, vocab_size=152064,
        num_heads=3, num_blocks=cfg.num_blocks, expansion=2,
    ).cuda()
    
    # 3. Initialize lm_heads from base model
    base_lm_head_weight = torch.load(cfg.base_lm_head_path)  # (V, H)
    model.init_lm_heads_from_base(base_lm_head_weight)
    
    # 4. Optimizer + scheduler (8-bit Adam — see §3 for rationale)
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.total_steps,
        min_lr_ratio=0.33,  # final_lr_multiplier
    )
    
    # 5. fp16 mixed precision (NOT bf16 — V100)
    scaler = GradScaler()
    
    # 6. Loop
    for step, batch in enumerate(train_loader):
        h_t    = batch['hidden'].cuda().half()           # (B, L, 3584) fp16
        tokens = batch['tokens'].cuda()                  # (B, L)
        
        with autocast(dtype=torch.float16):
            all_logits = model(h_t)
            losses = compute_loss(all_logits, tokens, weights=cfg.loss_weights)
        
        scaler.scale(losses['total_loss']).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
        
        if step % cfg.log_every == 0:
            log(losses, lr=scheduler.get_last_lr()[0], step=step)
        
        if step % cfg.eval_every == 0:
            eval_metrics = evaluate(model, val_loader)
            log(eval_metrics, step=step)
        
        if step % cfg.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, step, cfg.output_dir)
```

### 7.1 Sanity checks the training loop should print at startup

After init but before step 0, run one forward pass on a batch and assert:

1. `head_0_loss` should be **near-zero** (e.g. < 1.0). Reason: lm_head is copied from base, body ResBlocks are identity-init → head_0 forward ≈ base lm_head applied to h_t ≈ correctly predicts next token (the very task the cache was generated for).
2. `head_1_loss`, `head_2_loss` should be much higher (predicting 2-3 steps ahead from same h_t).
3. After 1 backward pass, gradient norm of `head_0` parameters should be **non-zero** even if only `loss_2` is backpropped. This proves the chain is connected. (Easy diagnostic: zero out `loss_0` and `loss_1`, check head_0 still has gradients.)

---

## 8. File organization

Create files under `~/medusa-qwen/` (existing codebase to migrate from):

```
~/medusa-qwen/
├── README.md                          # update with new arch description
├── config/
│   └── linked_medusa_default.yaml     # all hyperparams
├── data/
│   ├── __init__.py
│   ├── cached_dataset.py              # CachedVLMDataset
│   └── collate.py                     # collate_fn
├── model/
│   ├── __init__.py
│   ├── resblock.py                    # MLPResBlock
│   ├── linked_head.py                 # LinkedMedusaHead, LinkedMedusaHeads
│   └── init_utils.py                  # base lm_head copy logic
├── train/
│   ├── __init__.py
│   ├── loss.py                        # compute_loss
│   ├── trainer.py                     # main training loop
│   └── evaluate.py                    # evaluation metrics
├── scripts/
│   ├── train.py                       # entrypoint: python -m scripts.train --config ...
│   ├── extract_base_lm_head.py        # one-time: pull lm_head weight from base ckpt
│   └── debug_forward.py               # sanity-check forward pass (see §7.1)
├── pbs/
│   └── train_v100.pbs                 # NCI Gadi PBS submission script
└── tests/
    ├── test_resblock.py               # identity-init check, gradient flow
    ├── test_linked_head.py            # shape checks, gradient flow through chain
    └── test_loss.py                   # offset correctness, ignore_index handling
```

### 8.1 Migration notes from existing `~/medusa-qwen/` code

The existing codebase was for Qwen3-VL-2B. Changes needed:

- `hidden_dim`: 2048 → **3584**
- `vocab_size`: 151936 → **152064** (use lm_head padded physical dim, see §2)
- Base model: Qwen3-VL-2B → Qwen2.5-VL-7B-Instruct
- Cache path: read from `/scratch/li96/mz9869/cached_data/qwen25vl_long/` (NOT `/g/data/...`)
- Architecture: vanilla Medusa (independent heads) → **linked heads with hidden-state passing**

Keep what's reusable: data loading patterns, PBS script template, eval harness skeleton.

---

## 9. Environment and paths (NCI Gadi)

### 9.1 Storage rules (CRITICAL)

| What | Where | Why |
|------|-------|-----|
| Code | `~/medusa-qwen/` | Home dir, small footprint |
| Cache data (77 GB) | `/scratch/li96/mz9869/cached_data/qwen25vl_long/` | gdata quota insufficient |
| Training checkpoints | `/scratch/li96/mz9869/medusa_outputs/` | scratch is fine for transient outputs |
| Base model weights | `/g/data/li96/mz9869/models/qwen25vl-7b/` | Already there |
| PBS log files | `/scratch/li96/mz9869/logs/` | |

**Never** write training data or large outputs to `/g/data/` — quota will blow up.

### 9.2 Environment setup

```bash
module load python3/3.11.0 cuda/12.3.2
source ~/medusa-env/bin/activate
```

### 9.3 PBS script must include

```bash
#PBS -l storage=gdata/li96+scratch/li96
#PBS -l ncpus=12
#PBS -l mem=96GB
#PBS -l ngpus=1
#PBS -l walltime=12:00:00
#PBS -P li96
#PBS -q gpuvolta
```

Without the `-l storage=...` line, the job will fail with permission errors when reading the cache.

### 9.4 V100 specifics

- **Use fp16, NOT bf16**. V100 has no bf16 support.
- Use `torch.cuda.amp.GradScaler` for loss scaling.
- 32GB memory budget. With ~2.1B params, **must use bitsandbytes 8-bit AdamW** (see §3); standard AdamW OOMs at optimizer init.
- Memory budget breakdown (8-bit AdamW): weights 4.2 GB + grads 4.2 GB + optim 2.1 GB + activations ≤ 4 GB → ~15 GB total → batch_size 4-8 fits comfortably.
- With standard AdamW: weights 4.2 GB + grads 4.2 GB + optim **16.7 GB** + activations → 27+ GB → OOM at first optimizer step (confirmed during C4 debug).
- Install: `pip install bitsandbytes`; verify with `python -c "import bitsandbytes; print(bitsandbytes.__version__)"`.

---

## 10. Locked vs ablation knobs

### Locked in v1 (do not touch — these are the supervisor's confirmed design)

- Hidden state passing (NOT token embedding)
- Elementwise add for combination (NOT concat)
- 3 heads
- Independent lm_heads
- lm_head init from base
- Cache-only training
- CE loss with `ignore_index=-100`

### Parameterize via config (for later ablation)

- `num_blocks` (per head): default 2, ablate {1, 2, 4}
- `expansion`: default 2, ablate {1, 2, 4}
- `loss_weights`: default `[1.0, 0.8, 0.64]`, ablate uniform `[1, 1, 1]`
- `loss_type`: default `'ce'`, future `'teacher'` and `'mixed'`

Make these clean YAML config knobs, not hardcoded constants.

---

## 11. Validation checklist before submitting first training job

Run `scripts/debug_forward.py` interactively on a GPU node and verify:

- [ ] `LinkedMedusaHeads` forward pass produces 3 logits tensors of shape `(B, L, 152064)` *(synthetic OK)*
- [ ] After `init_lm_heads_from_base`, head_0 logits at position t closely match (or equal) base lm_head applied to h_t *(synthetic OK)*
- [ ] head_0 CE loss on a real batch is < 1.0 — sanity: identity init makes head_0 ≈ base predictor *(REQUIRES REAL CACHE — defer until cache uploaded)*
- [ ] head_1 and head_2 CE losses are higher than head_0 (predicting further ahead is harder) *(REQUIRES REAL CACHE — on synthetic random data, all three will be ~log(V); skip)*
- [ ] After `total_loss.backward()`, `head_0` parameters have non-zero `.grad` even when only head_2's loss is used (proves chain is connected) *(synthetic OK — this is the most important structural check)*
- [ ] Gradient norms across heads are sane (no NaN, no explosion) *(synthetic OK)*
- [ ] One full training step runs without OOM at the configured batch_size *(synthetic OK — actually use longest-possible synthetic seq length to stress-test memory)*

**Phase A (now, synthetic data)**: run all `(synthetic OK)` items. These cover code structure, gradient flow, shape correctness, and memory.

**Phase B (after cache arrives)**: run the two `REQUIRES REAL CACHE` items. If they pass, you're cleared to start full training.

### 11.1 Cache convention (verified)

The cache uses **TARGET convention**, verified empirically via `scripts/verify_cache_convention.py` on 3 test samples (TARGET 81.2% vs INPUT 0.0%):

- `tokens[t]` is the token the base model **EMITTED** at step t (its output / target).
- `hidden[t]` is the base model's last-layer hidden state immediately before emitting `tokens[t]`.
- Therefore the zero-latency prediction target for `hidden[t]` is `tokens[t]` itself (not `tokens[t+1]`).

This means head semantics are:

- `head_0`: predicts `tokens[t]` from `hidden[t]`. Since `head_0`'s lm_head is initialized from the base lm_head, and the base hidden state IS the input that base used to predict `tokens[t]`, **head_0 forward at init ≈ base lm_head(hidden[t]) ≈ argmax → tokens[t]**. Expected `head_0_loss < 1.0` at init.
- `head_k` (k ≥ 1): predicts `tokens[t+k]` from input at position t.

Loss offsets in `compute_loss`:

```python
# k = 0: head_0 predicts tokens[t] from logits_0[t] — no shift
# k >= 1: head_k predicts tokens[t+k] from logits_k[t] — shift by k

pred_k   = logits_k         if k == 0 else logits_k[:, :-k, :]
target_k = tokens           if k == 0 else tokens[:, k:]
```

**Sequence length requirement**: with K heads, the deepest head (k=K-1) requires `L >= K` (so `tokens[:, K-1:]` has at least 1 valid position).

### 11.2 Phase B sanity check (run after cache arrives)

After real cache is mounted and `dataset.kind: cached` is selected, the first training-loop startup sanity check must report:

- `head_0_loss < 1.0` on a real cache batch (because head_0 ≡ base lm_head at init, and target convention makes the lm_head output match `tokens[t]` directly)
- `head_1_loss, head_2_loss` should be in the 2-8 range — higher than head_0 (deeper-step prediction is harder) but lower than `log(V) ≈ 11.93` (because hidden states still carry information about future tokens through residual stream)

If `head_0_loss ≈ log(V)`, the convention is wrong — stop and re-run `verify_cache_convention.py` on real cache samples.

---

## 12. Out of scope for v1

The following are explicitly **NOT** part of this implementation. Do not add them:

- Tree attention / verification logic (needs base model live; Month 2-3 work)
- Acceptance length (σ) evaluation (needs base model verification loop)
- Token embedding passing (Hydra-style) — this is an ablation, not the baseline
- Teacher loss / self-distillation — ablation
- Uncertainty signal passing between heads — supervisor said "leave a placeholder, don't implement"
- PrefixMLP (extra decoder layer) — incompatible with last-layer cache
- Multi-GPU training — V100 single-GPU is the target

Keep `forward()` and the training loop clean of any code paths for these.

---

## 13. Quick reference: the design in three sentences

1. Three heads chained via hidden-state passing: `head_k` consumes `h_t + h_{k-1}'`, produces `(logits_k, h_k')`, where `h_k'` is the activation immediately before its lm_head.
2. The chain is fully differentiable end-to-end — gradients from `loss_2` reach `head_0` parameters through the residual hidden-state path.
3. Training is cache-only: the base model's last-layer hidden states are precomputed and stored on disk; we only train the 3 heads on top.
