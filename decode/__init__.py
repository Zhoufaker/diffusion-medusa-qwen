"""decode — shared speculative-decoding primitives for Linked Medusa.

Layering (per design review):
  common.py : model/head loaders, vocab masking (single source of truth),
              prompt filtering, image-input building, vanilla greedy,
              and the M-RoPE continuation-base helper. Used by BOTH the
              chain evaluator and the tree evaluator, and by P1 on-policy.
  tree.py   : tree CONSTRUCTION + attention mask/position_ids + KV reorg
              (reusable for inference AND training), plus greedy ACCEPT
              (inference-only; kept separate so P1 training is not forced
              into greedy-argmax matching semantics).
"""
