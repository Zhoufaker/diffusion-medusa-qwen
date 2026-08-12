"""GPU e2e: tree runner hard-cap ≡ vanilla_greedy[:max_new] (round-2 P1).

Skipped automatically without CUDA / ckpt / data. Run on a GPU node:
  pytest tests/test_max_new_cap_gpu.py -v
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

CKPT = "/scratch/li96/mz9869/medusa_outputs/linked_medusa_c1_eagle/ckpt_best.pt"
MANIFEST = "/g/data/li96/mz9869/data/llava_subset_2k.json"
IMAGES = "/g/data/li96/mz9869/data/coco_subset"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not Path(CKPT).is_file()
    or not Path(MANIFEST).is_file(),
    reason="CUDA + C1 ckpt + LLaVA manifest required",
)


@pytest.fixture(scope="module")
def env():
    os.environ.setdefault("HF_HOME", "/scratch/li96/mz9869/tmp_hf_download")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from decode.common import cfg_attr, filter_prompts, load_base, load_head, vanilla_greedy
    from scripts.eval_acceptance_tree import run_one_prompt_tree, run_one_prompt_tree_folded

    base, processor = load_base("Qwen/Qwen2.5-VL-7B-Instruct")
    head = load_head(CKPT, cfg_attr(base.config, "hidden_size"),
                     cfg_attr(base.config, "vocab_size"))
    eos_id = processor.tokenizer.eos_token_id
    prompts = filter_prompts(MANIFEST, min_ref_words=80, seed=42)[:1]
    images_dir = Path(IMAGES)
    return {
        "base": base, "head": head, "processor": processor, "eos_id": eos_id,
        "prompt": prompts[0], "images_dir": images_dir,
        "run_tree": run_one_prompt_tree,
        "run_folded": run_one_prompt_tree_folded,
        "vanilla_greedy": vanilla_greedy,
    }


@pytest.mark.parametrize("max_new", [1, 2, 3, 4, 5, 6])
def test_folded_emitted_eq_greedy_prefix(env, max_new):
    p = env["prompt"]
    g = env["vanilla_greedy"](
        env["base"], env["processor"], p["question"],
        env["images_dir"] / p["image"], max_new, env["eos_id"],
    )
    r = env["run_folded"](
        env["base"], env["head"], env["processor"], p, env["images_dir"],
        max_new, env["eos_id"], fanout=[1, 3, 2, 1, 0], max_nodes=16,
        tree_builder="static",
    )
    assert r["runner"] == "v2_hardcap"
    assert len(r["emitted_tokens"]) <= max_new
    assert r["emitted_tokens"] == g[:max_new]


@pytest.mark.parametrize("max_new", [1, 2, 3, 4])
def test_nonfolded_emitted_eq_greedy_prefix(env, max_new):
    p = env["prompt"]
    g = env["vanilla_greedy"](
        env["base"], env["processor"], p["question"],
        env["images_dir"] / p["image"], max_new, env["eos_id"],
    )
    r = env["run_tree"](
        env["base"], env["head"], env["processor"], p, env["images_dir"],
        max_new, env["eos_id"], fanout=[1, 1, 1, 0, 0], max_nodes=8,
        depth1_floor=True, reorg="gather",
    )
    assert r["runner"] == "v2_hardcap"
    assert len(r["emitted_tokens"]) <= max_new
    assert r["emitted_tokens"] == g[:max_new]
