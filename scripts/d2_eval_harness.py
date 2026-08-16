"""d2_eval_harness.py — D2 三臂计时 harness(reports/d2_eval_protocol.md)。

臂① AR greedy(decode.common.vanilla_greedy,旧线 clean-greedy 同路径)
臂② dyn_k8_n24 复跑(eval_acceptance_tree.run_one_prompt_tree_folded,
    dynamic builder,fanout [1,8,8,8,8] max_nodes 24——冠军 flags 逐项
    对照 job 175598529 日志头)
臂③ diffusion drafter(decode.dflash_vlm,block 16,best ckpt)

同进程交替;prompt 内三臂 Latin square 轮转;前 5 条 warmup 弃计;
cuda.synchronize 包夹计时;逐 prompt JSON 落盘;G0-G2 自动判定汇总。
不含 GPU 提交逻辑——由过审 PBS 驱动。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

_ROOT = Path(__file__).resolve().parent.parent
for p in (str(_ROOT), str(_ROOT / "external" / "dflash")):
    if p not in sys.path:
        sys.path.insert(0, p)

ARMS = ("ar_greedy", "dyn_k8_n24", "dflash_b16")
CHAMPION = dict(fanout=[1, 8, 8, 8, 8], max_nodes=24, tree_builder="dynamic",
                depth1_floor=True, skip_head0_lm_head=True)


def latin_order(i: int) -> tuple[int, int, int]:
    """3-arm Latin square by prompt index: each arm hits each slot n/3 times."""
    r = i % 3
    return tuple((r + k) % 3 for k in range(3))


def paired_bootstrap_ci(diffs: list[float], n_boot: int = 10000,
                        seed: int = 43, alpha: float = 0.05):
    """Percentile bootstrap CI of the mean of paired differences."""
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    point = sum(diffs) / n
    return point, lo, hi


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    _sync(); t = time.perf_counter()
    out = fn()
    _sync()
    return out, time.perf_counter() - t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/scratch/li96/mz9869/eval_manifests/manifest_300.json")
    ap.add_argument("--images-dir", required=True,
                    help="COCO eval images dir (e.g. /g/data/li96/mz9869/data/coco_subset)")
    ap.add_argument("--g0a-result", required=True,
                    help="G0a(V100 字节门)结果 JSON——协议 v1.1:harness 只读"
                         "其结论,不在 A100 上重比字节")
    ap.add_argument("--tree-ckpt", default="/scratch/li96/mz9869/medusa_outputs/linked_medusa_c1_eagle/ckpt_best.pt")
    ap.add_argument("--drafter-ckpt", required=True)
    ap.add_argument("--hf-snapshot", default="/scratch/li96/mz9869/tmp_hf_download/hub/"
                    "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/"
                    "cc594898137f460bfe9f0759e9844b3ce807cfb5")
    ap.add_argument("--lm-head", default="/scratch/li96/mz9869/medusa_assets/base_lm_head.safetensors")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-prompts", type=int, default=300)
    ap.add_argument("--max-new", type=int, default=256)      # 协议锁定值
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--g2-prompts", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    # 故障修复(2026-08-16,job 176390465 教训):
    # 1) 先 eager import train 包(scripts/train.py 与 train/ 同名,任何
    #    后续路径操纵都不得影响已缓存的包解析);
    # 2) scripts/ 只允许 append(CLAUDE.md 条款)——包名优先于同名模块。
    import train.train_drafter  # noqa: F401  (eager, anti-shadowing)
    from decode.common import (load_base, load_head, make_image_inputs,
                               vanilla_greedy, filter_prompts, cfg_attr)
    from decode.dflash_vlm import (load_drafter_for_inference,
                                   dflash_vlm_generate, run_twice_check)
    sys.path.append(str(_ROOT / "scripts"))
    from eval_acceptance_tree import run_one_prompt_tree_folded

    out = Path(args.out_dir); (out / "per_prompt").mkdir(parents=True, exist_ok=True)
    (out / "d3").mkdir(exist_ok=True)
    base, processor = load_base("Qwen/Qwen2.5-VL-7B-Instruct", args.device)
    head = load_head(args.tree_ckpt, cfg_attr(base.config, "hidden_size"),
                     cfg_attr(base.config, "vocab_size"), args.device)
    drafter, embed, lm_head, dcfg = load_drafter_for_inference(
        args.drafter_ckpt, args.hf_snapshot, args.lm_head,
        torch.device(args.device), torch.bfloat16)
    eos_id = processor.tokenizer.eos_token_id
    images_dir = Path(args.images_dir)
    # 与冠军跑同参:min_ref_words=80, seed=42, ordered
    prompts = filter_prompts(args.manifest, 80, 42, ordered=True)[: args.n_prompts]
    g0a = json.load(open(args.g0a_result))

    def arm_ar(p):
        return vanilla_greedy(base, processor, p["question"],
                              images_dir / p["image"], args.max_new, eos_id,
                              args.device)

    def arm_tree(p):
        r = run_one_prompt_tree_folded(
            base, head, processor, p, images_dir, args.max_new, eos_id,
            device=args.device, **CHAMPION)
        return r

    def arm_dflash(p):
        inputs = make_image_inputs(processor, p["question"],
                                   images_dir / p["image"], args.device)
        d3 = []
        seq, stats = dflash_vlm_generate(base, drafter, embed, lm_head,
                                         inputs, args.max_new, eos_id,
                                         d3_log=d3)
        return seq, stats, d3

    arm_fns = {0: arm_ar, 1: arm_tree, 2: arm_dflash}
    results = []
    for i, p in enumerate(prompts):
        rec = {"i": i, "id": p["id"], "image": p["image"],
               "order": latin_order(i), "warmup": i < args.warmup}
        for a in latin_order(i):
            if a == 0:
                toks, w = timed(lambda: arm_ar(p))
                rec["ar_greedy"] = {"wall_s": w, "n_tokens": len(toks),
                                    "tokens": toks}
            elif a == 1:
                tr, w = timed(lambda: arm_tree(p))
                # v1.2/B2:严格键访问(KeyError 即崩,不静默降级)
                rec["dyn_k8_n24"] = {"wall_s": w,
                                     "n_tokens": tr["total_emitted"],
                                     "tokens": tr["emitted_tokens"],
                                     "sigma": tr["sigma"]}
            else:
                (seq, stats, d3), w = timed(lambda: arm_dflash(p))
                rec["dflash_b16"] = {"wall_s": w, "n_tokens": len(seq),
                                     "tokens": seq, **{k: v for k, v in stats.items()
                                                       if k != "accept_lengths"}}
                json.dump({"warmup": i < args.warmup, "prompt_id": p["id"],
                           "cycles": d3},
                          open(out / "d3" / f"{p['id']}.json", "w"))
        # 跨臂匹配登记(协议 greedy 等价性声明)
        ta, td = rec["ar_greedy"]["tokens"], rec["dflash_b16"]["tokens"]
        n = min(len(ta), len(td))
        div = next((k for k in range(n) if ta[k] != td[k]), None)
        rec["cross_arm"] = {"exact": ta == td, "first_div": div,
                            "match_prefix_frac": (div if div is not None else n) / max(1, n)}
        json.dump(rec, open(out / "per_prompt" / f"{i:03d}_{p['id']}.json", "w"))
        results.append(rec)
        if i % 20 == 0:
            print(f"[d2] {i + 1}/{len(prompts)} done", flush=True)

    scored = [r for r in results if not r["warmup"]]
    # G1(v1.2): per-token paired speedup:speedup_X = (t1/n1)/(tX/nX)
    diffs = []
    for r in scored:
        pt1 = r["ar_greedy"]["wall_s"] / max(1, r["ar_greedy"]["n_tokens"])
        pt2 = r["dyn_k8_n24"]["wall_s"] / max(1, r["dyn_k8_n24"]["n_tokens"])
        pt3 = r["dflash_b16"]["wall_s"] / max(1, r["dflash_b16"]["n_tokens"])
        r["speedup_tree"] = pt1 / pt2
        r["speedup_dflash"] = pt1 / pt3
        diffs.append(r["speedup_dflash"] - r["speedup_tree"])
    point, lo, hi = paired_bootstrap_ci(diffs)
    g1 = {"point": point, "ci95": [lo, hi], "pass": point > 0 and lo > 0,
          "marginal": lo <= 0 <= hi and point > 0}
    # G0(协议 v1.1):G0a 结论从 V100 作业结果文件读入;G0b = A100 新基准
    g0 = {"G0a": {"pass": bool(g0a.get("pass")), "source": args.g0a_result,
                  "detail": {k: g0a.get(k) for k in
                             ("n_prompts", "n_byte_exact", "job")}},
          "G0b_tree_speedup_mean_new_baseline":
              sum(r["speedup_tree"] for r in scored) / len(scored),
          "sigma_batch_registered": 0.0537}
    # G2: 双跑一致
    g2_runs = []
    for r in scored[: args.g2_prompts]:
        p = prompts[r["i"]]
        inputs = make_image_inputs(processor, p["question"],
                                   images_dir / p["image"], args.device)
        g2_runs.append({"id": p["id"],
                        **run_twice_check(base, drafter, embed, lm_head,
                                          inputs, args.max_new, eos_id)})
    g2 = {"runs": g2_runs, "pass": all(x["identical"] for x in g2_runs)}
    tau1 = sum(r["dflash_b16"]["tau_accept_only"] for r in scored) / len(scored)
    tau2 = sum(r["dflash_b16"]["tau_with_bonus"] for r in scored) / len(scored)
    cross = {"exact_rate": sum(r["cross_arm"]["exact"] for r in scored) / len(scored),
             "div_sample_ids": [r["id"] for r in scored
                                if not r["cross_arm"]["exact"]][:10]}
    import subprocess
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     cwd=_ROOT, text=True).strip()
    summary = {"n_scored": len(scored), "code_commit": commit,
               "g0a_job_id": g0a.get("job"),
               "speedup_definition": "per-token (v1.2): (wall1/n1)/(wallX/nX)",
               "G0": g0, "G1": g1, "G2": g2,
               "tau_accept_only": tau1, "tau_with_bonus": tau2,
               "tau_footnote": "DFlash τ=6.5 对照须带 35K vs 800K 数据折扣脚注",
               "cross_arm": cross,
               "per_prompt_dir": str(out / "per_prompt"),
               "d3_dir": str(out / "d3")}
    json.dump(summary, open(out / "d2_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1)[:2000], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
