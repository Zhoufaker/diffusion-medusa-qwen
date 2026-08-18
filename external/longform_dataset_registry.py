"""Shared long-form VLM dataset loading utilities.

The paper needs a broader stress matrix than LLaVA-Bench detail plus one OOD
caption set. This module keeps that expansion in one place so cost profiling,
walltime scripts, and future training exporters do not grow separate dataset
loaders.

Every loader returns records with at least:
  image: PIL.Image.Image
  question: prompt string for the VLM
  reference: optional human/model long-form answer
  source: stable dataset/source name
  idx: integer within this sampled run
"""

import argparse
import io
import json
import os
import random
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_PROMPT = "Describe this image in detail."
LONGFORM_PROMPT = (
    "Describe this image in detail. Include the main objects, scene context, "
    "relationships, visual attributes, and any uncertainty."
)

DATASET_NAMES = (
    "detail",
    "detailcaps",
    "docci",
    "localized_narratives",
    "stanford_paragraph",
    "vqaonline",
    "vizwiz_lf",
    "jsonl",
    "synthetic",
)

IMAGE_FIELDS = (
    "image",
    "img",
    "image_path",
    "path",
    "filepath",
    "file_name",
    "filename",
    "image_url",
    "url",
    "binary",
)
QUESTION_FIELDS = ("question", "prompt", "query", "instruction")
REFERENCE_FIELDS = (
    "reference",
    "answer",
    "answers",
    "long_answer",
    "caption",
    "captions",
    "paragraph",
    "description",
    "text",
    "response",
)


class DatasetLoaderError(RuntimeError):
    pass


def _import_pil():
    try:
        from PIL import Image
    except Exception as exc:
        raise DatasetLoaderError("Pillow is required to load image datasets") from exc
    return Image


def _to_rgb(image: Any):
    Image = _import_pil()
    if isinstance(image, Image.Image):
        out = image
    elif isinstance(image, np.ndarray):
        out = Image.fromarray(image)
    elif isinstance(image, bytes):
        out = Image.open(io.BytesIO(image))
    elif hasattr(image, "read"):
        out = Image.open(image)
    else:
        out = Image.open(image)
    if out.mode != "RGB":
        out = out.convert("RGB")
    return out


def _load_image_url(url: str):
    import requests

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return _to_rgb(response.content)


def _field(record: Dict[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return None


def _string_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        strings = []
        for item in value:
            if isinstance(item, str):
                strings.append(item)
            elif isinstance(item, dict):
                nested = _string_or_none(_field(item, REFERENCE_FIELDS))
                if nested:
                    strings.append(nested)
        if not strings:
            return None
        return max(strings, key=len)
    if isinstance(value, dict):
        return _string_or_none(_field(value, REFERENCE_FIELDS))
    return str(value)


def _record_to_sample(
    record: Dict[str, Any],
    *,
    base_dir: Optional[str],
    source: str,
    idx: int,
    default_question: str,
) -> Optional[Dict[str, Any]]:
    image_value = _field(record, IMAGE_FIELDS)
    if image_value is None:
        return None

    image = None
    if isinstance(image_value, str):
        if image_value.startswith(("http://", "https://")):
            image = _load_image_url(image_value)
        else:
            path = image_value
            if base_dir and not os.path.isabs(path):
                path = os.path.join(base_dir, path)
            if not os.path.isabs(path) and not os.path.exists(path):
                return None
            if os.path.isabs(path) and not os.path.exists(os.path.expanduser(path)):
                return None
            image = _to_rgb(os.path.expanduser(path))
    else:
        image = _to_rgb(image_value)

    question = _string_or_none(_field(record, QUESTION_FIELDS)) or default_question
    reference = _string_or_none(_field(record, REFERENCE_FIELDS))
    sample_id = _string_or_none(record.get("id") or record.get("sample_id") or record.get("image_id"))
    return {
        "image": image,
        "question": question,
        "reference": reference,
        "source": source,
        "idx": idx,
        "sample_id": sample_id,
    }


def _iter_json_records(path: str) -> Iterator[Tuple[Dict[str, Any], Optional[str]]]:
    path = os.path.abspath(os.path.expanduser(path))
    base_dir = os.path.dirname(path) if os.path.isfile(path) else path
    if os.path.isdir(path):
        candidates = []
        for rel in ("samples.jsonl", "dataset.jsonl", "annotations.jsonl", "samples.json", "dataset.json", "annotations.json"):
            candidate = os.path.join(path, rel)
            if os.path.exists(candidate):
                candidates.append(candidate)
        ann_dir = os.path.join(path, "annotations")
        if os.path.isdir(ann_dir):
            for name in sorted(os.listdir(ann_dir)):
                if name.endswith((".jsonl", ".json")):
                    candidates.append(os.path.join(ann_dir, name))
        if not candidates:
            raise DatasetLoaderError(f"No JSON/JSONL annotation file found under {path}")
        for candidate in candidates:
            yield from _iter_json_records(candidate)
        return

    if path.endswith(".jsonl"):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line), base_dir
        return

    if path.endswith(".json"):
        with open(path) as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            for key in ("samples", "data", "annotations", "items"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise DatasetLoaderError(f"Unsupported JSON structure in {path}")
        for item in payload:
            if isinstance(item, dict):
                yield item, base_dir
        return

    raise DatasetLoaderError(f"Unsupported annotation file type: {path}")


def load_jsonl_samples(
    data_path: str,
    *,
    n_samples: int,
    seed: int,
    source: str = "jsonl",
    default_question: str = LONGFORM_PROMPT,
) -> List[Dict[str, Any]]:
    records = list(_iter_json_records(data_path))
    rng = random.Random(seed)
    rng.shuffle(records)
    samples = []
    for record, base_dir in records:
        sample = _record_to_sample(
            record,
            base_dir=base_dir,
            source=source,
            idx=len(samples),
            default_question=default_question,
        )
        if sample is not None:
            samples.append(sample)
        if len(samples) >= n_samples:
            break
    if not samples:
        raise DatasetLoaderError(f"No usable image/text samples found in {data_path}")
    return samples


def _load_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise DatasetLoaderError("`datasets` with pyarrow is required for HF dataset loading") from exc
    return load_dataset(*args, **kwargs)


def _hf_candidates(
    candidates: Sequence[Tuple[str, Optional[str], str, Optional[str]]],
    *,
    n_samples: int,
    seed: int,
    source: str,
    default_question: str,
    filter_fn=None,
) -> List[Dict[str, Any]]:
    errors = []
    for name, config, split, revision in candidates:
        try:
            kwargs: Dict[str, Any] = {"split": split}
            if revision:
                kwargs["revision"] = revision
            ds = _load_dataset(name, config, **kwargs) if config else _load_dataset(name, **kwargs)
            try:
                ds = ds.shuffle(seed=seed)
            except Exception:
                pass

            samples = []
            for raw in ds:
                if filter_fn is not None and not filter_fn(raw):
                    continue
                sample = _record_to_sample(
                    raw,
                    base_dir=None,
                    source=source or name,
                    idx=len(samples),
                    default_question=default_question,
                )
                if sample is not None:
                    samples.append(sample)
                if len(samples) >= n_samples:
                    return samples
            if samples:
                return samples
            errors.append(f"{name}:{split}: no records with loadable images")
        except Exception as exc:
            label = name if config is None else f"{name}/{config}"
            if revision:
                label = f"{label}@{revision}"
            errors.append(f"{label}:{split}: {exc}")
    raise DatasetLoaderError(" | ".join(errors) if errors else "No HF candidates were tried")


def load_synthetic_samples(n_samples: int, seed: int) -> List[Dict[str, Any]]:
    Image = _import_pil()
    samples = []
    for idx in range(n_samples):
        rng = np.random.default_rng(seed + idx)
        image = Image.fromarray(rng.integers(0, 255, (336, 336, 3), dtype=np.uint8))
        samples.append({
            "image": image,
            "question": DEFAULT_PROMPT,
            "reference": None,
            "source": "synthetic",
            "idx": idx,
            "sample_id": f"synthetic-{idx}",
        })
    return samples


def load_longform_samples(
    dataset: str,
    *,
    n_samples: int,
    seed: int = 42,
    data_path: Optional[str] = None,
    hf_name: Optional[str] = None,
    hf_config: Optional[str] = None,
    hf_split: Optional[str] = None,
    hf_revision: Optional[str] = None,
) -> List[Dict[str, Any]]:
    dataset = dataset.lower()
    if dataset not in DATASET_NAMES:
        raise DatasetLoaderError(f"Unknown dataset `{dataset}`. Choices: {', '.join(DATASET_NAMES)}")

    if dataset == "synthetic":
        return load_synthetic_samples(n_samples, seed)

    if data_path:
        return load_jsonl_samples(
            data_path,
            n_samples=n_samples,
            seed=seed,
            source=dataset,
            default_question=LONGFORM_PROMPT,
        )

    if dataset == "jsonl":
        raise DatasetLoaderError("`jsonl` requires --data-path or --dataset-jsonl")

    if hf_name:
        splits = [hf_split] if hf_split else ["test", "validation", "train"]
        return _hf_candidates(
            [(hf_name, hf_config, split, hf_revision) for split in splits],
            n_samples=n_samples,
            seed=seed,
            source=dataset,
            default_question=LONGFORM_PROMPT,
        )

    if dataset == "detail":
        return _hf_candidates(
            [
                ("lmms-lab/LLaVA-Bench-In-the-Wild", None, "train", None),
                ("liuhaotian/llava-bench-in-the-wild", None, "train", None),
            ],
            n_samples=n_samples,
            seed=seed,
            source="llava_bench_detail",
            default_question=DEFAULT_PROMPT,
            filter_fn=lambda raw: raw.get("category") in (None, "detail"),
        )

    if dataset == "detailcaps":
        return _hf_candidates(
            [
                ("foundation-multimodal-models/DetailCaps-4870", None, "test", None),
                ("foundation-multimodal-models/DetailCaps-4870", None, "train", None),
                ("foundation-multimodal-models/detail_caps-4870", None, "test", None),
                ("foundation-multimodal-models/detail_caps-4870", None, "train", None),
            ],
            n_samples=n_samples,
            seed=seed,
            source="detailcaps",
            default_question=LONGFORM_PROMPT,
        )

    if dataset == "docci":
        return _hf_candidates(
            [
                ("google/docci", None, "test_pic", None),
                ("google/docci", None, "train", "refs/convert/parquet"),
                ("google/docci", None, "validation", "refs/convert/parquet"),
            ],
            n_samples=n_samples,
            seed=seed,
            source="docci",
            default_question=LONGFORM_PROMPT,
        )

    if dataset == "localized_narratives":
        raise DatasetLoaderError(
            "Localized Narratives requires a local JSONL/images export in the current "
            "HF datasets environment. Prepare rows with image_path and caption/text/reference, "
            "then pass --data-path."
        )

    if dataset == "vqaonline":
        raise DatasetLoaderError(
            "VQAonline's public HF annotations expose image filenames but not "
            "loadable image content in this environment. Prepare local images as "
            "JSONL rows with image_path, question, and answer/reference, then pass --data-path."
        )

    if dataset == "stanford_paragraph":
        raise DatasetLoaderError(
            "Stanford Image Paragraph is not packaged consistently on HF. "
            "Prepare a local JSONL with image_path and paragraph/reference, then pass --data-path."
        )

    if dataset == "vizwiz_lf":
        raise DatasetLoaderError(
            "VizWiz-LF needs its released annotations/images locally. "
            "Prepare a JSONL with image_path, question, and long_answer/reference, then pass --data-path."
        )

    raise AssertionError(dataset)


def write_samples_jsonl(samples: Sequence[Dict[str, Any]], output_jsonl: str, image_dir: str) -> None:
    os.makedirs(os.path.abspath(os.path.expanduser(image_dir)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(os.path.expanduser(output_jsonl))), exist_ok=True)
    image_dir_abs = os.path.abspath(os.path.expanduser(image_dir))
    output_abs = os.path.abspath(os.path.expanduser(output_jsonl))
    with open(output_abs, "w") as f:
        for idx, sample in enumerate(samples):
            image_name = f"{idx:06d}.jpg"
            image_path = os.path.join(image_dir_abs, image_name)
            sample["image"].save(image_path, quality=95)
            row = {
                "id": sample.get("sample_id") or f"{sample.get('source', 'sample')}-{idx}",
                "source": sample.get("source"),
                "image_path": os.path.relpath(image_path, os.path.dirname(output_abs)),
                "question": sample.get("question") or DEFAULT_PROMPT,
                "reference": sample.get("reference"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Export long-form VLM samples to normalized JSONL.")
    parser.add_argument("--dataset", choices=DATASET_NAMES, required=True)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--hf-name", default=None)
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--hf-split", default=None)
    parser.add_argument("--hf-revision", default=None)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--image-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_longform_samples(
        args.dataset,
        n_samples=args.n_samples,
        seed=args.seed,
        data_path=args.data_path,
        hf_name=args.hf_name,
        hf_config=args.hf_config,
        hf_split=args.hf_split,
        hf_revision=args.hf_revision,
    )
    write_samples_jsonl(samples, args.output_jsonl, args.image_dir)
    print(json.dumps({
        "dataset": args.dataset,
        "n_samples": len(samples),
        "output_jsonl": os.path.abspath(os.path.expanduser(args.output_jsonl)),
        "image_dir": os.path.abspath(os.path.expanduser(args.image_dir)),
    }, indent=2))


if __name__ == "__main__":
    main()
