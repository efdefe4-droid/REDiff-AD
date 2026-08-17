import argparse
import csv
import os
from pathlib import Path

import torch
from torch_fidelity import calculate_metrics
from torch_fidelity.metric_isc import KEY_METRIC_ISC_MEAN, KEY_METRIC_ISC_STD


DEFAULT_CATEGORIES = [
    "capsule",
    "bottle",
    "carpet",
    "leather",
    "pill",
    "transistor",
    "tile",
    "cable",
    "zipper",
    "toothbrush",
    "metal_nut",
    "hazelnut",
    "screw",
    "grid",
    "wood",
]
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def split_list(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def has_images(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    return any(item.is_file() and item.suffix.lower() in IMAGE_EXTS for item in folder.iterdir())


def resolve_generated_image_dir(base: Path, category: str, defect: str) -> Path | None:
    candidates = [
        base / category / defect / "image",
        base / category / defect,
        base / defect / "image",
        base / defect,
    ]
    for candidate in candidates:
        if has_images(candidate):
            return candidate
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def run_fidelity_isc(image_dir: str, args) -> tuple[float, float]:
    use_cuda = False
    if not args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            print("CUDA is not available to torch-fidelity; falling back to CPU for IS.", flush=True)

    image_count = len([item for item in Path(image_dir).iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTS])
    effective_splits = max(1, min(args.isc_splits, image_count))
    print(f"ISC splits: requested={args.isc_splits} effective={effective_splits} images={image_count}", flush=True)
    metrics = calculate_metrics(
        input1=image_dir,
        isc=True,
        cuda=use_cuda,
        batch_size=args.batch_size,
        isc_splits=effective_splits,
        cache=False,
        verbose=True,
    )
    return float(metrics[KEY_METRIC_ISC_MEAN]), float(metrics[KEY_METRIC_ISC_STD])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_path", "--generated-path", type=str, default="generate_data_dir")
    parser.add_argument("--categories", type=str, default=None)
    parser.add_argument("--defects", type=str, default=None)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=64)
    parser.add_argument("--isc-splits", "--isc_splits", type=int, default=10)
    parser.add_argument("--output_csv", "--output-csv", type=str, default="IS.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    categories = split_list(args.categories) or DEFAULT_CATEGORIES
    selected_defects = split_list(args.defects)
    base = Path(args.generated_path)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    for category in categories:
        category_dir = base / category
        if selected_defects is not None:
            defects = selected_defects
        elif category_dir.is_dir():
            defects = sorted(item.name for item in category_dir.iterdir() if item.is_dir())
        elif len(categories) == 1 and base.is_dir():
            defects = sorted(item.name for item in base.iterdir() if item.is_dir())
        else:
            continue

        scores = []
        for defect in defects:
            image_dir = resolve_generated_image_dir(base, category, defect)
            if image_dir is None or not image_dir.is_dir():
                continue

            print(category, defect, flush=True)
            mean, std = run_fidelity_isc(str(image_dir), args)
            print(f"Inception Score {category}/{defect}: {mean:.7g} +/- {std:.7g}", flush=True)
            scores.append(mean)

        if not scores:
            continue

        with output_path.open("a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([category, str(float(sum(scores) / len(scores)))])


if __name__ == "__main__":
    main()
