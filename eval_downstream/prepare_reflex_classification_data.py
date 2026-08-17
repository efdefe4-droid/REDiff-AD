#!/usr/bin/env python
import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


FALLBACK_MASK_NAMES = (
    "generated_target_mask.png",
    "coarse_mask.png",
    "soft_mask.png",
    "target_mask_ca.png",
    "target_mask.png",
    "mask.png",
    "target_attention_map.png",
)


def numeric_key(path):
    name = path.name
    if name.isdigit():
        return (0, int(name))
    return (1, name)


def nested_numeric_key(path):
    parts = path.parts
    key = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part))
    return key


def read_mask(mask_path, image_size):
    mask = Image.open(mask_path).convert("L")
    if mask.size != image_size:
        mask = mask.resize(image_size, Image.Resampling.NEAREST)
    return mask


def find_mask(sample_dir, image_size, mask_name=None, require_mask_file=False):
    if mask_name:
        mask_path = sample_dir / mask_name
        if mask_path.exists():
            return read_mask(mask_path, image_size), mask_path
        if require_mask_file:
            raise FileNotFoundError(f"missing required mask: {mask_path}")
        return Image.new("L", image_size, color=0), None

    for name in FALLBACK_MASK_NAMES:
        mask_path = sample_dir / name
        if mask_path.exists():
            return read_mask(mask_path, image_size), mask_path

    if require_mask_file:
        raise FileNotFoundError(f"missing mask in {sample_dir}; candidates={FALLBACK_MASK_NAMES}")
    return Image.new("L", image_size, color=0), None


def mask_area_ratio(mask):
    arr = np.asarray(mask)
    return float((arr > 0).mean())


def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def link_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def image_size(path):
    with Image.open(path) as image:
        return image.size

def reflex_sample_dirs(args, anomaly):
    src_run_dir = (
        args.reflex_results_root
        / args.sample_name
        / f"{args.source_prefix}{anomaly}"
        / args.run_name
    )
    if not src_run_dir.exists():
        message = f"missing ReFlex run dir: {src_run_dir}"
        if args.skip_missing:
            print(f"[skip] {message}")
            return []
        raise FileNotFoundError(message)
    return sorted([path for path in src_run_dir.iterdir() if path.is_dir()], key=numeric_key)


def insert_anything_sample_dirs(args, anomaly):
    anomaly_dir = args.insert_anything_results_root / anomaly
    if not anomaly_dir.exists():
        message = f"missing Insert-Anything anomaly dir: {anomaly_dir}"
        if args.skip_missing:
            print(f"[skip] {message}")
            return []
        raise FileNotFoundError(message)

    direct_sample_dirs = [path for path in anomaly_dir.iterdir() if path.is_dir() and (path / args.image_name).exists()]
    if direct_sample_dirs:
        return sorted(direct_sample_dirs, key=numeric_key)

    nested_sample_dirs = []
    for ref_dir in sorted([path for path in anomaly_dir.iterdir() if path.is_dir()], key=numeric_key):
        for sample_dir in sorted([path for path in ref_dir.iterdir() if path.is_dir()], key=numeric_key):
            if (sample_dir / args.image_name).exists():
                nested_sample_dirs.append(sample_dir)
    return sorted(nested_sample_dirs, key=lambda path: nested_numeric_key(path.relative_to(anomaly_dir)))


def anomaly_diffusion_key(path):
    stem = path.stem
    prefix = stem.split("_", 1)[0]
    try:
        return (0, int(prefix))
    except ValueError:
        return (1, stem)


def find_anomaly_diffusion_mask(mask_dir, image_path):
    stem = image_path.stem
    prefix = stem.split("_", 1)[0]
    candidates = []
    if stem.startswith("gen_ano_"):
        mask_stem = stem.replace("gen_ano_", "gen_mask_", 1)
        candidates.extend([
            mask_dir / f"{mask_stem}.jpg",
            mask_dir / f"{mask_stem}.png",
            mask_dir / f"{mask_stem}.jpeg",
        ])
    try:
        idx = int(prefix)
        candidates.extend([
            mask_dir / f"{idx}.jpg",
            mask_dir / f"{idx}.png",
            mask_dir / f"{idx:03d}.jpg",
            mask_dir / f"{idx:03d}.png",
        ])
    except ValueError:
        pass
    candidates.extend([
        mask_dir / f"{prefix}.jpg",
        mask_dir / f"{prefix}.png",
        mask_dir / f"{stem}.jpg",
        mask_dir / f"{stem}.png",
    ])
    return next((path for path in candidates if path.exists()), None)


def anomaly_diffusion_sample_records(args, anomaly):
    anomaly_dir = args.insert_anything_results_root / anomaly
    image_dir = anomaly_dir / "image"
    mask_dir = anomaly_dir / "mask"
    if not image_dir.exists():
        message = f"missing AnomalyDiffusion image dir: {image_dir}"
        if args.skip_missing:
            print(f"[skip] {message}")
            return []
        raise FileNotFoundError(message)

    image_paths = sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ],
        key=anomaly_diffusion_key,
    )
    records = []
    for image_path in image_paths:
        mask_path = find_anomaly_diffusion_mask(mask_dir, image_path)
        if mask_path is None and args.require_mask_file:
            message = f"missing AnomalyDiffusion mask for image: {image_path}"
            if args.skip_missing:
                print(f"[skip] {message}")
                continue
            raise FileNotFoundError(message)
        records.append(
            {
                "source_dir": image_path.parent,
                "source_id": image_path.stem,
                "image": image_path,
                "mask": mask_path,
            }
        )
    return records


def find_o2mag_flat_mask(image_path):
    stem = image_path.stem
    if stem.endswith("_edit"):
        base = stem.removesuffix("_edit")
        candidates = [
            image_path.with_name(f"{base}_mask.png"),
            image_path.with_name(f"{base}_mask.jpg"),
            image_path.with_name(f"{base}_mask.jpeg"),
        ]
    else:
        candidates = [
            image_path.with_name(f"{stem}_mask.png"),
            image_path.with_name(f"{stem}_mask.jpg"),
            image_path.with_name(f"{stem}_mask.jpeg"),
        ]
    return next((path for path in candidates if path.exists()), None)


def o2mag_flat_sample_records(args, anomaly):
    anomaly_dir = args.insert_anything_results_root / anomaly
    if not anomaly_dir.exists():
        message = f"missing O2MAG anomaly dir: {anomaly_dir}"
        if args.skip_missing:
            print(f"[skip] {message}")
            return []
        raise FileNotFoundError(message)

    image_paths = sorted(
        [
            path
            for path in anomaly_dir.iterdir()
            if path.is_file()
            and path.stem.endswith("_edit")
            and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ],
        key=anomaly_diffusion_key,
    )
    records = []
    for image_path in image_paths:
        mask_path = find_o2mag_flat_mask(image_path)
        if mask_path is None and args.require_mask_file:
            message = f"missing O2MAG mask for image: {image_path}"
            if args.skip_missing:
                print(f"[skip] {message}")
                continue
            raise FileNotFoundError(message)
        records.append(
            {
                "source_dir": image_path.parent,
                "source_id": image_path.stem,
                "image": image_path,
                "mask": mask_path,
            }
        )
    return records


def tf_idg_sample_records(args, anomaly):
    object_root = args.insert_anything_results_root
    image_dir = object_root / "test" / anomaly
    mask_dir = object_root / "ground_truth" / anomaly
    if not image_dir.exists():
        message = f"missing TF-IDG image dir: {image_dir}"
        if args.skip_missing:
            print(f"[skip] {message}")
            return []
        raise FileNotFoundError(message)

    image_paths = sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ],
        key=anomaly_diffusion_key,
    )
    records = []
    for image_path in image_paths:
        candidates = [
            mask_dir / f"{image_path.stem}_mask.png",
            mask_dir / f"{image_path.stem}_mask.jpg",
            mask_dir / f"{image_path.stem}_mask.jpeg",
        ]
        mask_path = next((path for path in candidates if path.exists()), None)
        if mask_path is None and args.require_mask_file:
            message = f"missing TF-IDG mask for image: {image_path}"
            if args.skip_missing:
                print(f"[skip] {message}")
                continue
            raise FileNotFoundError(message)
        records.append(
            {
                "source_dir": image_path.parent,
                "source_id": image_path.stem,
                "image": image_path,
                "mask": mask_path,
            }
        )
    return records


def list_source_sample_dirs(args, anomaly):
    if args.input_layout == "insert-anything":
        sample_dirs = insert_anything_sample_dirs(args, anomaly)
    elif args.input_layout in {"anomaly-diffusion", "seas", "anostyle", "dualanodiff", "self-anomalydiffusion"}:
        sample_dirs = anomaly_diffusion_sample_records(args, anomaly)
    elif args.input_layout == "o2mag-flat":
        sample_dirs = o2mag_flat_sample_records(args, anomaly)
    elif args.input_layout == "tf-idg":
        sample_dirs = tf_idg_sample_records(args, anomaly)
    else:
        sample_dirs = reflex_sample_dirs(args, anomaly)
    if args.max_images is not None:
        sample_dirs = sample_dirs[: args.max_images]
    return sample_dirs


def convert_anomaly(args, anomaly):
    sample_dirs = list_source_sample_dirs(args, anomaly)

    dst_anomaly_dir = args.output_root / args.sample_name / anomaly
    if args.clean and dst_anomaly_dir.exists():
        shutil.rmtree(dst_anomaly_dir)
    dst_image_dir = dst_anomaly_dir / "image"
    dst_mask_dir = dst_anomaly_dir / "mask"
    dst_image_dir.mkdir(parents=True, exist_ok=True)
    dst_mask_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    out_idx = 0
    for source in sample_dirs:
        if isinstance(source, dict):
            sample_dir = source["source_dir"]
            source_id = source["source_id"]
            src_image = source["image"]
            fixed_src_mask = source["mask"]
        else:
            sample_dir = source
            source_id = sample_dir.name
            src_image = sample_dir / args.image_name
            fixed_src_mask = None
        if not src_image.exists():
            message = f"missing image: {src_image}"
            if args.skip_missing:
                print(f"[skip] {message}")
                continue
            raise FileNotFoundError(message)

        try:
            src_image_size = image_size(src_image)
            if fixed_src_mask is not None:
                mask = read_mask(fixed_src_mask, src_image_size)
                src_mask = fixed_src_mask
            else:
                mask, src_mask = find_mask(
                    sample_dir,
                    src_image_size,
                    mask_name=args.mask_name,
                    require_mask_file=args.require_mask_file,
                )
        except FileNotFoundError as exc:
            if args.skip_missing:
                print(f"[skip] {exc}")
                continue
            raise

        area = mask_area_ratio(mask)
        if area < args.min_mask_area_ratio or area > args.max_mask_area_ratio:
            print(f"[skip] mask area {area:.6f}: {sample_dir}")
            continue

        dst_image = dst_image_dir / f"{out_idx}.jpg"
        dst_mask = dst_mask_dir / f"{out_idx}.jpg"
        dst_localization_image = None

        image = None
        if args.link_files:
            link_file(src_image, dst_image)
            if src_mask is not None and image_size(src_mask) == src_image_size:
                link_file(src_mask, dst_mask)
            else:
                mask.save(dst_mask, quality=args.jpeg_quality)
        elif args.copy_raw:
            copy_file(src_image, dst_image)
            if src_mask is not None and image_size(src_mask) == src_image_size:
                copy_file(src_mask, dst_mask)
            else:
                mask.save(dst_mask, quality=args.jpeg_quality)
        else:
            image = Image.open(src_image).convert("RGB")
            image.save(dst_image, quality=args.jpeg_quality)
            mask.save(dst_mask, quality=args.jpeg_quality)

        if args.localization_compatible:
            dst_localization_image = dst_image_dir / f"{out_idx:03d}_{source_id}_triag.png"
            if args.link_files:
                link_file(src_image, dst_localization_image)
            elif args.copy_raw:
                copy_file(src_image, dst_localization_image)
            else:
                if image is None:
                    image = Image.open(src_image).convert("RGB")
                image.save(dst_localization_image)

        rows.append(
            {
                "sample_name": args.sample_name,
                "anomaly": anomaly,
                "index": out_idx,
                "source_dir": str(sample_dir),
                "source_image": str(src_image),
                "source_mask": str(src_mask) if src_mask is not None else "",
                "target_image": str(dst_image),
                "target_localization_image": str(dst_localization_image) if dst_localization_image else "",
                "target_mask": str(dst_mask),
                "mask_area_ratio": f"{area:.8f}",
            }
        )
        out_idx += 1

    print(f"[ok] {anomaly}: {out_idx} images -> {dst_anomaly_dir}")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Convert generated anomaly folders to eval_downstream image/mask layout."
    )
    parser.add_argument(
        "--input-layout",
        choices=[
            "reflex",
            "insert-anything",
            "anomaly-diffusion",
            "seas",
            "anostyle",
            "dualanodiff",
            "self-anomalydiffusion",
            "o2mag-flat",
            "tf-idg",
        ],
        default="reflex",
    )
    parser.add_argument(
        "--reflex-results-root",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument("--insert-anything-results-root", type=Path)
    parser.add_argument("--sample-name", type=str, required=True)
    parser.add_argument("--anomalies", nargs="+", required=True)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--source-prefix", type=str, default="diversity_")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("eval_downstream/generated_data/reflex_classification"),
    )
    parser.add_argument("--image-name", type=str, default="target.png")
    parser.add_argument("--mask-name", type=str, default=None)
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--min-mask-area-ratio", type=float, default=0.0)
    parser.add_argument("--max-mask-area-ratio", type=float, default=1.0)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--require-mask-file", action="store_true")
    parser.add_argument(
        "--copy-raw",
        action="store_true",
        help="Copy source image/mask bytes into the downstream layout instead of decoding and re-encoding.",
    )
    parser.add_argument(
        "--link-files",
        action="store_true",
        help="Symlink source image/mask files into the downstream layout instead of copying or re-encoding.",
    )
    parser.add_argument("--localization-compatible", dest="localization_compatible", action="store_true", default=True)
    parser.add_argument("--no-localization-compatible", dest="localization_compatible", action="store_false")
    args = parser.parse_args()

    args.reflex_results_root = args.reflex_results_root.expanduser().resolve()
    if args.insert_anything_results_root is not None:
        args.insert_anything_results_root = args.insert_anything_results_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()

    if args.input_layout != "reflex" and args.insert_anything_results_root is None:
        raise ValueError(f"--insert-anything-results-root is required for --input-layout {args.input_layout}")
    if args.input_layout == "reflex" and not args.run_name:
        raise ValueError("--run-name is required for --input-layout reflex")

    all_rows = []
    for anomaly in args.anomalies:
        all_rows.extend(convert_anomaly(args, anomaly))

    manifest_path = args.output_root / args.sample_name / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_name",
        "anomaly",
        "index",
        "source_dir",
        "source_image",
        "source_mask",
        "target_image",
        "target_localization_image",
        "target_mask",
        "mask_area_ratio",
    ]
    with open(manifest_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[done] wrote manifest: {manifest_path}")
    print(f"[done] total images: {len(all_rows)}")


if __name__ == "__main__":
    main()
