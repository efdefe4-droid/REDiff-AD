import argparse
import csv
import random
from pathlib import Path

import lpips
import torch
from PIL import Image
from torchvision import transforms


parser = argparse.ArgumentParser()
parser.add_argument("--real_path", "--real-path", type=str, default="/mvtec_anomaly_detection")
parser.add_argument("--generated_path", "--generated-path", type=str, default="/generate_data")
parser.add_argument("--categories", type=str, default=None)
parser.add_argument("--defects", type=str, default=None)
parser.add_argument("--cluster-size", "--cluster_size", type=int, default=50)
parser.add_argument("--lpips-batch-size", "--lpips_batch_size", type=int, default=16)
parser.add_argument("--lpips-pair-batch-size", "--lpips_pair_batch_size", type=int, default=16)
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--no-cpu-fallback", action="store_true")
parser.add_argument("--output_csv", "--output-csv", type=str, default="test.csv")
args = parser.parse_args()

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
device = "cuda" if torch.cuda.is_available() else "cpu"
lpips_fn = lpips.LPIPS(net="vgg").to(device)
lpips_fn.eval()
lpips_fn_cpu = None
preprocess = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def split_list(value):
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def list_images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(
        item for item in folder.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTS
    )


def has_images(folder):
    return bool(list_images(folder))


def resolve_generated_image_dir(base, category, defect):
    base = Path(base)
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


def load_tensor(path):
    image = Image.open(path).convert("RGB")
    return preprocess(image)


def chunks(items, size):
    size = max(1, size)
    for start in range(0, len(items), size):
        yield items[start:start + size]


def load_reference_tensors(real_dir):
    real_paths = list_images(real_dir)
    if not real_paths:
        return None

    # Keep the original script's len(real)/3 reference sampling rule, but fall
    # back to sorted filenames if 000.png, 001.png, ... are not available.
    reference_count = max(1, len(real_paths) // 3)
    selected_paths = []
    for idx in range(min(reference_count, len(real_paths))):
        numbered = Path(real_dir) / f"{idx:03d}.png"
        selected_paths.append(numbered if numbered.is_file() else real_paths[idx])

    return torch.stack([load_tensor(path) for path in selected_paths]).to(device)


def lpips_distances_once(batch_a, batch_b):
    with torch.no_grad():
        return lpips_fn(batch_a, batch_b).reshape(-1).detach().cpu()


def lpips_distances_cpu(batch_a, batch_b):
    global lpips_fn_cpu
    if lpips_fn_cpu is None:
        print("[oom-fallback] loading LPIPS on CPU", flush=True)
        lpips_fn_cpu = lpips.LPIPS(net="vgg").cpu()
        lpips_fn_cpu.eval()
    with torch.no_grad():
        return lpips_fn_cpu(batch_a.detach().cpu(), batch_b.detach().cpu()).reshape(-1).detach().cpu()


def lpips_distances_chunked(batch_a, batch_b):
    outputs = []
    total = batch_a.shape[0]
    start = 0
    default_chunk_size = max(1, args.lpips_pair_batch_size)
    chunk_size = min(default_chunk_size, total)

    while start < total:
        end = min(start + chunk_size, total)
        try:
            outputs.append(lpips_distances_once(batch_a[start:end], batch_b[start:end]))
            start = end
            chunk_size = min(default_chunk_size, total - start) if start < total else default_chunk_size
        except torch.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if chunk_size > 1:
                next_chunk_size = max(1, chunk_size // 2)
                print(
                    f"[oom-retry] reduce LPIPS pair batch {chunk_size} -> {next_chunk_size}",
                    flush=True,
                )
                chunk_size = next_chunk_size
                continue
            if args.no_cpu_fallback:
                raise
            print("[oom-fallback] GPU OOM at pair batch 1; computing this pair on CPU", flush=True)
            outputs.append(lpips_distances_cpu(batch_a[start:end], batch_b[start:end]))
            start = end
            chunk_size = min(default_chunk_size, total - start) if start < total else default_chunk_size

    return torch.cat(outputs)


def assign_clusters(generated_paths, reference_tensors):
    clusters = [[] for _ in range(reference_tensors.shape[0])]
    ref_count = reference_tensors.shape[0]

    with torch.no_grad():
        for batch_paths in chunks(generated_paths, args.lpips_batch_size):
            generated_tensors = torch.stack([load_tensor(path) for path in batch_paths]).to(device)
            batch_size = generated_tensors.shape[0]
            repeated_refs = reference_tensors.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            repeated_gens = generated_tensors.unsqueeze(1).expand(-1, ref_count, -1, -1, -1)
            distances = lpips_distances_chunked(
                repeated_refs.reshape(batch_size * ref_count, *reference_tensors.shape[1:]),
                repeated_gens.reshape(batch_size * ref_count, *generated_tensors.shape[1:]),
            ).reshape(batch_size, ref_count)
            nearest = distances.argmin(dim=1).detach().cpu().tolist()
            for path, cluster_idx in zip(batch_paths, nearest):
                clusters[cluster_idx].append(path)

    return clusters


def mean_pairwise_lpips(paths):
    if len(paths) < 2:
        return None

    tensors = torch.stack([load_tensor(path) for path in paths]).to(device)
    distances = []
    pair_a = []
    pair_b = []

    def flush_pairs():
        if not pair_a:
            return
        with torch.no_grad():
            batch_a = torch.stack(pair_a)
            batch_b = torch.stack(pair_b)
            distances.append(lpips_distances_chunked(batch_a, batch_b))
        pair_a.clear()
        pair_b.clear()

    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            pair_a.append(tensors[i])
            pair_b.append(tensors[j])
            if len(pair_a) >= args.lpips_batch_size:
                flush_pairs()
    flush_pairs()

    if not distances:
        return None
    return torch.cat(distances).mean().to(device)


def ic_lpips(sample_name, anomaly_name):
    print(sample_name, anomaly_name, flush=True)
    generated_dir = resolve_generated_image_dir(args.generated_path, sample_name, anomaly_name)
    if generated_dir is None:
        print(f"No generated images found for {sample_name}/{anomaly_name} under {args.generated_path}", flush=True)
        return None

    real_dir = Path(args.real_path) / sample_name / "test" / anomaly_name
    reference_tensors = load_reference_tensors(real_dir)
    if reference_tensors is None:
        print(f"No real images found for {sample_name}/{anomaly_name} under {real_dir}", flush=True)
        return None

    generated_paths = list_images(generated_dir)
    if not generated_paths:
        return None

    clusters = assign_clusters(generated_paths, reference_tensors)
    cluster_scores = []
    for idx, cluster_paths in enumerate(clusters):
        print(idx, flush=True)
        cluster_rng = random.Random(f"{args.seed}:{sample_name}:{anomaly_name}:{idx}")
        cluster_rng.shuffle(cluster_paths)
        selected_paths = cluster_paths[:args.cluster_size]
        score = mean_pairwise_lpips(selected_paths)
        if score is not None and not torch.isnan(score):
            cluster_scores.append(score)

    if not cluster_scores:
        return None
    stacked_scores = torch.stack(cluster_scores)
    mean_score = stacked_scores.mean()
    if len(cluster_scores) > 1:
        std_score = stacked_scores.std(unbiased=True)
    else:
        std_score = torch.zeros_like(mean_score)
    return {
        "mean": mean_score,
        "std": std_score,
        "n_clusters": len(cluster_scores),
    }


if __name__ == "__main__":
    sample_names = split_list(args.categories) or [
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
    selected_defects = split_list(args.defects)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    for sample_name in sample_names:
        sample_dir = Path(args.generated_path) / sample_name
        if selected_defects is not None:
            anomaly_names = selected_defects
        elif sample_dir.is_dir():
            anomaly_names = [item.name for item in sample_dir.iterdir() if item.is_dir()]
        elif len(sample_names) == 1 and Path(args.generated_path).is_dir():
            anomaly_names = [item.name for item in Path(args.generated_path).iterdir() if item.is_dir()]
        else:
            continue

        score_stats = []
        for anomaly_name in anomaly_names:
            stats = ic_lpips(sample_name, anomaly_name)
            if stats is not None:
                score_stats.append(stats)

        if not score_stats:
            continue
        means = torch.stack([stats["mean"] for stats in score_stats])
        mean_score = float(means.mean().detach().cpu())
        if len(score_stats) == 1:
            std_score = float(score_stats[0]["std"].detach().cpu())
        else:
            std_score = float(means.std(unbiased=True).detach().cpu())
        print("IC-LPIPS %s: %s +/- %s" % (sample_name, mean_score, std_score), flush=True)
        with output_path.open("a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([sample_name, str(mean_score), str(std_score)])
