import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import logging
import argparse
import csv
import zlib
from scipy.linalg import sqrtm

parser = argparse.ArgumentParser()
parser.add_argument("--real_path", "--real-path", type=str, default="/data1/Shared/Data/mvtec_anomaly_detection", help="Path to the real image dataset")
parser.add_argument("--generated_path", "--generated-path", type=str, default="./MCA-Ctrl/without_AGO_DAE", help="Path to the generated image dataset")
parser.add_argument("--categories", type=str, default=None, help="Comma-separated category list. Default: original MVTec list.")
parser.add_argument("--defects", type=str, default=None, help="Comma-separated defect list. Default: folders under each category.")
parser.add_argument("--kid-subsample-size", "--kid_subsample_size", type=int, default=50)
parser.add_argument("--kid-num-subsets", "--kid_num_subsets", type=int, default=200)
parser.add_argument("--kid-seed", "--kid_seed", type=int, default=2026)
parser.add_argument("--kid-batch-size", "--kid_batch_size", type=int, default=32)
parser.add_argument("--output_csv", "--output-csv", type=str, default=None)

args = parser.parse_args()

logging.basicConfig(filename=f'{args.generated_path}_kid_score_log.txt',
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InceptionV3FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(InceptionV3FeatureExtractor, self).__init__()
        self.model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.model.eval()

    def forward(self, x):
        x = self.model.Conv2d_1a_3x3(x)
        x = self.model.Conv2d_2a_3x3(x)
        x = self.model.Conv2d_2b_3x3(x)
        x = self.model.maxpool1(x)
        x = self.model.Conv2d_3b_1x1(x)
        x = self.model.Conv2d_4a_3x3(x)
        x = self.model.maxpool2(x)
        x = self.model.Mixed_5b(x)
        x = self.model.Mixed_5c(x)
        x = self.model.Mixed_5d(x)
        x = self.model.Mixed_6a(x)
        x = self.model.Mixed_6b(x)
        x = self.model.Mixed_6c(x)
        x = self.model.Mixed_6d(x)
        x = self.model.Mixed_6e(x)
        x = self.model.Mixed_7a(x)
        x = self.model.Mixed_7b(x)
        x = self.model.Mixed_7c(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x

def split_list(value):
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]

def get_image_paths_from_folder(folder):
    exts = ['.png', '.jpg', '.jpeg']
    image_paths = []
    if not os.path.isdir(folder):
        return image_paths
    for file in sorted(os.listdir(folder)):
        if any(file.lower().endswith(ext) for ext in exts):
            img_path = os.path.join(folder, file)
            if os.path.isfile(img_path):
                image_paths.append(img_path)
    return image_paths


def resolve_generated_class_folder(generated_folder_path, category, defect_class):
    candidates = [
        os.path.join(generated_folder_path, category, defect_class),
        os.path.join(generated_folder_path, defect_class),
        os.path.join(generated_folder_path, category, defect_class, 'image'),
        os.path.join(generated_folder_path, defect_class, 'image'),
    ]
    for candidate in candidates:
        if get_image_paths_from_folder(candidate):
            return candidate
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]

def get_activations(image_paths, model, batch_size=32):
    activations = []
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img = transform(img)
                    batch_images.append(img)
                except Exception as e:
                    print(f"Image loading error {path}: {e}")

            if batch_images:
                batch_images = torch.stack(batch_images).to(device)
                batch_activations = model(batch_images)
                activations.append(batch_activations.cpu().numpy())

    return np.concatenate(activations, axis=0) if activations else np.array([])

def stable_seed(base_seed, *parts):
    key = "/".join(str(part) for part in parts).encode("utf-8")
    return (int(base_seed) + zlib.crc32(key)) % (2**32)


def polynomial_kernel_matrix(x_feats, y_feats):
    feature_dim = x_feats.shape[1]
    return (1.0 + np.matmul(x_feats, y_feats.T) / feature_dim) ** 3


def mmd_squared(x_feats, y_feats):
    n = len(x_feats)
    m = len(y_feats)
    k_xx = polynomial_kernel_matrix(x_feats, x_feats)
    k_yy = polynomial_kernel_matrix(y_feats, y_feats)
    k_xy = polynomial_kernel_matrix(x_feats, y_feats)
    mean_xx = (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
    mean_yy = (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
    return mean_xx + mean_yy - 2.0 * k_xy.mean()


def calculate_kid(activations_real, activations_generated, subsample_size, num_subsets, rng):

    if activations_real.size == 0 or activations_generated.size == 0:
        return None

    kid_scores = []
    n_real = activations_real.shape[0]
    n_gen = activations_generated.shape[0]
    effective_subsample_size = min(max(2, subsample_size), n_real, n_gen)

    if effective_subsample_size < 2:
        return None

    for _ in range(num_subsets):
        idx_real = rng.choice(n_real, effective_subsample_size, replace=False)
        idx_gen = rng.choice(n_gen, effective_subsample_size, replace=False)
        x_feats = activations_real[idx_real]
        y_feats = activations_generated[idx_gen]
        kid_scores.append(mmd_squared(x_feats, y_feats))

    scores = np.asarray(kid_scores, dtype=np.float64)
    std = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
    sem = float(std / np.sqrt(len(scores))) if len(scores) > 1 else 0.0
    return {
        "mean": float(scores.mean()),
        "std": std,
        "sem": sem,
        "subsample_size": effective_subsample_size,
        "num_subsets": int(num_subsets),
    }

def get_categories_and_classes(folder_path):
    categories = split_list(args.categories) or [
        'bottle','cable','capsule','carpet','grid','hazelnut','leather',
        'metal_nut','pill','screw','tile','toothbrush','transistor',
        'wood','zipper'
    ]
    selected_defects = split_list(args.defects)
    category_class_dict = {}
    for category in categories:
        category_path = os.path.join(folder_path, category)
        if selected_defects is not None:
            category_class_dict[category] = selected_defects
            continue
        if not os.path.isdir(category_path):
            if len(categories) == 1 and os.path.isdir(folder_path):
                defect_classes = [d for d in os.listdir(folder_path)
                                  if os.path.isdir(os.path.join(folder_path, d))]
            else:
                defect_classes = []
            category_class_dict[category] = sorted(defect_classes)
            continue
        defect_classes = [d for d in os.listdir(category_path)
                          if os.path.isdir(os.path.join(category_path, d))]
        category_class_dict[category] = defect_classes
    return category_class_dict

def main():
    real_folder_path = args.real_path
    generated_folder_path = args.generated_path
    category_class_dict = get_categories_and_classes(generated_folder_path)
    model = InceptionV3FeatureExtractor().to(device)
    category_kid_scores = {}
    csv_rows = []

    for category, defect_classes in category_class_dict.items():
        logging.info(f"Processing category: {category}")
        print(f"Processing category: {category}")
        class_kid_scores = []

        for defect_class in defect_classes:
            real_class_folder = os.path.join(real_folder_path, category, 'test', defect_class)
            generated_class_folder = resolve_generated_class_folder(generated_folder_path, category, defect_class)
            real_image_paths = get_image_paths_from_folder(real_class_folder)
            generated_image_paths = get_image_paths_from_folder(generated_class_folder)

            if real_image_paths and generated_image_paths:
                act_real = get_activations(real_image_paths, model, batch_size=args.kid_batch_size)
                act_generated = get_activations(generated_image_paths, model, batch_size=args.kid_batch_size)

                if act_real.size == 0 or act_generated.size == 0:
                    logging.warning(f"No valid activations for {category}/{defect_class}.")
                    print(f"No valid activations for {category}/{defect_class}.")
                    continue

                rng = np.random.default_rng(stable_seed(args.kid_seed, category, defect_class))
                kid_stats = calculate_kid(
                    act_real,
                    act_generated,
                    subsample_size=args.kid_subsample_size,
                    num_subsets=args.kid_num_subsets,
                    rng=rng,
                )
                if kid_stats is not None:
                    kid_mean_x1000 = kid_stats["mean"] * 1000
                    kid_std_x1000 = kid_stats["std"] * 1000
                    kid_sem_x1000 = kid_stats["sem"] * 1000
                    class_kid_scores.append(kid_stats["mean"])
                    logging.info(
                        f"KID score for {category}/{defect_class}: "
                        f"{kid_mean_x1000} +/- {kid_std_x1000} std"
                    )
                    print(
                        f"KID score for {category}/{defect_class}: "
                        f"{kid_mean_x1000} +/- {kid_std_x1000} std "
                        f"(sem {kid_sem_x1000}, n={kid_stats['num_subsets']}, "
                        f"subsample={kid_stats['subsample_size']})"
                    )
                    csv_rows.append([
                        category,
                        defect_class,
                        kid_mean_x1000,
                        kid_std_x1000,
                        kid_sem_x1000,
                        len(real_image_paths),
                        len(generated_image_paths),
                        kid_stats["subsample_size"],
                        kid_stats["num_subsets"],
                        args.kid_seed,
                    ])
            else:
                logging.warning(f"No images found for {category}/{defect_class}.")
                print(f"No images found for {category}/{defect_class}.")

        if class_kid_scores:
            category_mean_kid = np.mean(class_kid_scores)
            category_kid_scores[category] = category_mean_kid * 1000
            print(f"Mean KID score for {category}: {category_mean_kid}")
            logging.info(f"Mean KID score for {category}: {category_mean_kid}")
        else:
            logging.warning(f"No valid images for category {category}.")
            print(f"No valid images for category {category}.")

    print("\nMean KID score per category:")
    logging.info("Mean KID score per category:")
    for category, kid in category_kid_scores.items():
        print(f"{category}: {kid}")
        logging.info(f"{category}: {kid}")

    if category_kid_scores:
        overall_mean_kid = np.mean(list(category_kid_scores.values()))
        print(f"\nOverall mean KID score: {overall_mean_kid}")
        logging.info(f"Overall mean KID score: {overall_mean_kid}")
        csv_rows.append(["overall", "mean", overall_mean_kid, "", "", "", "", "", "", args.kid_seed])

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "category",
                "defect",
                "kid_x1000",
                "kid_std_x1000",
                "kid_sem_x1000",
                "n_real",
                "n_generated",
                "kid_subsample_size",
                "kid_num_subsets",
                "kid_seed",
            ])
            writer.writerows(csv_rows)

if __name__ == "__main__":
    main()
