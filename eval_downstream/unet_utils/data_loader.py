import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
from pathlib import Path
import random
from PIL import Image
from torchvision import transforms
import math

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')


def _list_generated_anomaly_dirs(root_dir):
    return sorted([
        name
        for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name, 'image'))
        and os.path.isdir(os.path.join(root_dir, name, 'mask'))
    ])


def _numeric_file_sort_key(file_name):
    stem = os.path.splitext(file_name)[0]
    try:
        return int(stem[:3])
    except ValueError:
        return stem


def _list_image_files(folder):
    if not os.path.isdir(folder):
        return []
    files = [
        os.path.join(folder, file_name)
        for file_name in os.listdir(folder)
        if file_name.lower().endswith(IMAGE_EXTS)
    ]
    return sorted(files, key=lambda path: _numeric_file_sort_key(os.path.basename(path)))


def _list_good_files(mvtec_path, sample_name):
    train_good_path = os.path.join(mvtec_path, sample_name, 'train', 'good')
    good_files = _list_image_files(train_good_path)
    if good_files:
        return train_good_path, good_files

    source_path = os.path.join(mvtec_path, sample_name, 'source')
    source_files = []
    if os.path.isdir(source_path):
        for source_name in sorted(os.listdir(source_path)):
            source_files.extend(_list_image_files(os.path.join(source_path, source_name)))
        source_files = sorted(source_files, key=lambda path: _numeric_file_sort_key(os.path.basename(path)))
    if source_files:
        return source_path, source_files

    raise FileNotFoundError(
        f"Could not find normal images for {sample_name}. "
        f"Tried {train_good_path} and {source_path}."
    )


def _real_anomaly_pairs(mvtec_path, sample_name, anomaly):
    test_dir = os.path.join(mvtec_path, sample_name, 'test', anomaly)
    mask_dir = os.path.join(mvtec_path, sample_name, 'ground_truth', anomaly)
    image_files = _list_image_files(test_dir)
    if not image_files:
        return [], []

    selected_images = image_files[:max(1, len(image_files) // 3)]
    paired_images = []
    paired_masks = []
    for image_path in selected_images:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        mask_candidates = [
            os.path.join(mask_dir, f'{stem}_mask.png'),
            os.path.join(mask_dir, f'{stem}.png'),
            os.path.join(mask_dir, f'{stem}_mask.jpg'),
            os.path.join(mask_dir, f'{stem}.jpg'),
        ]
        mask_path = next((candidate for candidate in mask_candidates if os.path.isfile(candidate)), None)
        if mask_path is None:
            continue
        paired_images.append(image_path)
        paired_masks.append(mask_path)
    return paired_images, paired_masks


def _split_eval_files(files, split):
    if split == 'all':
        return files
    midpoint = len(files) // 2
    if split == 'val':
        return files[:midpoint]
    if split == 'test':
        return files[midpoint:]
    raise ValueError(f"Unsupported split: {split}")


class RandomAugmentation:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask):
        if random.random() < self.prob:
        
            width, height = image.size

            crop_x = random.randint(0, width // 16)
            crop_y = random.randint(0, height // 16)
            image = image.crop((crop_x, crop_y, width - crop_x, height - crop_y))
            mask = mask.crop((crop_x, crop_y, width - crop_x, height - crop_y))

            angle = random.uniform(-30, 30)
            image = image.rotate(angle, expand=True)
            mask = mask.rotate(angle, expand=True)

            dx = random.randint(-16, 16)
            dy = random.randint(-16, 16)
            translated_image = Image.new("RGB", (width, height), color="black")
            translated_mask = Image.new("L", (width, height), color=0)
            translated_image.paste(image, (dx, dy))
            translated_mask.paste(mask, (dx, dy))

            image, mask = translated_image, translated_mask

        return image, mask
        
class Real_RandomAugmentation:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask):
        if random.random() < self.prob:
        
            width, height = image.size
            
            crop_x = random.randint(0, width // 16)
            crop_y = random.randint(0, height // 16)
            image = image.crop((crop_x, crop_y, width - crop_x, height - crop_y))
            mask = mask.crop((crop_x, crop_y, width - crop_x, height - crop_y))

            angle = random.uniform(-30, 30)
            image = image.rotate(angle, expand=True)
            mask = mask.rotate(angle, expand=True)

            dx = random.randint(-16, 16)
            dy = random.randint(-16, 16)
            translated_image = Image.new("RGB", (width, height), color="black")
            translated_mask = Image.new("L", (width, height), color=0)
            translated_image.paste(image, (dx, dy))
            translated_mask.paste(mask, (dx, dy))

            image, mask = translated_image, translated_mask

        return image, mask


class MVTecDRAEMTestDataset_partial(Dataset):

    def __init__(self, root_dir, resize_shape=None, split='all'):
        if split not in ('all', 'val', 'test'):
            raise ValueError("split must be one of: all, val, test")
        self.root_dir = root_dir
        self.resize_shape = resize_shape
        self.images = []
        self.split = split
        self.anomaly_names = sorted(os.listdir(self.root_dir))
        for idx, anomaly_name in enumerate(self.anomaly_names):
            img_path = os.path.join(root_dir, anomaly_name)
            if not os.path.isdir(img_path):
                continue
            img_files = os.listdir(img_path)
            img_files.sort(key=_numeric_file_sort_key)
            if anomaly_name == 'good':
                selected_files = _split_eval_files(img_files, split)
            else:
                eval_files = img_files[len(img_files) // 3:]
                selected_files = _split_eval_files(eval_files, split)
            self.images += [os.path.join(img_path, file_name) for file_name in selected_files]

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx}

        return sample



class MVTec_Anomaly_Detection(Dataset):

    def __init__(self, args, sample_name, length=20000, anomaly_id=None, recon=False):
        self.recon = recon
        self.extra_real_anomaly_prob = getattr(args, 'extra_real_anomaly_prob', 0.1)
        self.good_path, self.good_files = _list_good_files(args.mvtec_path, sample_name)
        self.root_dir = '%s/%s' % (args.generated_data_path, sample_name)
        self.anomaly_names = _list_generated_anomaly_dirs(self.root_dir)
        # self.mask_dir = '%s/%s' % (args.mask_root, sample_name)

        # first 1/3 real MVTec-AD
        self.extra_anomaly_path = '%s/%s/ground_truth' % (args.mvtec_path, sample_name)
        self.extra_anomaly_names = sorted(os.listdir(self.extra_anomaly_path)) if os.path.isdir(self.extra_anomaly_path) else []

        self.augmentation = RandomAugmentation()
        self.real_augmentation = Real_RandomAugmentation()

        if anomaly_id is not None:
            self.anomaly_names = self.anomaly_names[anomaly_id:anomaly_id + 1]
            print('training subsets', self.anomaly_names)

        l = len(self.anomaly_names)
        self.anomaly_num = l
        self.img_paths = []
        self.mask_paths = []

        for idx, anomaly in enumerate(self.anomaly_names):
            img_path = []
            mask_path = []
            for i in range(min(len(os.listdir(os.path.join(self.root_dir, anomaly, 'mask'))), 1000)):
                # mask: 0.jpg / 1.jpg / ...
                # image: 000_003_triag.png / 001_007_triag.png / ...
                pattern = os.path.join(self.root_dir, anomaly, 'image', f'{i:03d}_*_triag.png')
                matched = sorted(glob.glob(pattern))
                if len(matched) == 0:
                    continue
                img_path.append(matched[0])
                mask_path.append(os.path.join(self.root_dir, anomaly, 'mask', f'{i}.jpg'))

            # Ablation experiment
            # new_img_dir = os.path.join(self.root_dir, anomaly)
            # new_msk_dir = os.path.join(self.mask_dir, anomaly)
            # for fname in os.listdir(new_img_dir):
            #     fl = fname.lower()
            #     if not fl.endswith(('.png', '.jpg', '.jpeg')):
            #         continue
            #     # 取文件名第一个下划线前的数字，如 '000_003_triag.png' -> '000' -> 0
            #     head = fname.split('_')[0]
            #     idx_num = int(head)
            #     msk_fp = None
            #     for ext in ('.png', '.jpg', '.jpeg'):
            #         cand = os.path.join(new_msk_dir, f"{idx_num}{ext}")
            #         if os.path.isfile(cand):
            #             msk_fp = cand; break
            #     if msk_fp is None:
            #         continue
            #     img_path.append(os.path.join(new_img_dir, fname))
            #     mask_path.append(msk_fp)

            self.img_paths.append(img_path.copy())
            self.mask_paths.append(mask_path.copy())

        extra_anomaly_names = []
        self.extra_img_paths = []
        self.extra_mask_paths = []

        for anomaly in self.extra_anomaly_names:
            img_path, mask_path = _real_anomaly_pairs(args.mvtec_path, sample_name, anomaly)
            if not img_path:
                continue
            extra_anomaly_names.append(anomaly)
            self.extra_img_paths.append(img_path)
            self.extra_mask_paths.append(mask_path)
        self.extra_anomaly_names = extra_anomaly_names

        for i in range(l):
            print(len(self.img_paths[i]), len(self.mask_paths[i]))

        for i in range(len(self.extra_anomaly_names)):
            print(len(self.extra_img_paths[i]), len(self.extra_mask_paths[i]))

        self.loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256, 256], antialias=True)
        ])

        self.length = length
        if self.length is None:
            self.length = len(self.good_files)

        #
        #self.image_save_dir = os.path.join(args.save_path_1, sample_name, 'image')
        #self.mask_save_dir = os.path.join(args.save_path_1, sample_name, 'mask')
        #os.makedirs(self.image_save_dir, exist_ok=True)
        #os.makedirs(self.mask_save_dir, exist_ok=True)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # --- 1. Define the probabilities for each branch here for easy configuration ---
        prob_good = 0.5
        configured_extra_prob = getattr(self, 'extra_real_anomaly_prob', 0.1)
        prob_extra_realanomaly = configured_extra_prob if self.extra_anomaly_names else 0.0
        # The probability for a regular anomaly is the remainder.
        
        # --- 2. Generate a single random number to determine which branch to take ---
        p = random.random()

        # --- 3. Use a clear if/elif/else structure based on the random number ---
        if p < prob_good:
            # Branch 1: Load a "good" (non-anomalous) sample.
            # This block is executed with a 50% probability (when p is in [0, 0.5)).
            
            image = self.loader(Image.open(self.good_files[idx % len(self.good_files)]).convert('RGB'))
            mask = torch.zeros((1, image.size(-2), image.size(-1)))
            has_anomaly = np.array([0], dtype=np.float32)
            sample = {'image': image, 'has_anomaly': has_anomaly, 'mask': mask, 'anomaly_id': -1}
            if self.recon:
                sample['source'] = image
                
        elif p < prob_good + prob_extra_realanomaly:
            # Branch 2: Load an "extra anomaly" sample.
            # This block is executed with a 10% probability (when p is in [0.5, 0.6)).
            
            anomaly_id = random.randint(0, len(self.extra_anomaly_names) - 1)
            img_path = self.extra_img_paths[anomaly_id][idx % len(self.extra_mask_paths[anomaly_id])]
            image = Image.open(img_path).convert('RGB')

            mask_path = self.extra_mask_paths[anomaly_id][idx % len(self.extra_mask_paths[anomaly_id])]
            mask = Image.open(mask_path).convert('L')

            image, mask = self.real_augmentation(image, mask)
            image = self.loader(image)
            mask = self.loader(mask)
            mask = (mask > 0.5).float()
            
            if mask.sum() == 0:
                has_anomaly = np.array([0], dtype=np.float32)
                anomaly_id = -1
            else:
                has_anomaly = np.array([1], dtype=np.float32)

            sample = {'image': image, 'has_anomaly': has_anomaly, 'mask': mask, 'anomaly_id': anomaly_id}

            if self.recon:
                img_path = img_path.replace('image', 'recon')
                ori_image = self.loader(Image.open(img_path).convert('RGB'))
                sample['source'] = ori_image

            # (The commented-out debugging code remains the same)
            #augmented_image_path = os.path.join(self.image_save_dir, f'augmented_extra_{idx}.jpg')
            # ...

        else:
            # Branch 3: Load a "regular anomaly" sample.
            # This block is executed with a 40% probability (when p is in [0.6, 1.0)).
            
            anomaly_id = random.randint(0, self.anomaly_num - 1)
            img_path = self.img_paths[anomaly_id][idx % len(self.mask_paths[anomaly_id])]
            image = Image.open(img_path).convert('RGB')

            mask_path = self.mask_paths[anomaly_id][idx % len(self.mask_paths[anomaly_id])]
            mask = Image.open(mask_path).convert('L')

            image, mask = self.augmentation(image, mask)
            image = self.loader(image)
            mask = self.loader(mask)
            mask = (mask > 0.5).float()

            if mask.sum() == 0:
                has_anomaly = np.array([0], dtype=np.float32)
                anomaly_id = -1
            else:
                has_anomaly = np.array([1], dtype=np.float32)

            sample = {'image': image, 'has_anomaly': has_anomaly, 'mask': mask,
                     'anomaly_id': anomaly_id}

            if self.recon:
                img_path = img_path.replace('image', 'recon')
                ori_image = self.loader(Image.open(img_path).convert('RGB'))
                sample['source'] = ori_image

            # (The commented-out debugging code remains the same)
            #augmented_image_path = os.path.join(self.image_save_dir, f'augmented_{idx}.jpg')
            # ...
            
        return sample


class MVTec_classification_test(Dataset):
    def __init__(self, args, sample_name, anomaly_names):
        root_dir = '%s/%s/test'%(args.mvtec_path, sample_name)
        self.anomaly_names=anomaly_names
        self.img_paths=[]
        self.labels=[]
        for idx, anomaly_name in enumerate(self.anomaly_names):
            img_path=os.path.join(root_dir, anomaly_name)
            img_files = os.listdir(img_path)
            img_files.sort(key=lambda x: int(x[:3]))
            l = len(img_files) // 3
            self.img_paths += [os.path.join(img_path, file_name) for file_name in img_files[l:]]
            self.labels+=[idx for file_name in img_files[l:]]
        self.loader=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256,256], antialias=True)
        ])
        self.length=len(self.img_paths)
    def __len__(self):
        return self.length
    def class_num(self):
        return len(self.anomaly_names)
    def __getitem__(self, idx):
        image=self.loader(Image.open(self.img_paths[idx%len(self.img_paths)]).convert('RGB'))
        label=self.labels[idx%len(self.img_paths)]
        return image, label

class MVTec_classification_train(Dataset):
    def __init__(self, args,sample_name):
        self.root_dir = '%s/%s'%(args.generated_data_path,sample_name)
        image_size = getattr(args, 'image_size', 256)
        self.train_repeat = getattr(args, 'train_repeat', 3)
        self.anomaly_names=_list_generated_anomaly_dirs(self.root_dir)
        self.img_paths=[]
        self.labels=[]
        for idx,anomaly in enumerate(self.anomaly_names):
            for i in range(min(len(os.listdir(os.path.join(self.root_dir,anomaly,'mask'))),500)):
                self.img_paths.append(os.path.join(self.root_dir,anomaly,'image','%d.jpg'%i))
                self.labels.append(idx)
        self.loader=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([image_size,image_size])
        ])
        self.length=len(self.img_paths)
    def __len__(self):
        return self.length*self.train_repeat
    def class_num(self):
        return len(self.anomaly_names)
    def return_anomaly_names(self):
        return self.anomaly_names
    def __getitem__(self, idx):
        image=self.loader(Image.open(self.img_paths[idx%len(self.img_paths)]).convert('RGB'))
        label=self.labels[idx%len(self.img_paths)]
        return image,label

class MVTec_classification_test(Dataset):
    def __init__(self, args,sample_name,anomaly_names):
        root_dir = '%s/%s/test'%(args.mvtec_path,sample_name)
        image_size = getattr(args, 'image_size', 256)
        test_split = getattr(args, 'test_split', 'all')
        self.anomaly_names=anomaly_names
        self.img_paths=[]
        self.labels=[]
        for idx, anomaly_name in enumerate(self.anomaly_names):
            img_path=os.path.join(root_dir,anomaly_name)
            if not os.path.isdir(img_path):
                available = sorted(
                    name
                    for name in os.listdir(root_dir)
                    if os.path.isdir(os.path.join(root_dir, name))
                ) if os.path.isdir(root_dir) else []
                raise FileNotFoundError(
                    f"Missing MVTec test directory: {img_path}. "
                    f"Check --mvtec_path. Available under {root_dir}: {available}"
                )
            img_files = os.listdir(img_path)
            img_files.sort(key=lambda x: int(x[:3]))
            if test_split == 'last_two_thirds':
                img_files = img_files[len(img_files) // 3:]
            self.img_paths += [os.path.join(img_path, file_name) for file_name in img_files]
            self.labels+=[idx for file_name in img_files]
        self.loader=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([image_size,image_size])
        ])
        self.length=len(self.img_paths)
    def __len__(self):
        return self.length
    def class_num(self):
        return len(self.anomaly_names)
    def __getitem__(self, idx):
        image=self.loader(Image.open(self.img_paths[idx%len(self.img_paths)]).convert('RGB'))
        label=self.labels[idx%len(self.img_paths)]
        return image,label
