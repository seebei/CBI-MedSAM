import os
join = os.path.join
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"  # 推荐方式，控制底层 C++ 输出
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
try:
    cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
except:
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except:
        pass
import albumentations as A
import math
import random

from torchvision import transforms
from monai.transforms import Resize, ToTensor
import glob

from typing import Tuple, Optional

from PIL import Image






def make_odd(num):
    num = math.ceil(num)
    if num % 2 == 0:
        num += 1
    return num

def apply_transform(image, mask):
    strategy = [(1, 2), (0, 3), (0, 2), (1, 1)]
    level = 5
    transform = A.Compose([
        A.ColorJitter(brightness=0.04 * level, contrast=0, saturation=0, hue=0, p=0.2 * level),
        A.ColorJitter(brightness=0, contrast=0.04 * level, saturation=0, hue=0, p=0.2 * level),
        A.Posterize(num_bits=math.floor(8 - 0.8 * level), p=0.2 * level),
        A.Sharpen(alpha=(0.04 * level, 0.1 * level), lightness=(1, 1), p=0.2 * level),
        A.GaussianBlur(blur_limit=(3, make_odd(3 + 0.8 * level)), p=0.2 * level),
        A.GaussNoise(var_limit=(2 * level, 10 * level), mean=0, per_channel=True, p=0.2 * level),
        A.Rotate(limit=4 * level, interpolation=1, border_mode=0, value=0, mask_value=None, rotate_method='largest_box',
                    crop_border=False, p=0.2 * level),
        A.HorizontalFlip(p=0.2 * level),
        A.VerticalFlip(p=0.2 * level),
        A.Affine(scale=(1 - 0.04 * level, 1 + 0.04 * level), translate_percent=None, translate_px=None, rotate=None,
                    shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
                    shear={'x': (0, 2 * level), 'y': (0, 0)}
                    , interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=0.2 * level),  # x
        A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
                    shear={'x': (0, 0), 'y': (0, 2 * level)}
                    , interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent={'x': (0, 0.02 * level), 'y': (0, 0)}, translate_px=None, rotate=None,
                    shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent={'x': (0, 0), 'y': (0, 0.02 * level)}, translate_px=None, rotate=None,
                    shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=0.2 * level)
    ])
    employ = random.choice(strategy)
    level, shape = random.sample(transform[:6], employ[0]), random.sample(transform[6:], employ[1])
    img_transform = A.Compose([*level, *shape])
    random.shuffle(img_transform.transforms)
    transformed = img_transform(image=image, mask=mask)
    return transformed['image'], transformed['mask']

class SessileDataLoader(Dataset):
    def __init__(self, data_root, train=True, resize_size=[384, 384], label_resize_size=[], train_ratio=1.0):
        self.train = train
        self.data_root = data_root
        self.resize_size = resize_size
        if len(label_resize_size) <= 0:
            self.label_resize_size = resize_size
        else:
            self.label_resize_size = label_resize_size

        self.img_files = []
        self.gt_files = []
        self.class_num=1
        self.image_size=resize_size[0]

        if train:
            self.scan_list = os.listdir(os.path.join(self.data_root,"train", "images"))
            self.scan_list.sort()
            self.scan_list = self.scan_list[:int(len(self.scan_list) * train_ratio)]
            for scan in self.scan_list:
                self.img_files.append(os.path.join(self.data_root,"train", "images", scan))
                self.gt_files.append(os.path.join(self.data_root,"train", "masks", scan))
        else:
            self.scan_list = os.listdir(os.path.join(self.data_root,"test", "images"))
            self.scan_list.sort()
            for scan in self.scan_list:
                self.img_files.append(os.path.join(self.data_root,"test", "images", scan))
                self.gt_files.append(os.path.join(self.data_root,"test", "masks", scan))
        
    
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img = cv2.imread(self.img_files[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.resize_size)

        gt2D = cv2.imread(self.gt_files[index], cv2.IMREAD_GRAYSCALE)
        gt2D = cv2.resize(gt2D, self.label_resize_size, cv2.INTER_NEAREST)

        if self.train:
            while True:
                img, gt2D = apply_transform(img, gt2D)
                if np.sum(gt2D) > 0:
                    break

        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        gt2D = gt2D.astype(np.uint8) / 255
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # add perturbation to bounding box coordinates
        if self.train:
            H, W = gt2D.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
        box = np.array([x_min, y_min, x_max, y_max])
        return torch.tensor(img).float(), torch.tensor(gt2D[None, :,:]).long(), torch.tensor(box).float(), torch.tensor(0).long(), self.img_files[index], self.gt_files[index]

class CVCDataLoader(Dataset):
    def __init__(self, data_root, train=True, resize_size=[384, 384], label_resize_size=[]):
        self.train = train
        self.data_root = os.path.join(data_root, "PNG")
        self.resize_size = resize_size
        if len(label_resize_size) <= 0:
            self.label_resize_size = resize_size
        else:
            self.label_resize_size = label_resize_size
        self.img_files = []
        self.gt_files = []
        self.class_num=1
        self.image_size=resize_size[0]

        scan_list = os.listdir(os.path.join(self.data_root, "Original"))
        scan_list.sort()
        if train:
            scan_list = scan_list[:]
        else:
            scan_list = scan_list[:]
        
        for scan in scan_list:
            self.img_files.append(os.path.join(self.data_root, "Original", scan))
            self.gt_files.append(os.path.join(self.data_root, "Ground Truth", scan))
    
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img = cv2.imread(self.img_files[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.resize_size)

        gt2D = cv2.imread(self.gt_files[index], cv2.IMREAD_GRAYSCALE)
        gt2D = cv2.resize(gt2D, self.label_resize_size, cv2.INTER_NEAREST)

        if self.train:
            while True:
                img, gt2D = apply_transform(img, gt2D)
                if np.sum(gt2D) > 0:
                    break

        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        gt2D = gt2D.astype(np.uint8) / 255
        
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # add perturbation to bounding box coordinates
        if self.train:
            H, W = gt2D.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
        box = np.array([x_min, y_min, x_max, y_max])
        
        return torch.tensor(img).float(), torch.tensor(gt2D[None, :,:]).long(), torch.tensor(box).float(), torch.tensor(0).long(), self.img_files[index], self.gt_files[index]



class ACDCDataLoader(Dataset):
    def __init__(self, data_root, train=True, resize_size=(384, 384), label_resize_size=None):
        self.train = train
        self.data_root = data_root
        self.resize_size = resize_size
        self.label_resize_size = label_resize_size if label_resize_size else resize_size

        self.data_list = sorted(glob.glob(os.path.join(data_root, "train" if train else "test", "*.npz")))
        self.class_num = 1
        self.image_size = resize_size[0]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_path = self.data_list[idx]
        data = np.load(file_path)
        image = data["img"]  # shape: (H, W)
        label = data["label"]  # shape: (H, W)

        # === Resize image and label ===
        image = cv2.resize(image, self.resize_size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, self.label_resize_size, interpolation=cv2.INTER_NEAREST)

        # === Convert to channels ===
        image = np.stack([image] * 3, axis=0)  # (3, H, W)
        label = np.expand_dims(label, axis=0)  # (1, H, W)

        # === Normalize ===
        image = image.astype(np.float32) / 255.0
        label = (label > 0).astype(np.uint8)

        # === Convert to torch tensor ===
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        # === Compute bounding box from label ===
        nz = (label[0] > 0).nonzero(as_tuple=True)
        if len(nz) != 2 or nz[0].numel() == 0:
            # If mask is empty, use minimal dummy box
            x_min, y_min, x_max, y_max = 0, 0, 1, 1
        else:
            y_indices, x_indices = nz
            x_min, x_max = x_indices.min().item(), x_indices.max().item()
            y_min, y_max = y_indices.min().item(), y_indices.max().item()

            if self.train:
                H, W = label.shape[1:]
                x_min = max(0, x_min - random.randint(0, 20))
                x_max = min(W, x_max + random.randint(0, 20))
                y_min = max(0, y_min - random.randint(0, 20))
                y_max = min(H, y_max + random.randint(0, 20))

        box = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

        return image, label, box, torch.tensor(0).long(), file_path, file_path


class BUSIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, split="train", image_size=384,
                 img_ext=".png", mask_ext=".png", transform=None):
        """
        适配 I-MedSAM 的 BUSI 数据集加载器
        """
        self.image_dir = image_dir              # 如 ./dataset/BUSI/images
        self.mask_dir = mask_dir                # 如 ./dataset/BUSI/masks
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.image_size = image_size
        self.transform = transform
        self.split = split
        self.class_num = 1                      # 单通道（合并前景）

        # 自动收集所有图像 ID（不含扩展名）
        all_files = [f for f in os.listdir(image_dir) if f.endswith(img_ext)]
        self.img_ids = sorted([os.path.splitext(f)[0] for f in all_files])

        # 简单按 8:2 划分 train/val
        split_ratio = 0.7
        split_index = int(split_ratio * len(self.img_ids))
        if self.split == "train":
            self.img_ids = self.img_ids[:split_index]
        else:
            self.img_ids = self.img_ids[split_index:]



    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]

        # --- 加载图像 ---
        img_path = os.path.join(self.image_dir, img_id + self.img_ext)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))

        # --- 加载 mask: 合并 mask/0 和 mask/1 ---
        mask0_path = os.path.join(self.mask_dir, "0", img_id + self.mask_ext)
        mask1_path = os.path.join(self.mask_dir, "1", img_id + self.mask_ext)

        m0 = cv2.imread(mask0_path, cv2.IMREAD_GRAYSCALE)
        m1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)

        if m0 is None:
            # 屏蔽 OpenCV 的警告输出
            # 原因：文件可能不存在或损坏，使用空 mask 替代
            m0 = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        else:
            m0 = cv2.resize(m0, (self.image_size, self.image_size))

        if m1 is None:
            # 屏蔽 OpenCV 的警告输出
            # 原因：文件可能不存在或损坏，使用空 mask 替代
            m1 = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        else:
            m1 = cv2.resize(m1, (self.image_size, self.image_size))

        # 合并为前景掩码（值为 0/255）
        merged_mask = np.clip(m0 + m1, 0, 255)
        mask = merged_mask[..., None]  # [H, W, 1]

        # --- 可选数据增强 ---
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # --- 标准化 ---
        image = image.astype('float32') / 255.0
        image = image.transpose(2, 0, 1)  # [C, H, W]

        mask = mask.astype('float32') / 255.0
        mask = mask.transpose(2, 0, 1)    # [1, H, W]

        # 为 IMedSAM 返回 cls (默认 0) 和 dummy box
        cls = np.array([0], dtype=np.int64)
        box = np.array([[0, 0, self.image_size, self.image_size]], dtype=np.float32)

        return image, mask, box, cls, img_path, mask0_path


class UniversalDataset(Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext=".png", mask_ext=".png",
                 dataset_type="isic", image_size=384, transform=None,class_num = 1):
        """
        通用 I-MedSAM 数据集加载器：支持 ISIC / GLAS / BUSI

        Args:
            img_ids (List[str]): 图像 ID（不带扩展名）
            img_dir (str): 图像目录
            mask_dir (str): 掩码目录
            img_ext (str): 图像后缀
            mask_ext (str): 掩码后缀
            dataset_type (str): ['isic', 'glas', 'busi']
            image_size (int): 输入图像缩放大小
            transform: albumentations transform
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.dataset_type = dataset_type.lower()
        self.image_size = image_size
        self.transform = transform
        self.class_num = class_num

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        # --- 读取图像 ---
        img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))

        # --- 读取掩码 ---
        if self.dataset_type == "isic":
            mask = []
            for i in range(self.class_num):
                mask_path = os.path.join(self.mask_dir, str(i), img_id + self.mask_ext)
                m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if m is None:
                    m = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
                else:
                    m = cv2.resize(m, (self.image_size, self.image_size))
                mask.append(m[..., None])
            mask = np.clip(np.sum(np.concatenate(mask, axis=-1), axis=-1, keepdims=True), 0, 255)

        elif self.dataset_type == "busi":
            m0_path = os.path.join(self.mask_dir, "0", img_id + self.mask_ext)
            m1_path = os.path.join(self.mask_dir, "1", img_id + self.mask_ext)
            m0 = cv2.imread(m0_path, cv2.IMREAD_GRAYSCALE)
            m1 = cv2.imread(m1_path, cv2.IMREAD_GRAYSCALE)
            m0 = cv2.resize(m0, (self.image_size, self.image_size)) if m0 is not None else np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            m1 = cv2.resize(m1, (self.image_size, self.image_size)) if m1 is not None else np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            mask = np.clip(m0 + m1, 0, 255)[..., None]

        elif self.dataset_type == "glas":
            mask = []
            for i in range(self.class_num):
                mask_path = os.path.join(self.mask_dir, str(i), img_id + "_anno" + self.mask_ext)
                gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if gt is None:
                    gt = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
                else:
                    gt = cv2.resize(gt, (self.image_size, self.image_size))
                mask.append(np.where(gt > 0, 255, 0)[..., None])
            mask = np.clip(np.sum(np.concatenate(mask, axis=-1), axis=-1, keepdims=True), 0, 255)

        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

        # --- 数据增强 ---
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # --- 标准化 ---
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        mask = mask.astype(np.float32) / 255.0
        mask = mask.transpose(2, 0, 1)

        # --- cls / box ---
        cls = np.array([0], dtype=np.int64)
        box = np.array([[0, 0, self.image_size, self.image_size]], dtype=np.float32)

        return image, mask, box, cls, img_path, mask_path if 'mask_path' in locals() else ""

class ACDCDataset(Dataset):
    def __init__(self, txt_file, npz_root, image_key='image', label_key='label'):
        with open(txt_file, 'r') as f:
            self.samples = [line.strip() for line in f if line.strip()]
        self.root = npz_root
        self.image_key = image_key
        self.label_key = label_key

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]
        path = os.path.join(self.root, name)
        data = np.load(path)
        image = data[self.image_key]
        mask = data[self.label_key]

        if image.ndim == 2:
            image = image[None]
        if mask.ndim == 2:
            mask = mask[None]

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        bbox = self._get_bbox_from_mask(mask)

        return image, mask, bbox, torch.tensor(0), name, name

    def _get_bbox_from_mask(self, mask):
        pos = torch.nonzero(mask[0], as_tuple=False)
        if pos.numel() == 0:
            return torch.tensor([[0, 0, 1, 1]])
        y_min, x_min = pos.min(dim=0)[0]
        y_max, x_max = pos.max(dim=0)[0]
        return torch.tensor([[x_min.item(), y_min.item(), x_max.item(), y_max.item()]])