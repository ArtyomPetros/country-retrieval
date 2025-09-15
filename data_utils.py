import os, glob, random
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

def list_images(root: str) -> Tuple[List[str], List[int], Dict[int,str]]:
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])
    class_to_idx = {c:i for i,c in enumerate(classes)}
    paths, labels = [], []
    for c in classes:
        cdir = os.path.join(root, c)
        for p in glob.glob(os.path.join(cdir, "**", "*"), recursive=True):
            if p.lower().endswith(IMG_EXTS):
                paths.append(p)
                labels.append(class_to_idx[c])
    return paths, labels, {v:k for k,v in class_to_idx.items()}

def build_tfms(img_size: int, train: bool):
    if train:
        tfm = A.Compose([
            A.LongestMaxSize(max_size=img_size*2, interpolation=1),
            A.RandomResizedCrop(size=(img_size, img_size),
                                scale=(0.6, 1.0),
                                ratio=(0.75, 1.33),
                                interpolation=1),
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=0.02, scale=(0.9, 1.1), rotate=(-5, 5), cval=0, p=0.7),
            A.OneOf([
                A.MotionBlur(p=0.3),
                A.GaussianBlur(p=0.3),
                A.GaussNoise(p=0.3),
                A.ImageCompression(quality_range=(40, 80), p=0.5),
            ], p=0.6),
            A.RandomBrightnessContrast(0.15, 0.15, p=0.6),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        tfm = A.Compose([
            A.LongestMaxSize(max_size=img_size, interpolation=1),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
            A.CenterCrop(height=img_size, width=img_size),
            A.Normalize(),
            ToTensorV2(),
        ])
    return tfm

class DocDataset(Dataset):
    def __init__(self, root: str, img_size: int=256, train: bool=True, indices: List[int]=None):
        paths, labels, idx_to_class = list_images(root)
        self.all_paths = paths
        self.all_labels = labels
        if indices is None:
            self.paths = paths
            self.labels = labels
        else:
            self.paths = [paths[i] for i in indices]
            self.labels = [labels[i] for i in indices]
        self.tfm = build_tfms(img_size, train)
        self.idx_to_class = idx_to_class

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        label = self.labels[i]
        img = Image.open(path).convert('RGB')
        img = np.array(img)
        aug = self.tfm(image=img)
        x = aug['image']
        return x, label

class PKSampler(Sampler):
    def __init__(self, labels: List[int], p_classes: int, k_samples: int):
        self.labels = np.array(labels)
        self.p = p_classes
        self.k = k_samples
        self.index_by_class = {}
        for idx, y in enumerate(self.labels):
            self.index_by_class.setdefault(int(y), []).append(idx)
        self.classes = list(self.index_by_class.keys())
        for y in self.classes:
            random.shuffle(self.index_by_class[y])

    def __iter__(self):
        c = self.classes.copy()
        random.shuffle(c)
        i_by_cls = {y:0 for y in self.classes}
        batch = []
        while True:
            if len(c) < self.p:
                c = self.classes.copy()
                random.shuffle(c)
            choose = c[:self.p]; c = c[self.p:]
            for y in choose:
                idxs = self.index_by_class[y]
                if len(idxs) < self.k:
                    sel = [random.choice(idxs) for _ in range(self.k)]
                else:
                    start = i_by_cls[y]
                    if start + self.k > len(idxs):
                        random.shuffle(idxs)
                        start = 0
                    sel = idxs[start:start+self.k]
                    i_by_cls[y] = start + self.k
                batch.extend(sel)
            if len(batch) == self.p * self.k:
                yield from batch
                batch = []

    def __len__(self):
        return len(self.labels)
