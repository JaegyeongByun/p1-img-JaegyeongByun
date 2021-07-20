"""
마스크 데이터셋을 읽고 전처리를 진행한 후
데이터를 하나씩 꺼내주는 Dataset 클래스를 구현한 파일
이곳에서 나만의 Data Augmentation 기법들을 구현하여 사용할 수 있다.
"""

import glob
import os

import random
from collections import defaultdict

from PIL import Image
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import albumentations as A
from torchvision import transforms
from torchvision.transforms import *

import timm
import torchvision.models as models

from tqdm.notebook import tqdm
import numpy as np
import pandas as pd


class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            CenterCrop((320, 256)),
            Resize(resize, Image.BILINEAR),
            ColorJitter(0.1, 0.1, 0.1, 0.1),
            ToTensor(),
            Normalize(mean=mean, std=std),
            AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)

class TrainDatasetForMulti(Dataset):
    class MaskLabels:
        mask = 0
        incorrect = 1
        normal = 2
        
    class GenderLabels:
        male = 0
        female = 1
        
    class AgeGroup:
        map_label = lambda x: 0 if int(x) < 30 else 1 if int(x) < 60 else 2
        
        
    _file_names = {
        "mask1": MaskLabels.mask,
        "mask2": MaskLabels.mask,
        "mask3": MaskLabels.mask,
        "mask4": MaskLabels.mask,
        "mask5": MaskLabels.mask,
        "incorrect_mask": MaskLabels.incorrect,
        "normal": MaskLabels.normal
    }
    
    img_paths = []
    age_labels = []
    gender_labels = []
    mask_labels = []

    def __init__(self, data_dir, upsampling=False, transform=None, val_ratio=0.2, tta=False):
        self.data_dir = data_dir  # '/opt/ml/input/data/train/images'
        self.indices = defaultdict(list)  # key: train or val, value: train index or val index
        
        self.val_ratio = val_ratio
        self.transform = transform

        self.upsampling = upsampling
        self.upsampling_labels = [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        self.upsampling_ratio = 2

        self.setup()

        self.tta = tta
        if self.tta:
            mean = (0.548, 0.504, 0.479)
            std = (0.237, 0.247, 0.246)

            self.transform_1 = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.CenterCrop(height=384, width=384, p=1),
                A.Normalize(mean=mean, std=std)
            ])
            self.transform_2 = A.Compose([
                A.RandomBrightnessContrast(p=0.2),
                A.CenterCrop(height=384, width=384, p=1),
                A.Normalize(mean=mean, std=std)
            ])
            self.transform_3 = A.Compose([
                A.ShiftScaleRotate(p=0.5),
                A.CenterCrop(height=384, width=384, p=1),
                A.Normalize(mean=mean, std=std)
            ])

    @staticmethod
    def split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)
        
        val_indices = list(range(n_val))
        train_indices = list(range(n_val, length))
        return {
            "train": train_indices,
            "val": val_indices
        }
        
    def get_label(self, img_path):
        folder_name, file_name = img_path.split('/')[-2:]
        file_name = file_name.split('.')[0]
        _, gender, _, age = folder_name.split('_')
        
        age_label = self.AgeGroup.map_label(age)
        gender_label = getattr(self.GenderLabels, gender)
        mask_label = self._file_names[file_name]
                        
        return (age_label, gender_label, mask_label)

    def setup(self):
        profiles = glob.glob(os.path.join(self.data_dir, '*'))
        split_profiles = self.split_profile(profiles, self.val_ratio)
        
        cnt = 0
        check = False
        for phase, indices in split_profiles.items():
            if phase=='train' and self.upsampling:
                check = True
            for idx in indices:
                img_dir = profiles[idx]
                seven_img_paths = glob.glob(os.path.join(img_dir, '*'))
                for one_img_path in seven_img_paths:
                    age_label, gender_label, mask_label = self.get_label(one_img_path)
                    label = age_label + gender_label * 3 + mask_label * 6
                    if (check == True) and (label in self.upsampling_labels):
                        for _ in range(self.upsampling_ratio-1):
                            self.img_paths.append(one_img_path)
                            self.age_labels.append(age_label)
                            self.gender_labels.append(gender_label)
                            self.mask_labels.append(mask_label)
                            self.indices[phase].append(cnt)
                            cnt += 1

                    self.img_paths.append(one_img_path)
                    self.age_labels.append(age_label)
                    self.gender_labels.append(gender_label)
                    self.mask_labels.append(mask_label)
                    self.indices[phase].append(cnt)
                    cnt += 1

    def set_transform(self, transform):
        self.transform = transform

    def get_tensor(self, image):
        image = image['image']
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)

        return image
        
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        age_label = self.age_labels[index]
        gender_label = self.gender_labels[index]
        mask_label = self.mask_labels[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.tta:
            img1 = self.transform_1(image=img)
            img1 = self.get_tensor(img1)

            img2 = self.transform_2(image=img)
            img2 = self.get_tensor(img2)

            img3 = self.transform_3(image=img)
            img3 = self.get_tensor(img3)

            return (img1, img2, img3), (age_label, gender_label, mask_label)

        else:
            img = self.transform(image=img)
            img = self.get_tensor(img)

            return img, (age_label, gender_label, mask_label)
    
    def __len__(self):
        return len(self.img_paths)

    @staticmethod
    def decode_multi_class(multi_class_label):
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label        
    
    def split_dataset(self):
        return [Subset(self, indices) for phase, indices in self.indices.items()]

class TrainDatasetForThreeModel(Dataset):
    class MaskLabels:
        mask = 0
        incorrect = 1
        normal = 2
        
    class GenderLabels:
        male = 0
        female = 1
        
    class AgeGroup:
        map_label = lambda x: 0 if int(x) < 30 else 1 if int(x) < 60 else 2
        
        
    _file_names = {
        "mask1": MaskLabels.mask,
        "mask2": MaskLabels.mask,
        "mask3": MaskLabels.mask,
        "mask4": MaskLabels.mask,
        "mask5": MaskLabels.mask,
        "incorrect_mask": MaskLabels.incorrect,
        "normal": MaskLabels.normal
    }
    
    def __init__(self, data_dir, category='all', upsampling=True, transform=None, val_ratio=0.2, tta=False):
        self.data_dir = data_dir  # '/opt/ml/input/data/train/images'
        self.indices = defaultdict(list)  # key: train or val, value: train index or val index
        self.img_paths = list()
        self.labels = list()
        
        self.upsampling = upsampling
        self.val_ratio = val_ratio
        self.transform = transform
        self.label_len = [0] * 18
        self.tta = tta

        if self.tta:
            mean = (0.548, 0.504, 0.479)
            std = (0.237, 0.247, 0.246)

            self.transform_1 = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.CenterCrop(height=384, width=384, p=1),
                A.Normalize(mean=mean, std=std)
            ])
            self.transform_2 = A.Compose([
                A.RandomBrightnessContrast(p=0.2),
                A.CenterCrop(height=384, width=384, p=1),
                A.Normalize(mean=mean, std=std)
            ])
            self.transform_3 = A.Compose([
                A.ShiftScaleRotate(p=0.5),
                A.CenterCrop(height=384, width=384, p=1),
                A.Normalize(mean=mean, std=std)
            ])

        if category == 'all':
            self.num_classes = 3*2*3
            self.weights = [1.2, 2, 10, 1, 1, 10, 10, 10, 20, 7, 5, 20, 10, 10, 20, 7, 5, 20]
            #self.label_len_min = 83
            self.augmentation_labels = [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
            self.augmentation_ratio = 4

        elif category == 'age':
            self.num_classes = 3
            self.weights = [1, 1, 6]
            #self.label_len_min = 192 * 7
            self.augmentation_labels = [2]
            self.augmentation_ratio = 5

        elif category == 'gender':
            self.num_classes = 2
            self.weights = [3, 2]
            #self.label_len_min = 1024 * 7
            self.augmentation_labels = [1]
            self.augmentation_ratio = 2
        
        elif category == 'mask':
            self.num_classes = 3
            self.weights = [1, 5, 5]
            #self.label_len_min = 18900 // 5
            self.augmentation_labels = [1, 2]
            self.augmentation_ratio = 5

        elif category == 'age_gender':
            self.num_classes = 6
            self.augmentation_labels = [2]
            self.augmentation_ratio = 2
        
        else:
            raise ValueError("category can have only 'all', 'age', 'gender' or 'mask'")
        self.category = category

        self.setup()

    @staticmethod
    def split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)
        
        #val_indices = set(random.choices(range(length), k=n_val))
        val_indices = list(range(n_val))
        #train_indices = set(range(length)) - val_indices
        train_indices = list(range(n_val, length))
        return {
            "train": train_indices,
            "val": val_indices
        }
        
    def get_label(self, img_path):
        folder_name, file_name = img_path.split('/')[-2:]
        file_name = file_name.split('.')[0]
        _, gender, _, age = folder_name.split('_')
        
        if self.category == 'all':
            age_label = self.AgeGroup.map_label(age)
            gender_label = getattr(self.GenderLabels, gender)
            mask_label = self._file_names[file_name]
            label = age_label + gender_label*3 + mask_label*6
            
        elif self.category == 'age_gender':
            age_label = self.AgeGroup.map_label(age)
            gender_label = getattr(self.GenderLabels, gender)
            label = age_label + gender_label*3

        elif self.category == 'age':
            label = self.AgeGroup.map_label(age)
                
        elif self.category == 'gender':
            label = getattr(self.GenderLabels, gender)
            
        elif self.category == 'mask':
            label = self._file_names[file_name]
            
        return label

    def setup(self):
        profiles = glob.glob(os.path.join(self.data_dir, '*'))
        # /opt/ml/input/data/train/images/Asia_30, 
        # /opt/ml/input/data/train/images/Asia_54, ...
        split_profiles = self.split_profile(profiles, self.val_ratio)
        
        cnt = 0
        for phase, indices in split_profiles.items():
            check = True if (phase=="train" and self.upsampling) else False
            for idx in indices:
                img_dir = profiles[idx]
                seven_img_paths = glob.glob(os.path.join(img_dir, '*'))
                for one_img_path in seven_img_paths:
                    label = self.get_label(one_img_path)
                    if check and label in self.augmentation_labels:
                        for _ in range(self.augmentation_ratio-1):
                            self.img_paths.append(one_img_path)
                            self.labels.append(label)
                            self.indices[phase].append(cnt)
                            cnt += 1

                    self.img_paths.append(one_img_path)
                    self.labels.append(label)
                    self.indices[phase].append(cnt)
                    cnt += 1

    def set_transform(self, transform):
        self.transform = transform

    def get_tensor(self, image):
        image = image['image']
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)

        return image
        
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.tta:
            img1 = self.transform_1(image=img)
            img1 = self.get_tensor(img1)

            img2 = self.transform_2(image=img)
            img2 = self.get_tensor(img2)

            img3 = self.transform_3(image=img)
            img3 = self.get_tensor(img3)

            return (img1, img2, img3, label)

        else:
            img = self.transform(image=img)
            img = self.get_tensor(img)

            return img, label
    
    def __len__(self):
        return len(self.img_paths)

    @staticmethod
    def decode_multi_class(multi_class_label):
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label        
    
    def split_dataset(self):
        return [Subset(self, indices) for phase, indices in self.indices.items()]
    

class TestDataset(Dataset):
    def __init__(self, img_paths, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):  # resize
        self.img_paths = img_paths
        # self.transform = transforms.Compose([
        #     # Resize(resize, Image.BILINEAR),
        #     ToTensor(),
        #     # Normalize(mean=mean, std=std),
        # ])

        # self.transform = A.Compose([
        #     A.HorizontalFlip(p=0.5),
        #     A.RandomBrightnessContrast(p=0.2),
        #     A.ShiftScaleRotate(p=0.5),
        #     A.CenterCrop(height=384, width=384, p=1),
        #     A.Normalize(mean=mean, std=std)
        # ])

        self.transform = A.Compose([
        A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246))
        ])

    def __getitem__(self, index):
        # image = Image.open(self.img_paths[index])

        image = cv2.imread(self.img_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            # image = self.transform(image)

            image = self.transform(image=image)
            image = image['image']
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1)

        return image

    def __len__(self):
        return len(self.img_paths)


class TestDatasetForTTA(Dataset):
    def __init__(self, img_paths, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):  # resize
        self.img_paths = img_paths
        # self.transform = transforms.Compose([
        #     # Resize(resize, Image.BILINEAR),
        #     ToTensor(),
        #     # Normalize(mean=mean, std=std),
        # ])

        self.transform_1 = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=mean, std=std)
        ])
        self.transform_2 = A.Compose([
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=mean, std=std)
        ])
        
        self.transform_3 = A.Compose([
            A.ShiftScaleRotate(p=0.5),
            A.Normalize(mean=mean, std=std)
        ])

    def get_tensor(self, image):
        image = image['image']
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)

        return image

    def __getitem__(self, index):
        image = cv2.imread(self.img_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_1 = self.transform_1(image=image)
        image_1 = self.get_tensor(image_1)

        image_2 = self.transform_2(image=image)
        image_2 = self.get_tensor(image_2)

        image_3 = self.transform_3(image=image)
        image_3 = self.get_tensor(image_3)

        return (image_1, image_2, image_3)

    def __len__(self):
        return len(self.img_paths)