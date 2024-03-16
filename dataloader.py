import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from PIL import Image
import cv2
import numpy as np

class CustomDataLoader(Dataset):
    def __init__(self, split, transform):
        self.split = split
        if split == 'train':
          self.root = 'C:/Users/nyles/Downloads/chest_xray/train'
          self.norm_img_dir = os.listdir(os.path.join(self.root, 'NORMAL'))
          self.pnum_img_dir = os.listdir(os.path.join(self.root, 'PNEUMONIA'))
        else:
          self.root = 'C:/Users/nyles/Downloads/chest_xray/test'
          self.norm_img_dir = os.listdir(os.path.join(self.root, 'NORMAL'))
          self.pnum_img_dir = os.listdir(os.path.join(self.root, 'PNEUMONIA'))

        self.img_dir = self.norm_img_dir + self.pnum_img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img_path = self.img_dir[idx]
        if 'bacteria' in img_path or 'virus' in img_path:
            image = read_image(os.path.join(f'{self.root}/PNEUMONIA/{img_path}'))
            label = 1
        else:
            image = read_image(os.path.join(f'{self.root}/NORMAL/{img_path}'))
            label = 0
        image = image / 255.
        if self.transform:
            image = self.transform(image)
        if image.shape[0] == 1:
            image = torch.stack([image, image, image], dim=1).squeeze()
        return image, label