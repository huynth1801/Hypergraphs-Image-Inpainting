import os
import numpy as np
import glob
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


class InpaintingData(Dataset):
    def __init__(self, opt):
        super(InpaintingData, self).__init__()
        self.w = self.h = opt.image_size
        self.mask_type = opt.mask_type

        # Image and mask
        self.image_path = []
        for ext in ['*.jpg', '*.png']:
            self.image_path.extend(glob(os.path.join(opt.train_dir, ext)))
        self.mask_path = glob(os.path.join(opt.dir_mask, opt.mask_type, '*.png'))

        # Augumentation
        self.img_transform = transforms.Compose([
                            transforms.RandomResizedCrop(opt.image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        self.mask_trans = transforms.Compose([
            transforms.Resize(opt.image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomRotation(
                (0, 45), interpolation=transforms.InterpolationMode.NEAREST),
        ])


    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        # Load image
        image = Image.open(self.image_path[index]).convert('RGB')
        filename = os.path.basename(self.image_path[index])

        if self.mask_type == '1':
            index = np.random.randint(0, len(self.mask_path))
            mask = Image.open(self.mask_path[index])
            mask = mask.convert('L')
        else:
            mask = np.zeros((self.h, self.w)).astype(np.uint8)
            mask[self.h//4:self.h//4*3, self.w//4:self.w//4*3] = 1
            mask = Image.fromarray(mask).convert('L')

        # augment
        image = self.img_trans(image) * 2. - 1.
        mask = F.to_tensor(self.mask_trans(mask))
        return image, mask, filename


