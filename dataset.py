import os
import glob
from PIL import Image

import torch
from torch.utils.data import Dataset

class ImageFolder(Dataset):
    
    def __init__(self, Y0_dir, Y1_dir, transform=None):
        self.Y0 = glob.glob(os.path.join(Y0_dir, '*.jpg'))
        self.Y1 = glob.glob(os.path.join(Y1_dir, '*.jpg'))
        
        self.files = self.Y0 + self.Y1
        self.labels = [0] * len(self.Y0) + [1] * len(self.Y1)
        self.transform = transform
    
    def __getitem__(self, index):
        img = Image.open(self.files[index])
        label = self.labels[index]
        
        if self.transform is not None:
            return self.transform(img), label
        else:
            return img, label

    def __len__(self):
        return len(self.files)
