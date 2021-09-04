import os
import torch

import torchvision.transforms as transforms
import numpy as np

from torch.utils.data import Dataset
from abc import ABC, abstractmethod

class BaseDataset(Dataset, ABC):

    def __init__(self, root, transform=None, is_training=False):
        self.is_training = is_training
        self.root = root
        self.transform = transform
    
    def transforms(self, transform=None):
        if transform is not None and isinstance(transform, transforms.Compose):
            self.transforms = transforms
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()])

    @abstractmethod
    def __len__(self):
        pass
    
    @abstractmethod
    def __getitem__(self, idx):
        pass