import os
import pickle
import torch

import torchvision.transforms as transforms
import numpy as np

from torch.utils.data import Dataset
from .base_dataset import BaseDataset

class Cifar10(BaseDataset):
    def __init__(self, root, is_training=False, transform=None):
        super(Cifar10, self).__init__(root, is_training, transform)
        # set transforms
        super(Cifar10, self).transforms(transform)

        self.is_training = is_training
        self.images, self.labels = self.get_cifar_data(self.root)
        self.classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    def get_cifar_data(self, root):
        images, labels = [], []
        if self.is_training:
            filename = "data_batch"
        else:
            filename = "test_batch"

        batches = [os.path.join(root, databatch) 
                    for databatch in os.listdir(root) 
                    if filename in databatch]
        for batch in batches:
            with open(batch, 'rb') as f:
                datadict = pickle.load(f, encoding='latin1')
                images.append(datadict['data'])
                labels.extend(datadict['labels'])
        images = np.vstack(images).reshape(-1, 3, 32, 32)
        images = images.transpose((0, 2, 3, 1))
        return images, labels

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # get the image and label
        image = self.images[index]
        label = self.labels[index]
        # if training apply transforms
        #if self.is_training:
        image = self.transforms(image)
        return {"image" : image, "label" : label, "class" : self.classes[label]}

if __name__=='__main__':
    trainset = Cifar10("C:\\Users\\Karthik\\Desktop\\GAN\\data\\cifar-10-batches-py", is_training=True, transform=None)
    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=64,
                                            num_workers=2,
                                            drop_last=True)
    for i, batch in enumerate(trainloader):
        print(batch['image'].dtype)
        break