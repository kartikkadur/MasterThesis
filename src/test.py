import torch.nn as nn
import os
import sys
import torch

from dataset.cifar10 import Cifar10
from models.base_model import Model
import torch.nn.functional as F
from models.networks import ResnetBlock
"""
class ResnetBlock1(nn.Module):
    '''Define a Resnet block'''

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        '''Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        '''
        super(ResnetBlock1, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        '''Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        '''
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
#        Forward function (with skip connections)
        out = x + self.conv_block(x)  # add skip connections
        return out

class Test(Model):
    def __init__(self):
        super(Test, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def set_inputs(self, inputs):
        self.inputs = inputs['image']
        self.labels = inputs['label']
    
    def forward(self):
        x = self.conv1(self.inputs)
        x = self.pool(x)
        x = F.relu(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    model = Test()
    trainset = Cifar10(r"C:\\Users\\Karthik\\Desktop\\GAN\\data\\cifar-10-batches-py", is_training=True, transform=None)
    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=64,
                                            num_workers=1,
                                            drop_last=True)
    valset = Cifar10(r"C:\\Users\\Karthik\\Desktop\\GAN\\data\\cifar-10-batches-py", is_training=False, transform=None)
    valloader = torch.utils.data.DataLoader(valset,
                                            batch_size=64,
                                            num_workers=4,
                                            drop_last=True)
    
    model.compile(optimizer='Adam', criterion='CrossEntropyLoss')
    model.fit(trainloader, valloader, epochs=5, val_freq=2)
    #losses = model.get_losses()
"""

if __name__ == '__main__':
    from arguments.train_arguments import TrainArguments
    from dataset.multiclass_dataset import MultiClassDataset
    from models.attentionGAN import AttentionGANModel
    args = TrainArguments().parse()
    dataset = MultiClassDataset(args)
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=1)
    model = args.model(args)
    model.fit(dataloader, epochs=1, print_freq=1)