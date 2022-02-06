import os
import time
import torch
import inspect
import re
import numpy as np
import matplotlib.cm as cm

from inspect import isclass
from collections import OrderedDict
from PIL import Image


######################
### Helper methods ###
######################

def get_modules(module, superclass=None, filter=None):
    if superclass:
        modules = dict([(x, getattr(module, x)) for x in dir(module)
                 if isclass(getattr(module, x)) and issubclass(getattr(module, x), superclass)]).keys()
    else:
        modules = dict([(x, getattr(module, x)) for x in dir(module)
                 if isclass(getattr(module, x))]).keys()
    if filter:
        modules = [m for m in modules if filter in m]
    return modules

def module_to_dict(module, exclude=[]):
    return dict([(x, getattr(module, x)) for x in dir(module)
                 if x not in exclude and isclass(getattr(module, x))
                 and getattr(module, x) not in exclude])

def param_to_str(**kwargs):
    """
    returns a string with [parameter_name1: value1, ...] format
    """
    return str([f"{key}: {value}" for key, value in kwargs.items()]).strip('[]')

@torch.no_grad()
def make_grid(tensor, nrow=1):
    """
    makes a grid from the tensor.
    code adapted from pytorch's official code :
    https://pytorch.org/vision/stable/_modules/torchvision/utils.html
    """
    if not (isinstance(tensor, torch.Tensor) or
            (isinstance(tensor, list) and all(isinstance(t, torch.Tensor) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:
        if tensor.size(0) == 1:
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)
    batch = tensor.size(0)
    nrows = min(nrow, batch)
    ncols = int(np.ceil(float(batch) / nrows))
    height, width = int(tensor.size(2)), int(tensor.size(3))
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ncols, width * nrows), 0)
    k = 0
    for y in range(ncols):
        for x in range(nrows):
            if k >= batch:
                break
            grid.narrow(1, y * height, height).narrow(2, x * width, width).copy_(tensor[k])
            k = k + 1
    return grid

@torch.no_grad()
def tensor_to_image(tensor, imtype=np.uint8):
    """
    converts a torch tensor to a numpy image
    """
    tensor = make_grid(tensor/ 2.0 + 0.5)
    image_numpy = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return image_numpy.astype(imtype)

@torch.no_grad()
def tensor_to_mask(tensor, imtype=np.uint8):
    """
    converts a torch tensor to a segmentation mask
    """
    tensor = make_grid(tensor)
    if len(tensor.shape) == 4:
        image_numpy = tensor.squeeze().data.cpu().float().numpy()
    else:
        image_numpy = tensor.data.cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    return image_numpy.astype(imtype)

def resize(img, size):
    """
    resize image
    """
    image = Image.fromarray(img)
    image = image.resize(size)
    return np.array(image)

@torch.no_grad()
def save_image(tensor, image_path):
    """
    saves image to disc as a PIL image
    """
    image_numpy = tensor_to_image(tensor)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

@torch.no_grad()
def save_images(images, names):
    for img, name in zip(images, names):
        save_image(img, name)

#########################
#### Helper classes #####
#########################

class AttributeDict(OrderedDict):
    """
    class that provides attribute like access to OrderdDict objects
    """
    def __init__(self, *args, **kwargs):
        super(AttributeDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(AttributeDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

class TimerBlock:
    def __init__(self, title):
        print(("{}".format(title)))

    def __enter__(self):
        self.start = time.process_time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.process_time()
        self.interval = self.end - self.start

        if exc_type is not None:
            self.log("Operation failed\n")
        else:
            self.log("Operation finished\n")

    def log(self, string):
        duration = time.process_time() - self.start
        units = 's'
        if duration > 60:
            duration = duration / 60.
            units = 'm'
        print("  [{:.3f}{}] {}".format(duration, units, string), flush = True)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)