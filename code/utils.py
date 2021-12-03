import os
import time
import torch
import random
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
def module_to_dict(module, exclude=[]):
    return dict([(x, getattr(module, x)) for x in dir(module)
                 if x not in exclude and isclass(getattr(module, x))
                 and getattr(module, x) not in exclude])

def param_to_str(**kwargs):
    """
    returns a string with [parameter_name1: value1, ...] format
    """
    return str([f"{key}: {value}" for key, value in kwargs.items()]).strip('[]')

def tensor_to_image(tensor, imtype=np.uint8):
    """
    converts a torch tensor to a numpy image
    """
    if isinstance(tensor, torch.Tensor):
        if tensor.dim() == 4:
            image_numpy = tensor[0,:,:,:].data.cpu().float().numpy()
        else:
            image_numpy = tensor.data.cpu().float().numpy()
    else:
        if len(tensor.shape) == 4:
            image_numpy = tensor[0,:,:,:]
        else:
            image_numpy = tensor
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def tensor_to_mask(tensor, imtype=np.uint8):
    """
    converts a torch tensor to a segmentation mask
    """
    if len(tensor.shape) == 4:
        image_numpy = tensor.squeeze().data.cpu().float().numpy()
    else:
        image_numpy = tensor.data.cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    return image_numpy.astype(imtype)

def mask_to_image(img):
    """
    converts a mask to an image
    """
    return np.stack((img[:,:,0],img[:,:,0],img[:,:,0]),axis=2)

def mask_to_heatmap(tensor, imtype=np.uint8, size=None):
    """
    converts a mask to a heatmap
    """
    image_numpy = tensor[0,0].cpu().float().numpy()
    if size:
        image_numpy=resize(image_numpy,size)
    heatmap_marked = cm.jet(image_numpy)[..., :3] * 255.0
    return heatmap_marked.transpose((2,0,1)).astype(imtype)

def overlay(seed_img, heatmap_marked, alpha=0.5, imtype=np.uint8):
    """
    overlays heatmap on the seed image
    """
    if isinstance(seed_img, torch.Tensor):
        seed_img = seed_img.squeeze().data.cpu().float().numpy()
    img = seed_img * alpha + heatmap_marked * (1. - alpha)
    return img.astype(imtype)

def resize(img, size):
    """
    resize image
    """
    image = Image.fromarray(img)
    image = image.resize(size)
    return np.array(image)

def save_image(image_numpy, image_path):
    """
    saves image to disc as a PIL image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_images(images, names, path):
    os.makedirs(path, exist_ok=True)
    for img, name in zip(images, names):
        img = tensor_to_image(img)
        img = Image.fromarray(img)
        img.save(os.path.join(path, name + '.png'))

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

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
                for k, v in arg.iteritems():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
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