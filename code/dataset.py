import os
import torch

from numpy import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class ImageFolder(Dataset):
    """reads images from a single folder"""
    def __init__(self, args, return_paths=False):
        super(ImageFolder, self).__init__()
        self.args = args
        self.root = self.args.dataroot
        self.dataset = self._make_dataset(self.root)
        transform = [transforms.Resize((args.load_size, args.load_size), Image.BICUBIC)]
        if args.mode == 'train':
            transform.append(transforms.RandomCrop(args.crop_size))
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = transforms.Compose(transform)
    
    def _make_dataset(self, root):
        return [os.path.join(root, img) for img in os.listdir(root) if is_image_file(img)]

    def load_image(self, img_name, dim=3):
        img = Image.open(img_name).convert('RGB')
        img = self.transforms(img)
        if dim == 1:
            img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = img.unsqueeze(0)
        return img

    def __getitem__(self, index):
        x = self.dataset[index]
        x = self.load_image(x)
        return x

class SingleDataset(Dataset):
    """Returns a single domain image and label"""
    def __init__(self, args, return_paths=False):
        super(SingleDataset, self).__init__()
        self.args = args
        self.root = os.path.join(args.dataroot, args.mode)
        self.dataset, self.targets, self.target_names = self._make_dataset(self.root)
        self.num_domains = len(self.targets)
        self.return_paths = return_paths
        self.size = max(map(len, self.dataset.values()))
        transform = [transforms.Resize((args.load_size, args.load_size), Image.BICUBIC)]
        if args.mode == 'train':
            transform.append(transforms.RandomCrop(args.crop_size))
        else:
            transform.append(transforms.CenterCrop(args.crop_size))
        if not args.no_flip:
            transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = transforms.Compose(transform)

    def _make_dataset(self, root):
        domains = os.listdir(root)
        dataset = {}
        for i, domain in enumerate(sorted(domains)):
            domain_dir = os.path.join(root, domain)
            fnames = [os.path.join(domain_dir,f) for f in os.listdir(domain_dir) if is_image_file(f)]
            dataset[i] = fnames
        return dataset, sorted(dataset.keys()), domains

    def load_image(self, img_name, dim=3):
        img = Image.open(img_name).convert('RGB')
        img = self.transforms(img)
        if dim == 1:
            img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = img.unsqueeze(0)
        return img

    def get_onehot(self, index, shape):
        vector = torch.zeros(shape)
        vector[index] = 1
        return vector

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # sample a random class
        y_src = random.choice(self.targets)
        # generate one-hot label
        y = self.get_onehot(y_src, (self.args.num_domains,))
        # get image
        x_src = self.dataset[y_src][index % len(self.dataset[y_src])]
        x = self.load_image(x_src)
        if self.return_paths:
            return {'x':x, 'y':y, 'x_path':x_src}
        return {'x':x, 'y':y}

class AlignedDataset(SingleDataset):
    """
    This dataset returns images from two domains.
    """
    def __init__(self, args, return_paths=False):
        super(AlignedDataset, self).__init__(args, return_paths)
        if self.args.class_a and self.args.class_b:
            self.targets = sorted([self.target_names.index(self.args.class_a),
                            self.target_names.index(self.args.class_b)])

    def __getitem__(self, index):
        # sample a random class
        if len(self.targets) == 2:
            y1_src, y2_src = self.targets[0], self.targets[1]
        else:
            y1_src, y2_src = random.choice(self.targets, 2, replace=False)
        # generate one-hot label
        y1 = self.get_onehot(y1_src, (self.args.num_domains,))
        y2 = self.get_onehot(y2_src, (self.args.num_domains,))
        # get image
        x1_src = self.dataset[y1_src][index % len(self.dataset[y1_src])]
        x2_src = self.dataset[y2_src][index % len(self.dataset[y2_src])]
        x1 = self.load_image(x1_src)
        x2 = self.load_image(x2_src)
        if self.return_paths:
            return {'x1':x1, 'x2':x2, 'y1':y1, 'y2':y2, 'x1_path':x1_src, 'x2_path':x2_src}
        return {'x1':x1, 'x2':x2, 'y1':y1, 'y2':y2}