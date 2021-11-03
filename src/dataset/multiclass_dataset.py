import os.path
import torch

from numpy import random
from dataset.base_dataset import Dataset, get_transform
from dataset import is_image_file
from PIL import Image


class SingleDataset(Dataset):
    """Returns a single domain image and label"""
    @staticmethod
    def modify_commandline_arguments(parser, is_train):
        return parser

    def __init__(self, args, return_paths=False):
        super(SingleDataset, self).__init__(args)
        self.root = os.path.join(args.dataroot, args.mode)
        self.dataset, self.targets, self.target_names = self._make_dataset(self.root)
        self.num_domains = len(self.targets)
        self.transform = get_transform(self.args, grayscale=(args.input_nc == 1))
        self.return_paths = return_paths
        self.size = max(map(len, self.dataset.values()))

    def _make_dataset(self, root):
        domains = os.listdir(root)
        dataset = {}
        for i, domain in enumerate(sorted(domains)):
            domain_dir = os.path.join(root, domain)
            fnames = [os.path.join(domain_dir,f) for f in os.listdir(domain_dir) if is_image_file(f)]
            dataset[i] = fnames
        return dataset, sorted(dataset.keys()), domains

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # sample a random class
        y_src = random.choice(self.targets)
        # generate one-hot label
        y = torch.zeros((self.num_domains,))
        y[y_src] = 1
        # get image
        x_src = self.dataset[y_src][index % len(self.dataset[y_src])]
        x = Image.open(x_src).convert('RGB')
        if self.transform is not None:
            x = self.transform(x)
        if self.return_paths:
            return {'x':x, 'y':y, 'x_path':x_src}
        return {'x':x, 'y':y}


class AlignedDataset(SingleDataset):
    """
    This dataset returns images from two domains. The domains are selected at
    random if the arguments class_a and class_b are not passed
    """
    @staticmethod
    def modify_commandline_arguments(parser, is_train):
        parser.add_argument('--class_a', type=str, default=None, help='name of the directory containing class A images')
        parser.add_argument('--class_b', type=str, default=None, help='name of the directory containing class B images')
        return parser

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
        y1 = torch.zeros((self.num_domains,))
        y2 = torch.zeros((self.num_domains,))
        y1[y1_src] = 1
        y2[y2_src] = 1
        # get image
        x1_src = self.dataset[y1_src][index % len(self.dataset[y1_src])]
        x2_src = self.dataset[y2_src][index % len(self.dataset[y2_src])]
        x1 = Image.open(x1_src).convert('RGB')
        x2 = Image.open(x2_src).convert('RGB')
        if self.transform is not None:
            x1 = self.transform(x1)
            x2 = self.transform(x2)
        if self.return_paths:
            return {'x1':x1, 'x2':x2, 'y1':y1, 'y2':y2, 'x1_path':x1_src, 'x2_path':x2_src}
        return {'x1':x1, 'x2':x2, 'y1':y1, 'y2':y2}


class ReferenceDataset(SingleDataset):
    """
    This dataset returns images from two domains with a additional reference image from the second somain.
    The domains are selected at random if the arguments class_a and class_b are not passed.
    """
    @staticmethod
    def modify_commandline_arguments(parser, is_train):
        parser.add_argument('--class_a', type=str, default=None, help='name of the directory containing class A images')
        parser.add_argument('--class_b', type=str, default=None, help='name of the directory containing class B images')
        return parser

    def __init__(self, args, return_paths=False):
        super(ReferenceDataset, self).__init__(args, return_paths)
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
        y1 = torch.zeros((self.num_domains,))
        y2 = torch.zeros((self.num_domains,))
        y2_1 = torch.zeros((self.num_domains,))
        y1[y1_src] = 1
        y2[y2_src] = 1
        y2_1[y2_src] = 1
        # get image
        x1_src = self.dataset[y1_src][index % len(self.dataset[y1_src])]
        x2_src = self.dataset[y2_src][index % len(self.dataset[y2_src])]
        x2_src2 = self.dataset[y2_src][random.randint(len(self.dataset[y2_src]))]
        x1 = Image.open(x1_src).convert('RGB')
        x2 = Image.open(x2_src).convert('RGB')
        x2_1 = Image.open(x2_src2).convert('RGB')
        if self.transform is not None:
            x1 = self.transform(x1)
            x2 = self.transform(x2)
            x2_1 = self.transform(x2_1)
        if self.return_paths:
            return {'x1':x1, 'x2':x2, 'x2_1':x2_1, 'y1':y1, 'y2':y2, 'y2_1':y2_1, 'x1_path':x1_src, 'x2_path':x2_src, 'x2_1_path':x2_src2}
        return {'x1':x1, 'x2':x2, 'x2_1':x2_1, 'y1':y1, 'y2':y2, 'y2_1':y2_1}