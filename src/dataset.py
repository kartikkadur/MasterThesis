import os
import torch
import cv2
import warnings

from numpy import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class ImageList(Dataset):
    """reads images into a single list by unwraping the folders"""
    def __init__(self, root, return_paths=False, transform=None) -> None:
        super(ImageList, self).__init__()
        self.root = root
        self.return_paths = return_paths
        self.dataset = self._make_dataset(self.root)
        self.transforms = transform
        if self.transforms is None:
            self.transforms = [transforms.ToTensor()]
            self.transforms = transforms.Compose(self.transforms)

    def _make_dataset(self, root) -> list:
        return [os.path.join(fdir, fname) for fdir, _, fnames in os.walk(root) for fname in fnames if is_image_file(fname)]

    def load_image(self, img_name, dim=3):
        img = Image.open(img_name).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        if dim == 1:
            img = img.unsqueeze(0)
        return img

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_pth = self.dataset[index]
        img = self.load_image(img_pth)
        if self.return_paths:
            return img, img_pth
        return img

class ImageFolder(Dataset):
    """
    reads images from a root folder and returns the raw images and its class
    folder structure:
    root:
        domain1:
            img1.jpg
            img2.jpg
            .
            .
        domain2:
            img1.jpg
            img2.jpg
            .
            .
        domain3:
            ...
    """
    def __init__(self, args, return_paths=False, transforms=None) -> None:
        super(ImageFolder, self).__init__()
        self.args = args
        self.root = self.args.dataroot
        self.dataset = self._make_dataset(self.root)
        self.transforms = transforms
    
    def _make_dataset(self, root):
        dataset = []
        domains = os.listdir(root)
        for i, d in enumerate(sorted(domains)):
            dataset += [(os.path.join(root, d, f), i) for f in os.listdir(os.path.join(root, d))]
        return dataset

    def load_image(self, img_name, dim=3):
        img = Image.open(img_name).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        if dim == 1:
            img = img.unsqueeze(0)
        return img

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = self.load_image(x)
        return x, y
    
    def __len__(self):
        return len(self.dataset)

class SingleDataset(Dataset):
    """Returns a single domain image and label"""
    def __init__(self, args, return_paths=False) -> None:
        super(SingleDataset, self).__init__()
        self.args = args
        self.root = os.path.join(args.dataroot, args.mode)
        self.dataset, self.targets, self.target_names = self._make_dataset(self.root, self.args.select_domains)
        assert self.args.num_domains == len(self.targets)
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

    def _make_dataset(self, root, select_domains=None):
        if select_domains is not None:
            assert set(select_domains) <= set(os.listdir(root)), 'Provided domain directories could not be found'
            domains = select_domains
        else:
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

class PairedDataset(SingleDataset):
    """
    This dataset returns images from two domains.
    """
    def __init__(self, args, return_paths=False) -> None:
        super(PairedDataset, self).__init__(args, return_paths)
        if self.args.select_domains is not None:
            assert len(self.args.select_domains) >= 2 

    def __getitem__(self, index):
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

class PairedImageDataset(SingleDataset):
    """
    This dataset return images and their labels as an index
    """
    def __init__(self, args, return_paths=False) -> None:
        self.args = args
        self.root = os.path.join(args.dataroot, args.mode)
        self.dataset, self.targets, self.target_names = self._make_dataset(self.root, self.args.select_domains)
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

    def __getitem__(self, index):
        y1, y2 = random.choice(self.targets, 2, replace=False)
        # get image
        x1_src = self.dataset[y1][index % len(self.dataset[y1])]
        x2_src = self.dataset[y2][index % len(self.dataset[y2])]
        x1 = self.load_image(x1_src)
        x2 = self.load_image(x2_src)
        y1 = torch.tensor(y1, dtype=torch.long)
        y2 = torch.tensor(y2, dtype=torch.long)
        if self.return_paths:
            return {'x1':x1, 'x2':x2, 'y1':y1, 'y2':y2, 'x1_path':x1_src, 'x2_path':x2_src}
        return {'x1':x1, 'x2':x2, 'y1':y1, 'y2':y2}

class VideoDataset(Dataset):
    """reads the video and returns frames"""
    def __init__(self, root, transform=None):
        super(VideoDataset, self).__init__()
        self.filepath = root
        self.transforms = transform
        if self.transforms is None:
            self.transforms = transforms.Compose([transforms.ToTensor()])
        self.cam = cv2.VideoCapture(self.filepath)
    
    def __len__(self):
        return int(self.cam.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    def __getitem__(self, index):
        # loop back to index 0 if run out of frames
        index = index % len(self)
        if self.cam.isOpened():
            self.cam.set(1, index)
            out, frame = self.cam.read()
            if out:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = self.transforms(frame)
            else:
                raise RuntimeError('Frame not read. Please check the frame number')
        else:
            raise RuntimeError("Camera is not opened")
        return frame