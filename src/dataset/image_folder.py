import os
import torch.utils.data as data

from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(path, max_dataset_size=float("inf")):
    assert os.path.isdir(path), '%s is not a valid directory' % path

    images = [
                os.path.join(fdir, fname)
                for fdir, _, fnames in sorted(os.walk(path)) 
                for fname in fnames if is_image_file(fname)
            ]
    return images[:min(max_dataset_size, len(images))]


def make_dataset_dict(path, max_dataset_size=float('inf')):
    assert os.path.isdir(path), '%s is not a valid directory' % path

    images = {}
    for fdir, _, fnames in sorted(os.walk(path)):
        if not fdir.startswith('.'):
            for fname in fnames:
                img_file = os.path.join(fdir, fname)
                if is_image_file(img_file):
                    if os.path.basename(fdir) in images.keys():
                        images[os.path.basename(fdir)] += [img_file]
                    else:
                        images[os.path.basename(fdir)] = [img_file]
    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)