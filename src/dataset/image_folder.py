import os
import torch.utils.data as data

from PIL import Image
from dataset.base_dataset import Dataset, get_transform
from dataset import make_image_dataset, default_loader

class ImageFolder(Dataset):

    def __init__(self, args, return_paths=True):
        super(ImageFolder, self).__init__(args)
        self.root = os.path.join(args.dataroot, args.mode)
        self.dataset = make_image_dataset(self.root)
        if len(self.dataset) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.transform = get_transform(self.args, grayscale=(args.input_nc == 1))
        self.return_paths = return_paths

    def __getitem__(self, index):
        path, label = self.dataset[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, label, path
        else:
            return img, label

    def __len__(self):
        return len(self.dataset)