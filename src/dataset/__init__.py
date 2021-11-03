import importlib
import torch.utils.data
import dataset
import os.path

from dataset.base_dataset import Dataset
from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def create_dataset(args):
    """
    create a dataloader and returns the loader object
    """
    dataloader = CustomDatasetDataLoader(args)
    dataset = dataloader.load_data()
    return dataset

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_image_dataset(path, max_dataset_size=float("inf")):
    assert os.path.isdir(path), '%s is not a valid directory' % path
    labels = sorted(os.listdir(path))
    images = [
                (os.path.join(fdir, fname), labels.index(os.path.basename(fdir)))
                for fdir, _, fnames in sorted(os.walk(path))
                for fname in fnames if is_image_file(fname)
            ]
    return images[:min(max_dataset_size, len(images))]


def make_image_dataset_dict(path, max_dataset_size=float('inf')):
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

############################
### dataset class imports###
############################

from dataset.multiclass_dataset import SingleDataset, AlignedDataset, ReferenceDataset
from dataset.classification_dataset import ClassificationDataset

#########################
### Custom dataloaders ###
#########################

class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, args):
        """
        Initialize this class
        """
        self.args = args
        self.dataset = args.dataset(args)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        # create dataloader
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            drop_last=True)
        # if max_dataset_size is provided calculate maximum batches
        self.max_batches = float('inf')
        if self.args.max_dataset_size != float('inf'):
            self.max_batches = self.args.max_dataset_size // self.args.batch_size

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataloader), self.max_batches)

    def __iter__(self):
        """Return a batch of data"""
        for i, batch in enumerate(self.dataloader):
            if i >= self.max_batches:
                break
            yield batch