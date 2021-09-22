import importlib
import torch.utils.data
import dataset

from dataset.base_dataset import Dataset
from dataset.multiclass_dataset import MultiClassDataset
from dataset.classification_dataset import ClassificationDataset
from utils.tools import module_to_dict


def create_dataset(args):
    """
    create a dataloader and returns the loader object
    """
    data_loader = CustomDatasetDataLoader(args)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, args):
        """
        Initialize this class
        """
        self.mode = args.mode
        self.batch_size = args.batch_size
        self.dataset = args.dataset(args)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        # create dataloader
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers)
        # get the dataset size
        self.dataset_size = min(len(self.dataset), args.max_dataset_size)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return self.dataset_size

    def __iter__(self):
        """Return a batch of data"""
        for i, batch in enumerate(self.dataloader):
            if i * self.batch_size >= self.dataset_size:
                break
            yield batch
