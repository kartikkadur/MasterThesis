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
    dataloader = CustomDatasetDataLoader(args)
    dataset = dataloader.load_data()
    return dataset


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
