import importlib
import torch.utils.data
import dataset

from dataset.base_dataset import Dataset
from dataset.multiclass_dataset import MultiClassDataset
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
        """Initialize this class
        dataset(str) : the name of the dataset py file
        mode(str) : one of 'train' or 'test' mode
        batch_size(int) : the batch size to which the dataset will be grouped into
        num_workers(int) : number of workers that should load the data
        shuffle(bool) : True if you want to shuffle else False
        dataset_size(int) : the size of the dataset that is to be considered. default : len(dataset)
        """
        self.mode = args.phase
        self.batch_size = args.batch_size

        dataset_class = module_to_dict(dataset)[args.dataset]
        self.dataset = dataset_class(dataset)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        # create dataloader
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers)
        # get the dataset size
        self.dataset_size = dataset_size if dataset_size is not None else len(self.dataset)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.batch_size >= self.dataset_size:
                break
            yield data
