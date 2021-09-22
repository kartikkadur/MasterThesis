import os
import torch

from dataset.base_dataset import Dataset, get_transform
from dataset.image_folder import make_dataset
from PIL import Image
from random import shuffle


class ClassificationDataset(Dataset):
    """
    This dataset class can load data from multiple folders/classes.
    It returns two classes passed as command line arguments --class_a and --class_b.
    """
    @staticmethod
    def modify_commandline_arguments(parser, is_train):
        return parser

    def __init__(self, args):
        """Initialize this dataset class.
        Parameters:
            args (Arguments class) -- stores all the experiment flags; needs to be a subclass of Arguments
        """
        super(ClassificationDataset, self).__init__(args)
        # variable indicating weather to select random class or not
        self.random = False
        # read the paths
        path = os.path.join(args.dataroot, args.mode)
        self.classes = [c for c in os.listdir(path) if not c.startswith('.')]
        self.paths = make_dataset(path)
        shuffle(self.paths)
        # transforms
        input_nc = self.args.input_nc
        self.transform_A = get_transform(self.args, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index (int)      -- a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_class (str)    -- class A name
            B_class (str)    -- class B name
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        # get paths
        path = self.paths[index]
        label = os.path.basename(os.path.dirname(path))
        # read images
        img = Image.open(path).convert('RGB')
        # apply image transformation
        img = self.transform_A(img)

        return {'image': img, 'label': torch.tensor(self.classes.index(label))}

    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return len(self.paths)