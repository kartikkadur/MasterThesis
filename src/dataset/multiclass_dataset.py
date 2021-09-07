import os.path
from dataset.base_dataset import Dataset, get_transform
from dataset.image_folder import make_dataset
from PIL import Image
import random


class MultiClassDataset(Dataset):
    """
    This dataset class can load data from multiple folders/classes.
    It returns two classes passed as command line arguments --class_a and --class_b.
    """
    @staticmethod
    def modify_commandline_arguments(parser, is_train):
        parser.add_argument('--class_a', type=str, default='A', help='name of the directory containing class A images')
        parser.add_argument('--class_b', type=str, default='B', help='name of the directory containing class B images')
        return parser

    def __init__(self, args):
        """Initialize this dataset class.
        Parameters:
            args (Arguments class) -- stores all the experiment flags; needs to be a subclass of BaseArguments
        """
        Dataset.__init__(self, args)
        self.dir_A = os.path.join(args.dataroot, args.mode, args.class_a)
        self.dir_B = os.path.join(args.dataroot, args.mode, args.class_b)

        self.A_paths = sorted(make_dataset(self.dir_A, args.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, args.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        btoA = self.args.direction == 'BtoA'
        input_nc = self.args.output_nc if btoA else self.args.input_nc
        output_nc = self.args.input_nc if btoA else self.args.output_nc
        self.transform_A = get_transform(self.args, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.args, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index (int)      -- a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]
        if not self.args.shuffle:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return max(self.A_size, self.B_size)