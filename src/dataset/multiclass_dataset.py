import os.path
from dataset.base_dataset import Dataset, get_transform
from dataset.image_folder import make_dataset
from dataset.image_folder import make_dataset_dict
from PIL import Image
import random


class MultiClassDataset(Dataset):
    """
    This dataset class can load data from multiple folders/classes.
    It returns two classes passed as command line arguments --class_a and --class_b.
    """
    @staticmethod
    def modify_commandline_arguments(parser, is_train):
        parser.add_argument('--class_a', type=str, default=None, help='name of the directory containing class A images')
        parser.add_argument('--class_b', type=str, default=None, help='name of the directory containing class B images')
        return parser

    def __init__(self, args):
        """Initialize this dataset class.
        Parameters:
            args (Arguments class) -- stores all the experiment flags; needs to be a subclass of Arguments
        """
        Dataset.__init__(self, args)
        # variable indicating weather to select random class or not
        self.random = False
        # read the paths
        if args.class_a is None or args.class_b is None:
            path = os.path.join(args.dataroot, args.mode)
            self.classes = [c for c in os.listdir(path) if not c.startswith('.')]
            self.paths = make_dataset_dict(path)
            self.random = True
        else:
            dir_A = os.path.join(args.dataroot, args.mode, args.class_a)
            dir_B = os.path.join(args.dataroot, args.mode, args.class_b)
            A_paths = sorted(make_dataset(dir_A, args.max_dataset_size))
            B_paths = sorted(make_dataset(dir_B, args.max_dataset_size))
            self.classes = [args.class_a, args.class_b]
            self.paths = {args.class_a : A_paths, args.class_b : B_paths}

        btoA = self.args.direction == 'BtoA'
        # dataset size will be the folder containing max images
        self.size = sum(map(len, self.paths.values())) // args.batch_size
        # transforms
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
            A_class (str)    -- class A name
            B_class (str)    -- class B name
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        if self.random:
            classes = list(self.paths.keys())
            A_class = random.choice(classes)
            B_class = random.choice([c for c in classes if c != A_class])
            # get paths
            A_path = self.paths[A_class][index % len(self.paths[A_class])]
            B_path = self.paths[B_class][index % len(self.paths[B_class])]
        else:
            A_class = self.classes[0]
            B_class = self.classes[1]
            A_path = self.paths[A_class][index % len(self.paths[A_class])]
            if not self.args.shuffle:
                index_B = index % len(self.paths[B_class])
            else:
                index_B = random.randint(0, len(self.paths[B_class]) - 1)
            B_path = self.paths[B_class][index_B]
        # read images
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_class' : self.classes.index(A_class), 'B_class' : self.classes.index(B_class), 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return self.size