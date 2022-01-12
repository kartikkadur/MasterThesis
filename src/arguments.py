import argparse
import os
import dataset
import models

from inspect import isclass
from datetime import datetime
from utils import module_to_dict
from utils import get_modules

class Arguments(object):
    """
    This is the base argument class. Subclass this class to create different arguments
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser("Arguments for the program")
        # parse args
        self.parser.add_argument('--dataroot', help='root folder of the dataset')
        self.parser.add_argument('--name', type=str, default=f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}', help='name of the experiment. It decides where to store samples and model')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--exp_dir', type=str, default='../exps', help='custom directory for storing experiment results')
        # model parameters
        self.parser.add_argument('--model', type=str, default='BaseModel', help='chooses which model to use.')
        self.parser.add_argument('--input_dim', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        self.parser.add_argument('--output_dim', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        self.parser.add_argument('--dim', type=int, default=64, help='# of gen filters in the last conv layer')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization.')
        self.parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--num_domains', type=int, default=2, help='number of domains in the dataset')
        self.parser.add_argument('--mode', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--concat', action='store_true', help='concatenate attribute features for translation')
        self.parser.add_argument('--use_dis_content', action='store_true', help='weather to use content discriminator')
        self.parser.add_argument('--latent_dim', type=int, default=8, help='size of latent dimention')
        self.parser.add_argument('--up_type', type=str, default='transpose', choices=['transpose', 'nearest', 'pixelshuffle'],
                                                                                    help='type of upsample layer to be used in decoder')
        # dataset parameters
        self.parser.add_argument('--dataset', type=str, default='PairedDataset', choices=get_modules(dataset), help='chooses how datasets are loaded.')
        self.parser.add_argument('--shuffle', action='store_true', help='if true, takes the batches randomly, else takes them in serial fashion')
        self.parser.add_argument('--num_workers', default=4, type=int, help='# threads for loading data')
        self.parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        self.parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--no_flip', action='store_true', help='then crop to this size')
        self.parser.add_argument('--select_domains', default=None, type=str, nargs='+', help='specify the perticular domains to select')
        # additional parameters
        self.parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to load')
        self.parser.add_argument('--save_logs', action='store_true', help='value indicating weather to save logs or not')

    def parse(self):
        args = self.parser.parse_args()
        # assign classes
        args.dataset = module_to_dict(dataset)[args.dataset]
        args.model = module_to_dict(models)[args.model]
        # experiment dir
        args.exp_dir = os.path.join(args.exp_dir, args.name)
        os.makedirs(args.exp_dir, exist_ok=True)
        # create checkpoint dir
        args.checkpoint_dir = os.path.join(args.exp_dir, 'checkpoints')
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        # create logs dir
        args.logdir = os.path.join(args.exp_dir, 'logs')
        os.makedirs(args.logdir, exist_ok=True)
        # create image display directory
        args.display_dir = os.path.join(args.exp_dir, 'images')
        os.makedirs(args.display_dir, exist_ok=True)
        # set GPU ids
        args.gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',') if int(gpu_id) >= 0]
        arguments = vars(args)
        with open(os.path.join(args.exp_dir, 'args.txt'), 'a') as f:
            print('\n--- Loaded arguments ---')
            for name, value in sorted(arguments.items()):
                print('%s: %s' % (str(name), str(value)))
                f.write('%s: %s\n' % (str(name), str(value)))
        return args

class TrainArguments(Arguments):
    """arguments specific for training"""
    def __init__(self):
        super(TrainArguments, self).__init__()
        self.parser.add_argument('--gen_norm', type=str, default=None, choices=['batch','instance'], help='normalization layer in generator')
        self.parser.add_argument('--dis_norm', type=str, default=None, choices=['batch','instance'], help='normalization layer in discriminator')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='learning rate parameter')
        self.parser.add_argument('--wd', type=float, default=0.0001, help='weight decay parameter')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='type of learn rate decay')
        self.parser.add_argument('--n_epoch', type=int, default=200, help='number of epochs to train')
        self.parser.add_argument('--start_epoch', type=int, default=1, help='start epoch number')
        self.parser.add_argument('--d_iter', type=int, default=3, help='num iteration to update content discriminator')
        self.parser.add_argument('--n_epoch_decay', type=int, default=100, help='epoch start decay learning rate, set -1 if no decay')
        self.parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
        self.parser.add_argument('--lambda_cls', type=float, default=1.0, help='weight for classification loss for Discriminator')
        self.parser.add_argument('--lambda_cls_G', type=float, default=5.0, help='weight for classification loss for Generator')
        self.parser.add_argument('--print_freq', type=int, default=1000, help='frquency at which the logs have to be printed to console')
        self.parser.add_argument('--save_freq', type=int, default=1000, help='frequency at which the model checkpoint has to be saved')
        self.parser.add_argument('--display_freq', type=int, default=1000, help='frequency at which the images are to be saved')
        self.parser.add_argument('--max_iter', type=float, default=float('inf'), help='maximum number of global iterations to be performed')
        self.parser.add_argument('--train_n_batch', type=float, default=float('inf'), help='max number of batches to train')
        self.parser.add_argument('--gan_mode', type=str, default='vanilla', help='which type of loss to be used for adversarial training')
        self.parser.add_argument('--resume_opt', type=str, default=None, help='path to checkpoint to load for optimizer')
        # discriminator params
        self.parser.add_argument('--ms_dis', action='store_true', help='use multiscale discriminator instead of the normal discriminator')
        self.parser.add_argument('--dis_sn', action='store_true', help='use spectral normalization in discriminator')
        self.parser.add_argument('--num_scales', type=int, default=3, help='number of downsampling to be performed in ms discriminator')
        # perceptual loss parameters
        self.parser.add_argument('--lambda_perceptual', type=float, default=1.0, help='weight for perceptual loss for Generator')
        self.parser.add_argument('--vgg_type', type=str, default='vgg19', help='vgg model to be used to calculate perceptual loss')
        self.parser.add_argument('--vgg_loss', type=str, default=None, help='loss to be used to calculate perceptual loss')
        self.parser.add_argument('--vgg_layers', type=str, nargs='+', default=['conv5_4'], help='layers to consider for perceptual loss')
        self.parser.add_argument('--layer_weights', type=float, nargs='+', default=[1.0], help='layers to consider for perceptual loss')

class TestArguments(Arguments):
    """arguments specific for test run"""
    def __init__(self):
        super(TestArguments, self).__init__()
        self.parser.add_argument('--num', type=int, default=5, help='number of outputs per image')
        self.parser.add_argument('--result_dir', type=str, default='./outputs', help='path for saving result images and models')
        self.parser.add_argument('--out_fmt', type=str, default='image', help='type of output format. one of [image, video]')
        self.parser.add_argument('--vid_fname', type=str, default='video.avi', help='name of the video file')
        self.parser.add_argument('--reference', type=str, default=None, help='path to the reference image for extracting the style from')
        self.parser.add_argument('--trg_cls', type=int, default=-1, help='required target class')

    def parse(self):
        args = self.parser.parse_args()
        arguments = vars(args)
        # create directories
        os.makedirs(args.result_dir, exist_ok=True)
        # display directory
        if 'image' in args.out_fmt:
            args.display_dir = os.path.join(args.result_dir, 'images')
        elif 'video' in args.out_fmt:
            args.display_dir = os.path.join(args.result_dir, 'videos')
        os.makedirs(args.display_dir, exist_ok=True)
        # set gpu ids
        args.gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',') if int(gpu_id) >= 0]
        # set mode to test
        args.mode = 'test'
        # print args
        print('\n--- Load test arguments ---')
        for name, value in sorted(arguments.items()):
            print('%s: %s' % (str(name), str(value)))
        args.dis_scale = 3
        args.dis_norm = None
        args.dis_sn = False
        # assign model class
        args.model = module_to_dict(models)[args.model]
        with open(os.path.join(args.result_dir, 'args.txt'), 'a') as f:
            print('\n--- Loaded arguments ---')
            for name, value in sorted(arguments.items()):
                print('%s: %s' % (str(name), str(value)))
                f.write('%s: %s\n' % (str(name), str(value)))
        return args