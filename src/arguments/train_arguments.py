from .base_arguments import Arguments


class TrainArguments(Arguments):
    """This class includes training arguments.
    It also includes shared arguments defined in Arguments.
    """

    def initialize(self, parser):
        parser = Arguments.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [args.checkpoints-dir]/[args.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest', type=int, default=5000, help='frequency of saving the latest results comparing the iteration')
        parser.add_argument('--save_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--load_epoch', type=int, default='0', help='which epoch to load?')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load model by iter_[load_iter]; otherwise, the code will load model by [epoch]')
        parser.add_argument('--start_epoch', type=int, default=1, help='the starting epoch count')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs with the initial learning rate')
        parser.add_argument('--epoch_decay', type=int, default=250, help='epoch number to start the linear decay of learning rate to zero')
        parser.add_argument('--steps_per_epoch', type=int, default=None, help='Number of iteration to run per epoch')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', choices=['vanilla', 'lsgan', 'wgangp'], help='the type of GAN objective. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr-decay-iters iterations')
        # validation parameters
        parser.add_argument('--num_val_steps', type=int, default=None, help='number of validation steps to be done.')
        parser.add_argument('--val_freq', type=int, default=None, help='number indicating the frequency of running validation')
        self.isTrain = True
        return parser