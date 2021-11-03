import torch
import itertools

from torch.autograd import Variable
from model.networks import init_net
from utils.tools import ImagePool
from utils.tools import tensor_to_image, mask_to_heatmap, overlay, save_image
from .base_model import Model
from . import networks
from . import loss
from PIL import Image
from torchvision import transforms


class AttentionGAN(Model):
    """
    Implementation of AttentionGAN model
    """
    @staticmethod
    def modify_commandline_arguments(parser, is_train=True):
        parser.set_defaults(no_dropout=False)
        parser.add_argument('--n_updates', type=int, default=5, help='n discriminator updates per single generator update')
        parser.add_argument('--lambda_cycle', type=float, default=10.0, help='weight for cycle loss')
        parser.add_argument('--lambda_idt', type=float, default=0.5, help='weight for identity loss')
        parser.add_argument('--lambda_cls_D', type=float, default=1.0, help='classification loss for Discriminator')
        parser.add_argument('--lambda_cls_G', type=float, default=5.0, help='classification loss for Generator')
        return parser

    def __init__(self, args):
        super(AttentionGAN, self).__init__(args)
        # loss names
        self.print_losses = ['g', 'cycle', 'cls', 'feat', 'idt']
        self.visuals = ['real_src', 'real_ref', 'fake', 'rec', 'src_att']
        # define generator
        #self.models.g = networks.UnetGenerator(args.input_nc, args.output_nc, args.ngf, norm_layer=args.norm_layer, input_size=args.crop_size, num_domains=args.num_domains, feature=True)
        self.models.g = networks.ResnetGenerator(args.input_nc, args.output_nc, args.ngf, n_downsampling=2, n_blocks=9, 
                                                 norm_layer=args.norm_layer, num_domains=args.num_domains, attention=1, use_dropout=True)
        # define discriminator
        if self.isTrain:
            self.models.d = networks.NLayerDiscriminator(args.input_nc, args.ndf, norm_layer=args.norm_layer, input_size=args.crop_size, n_domain=args.num_domains)
            # define loss functions
            self.criterion.gan = loss.GANLoss(args.gan_mode).to(self.device)
            self.criterion.cycle = torch.nn.L1Loss()
            self.criterion.idt = torch.nn.L1Loss()
            self.criterion.classification = torch.nn.BCEWithLogitsLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer.g = torch.optim.Adam(self.models.g.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)
            self.optimizer.d = torch.optim.Adam(self.models.d.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)
            # image pool to hold fake images
            self.fake_pool = ImagePool(args.pool_size)
            # compile model
            super(AttentionGAN, self).compile()
        # init net
        for net in self.models:
            self.models[net] = init_net(self.models[net], args.init_type, args.init_gain, args.gpu_ids)

    def set_inputs(self, inputs):
        self.real_src = inputs['x1'].to(self.device)
        self.real_ref = inputs['x2'].to(self.device)
        self.cls_src = inputs['y1'].to(self.device)
        self.cls_ref = inputs['y2'].to(self.device)

    def forward(self):
        self.fake, self.att = self.models.g(self.real_src, self.cls_ref)

    def backward_D(self):
        # loss with real images
        pred_real, cls_real = self.models.d(self.real_ref)
        loss_d_real = self.criterion.gan(pred_real, True)
        loss_cls = self.criterion.classification(cls_real, self.cls_ref)
        # Fake
        fake = self.fake_pool.query(self.fake)
        pred_fake, cls_fake = self.models.d(fake.detach())
        loss_d_fake = self.criterion.gan(pred_fake, False)
        # add losses
        loss_d = (loss_d_real + loss_d_fake) * 0.5 + loss_cls * self.args.lambda_cls_D
        loss_d.backward()
        self.loss.d = loss_d.item()

    def backward_G(self):
        # identity loss
        fake, _ = self.models.g(self.real_src, self.cls_src)
        self.loss.idt = self.criterion.idt(self.real_src, fake) * self.args.lambda_idt
        # update generator
        d_out, cls_fake = self.models.d(self.fake.detach())
        self.loss.g = self.criterion.gan(d_out, True)
        self.loss.cls = self.criterion.classification(cls_fake, self.cls_ref) * self.args.lambda_cls_G
        # cycle consistency loss
        self.rec, _ = self.models.g(self.fake, self.cls_src)
        self.loss.cycle = self.criterion.cycle(self.rec, self.real_src) * self.args.lambda_cycle
        # combined loss and calculate gradients
        loss_g = self.loss.g + self.loss.idt + self.loss.cycle + self.loss.cls + self.loss.feat
        loss_g.backward()

    def optimize_G(self):
        self.set_requires_grad(self.models.d, self.models.d_c, grad=False)
        self._zero_grad()
        self.backward_G()
        self.optimizer.g.step()

    def optimize_D(self):
        self.set_requires_grad(self.models.d, self.models.d_c, grad=True)
        self._zero_grad()
        # discriminator
        self.backward_D()
        self.optimizer.d.step()

    def optimize_parameters(self, *args):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()
        # optimizer discriminator
        self.optimize_G()
        # optimizer generator
        self.optimize_D()

    def compute_visuals(self):
        att_src = mask_to_heatmap(self.att.data)
        self.src_att = overlay(self.real_src, att_src)