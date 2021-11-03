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


class StarGAN(Model):
    """
    Implementation of AttentionGAN model
    """
    @staticmethod
    def modify_commandline_arguments(parser, is_train=True):
        parser.add_argument('--lambda_style', type=float, default=1, help='weight for style loss')
        parser.add_argument('--lambda_ds', type=float, default=1, help='weight for diversity loss')
        parser.add_argument('--lambda_cycle', type=float, default=1, help='weight for cycle loss')
        parser.add_argument('--lambda_reg', type=float, default=1, help='weight for regression loss')
        parser.add_argument('--latent_dim', type=int, default=16, help='size of the latent dimention')
        parser.add_argument('--style_dim', type=int, default=64, help='size of the style dimention')
        parser.add_argument('--m_lr', type=float, default=1e-6, help='learning rate for mapping network')
        return parser

    def __init__(self, args):
        super(StarGAN, self).__init__(args)
        self.print_losses = ['gen', 'dis', 'style', 'ds', 'cycle']
        self.visuals = ['x_real', 'x_fake', 'x_ref', 'x_rec']
        # generator networks
        self.models.gen = networks.StarGANGenerator(style_dim=args.style_dim, img_size=args.crop_size)
        # mapping networks
        self.models.mapnet = networks.MappingNetwork(args.latent_dim, args.style_dim, args.num_domains)
        # Style encoder networks
        self.models.style_enc = networks.StyleEncoder(args.style_dim, num_domains=args.num_domains, img_size=args.crop_size)
        if self.isTrain:
            # discriminator network
            self.models.dis = networks.StarGANDiscriminator(num_domains=args.num_domains, img_size=args.crop_size)
            # define optimizers
            self.optimizer.g = torch.optim.Adam(self.models.gen.parameters(), lr=args.lr, 
                                                betas=(args.beta1, 0.99), weight_decay=args.weight_decay)
            self.optimizer.m = torch.optim.Adam(self.models.mapnet.parameters(), lr=args.m_lr,
                                                betas=(args.beta1, 0.99), weight_decay=args.weight_decay)
            self.optimizer.e = torch.optim.Adam(self.models.style_enc.parameters(), lr=args.lr,
                                                betas=(args.beta1, 0.99), weight_decay=args.weight_decay)
            self.optimizer.d = torch.optim.Adam(self.models.dis.parameters(), lr=args.lr,
                                                betas=(args.beta1, 0.99), weight_decay=args.weight_decay)
            # define losses
            self.criterion.gan = loss.GANLoss(args.gan_mode).to(self.device)
            self.criterion.reg = loss.RegressionLoss()
        # initialize network
        for net in self.models:
            self.models[net] = init_net(self.models[net], args.init_type, args.init_gain, args.gpu_ids)
        # call compile
        super(StarGAN, self).compile(loss_names=self.print_losses)


    def set_inputs(self, inputs):
        self.x_real, self.y_org = inputs['x'].to(self.device), inputs['y'].to(self.device)
        self.x_ref1, self.x_ref2, self.y_trg = inputs['x_ref1'].to(self.device), inputs['x_ref2'].to(self.device), inputs['y_ref'].to(self.device)
        self.z_trg1, self.z_trg2 = inputs['z_trg1'].to(self.device), inputs['z_trg2'].to(self.device)

    def forward(self):
        pass

    def backward_D(self, style, masks=None):
        self.x_real.requires_grad_()
        out = self.models.dis(self.x_real, self.y_org)
        self.loss.loss_real = self.criterion.gan(out, 1)
        self.loss.loss_reg = self.criterion.reg(out, self.x_real)

        # with fake images
        with torch.no_grad():
            if not style:
                s_trg = self.models.mapnet(self.z_trg1, self.y_trg)
            else:
                s_trg = self.models.style_enc(self.x_ref1, self.y_trg)

            x_fake = self.models.gen(self.x_real, s_trg, masks=masks)
        out = self.models.dis(x_fake, self.y_trg)
        self.loss.loss_fake = self.criterion.gan(out, 0)
        self.loss.dis = self.loss.loss_real + self.loss.loss_fake + self.args.lambda_reg * self.loss.loss_reg
        self._zero_grad()
        self.loss.dis.val.backward()
        self.optimizer.d.step()

    def backward_G(self, style=False, masks=None):
        """Calculate the loss for generators G_A and G_B"""
        # adversarial loss
        if style:
            s_trg = self.models.style_enc(self.x_ref1, self.y_trg)
        else:
            s_trg = self.models.mapnet(self.z_trg1, self.y_trg)

        self.x_fake = self.models.gen(self.x_real, s_trg, masks=masks)
        out = self.models.dis(self.x_fake, self.y_trg)
        # compute adversarial loss
        self.loss.adv = self.criterion.gan(out, 1)

        # style reconstruction loss
        s_pred = self.models.style_enc(self.x_fake, self.y_trg)
        self.loss.style = torch.mean(torch.abs(s_pred - s_trg))

        # diversity sensitive loss
        if style:
            s_trg2 = self.models.style_enc(self.x_ref2, self.y_trg)
        else:
            s_trg2 = self.models.mapnet(self.z_trg2, self.y_trg)
            
        self.x_fake2 = self.models.gen(self.x_real, s_trg2, masks=masks).detach()
        self.loss.ds = torch.mean(torch.abs(self.x_fake - self.x_fake2))

        # cycle-consistency loss
        #masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
        s_org = self.models.style_enc(self.x_real, self.y_org)
        self.x_rec = self.models.gen(self.x_fake, s_org, masks=masks)
        self.loss.cycle = torch.mean(torch.abs(self.x_rec - self.x_real))

        self.loss.gen = self.loss.adv + self.args.lambda_style * self.loss.style.val \
                        - self.args.lambda_ds * self.loss.ds.val + self.args.lambda_cycle * self.loss.cycle.val
        self._zero_grad()
        self.loss.gen.val.backward()
        # step optimizer
        self.optimizer.g.step()
        if not style:
            self.optimizer.m.step()
            self.optimizer.e.step()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # train disscriminators
        self.backward_D(style=False)
        self.backward_D(style=True)
        # train generators
        self.backward_G(style=False)
        self.backward_G(style=True)
        initial_lambda_ds = self.args.lambda_ds
        if self.args.lambda_ds > 0:
                self.args.lambda_ds -= (initial_lambda_ds / self.args.n_epochs * 1000)

    def compute_visuals(self):
        pass

    def predict(self):
        pass
