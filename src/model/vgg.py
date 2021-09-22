import torch

from model.networks import init_net
from .base_model import Model
from . import networks
from collections import OrderedDict


class VGG(Model):
    """
    use predifined VGG models from torchvision
    """
    @staticmethod
    def modify_commandline_arguments(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        parser.add_argument('--vgg_type', type=str, default='vgg19', choices=['vgg11', 'vgg13', 'vgg16', 'vgg19'], help='the type of vgg model to use')
        parser.add_argument('--vgg_bn', action='store_true', help='boolian indicating weather to use batch norm layer or not.')
        return parser

    def __init__(self, args):
        super(VGG, self).__init__(args)
        self.models.vgg = init_net(networks.VGGGenerator(args.vgg_type, args.num_classes, args.vgg_bn))
        self.criterion.loss = torch.nn.CrossEntropyLoss()
        self.optimizer.opt = torch.optim.Adam(self.models.vgg.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
        self.print_losses = ['loss']
        self.print_metrics = ['acc']
        self.visuals = []
        super(VGG, self).compile(loss_names=self.print_losses,
                                 metrics=self.print_metrics)

    def set_inputs(self, inputs):
        self.image = inputs['image'].to(self.device)
        self.label = inputs['label'].to(self.device)

    def forward(self):
        self.pred = self.models.vgg(self.image)

    def optimize_parameters(self):
        self.forward()
        self.loss.loss = self.criterion.loss(self.pred, self.label)
        # optimize parameters only if model is training
        if self.isTrain:
            self.optimizer.opt.zero_grad()
            self.loss.loss.val.backward()
            self.optimizer.opt.step()
        self.metrics.acc(self.pred, self.label)
    
    def load(self, checkpoint):
        ckpt_state = torch.load(checkpoint)
        model_state = self.models.vgg.state_dict()
        state_dict = OrderedDict()
        for c, m in zip(ckpt_state, model_state):
            if ckpt_state[c].shape == model_state[m].shape:
                state_dict[c] = ckpt_state[c]
        # load weights
        self.models.vgg.load_state_dict(state_dict, strict=False)

    def save(self, epoch):
        torch.save(self.models.vgg.state_dict(),
                   os.path.join(self.args.checkpoints_dir, f"{epoch}")
                  )
