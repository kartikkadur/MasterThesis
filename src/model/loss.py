import torch
import torch.nn as nn

from model.networks import FeatureExtractor, VGGGenerator, init_net


class GANLoss(nn.Module):
    """
    Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'bce':
            self.loss = nn.BCELoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla', 'bce']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class RegressionLoss(object):
    """
    computes regression loss
    """
    def __init__(self):
        pass
    
    def __call__(self, d_out, x_real):
        batch_size = x_real.size(0)
        grad_dout = torch.autograd.grad(
                    outputs=d_out.sum(), inputs=x_real,
                    create_graph=True, retain_graph=True, only_inputs=True
                    )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_real.size())
        reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
        return reg


class PerceptualLoss(nn.Module):
    """
    Constructs a VGG 19 model used as a loss network
    """
    def __init__(self, model:str, layers:int, use_l1:bool = False, checkpoint:str = None, url:str = None):
        """
        Parameters:
            model : the vgg model used for calculating feature loss, options : [vgg13, vgg16, vgg19]
            layes : the layer number from which the feature is extracted.
            checkpoint : pretrained checkpoint of model
            url : checkpoint url to download and load.
        """
        super(PerceptualLoss, self).__init__()
        with torch.no_grad():
            model = VGGGenerator(model, checkpoint=checkpoint, batch_norm=True)
            model = self.unwrap_model(model)
            blocks = []
            for layer in layers:
                block = init_net(nn.Sequential(*model[:layer]), init_type=None)
                blocks.append(block)

        # extracted blocks
        self.blocks = blocks

        # define loss type
        if use_l1:
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.MSELoss()

    def unwrap_model(self, model, layers=[]):
        """
        unwraps the nn.Sequential layers and returns individual layers of the model.
        Parameters:
            model : the model to be unwraped
            layers : always an empty list
        """
        for module in model.children():
            # recursively unwrap if the module is nn.Sequential
            if isinstance(module, nn.Sequential):
                self.unwrap_model(module, layers)
            # if its a child module then add to the final list
            if list(module.children()) == []:
                layers.append(module)
        return layers
    
    def __call__(self, x, y):
        """
        Parameters:
            x : input tensor
            y : target tensor
        """
        loss = 0.0
        for block in self.blocks:
            x_out = block(x)
            y_out = block(y)
            loss += self.loss(x_out ,y_out)
        return loss
        
