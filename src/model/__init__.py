import importlib
import model
from model.base_model import Model
from model.attentionGAN import AttentionGANModel
from utils.tools import module_to_dict


def create_model(args):
    """
    Creates a model given the command line arguments
    """
    model = module_to_dict(model)[args.model]
    instance = model(args)
    print("model [%s] was created" % type(instance).__name__)
    return instance