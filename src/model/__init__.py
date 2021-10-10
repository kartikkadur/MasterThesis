import importlib
import model
from model.base_model import Model
from model.attention_gan import AttentionGANModel
from model.aggan import AGGAN
from model.spagan import SPAGAN
from model.weatherGAN import WeatherGANModel
from model.vgg import VGG
from utils.tools import module_to_dict

def create_model(args):
    """
    create model and load weights if provided
    """
    model = args.model(args)
    if args.load_checkpoint:
        model.load(args.load_checkpoint)
    return model
    