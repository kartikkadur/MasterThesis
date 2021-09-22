import importlib
import model
from model.base_model import Model
from model.attentionGAN import AttentionGANModel
from model.weatherGAN import WeatherGANModel
from model.vgg import VGG
from utils.tools import module_to_dict

def create_model(args):
    """
    create model and load weights if provided
    """
    model = args.model(args)
    