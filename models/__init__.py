import os

from .base_model import BaseModel
from ..utils.utils import get_class

def get_model(name):
    return get_class(name, __name__, os.path.dirname(__file__), BaseModel)