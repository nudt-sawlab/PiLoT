import os
from .base_dataset import BaseDataset
from ..utils.utils import get_class

def get_dataset(name):
    return get_class(name, __name__, os.path.dirname(__file__), BaseDataset)