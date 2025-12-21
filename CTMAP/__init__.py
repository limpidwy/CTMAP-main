# ctmap/__init__.py
from .model import CTMAP
from .dataprocess import cell_type_encoder, anndata_preprocess, generate_dataloaders

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    "CTMAP",
    "cell_type_encoder",
    "anndata_preprocess",
    "generate_dataloaders",
]