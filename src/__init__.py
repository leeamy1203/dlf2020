import warnings
import os

from src.conf import config

__version__ = '0.1.0'

warnings.simplefilter(action='ignore', category=FutureWarning)

config.setup_logger()

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
