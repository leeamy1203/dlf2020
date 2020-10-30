import logging
import os
import yaml

from logging import config

logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def setup_logger():
    """
    Sets up the python logging module with a .yaml config
    """
    with open(os.path.join(CURRENT_DIR, 'logging.yaml'), 'r') as f:
        dict_config = yaml.safe_load(f.read())
        logging.config.dictConfig(dict_config)
