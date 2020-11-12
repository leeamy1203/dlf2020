import os

import numpy as np
import pandas as pd

from src.data import DATA_DIR


def read_3d_txt_file(filename: str) -> np.ndarray:
    """
    Reads the text file and returns an numpy array of (frames, 50=number of coordinates for body part, 3=(x,y,z))
    """
    data = np.loadtxt(os.path.join(DATA_DIR, filename))
    frames = data.shape[0]
    return data.reshape(frames, -1, 3)


def get_wlasl_words() -> np.ndarray:
    """
    Returns an array of words in the WLASL dataset
    """
    meta = pd.read_json(os.path.join(DATA_DIR, 'meta', 'WLASL_v0.3.json'))  # meta data with first column = gloss
    words = meta['gloss'].values
    return words

    
