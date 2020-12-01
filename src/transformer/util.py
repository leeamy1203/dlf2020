import copy

import numpy as np
import torch
from torch import nn


def clones(module, N):
    """
    Produces N layers
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    """
    Mask out subsequent positions.
    This masking ensures that the predictions for position i can only depend on the known outputs
    at positions less than i

    This is used in training
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
