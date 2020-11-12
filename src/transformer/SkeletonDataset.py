import torch
import numpy as np
from torch.utils.data import Dataset


class SkeletonDataset(Dataset):
    """
    Dataset class
    followed: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """
    def __init__(self, data, word_embedding_size, skeleton_size):
        self.data = data
        self.word_embedding_size = word_embedding_size
        self.skeleton_size = skeleton_size
        
    def __len__(self):
        return len(self.data)
    
    def pad_data(self, skeletons):
        padded_data = np.zeros((skeletons.shape[0], self.word_embedding_size))
        padded_data[:skeletons.shape[0], :skeletons.shape[1]] = skeletons
        return padded_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx]
        data["skeletons"] = self.pad_data(data['skeletons'])
        return data
