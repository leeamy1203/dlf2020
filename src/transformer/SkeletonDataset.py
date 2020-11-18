import numpy as np
import torch
from torch.utils.data import Dataset


class SkeletonDataset(Dataset):
    """
    Dataset class
    followed: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """
    def __init__(self, data, word_embedding_size, skeleton_size):
        self.data = data
        self.pad_num = 100
        all_lengths = np.array([m["skeletons"].shape[0] for m in data])
        self.max_length = max(all_lengths)
        self.word_embedding_size = word_embedding_size
        self.skeleton_size = skeleton_size
        
    def __len__(self):
        return len(self.data)

    def pad_data(self, skeletons):
        padded_data = np.zeros((self.max_length, self.word_embedding_size))
        padded_data[:skeletons.shape[0], :skeletons.shape[1]] = skeletons
        padded_data[skeletons.shape[0]:] = self.pad_num
        return padded_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx]
        skeletons = self.pad_data(data['skeletons'])
        output = dict()
        output["src"] = data["embedding"]
        output["trg"] = skeletons
        return output
