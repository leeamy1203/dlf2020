from copy import deepcopy
import time
import os
import logging
import math

import numpy as np
import pickle
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch

from src.transformer.layers import MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding
from src.transformer.model import EncoderDecoder, Encoder, Decoder, EncoderLayer, DecoderLayer, Generator
from src.transformer.optimizer import ScheduledOptim
from src.transformer.SkeletonDataset import SkeletonDataset
from src.data import DATA_DIR
from src.train import MODEL_DIR

logger = logging.getLogger(__name__)


def create_tranformer(n: int = 6,
                      input_size: int = 300,
                      hidden_size: int = 2048,
                      h: int = 10,
                      dropout: float = 0.1):
    """
    Construct transformer with the given hyperparameters
    """
    self_attn = MultiHeadAttention(n_head=h, d_model=input_size)
    src_attn = MultiHeadAttention(n_head=h, d_model=input_size)

    ff = PositionwiseFeedForward(d_in=input_size, d_hid=hidden_size,
                                 dropout=dropout)

    transformer = EncoderDecoder(
        Encoder(EncoderLayer(size=input_size,
                             feed_forward=deepcopy(ff),
                             dropout=dropout), n=1),
        Decoder(DecoderLayer(size=input_size,
                             self_attn=deepcopy(self_attn),
                             src_attn=deepcopy(src_attn),
                             feed_forward=deepcopy(ff),
                             dropout=dropout),
                PositionalEncoding(d_model=input_size, dropout=dropout),
                n=n),
        Generator(d_model=input_size, output=input_size)
    )
    return transformer
    
    
def split_data(data = list,
               val_size: float = 0.2,
               random_state: int = 42):
    train, val = train_test_split(data, test_size=val_size, random_state=random_state)
    with open(os.path.join(DATA_DIR, 'processed', 'train.pkl'), 'wb') as output:
        pickle.dump(train, output)
    with open(os.path.join(DATA_DIR, 'processed', 'val.pkl'), 'wb') as output:
        pickle.dump(val, output)
    return train, val

    
def train():
    """
    Define and train the transformer
    """
    input_size = 300
    n_warmup_steps = 4000
    batch_size = 1
    epochs = 100

    # get data
    with open(os.path.join(DATA_DIR, 'interim', '2dskeleton.pkl'), 'rb') as f:
        data = pickle.load(f)
    train, val = split_data(data)

    train_data = get_dataloader(train, batch_size=batch_size)
    val_data = get_dataloader(val, batch_size=batch_size)
    
    transformer = create_tranformer()
    print(transformer)
    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        2.0, d_model=input_size, n_warmup_steps=n_warmup_steps)
    loss = nn.MSELoss()
    
    val_loss = []
    training_loss = []
    
    for epoch_i in range(epochs):
        logger.info(f"Epoch = {epoch_i}")
        train_loss = train_epoch(transformer, train_data, loss, optimizer)
        logger.info(f"loss: {train_loss / len(train)}")
        training_loss.append([epoch_i, train_loss / len(train)])
    torch.save(transformer.state_dict(), os.path.join(MODEL_DIR, 'model_weights.pt'))
    
    
def train_epoch(model: nn.Module,
                data: DataLoader,
                loss_func,
                optimizer: ScheduledOptim):
    """
    Method for training one epoch
    """
    model.train()
    total_loss = 0
    
    for batch in tqdm(data, mininterval=2, leave=False):
        src = batch['embedding']
        trg = batch['skeletons']
        
        trg_input = trg[:, :-1]
        targets = trg[:, 1:]

        src_mask = None
        trg_mask = get_subsequent_mask(trg_input)
        
        # forward
        optimizer.zero_grad()
        pred = model(src.float(), trg_input.float(), src_mask, trg_mask)
        loss = loss_func(pred[:, :, 0:240], targets[:, :, 0:240].float())
        # backward
        loss.backward()
        optimizer.step_and_update_lr()
        if torch.isnan(loss):
            logger.error(f"NAN ")
        total_loss += loss
    return total_loss.detach().numpy()
        
    
def get_dataloader(train,
                   batch_size: int):
    skeleton_size = train[0]["skeletons"].shape[-1]
    word_embedding_size = train[0]["embedding"].shape[0]
    dataset = SkeletonDataset(data=train, word_embedding_size=word_embedding_size, skeleton_size=skeleton_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    len_s = seq.size()[1]
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)