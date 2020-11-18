from copy import deepcopy
import time
import os
import logging

import numpy as np
import pickle
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable


from src.transformer.layers import MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding
from src.transformer.model import EncoderDecoder, Encoder, Decoder, EncoderLayer, DecoderLayer, Generator
from src.transformer.optimizer import ScheduledOptim
from src.transformer.SkeletonDataset import SkeletonDataset
from src.data import DATA_DIR
from src.train import MODEL_DIR

PAD_NUM = 100
logger = logging.getLogger(__name__)


def create_transformer(n: int = 2,
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

    
def train(input_size: int = 300,
          n_warmup_steps: int = 4000,
          batch_size: int = 64,
          epochs: int = 100,
          device: str = 'cpu'):
    """
    Define and train the transformer
    """
    if torch.cuda.is_available():
        device = 'cuda'

    # get data
    with open(os.path.join(DATA_DIR, 'interim', '2dskeleton.pkl'), 'rb') as f:
        data = pickle.load(f)
    data = data[0:10]

    train_data = get_dataloader(data, batch_size=batch_size)
    eval_data = get_dataloader(data, batch_size=batch_size)
    
    transformer = create_transformer()
    
    if torch.cuda.is_available():
        transformer.cuda()
    
    print(transformer)
    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        2.0, d_model=input_size, n_warmup_steps=n_warmup_steps)
    loss = nn.MSELoss()
    
    training_loss = []
    
    for epoch_i in range(epochs):
        logger.info(f"Epoch = {epoch_i}")
        train_loss = train_epoch(transformer, train_data, loss, optimizer, device)
        logger.info(f"loss: {train_loss / len(data)}")
        training_loss.append([epoch_i, train_loss / len(data)])
        eval_epoch(transformer, eval_data)
    torch.save(transformer.state_dict(), os.path.join(MODEL_DIR, 'model_weights.pt'))
    
    
def train_epoch(model: nn.Module,
                data: DataLoader,
                loss_func,
                optimizer: ScheduledOptim,
                device: str = 'cpu'):
    """
    Method for training one epoch
    """
    model.train()
    total_loss = 0
    
    for batch in tqdm(data, mininterval=2, leave=False):
        src = batch['src'].to(device)
        trg = batch["trg"].to(device)
        trg_input = trg[:, :-1]
        targets = trg[:, 1:]
        src_mask = None
        trg_mask = get_subsequent_mask(trg_input, PAD_NUM)
        
        # forward
        optimizer.zero_grad()
        pred = model(src.float().to(device), trg_input.float().to(device), src_mask, trg_mask)
        
        # loss calculation
        skeleton_pred = pred[:, :, 0:240].contiguous().view(-1)
        skeleton_target = targets[:, :, 0:240].contiguous().view(-1)
        padding_mask = skeleton_target != PAD_NUM
        skeleton_loss = loss_func(skeleton_pred[padding_mask].float(), skeleton_target[padding_mask].float())
        
        counter_pred = pred[:, :, 240].contiguous().view(-1)
        counter_target = targets[:, :, 240].contiguous().view(-1)
        padding_mask = counter_target != PAD_NUM
        counter_loss = loss_func(counter_pred[padding_mask].float(), counter_target[padding_mask].float())
        
        loss = skeleton_loss * 0.8 + counter_loss * 0.2
        
        # backward
        loss.backward()
        optimizer.step_and_update_lr()
        total_loss += loss
    return total_loss.detach().numpy()


def eval_epoch(model, validation_data, device='cpu'):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    desc = '  - (Validation) '
    with torch.no_grad():
        all_loss = 0
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):
            src = batch['src'].to(device)
            trg = batch["trg"].to(device)
            frame_cnt = trg.shape[1]
            decoder_input = torch.zeros((1, 300)).unsqueeze(0).to(device)
            total_loss = 0
            for i in range(frame_cnt - 1):
                pred = model(src, decoder_input, None, None)
                actual = trg[0][1:i+2].unsqueeze(0)
                
                skeleton_pred = pred[:, :, 0:240].contiguous().view(-1)
                skeleton_actual = actual[:, :, 0:240].contiguous().view(-1)
                padding_mask = skeleton_actual != PAD_NUM
                loss = nn.functional.mse_loss(skeleton_pred[padding_mask].float(),
                                              skeleton_actual[padding_mask].float())
                last_pred = pred[:, -1].unsqueeze(0)
                decoder_input = torch.cat((decoder_input, last_pred), 1)
                total_loss += loss.item()
            all_loss += total_loss / frame_cnt
    return all_loss


def get_subsequent_mask(seq, pad):
    ''' For masking out the subsequent info. '''
    len_s = seq.size()[1]
    len_b = seq.size()[0]
    subsequent_mask = (1 - torch.triu(
        torch.ones((len_b, len_s, len_s), device=seq.device), diagonal=1)).bool()
    find_padding = seq[:, :, 0] != pad
    for i in range(subsequent_mask.shape[0]):
        pad_mask = np.multiply(find_padding[i][np.newaxis, :],
                               find_padding[i][:, np.newaxis]).bool()
        subsequent_mask[i] = subsequent_mask[i] & pad_mask
    return subsequent_mask


    
    
def get_dataloader(data,
                   batch_size: int):
    skeleton_size = data[0]["skeletons"].shape[-1]
    word_embedding_size = data[0]["embedding"].shape[0]
    dataset = SkeletonDataset(data=data, word_embedding_size=word_embedding_size, skeleton_size=skeleton_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
