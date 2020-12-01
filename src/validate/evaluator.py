import os
import logging
import pickle

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from src.data import DATA_DIR
from src.train import MODEL_DIR
from src.train.train_transformer import create_transformer


logger = logging.getLogger(__name__)


def load_model(model_file: str = 'model_weights.pt'):
    model = create_transformer()
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, model_file)))
    return model


def evaluate_word(word: str):
    
    with open(os.path.join(DATA_DIR, 'meta', 'word_to_embedding.pkl'), 'rb') as f:
        embedding_map = pickle.load(f)
        
    if word not in embedding_map:
        logger.info("Word doesn't exist in the embedding mapper.")
        return
    
    word_embedding = embedding_map[word]
    encoder_input = torch.tensor(word_embedding, dtype=torch.float32).unsqueeze(0)
    decoder_input = torch.zeros((185, 300)).unsqueeze(0)
    
    model = load_model()
    model.eval()
    
    i = 0
    with torch.no_grad():
        while (decoder_input[0, i, 360] <= 1 and i <= 183):
            src = Variable(encoder_input, requires_grad=False).cuda()
            trg = Variable(decoder_input, requires_grad=False).cuda()
            
            out = model.forward(src, trg, None)
            decoder_input[0, i + 1] = out[0, i]
            i += 1
            
