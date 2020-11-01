import logging
import os
import pickle

from torchnlp.word_to_vector import FastText
import numpy as np

from src.data.importer import get_wlasl_words
from src.data import DATA_DIR

logger = logging.getLogger(__name__)


def create_trainable() -> None:
    """
    Create trainable data
     1. create word embeddings
    """
    logger.info("Transforming words to word vectors using FAIR's FastText.")
    embeddings, words = get_word_vectors()
    logger.info("Saving embeddings and words as pickle files.")
    
    with open(os.path.join(DATA_DIR, 'interim', 'embeddings.pkl'), 'wb') as emb_file:
        pickle.dump(embeddings, emb_file)
    with open(os.path.join(DATA_DIR, 'interim', 'words.pkl'), 'wb') as word_file:
        pickle.dump(words, word_file)


def get_word_vectors() -> (np.ndarray, np.ndarray):
    """
    Goes through the meta data of WLASL and creates word embeddings for each using FastText.
    See: https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.word_to_vector.html#torchnlp.word_to_vector.FastText
    Returns an array of size (# of words=2000 x dimension of embedding=300) and an array of words

    This takes awhile because it needs to load the pretrained vectors which is about 6GB of data
    """
    logger.info("Loading FastText word vectors. This will take about 15min")
    vectors = FastText()  # loading vectors this take about 15 min
    words = get_wlasl_words()
    
    embeddings = []
    for w in words:
        embeddings.append(vectors[w].numpy())
    embeddings = np.array(embeddings)
    return embeddings, np.array(words)
