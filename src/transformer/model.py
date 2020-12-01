from torch import nn
from src.transformer.layers import LayerNorm, SublayerConnection, PositionalEncoding
from src.transformer.util import clones
from torch.functional import F


class Encoder(nn.Module):
    """
    Core encoder is a stack of n layers.
    Given that we are only encoding a word, we'll just need one layer n =1
    """
    
    def __init__(self, layer, n=1):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
    
    
class EncoderLayer(nn.Module):
    """
    Encoder is usually made up of self-attn and feed forward.
    Going to remove self_sttention but leave the feed_forward (non linearity) and one residual connection
    TODO: might want to update this to be just the word embedding layer
    """
    
    def __init__(self, size, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.feed_forward = feed_forward
        self.sublayer = SublayerConnection(size, dropout)
        self.size = size

    def forward(self, x):
        return self.sublayer(x, self.feed_forward)


class Decoder(nn.Module):
    """
    N layer decoder with masking. N = 6
    """
    
    def __init__(self, layer, position_enc, n=6):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)
        self.position_enc = position_enc
    
    def forward(self, x, memory, src_mask, tgt_mask):
        dec_output = self.position_enc(x)
        for layer in self.layers:
            x = layer(dec_output, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    Decoder consists of self-attn, src-attn, and feed forward
    """
    
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    
    def __init__(self, encoder, decoder, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(tgt, memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """
    Define standard linear generation step. No need for softamx at the end
    """
    def __init__(self, d_model, output):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, output)

    def forward(self, x):
        return self.proj(x)
    