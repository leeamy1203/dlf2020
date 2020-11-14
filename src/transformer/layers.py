import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.autograd import Variable


class LayerNorm(nn.Module):
    """
    Construct a LayerNorm module.
    From http://nlp.seas.harvard.edu/2018/04/03/attention.html#full-model
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm. This is the LayerNorm(x + Sublayer(x))
    From http://nlp.seas.harvard.edu/2018/04/03/attention.html#full-model
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        """
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    """
    Fully connected feed-forward network (Linear + ReLU + Linear)
    The dimensionality of input and output is d_model = 512 and the inner-layer has dimensionality d_ff = 2048
    From http://nlp.seas.harvard.edu/2018/04/03/attention.html#full-model
    """
    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Typical dropout is added after activation function (ReLU)
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Modules.py
    """
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class PositionalEncoding(nn.Module):
    """
    Position Encoding
    From: http://nlp.seas.harvard.edu/2018/04/03/attention.html#full-model
    
    Should have same dimension as the embeddings so that the two can be summed
    """
    
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
    
    
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module
    From: https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py
    and http://nlp.seas.harvard.edu/2018/04/03/attention.html#full-model
    
    
    
    """

    def __init__(self, n_head, d_model):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        # making an assumption that key and value have same dimension
        self.d_k = d_model // n_head
        self.d_v = self.d_k

        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=self.d_v ** 0.5)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, -1, n_head, d_k)
        k = self.w_ks(k).view(sz_b, -1, n_head, d_k)
        v = self.w_vs(v).view(sz_b, -1, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, -1, self.d_model)
        return self.fc(q)
